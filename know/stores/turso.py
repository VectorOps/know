import threading
import os
import libsql_client
import math
import queue
from concurrent.futures import Future
from typing import Optional, Dict, Any, List, Generic, TypeVar, Callable
import importlib.resources as pkg_resources
from pypika import Table, Query, AliasedQuery, QmarkParameter, CustomFunction, functions, analytics, Order, Case
from pypika.terms import LiteralValue, ValueWrapper, Field

from pydantic import BaseModel
from know.logger import logger

from know.models import (
    Repo,
    Package,
    File,
    Node,
    NodeSignature,
    ImportEdge,
    NodeRef,
    Modifier,
    Project,
)
from know.data import (
    AbstractRepoRepository,
    AbstractPackageRepository,
    AbstractFileRepository,
    AbstractNodeRepository,
    AbstractImportEdgeRepository,
    AbstractNodeRefRepository,
    AbstractDataRepository,
    NodeSearchQuery,
    PackageFilter,
    include_direct_descendants,
    resolve_node_hierarchy,
    NodeFilter,
    NodeRefFilter,
    ImportEdgeFilter,
    AbstractProjectRepository,
    AbstractProjectRepoRepository,
    RepoFilter,
    FileFilter,
)
from know.helpers import generate_id
from know.stores.helpers import BaseQueueWorker
from know.stores.sql import BaseSQLRepository, RawValue, apply_migrations

T = TypeVar("T", bound=BaseModel)

CREATE_MIGRATIONS_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS __migrations__ (
        name TEXT PRIMARY KEY,
        applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
"""
GET_APPLIED_MIGRATIONS_SQL = "SELECT name FROM __migrations__"

VectorDistanceCosFn = CustomFunction("vector_distance_cos", ["vec", "param"])
VectorFn = CustomFunction("vector32", ["param"])

# helpers
def _result_set_to_dict(rs: libsql_client.ResultSet) -> list[dict[str, Any]]:
    """Convert a libSQL result set to a list of dicts."""
    return [dict(zip(rs.columns, row)) for row in rs.rows]


class TursoClientWrapper(BaseQueueWorker):
    """
    Serialize all accesses to Turso to a separate worker thread.
    """
    def __init__(self, url: str, auth_token: Optional[str] = None):
        super().__init__()
        self._url = url
        self._auth_token = auth_token
        self._client: Optional[libsql_client.Client] = None

    def _initialize_worker(self) -> None:
        self._client = libsql_client.create_client_sync(self._url, auth_token=self._auth_token)

        def execute_fn(sql: str, params: Optional[list[Any]] = None):
            self._client.execute(sql, params if params else [])

        def query_fn(sql: str, params: Optional[list[Any]] = None) -> list[dict[str, Any]]:
            rs = self._client.execute(sql, params if params else [])
            return _result_set_to_dict(rs)

        apply_migrations(
            execute_fn,
            query_fn,
            "know.migrations.turso",
            CREATE_MIGRATIONS_TABLE_SQL,
            GET_APPLIED_MIGRATIONS_SQL,
        )

    def _handle_item(self, item: Any) -> None:
        sql, params, fut = item
        if fut.set_running_or_notify_cancel():
            try:
                assert self._client is not None
                rs = self._client.execute(sql, params)
                fut.set_result(_result_set_to_dict(rs))
            except Exception as exc:
                fut.set_exception(exc)

    def _cleanup(self) -> None:
        if self._client:
            self._client.close()
            self._client = None

    def execute(self, sql, params=None):
        fut = Future()
        self._queue.put((sql, params, fut))
        return fut.result()

# generic base repository
class _TursoBaseRepo(BaseSQLRepository[T]):
    table: str

    def __init__(self, client: "TursoClientWrapper"):
        self.client = client
        self._table = Table(self.table)

    def _get_query(self, q):
        parameter = QmarkParameter()
        sql = q.get_sql(parameter=parameter)
        return sql, parameter.get_parameters()

    def _execute(self, q):
        sql, args = self._get_query(q)
        return self.client.execute(sql, args)

    # CRUD
    def get_by_id(self, item_id: str) -> Optional[T]:
        q = Query.from_(self._table).select("*").where(self._table.id == item_id)
        rows = self._execute(q)
        return self.model(**self._deserialize_data(rows[0])) if rows else None

    def get_list_by_ids(self, item_ids: list[str]) -> list[T]:
        if not item_ids:
            return []
        q = Query.from_(self._table).select("*").where(self._table.id.isin(item_ids))
        rows = self._execute(q)
        return [self.model(**self._deserialize_data(r)) for r in rows]

    def create(self, item: T) -> T:
        data = self._serialize_data(item.model_dump(exclude_none=True))

        keys = data.keys()
        q = Query.into(self._table).columns([self._table[k] for k in keys]).insert([data[k] for k in keys])
        self._execute(q)

        return item

    def update(self, item_id: str, data: Dict[str, Any]) -> Optional[T]:
        if not data:
            return self.get_by_id(item_id)

        data = self._serialize_data(data)

        q = Query.update(self._table).where(self._table.id == item_id)
        for k, v in data.items():
            q = q.set(k, v)

        self._execute(q)

        return self.get_by_id(item_id)

    def delete(self, item_id: str) -> bool:
        q = Query.from_(self._table).where(self._table.id == item_id).delete()
        self._execute(q)
        return True

class TursoProjectRepo(_TursoBaseRepo[Project], AbstractProjectRepository):
    table = "projects"
    model = Project

    def get_by_name(self, name: str) -> Optional[Project]:
        q = Query.from_(self._table).select("*").where(self._table.name == name)
        rows = self._execute(q)
        return self.model(**self._deserialize_data(rows[0])) if rows else None


class TursoProjectRepoRepo(AbstractProjectRepoRepository):
    table = "project_repos"

    def __init__(self, client: "TursoClientWrapper"):
        self.client = client
        self._table = Table(self.table)

    def _get_query(self, q):
        parameter = QmarkParameter()
        sql = q.get_sql(parameter=parameter)
        return sql, parameter.get_parameters()

    def _execute(self, q):
        sql, args = self._get_query(q)
        return self.client.execute(sql, args)

    def get_repo_ids(self, project_id: str) -> List[str]:
        q = (
            Query.from_(self._table)
            .select(self._table.repo_id)
            .where(self._table.project_id == project_id)
        )
        rows = self._execute(q)
        return [r["repo_id"] for r in rows]

    def add_repo_id(self, project_id: str, repo_id: str) -> None:
        q = (
            Query.into(self._table)
            .columns(self._table.id, self._table.project_id, self._table.repo_id)
            .insert(generate_id(), project_id, repo_id)
        )
        try:
            self._execute(q)
        except Exception as e:
            # This can happen if the repo is already associated with the project,
            # which is fine. Check for unique constraint violation.
            if "UNIQUE constraint failed" not in str(e):
                raise

# ---------------------------------------------------------------------------
# concrete repositories
# ---------------------------------------------------------------------------

class TursoRepoRepo(_TursoBaseRepo[Repo], AbstractRepoRepository):
    table = "repos"
    model = Repo

    def get_by_name(self, name: str) -> Optional[Repo]:
        """Get a repo by its name."""
        q = Query.from_(self._table).select("*").where(self._table.name == name)
        rows = self._execute(q)
        return self.model(**self._deserialize_data(rows[0])) if rows else None

    def get_by_path(self, root_path: str) -> Optional[Repo]:
        q = Query.from_(self._table).select("*").where(self._table.root_path == root_path)
        rows = self._execute(q)
        return self.model(**self._deserialize_data(rows[0])) if rows else None

    def get_list(self, flt: RepoFilter) -> list[Repo]:
        q = Query.from_(self._table).select(self._table.star)
        if flt.project_id:
            project_repos = Table("project_repos")
            subq = (
                Query.from_(project_repos)
                .select(project_repos.repo_id)
                .where(project_repos.project_id == flt.project_id)
            )
            q = q.where(self._table.id.isin(subq))

        rows = self._execute(q)
        return [self.model(**self._deserialize_data(r)) for r in rows]


class TursoPackageRepo(_TursoBaseRepo[Package], AbstractPackageRepository):
    table = "packages"
    model = Package

    def __init__(self, client, file_repo: "TursoFileRepo"):  # type: ignore
        super().__init__(client)
        self._file_repo = file_repo

    def get_by_physical_path(self, repo_id: str, root_path: str) -> Optional[Package]:
        q = Query.from_(self._table).select("*").where(
            (self._table.repo_id == repo_id) & (self._table.physical_path == root_path)
        )
        rows = self._execute(q)
        return self.model(**self._deserialize_data(rows[0])) if rows else None

    def get_by_virtual_path(self, repo_id: str, root_path: str) -> Optional[Package]:
        q = Query.from_(self._table).select("*").where(
            (self._table.repo_id == repo_id) & (self._table.virtual_path == root_path)
        )
        rows = self._execute(q)
        return self.model(**self._deserialize_data(rows[0])) if rows else None

    def get_list(self, flt: PackageFilter) -> list[Package]:
        q = Query.from_(self._table).select("*")

        if flt.repo_ids:
            q = q.where(self._table.repo_id.isin(flt.repo_ids))

        rows = self._execute(q)

        return [self.model(**self._deserialize_data(r)) for r in rows]

    def delete_orphaned(self) -> None:
        files_tbl = Table("files")
        subq = Query.from_(files_tbl).select(files_tbl.package_id).where(files_tbl.package_id.notnull())

        q = Query.from_(self._table).where(self._table.id.notin(subq)).delete()
        self._execute(q)


class TursoFileRepo(_TursoBaseRepo[File], AbstractFileRepository):
    table = "files"
    model = File

    def get_by_path(self, repo_id: str, path: str) -> Optional[File]:
        q = Query.from_(self._table).select("*").where(
            (self._table.path == path) & (self._table.repo_id == repo_id)
        )
        rows = self._execute(q)
        return self.model(**self._deserialize_data(rows[0])) if rows else None

    def get_list(self, flt: FileFilter) -> list[File]:
        q = Query.from_(self._table).select("*")

        if flt.repo_ids:
            q = q.where(self._table.repo_id.isin(flt.repo_ids))
        if flt.package_id:
            q = q.where(self._table.package_id == flt.package_id)

        rows = self._execute(q)
        return [self.model(**self._deserialize_data(r)) for r in rows]


class TursoNodeRepo(_TursoBaseRepo[Node], AbstractNodeRepository):
    table = "nodes"
    fts_table_name = "nodes_fts"
    model = Node

    _json_fields = {"signature", "modifiers"}
    _compress_fields = {"body"}
    _field_parsers = {
        "signature": lambda v: NodeSignature(**v) if v is not None else None,
        "modifiers": lambda v: [Modifier(m) for m in v] if v is not None else [],
    }

    RRF_K: int = 60          # tuning-parameter k (see RRF paper)
    RRF_CODE_WEIGHT: float = 0.7
    RRF_FTS_WEIGHT:  float = 0.3

    def __init__(self, client: "TursoClientWrapper"):
        super().__init__(client)
        self._fts_table = Table(self.fts_table_name)

    def create(self, item: Node) -> Node:
        return super().create(item)

    def update(self, item_id: str, data: Dict[str, Any]) -> Optional[Node]:
        return super().update(item_id, data)

    def delete(self, item_id: str) -> bool:
        return super().delete(item_id)

    def search(self, query: NodeSearchQuery) -> list[Node]:
        q = Query.from_(self._table)

        # Candidates
        candidates = (
            Query.
            from_(self._table).
            select(self._table.id, self._table.repo_id, self._table.embedding_code_vec)
        )

        if query.repo_ids:
            q = q.where(self._table.repo_id.isin(query.repo_ids))
        if query.symbol_name:
            q = q.where(functions.Lower(self._table.name) == query.symbol_name.lower())
        if query.kind:
            q = q.where(self._table.kind == query.kind)
        if query.visibility:
            q = q.where(self._table.visibility == query.visibility)

        # Determine which search dimensions are provided
        has_fts = bool(query.doc_needle)
        has_embedding = bool(query.embedding_query)

        q = q.with_(candidates, "candidates")
        aliased_candidates = AliasedQuery("candidates")

        # unified CTE / query construction
        if has_embedding:
            embedding_query_str = str(query.embedding_query)
            rank_code_scores = (
                Query.
                from_(aliased_candidates).
                select(
                    aliased_candidates.id,
                    VectorDistanceCosFn(
                        Field("embedding_code_vec"),
                        VectorFn(RawValue(embedding_query_str))
                    ).as_("dist"))
            )

            aliased = AliasedQuery("rank_code_scores")

            rank_code = (
                Query.
                from_(aliased).
                select(aliased.id, analytics.RowNumber().orderby(aliased.dist, order=Order.asc).as_("code_rank")).
                where(aliased.dist <= 0.6) # cosine distance
            )

            q = q.with_(rank_code_scores, "rank_code_scores").with_(rank_code, "rank_code")

        if has_fts:
            rank_fts_scores = (
                Query.
                from_(self._fts_table).
                join(aliased_candidates).on(self._fts_table.id == aliased_candidates.id).
                select(
                    self._fts_table.id,
                    RawValue("bm25(nodes_fts)").as_("score")
                ).
                where(self._fts_table.match(query.doc_needle))
            )

            aliased = AliasedQuery("rank_fts_scores")

            rank_fts = (
                Query.
                from_(aliased).
                select(aliased.id, analytics.RowNumber().orderby(aliased.score, order=Order.desc).as_("fts_rank")).
                where(aliased.score.notnull())
            )

            q = q.with_(rank_fts_scores, "rank_fts_scores").with_(rank_fts, "rank_fts")

        has_ranking = has_fts or has_embedding
        if has_ranking:
            union_parts = []

            if has_embedding:
                aliased = AliasedQuery("rank_code")
                union_parts.append(
                    Query.
                    from_(aliased).
                    select(
                        aliased.id,
                        (LiteralValue(self.RRF_CODE_WEIGHT) / (LiteralValue(self.RRF_K) + aliased.code_rank)).as_("score"))
                )

            if has_fts:
                aliased = AliasedQuery("rank_fts")
                union_parts.append(
                    Query.
                    from_(aliased).
                    select(
                        aliased.id,
                        (LiteralValue(self.RRF_FTS_WEIGHT) / (LiteralValue(self.RRF_K) + aliased.fts_rank)).as_("score")
                    )
                )

            rrf_scores = union_parts[0]
            for p in union_parts[1:]:
                rrf_scores = rrf_scores.union_all(p)  # type: ignore[assignment]

            aliased = AliasedQuery("rrf_scores")

            rrf_final = (
                Query.
                from_(aliased).
                select(aliased.id, functions.Sum(aliased.score).as_("score")).
                groupby(aliased.id)
            )

            aliased_scores = AliasedQuery("rrf_final")
            aliased_fused = AliasedQuery("fused")

            score_col = aliased_scores.score
            if query.boost_repo_id and query.repo_boost_factor != 1.0:
                score_col = Case().when(
                    aliased_candidates.repo_id == query.boost_repo_id,
                    aliased_scores.score * query.repo_boost_factor
                ).else_(aliased_scores.score)


            fused = (
                Query.
                from_(aliased_candidates).
                join(aliased_scores).on(aliased_candidates.id == aliased_scores.id).
                select(aliased_scores.id, score_col.as_("rrf_score"))
            )

            q = (
                q.with_(rrf_scores, "rrf_scores").
                with_(rrf_final, "rrf_final").
                with_(fused, "fused").
                join(aliased_fused).on(self._table.id == aliased_fused.id).
                select(self._table.star, aliased_fused.rrf_score).
                orderby(aliased_fused.rrf_score, order=Order.desc).
                orderby(self._table.name)
            )
        else:
            q = (
                q.select("*").
                join(aliased_candidates).on(self._table.id == aliased_candidates.id).
                orderby(self._table.name)
            )

        limit  = query.limit  if query.limit  is not None else 20
        offset = query.offset if query.offset is not None else 0

        q = q.limit(limit).offset(offset)

        rows = self._execute(q)
        syms = [self.model(**self._deserialize_data(r)) for r in rows]
        syms = include_direct_descendants(self, syms)
        return syms

    def delete_by_file_id(self, file_id: str) -> None:
        q = Query.from_(self._table).where(self._table.file_id == file_id).delete()
        self._execute(q)

    def get_list_by_ids(self, symbol_ids: list[str]) -> list[Node]:
        syms = super().get_list_by_ids(symbol_ids)
        resolve_node_hierarchy(syms)
        return syms

    def get_list(self, flt: NodeFilter) -> list[Node]:
        q = Query.from_(self._table).select("*")

        if flt.parent_ids:
            q = q.where(self._table.parent_node_id.isin(flt.parent_ids))
        if flt.repo_ids:
            q = q.where(self._table.repo_id.isin(flt.repo_ids))
        if flt.file_id:
            q = q.where(self._table.file_id == flt.file_id)
        if flt.package_id:
            q = q.where(self._table.package_id == flt.package_id)
        if flt.kind:
            q = q.where(self._table.kind == flt.kind)
        if flt.visibility:
            q = q.where(self._table.visibility == flt.visibility)
        if flt.has_embedding is True:
            q = q.where(self._table.embedding_code_vec.notnull())
        elif flt.has_embedding is False:
            q = q.where(self._table.embedding_code_vec.isnull())
        if flt.top_level_only:
            q = q.where(self._table.parent_node_id.isnull())

        if flt.offset:
            q = q.offset(flt.offset)
        if flt.limit:
            q = q.limit(flt.limit)

        rows = self._execute(q)
        syms = [self.model(**self._deserialize_data(r)) for r in rows]
        resolve_node_hierarchy(syms)
        return syms


class TursoImportEdgeRepo(_TursoBaseRepo[ImportEdge], AbstractImportEdgeRepository):
    table = "import_edges"
    model = ImportEdge

    def get_list(self, flt: ImportEdgeFilter) -> list[ImportEdge]:
        q = Query.from_(self._table).select("*")

        if flt.source_package_id:
            q = q.where(self._table.from_package_id == flt.source_package_id)
        if flt.source_file_id:
            q = q.where(self._table.from_file_id == flt.source_file_id)
        if flt.repo_ids:
            q = q.where(self._table.repo_id.isin(flt.repo_ids))

        rows = self._execute(q)
        return [self.model(**self._deserialize_data(r)) for r in rows]


class TursoNodeRefRepo(_TursoBaseRepo[NodeRef], AbstractNodeRefRepository):
    table = "node_refs"
    model = NodeRef

    def get_list(self, flt: NodeRefFilter) -> list[NodeRef]:
        q = Query.from_(self._table).select("*")

        if flt.file_id:
            q = q.where(self._table.file_id == flt.file_id)
        if flt.package_id:
            q = q.where(self._table.package_id == flt.package_id)
        if flt.repo_ids:
            q = q.where(self._table.repo_id.isin(flt.repo_ids))

        rows = self._execute(q)
        return [self.model(**self._deserialize_data(r)) for r in rows]

    def delete_by_file_id(self, file_id: str) -> None:
        q = Query.from_(self._table).where(self._table.file_id == file_id).delete()
        res = self._execute(q)

# Data-repository
class TursoDataRepository(AbstractDataRepository):
    """
    Main entry point for Turso/libSQL.
    """
    def __init__(self, db_url: str | None = None, auth_token: Optional[str] = None):
        """
        Parameters
        ----------
        db_url : str | None
            URL to the database (e.g., "file:local.db", "libsql://...")
            If None, an in-memory database is used.
        auth_token: str | None
            Auth token for Turso platform databases.
        """
        if db_url is None:
            db_url = "file::memory:"

        if db_url.startswith("file:") and not db_url == "file::memory:":
            db_path = db_url.split(":", 1)[1]
            os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)

        self._client = TursoClientWrapper(url=db_url, auth_token=auth_token)
        self._client.start()

        # build repositories (some need cross-references)
        self._project_repo = TursoProjectRepo(self._client)
        self._prj_repo_repo = TursoProjectRepoRepo(self._client)
        self._file_repo = TursoFileRepo(self._client)
        self._package_repo = TursoPackageRepo(self._client, self._file_repo)
        self._repo_repo = TursoRepoRepo(self._client)
        self._symbol_repo = TursoNodeRepo(self._client)
        self._edge_repo = TursoImportEdgeRepo(self._client)
        self._symbolref_repo = TursoNodeRefRepo(self._client)

    def close(self):
        self._client.close()

    @property
    def project(self) -> AbstractProjectRepository:
        return self._project_repo

    @property
    def prj_repo(self) -> AbstractProjectRepoRepository:
        return self._prj_repo_repo

    @property
    def repo(self) -> AbstractRepoRepository:
        return self._repo_repo

    @property
    def package(self) -> AbstractPackageRepository:
        return self._package_repo

    @property
    def file(self) -> AbstractFileRepository:
        return self._file_repo

    @property
    def symbol(self) -> AbstractNodeRepository:
        return self._symbol_repo

    @property
    def importedge(self) -> AbstractImportEdgeRepository:
        return self._edge_repo

    @property
    def symbolref(self) -> AbstractNodeRefRepository:
        return self._symbolref_repo

    def refresh_full_text_indexes(self) -> None:
        pass
