import threading
import os
import duckdb
import json
import pandas as pd
import math
import zlib
import queue
from concurrent.futures import Future
from typing import Optional, Dict, Any, List, Generic, TypeVar, Type, Callable
import importlib.resources as pkg_resources
from datetime import datetime, timezone
from pypika import Table, Query, AliasedQuery, QmarkParameter, CustomFunction, functions, analytics, Order
from pypika.terms import ValueWrapper, LiteralValue

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
    ImportEdgeFilter,
    AbstractProjectRepository,
    AbstractProjectRepoRepository,
    RepoFilter,
)
from know.data import NodeRefFilter

T = TypeVar("T", bound=BaseModel)

MatchBM25Fn = CustomFunction('fts_main_nodes.match_bm25', ['id', 'query'])
ArrayCosineSimilarityFn = CustomFunction("array_cosine_similarity", ["vec", "param"])


class RawValue(ValueWrapper):
     def get_value_sql(self, **kwargs: Any) -> str:
        return self.value


# helpers
def _apply_migrations(conn: duckdb.DuckDBPyConnection) -> None:
    # ensure bookkeeping table exists
    conn.execute("""
        CREATE TABLE IF NOT EXISTS __migrations__ (
            name TEXT PRIMARY KEY,
            applied_at TIMESTAMP DEFAULT NOW()
        );
    """)
    applied_rows = _row_to_dict(conn.execute("SELECT name FROM __migrations__"))
    already_applied = {r["name"] for r in applied_rows}

    with pkg_resources.as_file(pkg_resources.files("know.migrations.duckdb")) as mig_root:
        sql_files = sorted(p for p in mig_root.iterdir() if p.suffix == ".sql")

        for file_path in sql_files:
            if file_path.name in already_applied:
                continue
            sql = file_path.read_text()
            conn.execute(sql)

            conn.execute("INSERT INTO __migrations__(name, applied_at) VALUES (?, ?)",
                         [file_path.name, datetime.now(timezone.utc)])


def _row_to_dict(rel) -> list[dict[str, Any]]:
    """
    Convert a DuckDB relation to List[Dict] via a pandas DataFrame.
    Using DataFrame avoids the manual column-name handling and is faster.
    """
    df = rel.df()
    records = df.to_dict(orient="records")  # [] when df is empty
    for row in records:
        for k, v in row.items():
            # convert scalar NaN / pandas.NA → None, but skip sequences/arrays
            if isinstance(v, float) and math.isnan(v):
                row[k] = None
                continue
            try:
                is_na = pd.isna(v)          # may return array for sequences
            except Exception:
                is_na = False
            if isinstance(is_na, bool) and is_na:
                row[k] = None
    return records


class DuckDBThreadWrapper:
    """
    DuckDB is not great with threading, initialize connection and serialize all accesses to DuckDB
    to a separate worker thread. Without this random lockups in execute() were happening even
    with cursor.
    """
    def __init__(self, db_path: Optional[str] = None):
        self._queue: queue.Queue[Optional[tuple[str, Any, Future]]] = queue.Queue()
        self._db_path = db_path
        self._conn = None
        self._thread = None

    def start(self):
        self._init = Future()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        return self._init.result()

    def _worker(self):
        try:
            self._conn = duckdb.connect()

            self._conn.execute("INSTALL vss")
            self._conn.execute("LOAD vss")

            self._conn.execute("INSTALL fts")
            self._conn.execute("LOAD fts")

            # TODO: SQL injection?
            if self._db_path:
                self._conn.execute(f"ATTACH '{self._db_path}' as db")
                self._conn.execute("USE db")

            _apply_migrations(self._conn)

            self._init.set_result(True)
        except Exception as ex:
            self._init.set_exception(ex)
            return

        while True:
            item = self._queue.get()
            if item is None:
                break
            sql, params, fut = item
            if fut.set_running_or_notify_cancel():
                 try:
                      fut.set_result(_row_to_dict(self._conn.execute(sql, params)))
                 except Exception as exc:
                      fut.set_exception(exc)
            self._queue.task_done()

        if self._conn:
            self._conn.close()

    def execute(self, sql, params=None):
        fut = Future()
        self._queue.put((sql, params, fut))
        return fut.result()

    def close(self):
        if self._thread:
            self._queue.put(None)
            self._thread.join()


# binary-compression helpers
_UNCOMPRESSED_PREFIX = b"\x00"           # 1-byte marker → raw payload follows
_COMPRESSED_PREFIX   = b"\x01"           # 1-byte marker → zlib-compressed payload
_MIN_COMPRESS_LEN    = 50                # threshold in *bytes*

def _compress_blob(data: bytes) -> bytes:
    """Return data prefixed & (optionally) zlib-compressed for storage."""
    if len(data) <= _MIN_COMPRESS_LEN:
        return _UNCOMPRESSED_PREFIX + data
    return _COMPRESSED_PREFIX + zlib.compress(data)

def _decompress_blob(blob: bytes) -> bytes:
    """Undo `_compress_blob`."""
    if not blob:
        return blob
    prefix, payload = blob[:1], blob[1:]
    if prefix == _COMPRESSED_PREFIX:
        return zlib.decompress(payload)
    if prefix == _UNCOMPRESSED_PREFIX:
        return payload
    # legacy / unknown prefix → return as-is
    return blob

# generic base repository
class _DuckDBBaseRepo(Generic[T]):
    table: str
    model: Type[T]

    _json_fields: set[str] = set()
    _field_parsers: dict[str, Callable[[Any], Any]] = {}
    _compress_fields: set[str] = set()

    def __init__(self, conn: "DuckDBThreadWrapper"):
        self.conn = conn
        self._table = Table(self.table)

    def _serialize_data(self, data: dict[str, Any]) -> dict[str, Any]:
        for fld in self._json_fields:
            if fld in data and data[fld] is not None:
                val = data[fld]
                # support Pydantic models
                if hasattr(val, "model_dump"):
                    val = val.model_dump(exclude_none=False)
                data[fld] = json.dumps(val)

        for fld in self._compress_fields:
            if fld in data and data[fld] is not None:
                raw = data[fld]
                if isinstance(raw, str):
                    raw = raw.encode("utf-8")
                data[fld] = _compress_blob(bytes(raw))

        for k, v in data.items():
            if isinstance(v, list):
                data[k] = RawValue(v)

        return data

    def _deserialize_data(self, row: dict[str, Any]) -> dict[str, Any]:
        for fld in self._json_fields:
            if fld in row and row[fld] is not None:
                parsed = json.loads(row[fld])
                parser = self._field_parsers.get(fld)
                row[fld] = parser(parsed) if parser else parsed

        for fld in self._compress_fields:
            if fld in row and row[fld] is not None:
                blob = bytes(row[fld])
                text = _decompress_blob(blob)
                try:
                    row[fld] = text.decode("utf-8")
                except Exception:
                    row[fld] = text

        return row

    def _get_query(self, q):
        parameter = QmarkParameter()
        sql = q.get_sql(parameter=parameter)
        return sql, parameter.get_parameters()

    def _execute(self, q):
        sql, args = self._get_query(q)
        return self.conn.execute(sql, args)

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

        # TODO: Unify helpers
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

class DuckDBProjectRepo(_DuckDBBaseRepo[Project], AbstractProjectRepository):
    table = "projects"
    model = Project

    def get_by_name(self, name: str) -> Optional[Project]:
        q = Query.from_(self._table).select("*").where(self._table.name == name)
        rows = self._execute(q)
        return self.model(**self._deserialize_data(rows[0])) if rows else None


class DuckDBProjectRepoRepo(AbstractProjectRepoRepository):
    table = "project_repos"

    def __init__(self, conn: "DuckDBThreadWrapper"):
        self.conn = conn
        self._table = Table(self.table)

    def _get_query(self, q):
        parameter = QmarkParameter()
        sql = q.get_sql(parameter=parameter)
        return sql, parameter.get_parameters()

    def _execute(self, q):
        sql, args = self._get_query(q)
        return self.conn.execute(sql, args)

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
            .columns(self._table.project_id, self._table.repo_id)
            .insert(project_id, repo_id)
        )
        self._execute(q)


# ---------------------------------------------------------------------------
# concrete repositories
# ---------------------------------------------------------------------------

class DuckDBRepoRepo(_DuckDBBaseRepo[Repo], AbstractRepoRepository):
    table = "repos"
    model = Repo

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


class DuckDBPackageRepo(_DuckDBBaseRepo[Package], AbstractPackageRepository):
    table = "packages"
    model = Package

    def __init__(self, conn, file_repo: "DuckDBFileRepo"):  # type: ignore
        super().__init__(conn)
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

        if flt.repo_id:
            q = q.where(self._table.repo_id.isin(flt.repo_id))

        rows = self._execute(q)

        return [self.model(**self._deserialize_data(r)) for r in rows]

    def delete_orphaned(self) -> None:
        files_tbl = Table("files")
        subq = Query.from_(files_tbl).select(files_tbl.package_id).where(files_tbl.package_id.notnull())

        q = Query.from_(self._table).where(self._table.id.notin(subq)).delete()
        self._execute(q)


from know.data import FileFilter      # already present – keep / ensure

class DuckDBFileRepo(_DuckDBBaseRepo[File], AbstractFileRepository):
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

        if flt.repo_id:
            q = q.where(self._table.repo_id.isin(flt.repo_id))
        if flt.package_id:
            q = q.where(self._table.package_id == flt.package_id)

        rows = self._execute(q)
        return [self.model(**self._deserialize_data(r)) for r in rows]


class DuckDBNodeRepo(_DuckDBBaseRepo[Node], AbstractNodeRepository):
    table = "nodes"
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

    def search(self, repo_id: str, query: NodeSearchQuery) -> list[Node]:
        q = Query.from_(self._table)

        # Candidates
        candidates = (
            Query.
            from_(self._table).
            select(self._table.id, self._table.embedding_code_vec).
            where(self._table.repo_id == repo_id)
        )

        if query.symbol_name:
            q = q.where(functions.Lower(self._table.name) == query.symbol_name.lower())

        if query.symbol_fqn:
            q = q.where(functions.Lower(self._table.fqn).like(f"%{query.symbol_fqn.lower()}%"))

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
            rank_code_scores = (
                Query.
                from_(aliased_candidates).
                select(
                    aliased_candidates.id,
                    ArrayCosineSimilarityFn(
                        aliased_candidates.embedding_code_vec,
                        functions.Cast(ValueWrapper(query.embedding_query), "FLOAT[1024]")
                    ).as_("dist"))
            )

            aliased = AliasedQuery("rank_code_scores")

            rank_code = (
                Query.
                from_(aliased).
                select(aliased.id, analytics.RowNumber().orderby(aliased.dist, order=Order.desc).as_("code_rank")).
                where(aliased.dist >= 0.4)
            )

            q = q.with_(rank_code_scores, "rank_code_scores").with_(rank_code, "rank_code")

        if has_fts:
            rank_fts_scores = (
                Query.
                from_(aliased_candidates).
                select(
                    aliased_candidates.id,
                    MatchBM25Fn(
                        aliased_candidates.id,
                        query.doc_needle
                    ).as_("score"))
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

            fused = (
                Query.
                from_(aliased_candidates).
                join(aliased_scores).on(aliased_candidates.id == aliased_scores.id).
                select(aliased_scores.id, aliased_scores.score.as_("rrf_score"))
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

    def delete_by_file_id(self, file_id: str) -> int:
        q = Query.from_(self._table).where(self._table.file_id == file_id).delete()
        res = self._execute(q)
        return res[0]["count_star()"] if res else 0

    def get_list_by_ids(self, symbol_ids: list[str]) -> list[Node]:
        syms = super().get_list_by_ids(symbol_ids)
        resolve_node_hierarchy(syms)
        return syms

    def get_list(self, flt: NodeFilter) -> list[Node]:
        q = Query.from_(self._table).select("*")

        if flt.parent_ids:
            q = q.where(self._table.parent_node_id.isin(flt.parent_ids))
        if flt.repo_id:
            q = q.where(self._table.repo_id.isin(flt.repo_id))
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


class DuckDBImportEdgeRepo(_DuckDBBaseRepo[ImportEdge], AbstractImportEdgeRepository):
    table = "import_edges"
    model = ImportEdge

    def get_list(self, flt: ImportEdgeFilter) -> list[ImportEdge]:
        q = Query.from_(self._table).select("*")

        if flt.source_package_id:
            q = q.where(self._table.from_package_id == flt.source_package_id)
        if flt.source_file_id:
            q = q.where(self._table.from_file_id == flt.source_file_id)
        if flt.repo_id:
            q = q.where(self._table.repo_id.isin(flt.repo_id))

        rows = self._execute(q)
        return [self.model(**self._deserialize_data(r)) for r in rows]


class DuckDBNodeRefRepo(_DuckDBBaseRepo[NodeRef], AbstractNodeRefRepository):
    table = "node_refs"
    model = NodeRef

    def get_list(self, flt: NodeRefFilter) -> list[NodeRef]:
        q = Query.from_(self._table).select("*")

        if flt.file_id:
            q = q.where(self._table.file_id == flt.file_id)
        if flt.package_id:
            q = q.where(self._table.package_id == flt.package_id)
        if flt.repo_id:
            q = q.where(self._table.repo_id.isin(flt.repo_id))

        rows = self._execute(q)
        return [self.model(**self._deserialize_data(r)) for r in rows]

    def delete_by_file_id(self, file_id: str) -> int:
        q = Query.from_(self._table).where(self._table.file_id == file_id).delete()
        res = self._execute(q)
        return res[0]["count_star()"] if res else 0

# Data-repository
class DuckDBDataRepository(AbstractDataRepository):
    """
    Main entry point.  Automatically applies pending SQL migrations on first
    construction.
    """

    def __init__(self, db_path: str | None = None):
        """
        Parameters
        ----------
        db_path : str | None
            Filesystem path to the DuckDB database.
            • None  ->  use in-memory database.
        """
        if db_path is None:
            db_path = ":memory:"
        # ensure parent dir exists for file-based DBs
        if db_path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)

        self._conn = DuckDBThreadWrapper(db_path)
        self._conn.start()

        # build repositories (some need cross-references)
        self._project_repo = DuckDBProjectRepo(self._conn)
        self._prj_repo_repo = DuckDBProjectRepoRepo(self._conn)
        self._file_repo = DuckDBFileRepo(self._conn)
        self._package_repo = DuckDBPackageRepo(self._conn, self._file_repo)
        self._repo_repo = DuckDBRepoRepo(self._conn)
        self._symbol_repo = DuckDBNodeRepo(self._conn)
        self._edge_repo = DuckDBImportEdgeRepo(self._conn)
        self._symbolref_repo = DuckDBNodeRefRepo(self._conn)

    def close(self):
        self._conn.close()

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
        try:
            self._conn.execute("PRAGMA drop_fts_index('nodes');")
            self._conn.execute(
                "PRAGMA create_fts_index('nodes', "
                "'id', 'name', 'fqn', 'docstring', 'comment');"
            )
        except Exception as ex:
            logger.debug("Failed to refresh DuckDB FTS index", ex=ex)
