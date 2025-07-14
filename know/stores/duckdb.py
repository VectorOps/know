from __future__ import annotations
import threading
import os
import duckdb
import json
import pandas as pd   # new – required for .df() conversion
import math          # needed for math.isnan
from typing import Optional, Dict, Any, Generic, TypeVar, Type, Callable
import importlib.resources as pkg_resources
from datetime import datetime, timezone

from know.models import (
    RepoMetadata,
    PackageMetadata,
    FileMetadata,
    SymbolMetadata,
    SymbolSignature,
    ImportEdge,
    SymbolRef,
    Modifier,
)
from know.data import (
    AbstractRepoMetadataRepository,
    AbstractPackageMetadataRepository,
    AbstractFileMetadataRepository,
    AbstractSymbolMetadataRepository,
    AbstractImportEdgeRepository,
    AbstractSymbolRefRepository,
    AbstractDataRepository,
    SymbolSearchQuery,
    PackageFilter,
    include_direct_descendants,
    SymbolFilter,
    ImportFilter,
)

T = TypeVar("T")

# helpers
def _row_to_dict(rel) -> list[dict[str, Any]]:
    """
    Convert a DuckDB relation to List[Dict] via a pandas DataFrame.
    Using DataFrame avoids the manual column-name handling and is faster.
    """
    df = rel.df()              # pandas DataFrame
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

# generic base repository
class _DuckDBBaseRepo(Generic[T]):
    table: str
    model: Type[T]

    _json_fields: set[str] = set()                    # columns stored as JSON
    _json_parsers: dict[str, Callable[[Any], Any]] = {}   # field → decode-fn

    def cursor(self):
        cur = self.conn.cursor()
        cur.execute("USE db")
        return cur

    def __init__(self, conn: duckdb.DuckDBPyConnection):
        self.conn = conn

    def _serialize_json(self, data: dict[str, Any]) -> dict[str, Any]:
        """Return *data* with JSON fields dumped to str."""
        for fld in self._json_fields:
            if fld in data and data[fld] is not None:
                val = data[fld]
                # support Pydantic models
                if hasattr(val, "model_dump"):
                    val = val.model_dump(exclude_none=False)
                data[fld] = json.dumps(val)
        return data

    def _deserialize_row(self, row: dict[str, Any]) -> dict[str, Any]:
        """Return *row* with JSON fields loaded & parsed."""
        for fld in self._json_fields:
            if fld in row and row[fld] is not None:
                parsed = json.loads(row[fld])
                parser = self._json_parsers.get(fld)
                row[fld] = parser(parsed) if parser else parsed
        return row

    # CRUD
    def get_by_id(self, item_id: str) -> Optional[T]:
        rows = _row_to_dict(self.cursor().execute(f"SELECT * FROM {self.table} WHERE id = ?", [item_id]))
        return self.model(**self._deserialize_row(rows[0])) if rows else None

    def get_list_by_ids(self, item_ids: list[str]) -> list[T]:
        if not item_ids:
            return []
        placeholders = ", ".join("?" for _ in item_ids)
        rows = _row_to_dict(self.cursor().execute(f"SELECT * FROM {self.table} WHERE id IN ({placeholders})", item_ids))
        return [self.model(**self._deserialize_row(r)) for r in rows]

    def create(self, item: T) -> T:
        data = self._serialize_json(item.model_dump(exclude_none=True))
        cols = ", ".join(data.keys())
        placeholders = ", ".join("?" for _ in data)
        self.cursor().execute(f"INSERT INTO {self.table} ({cols}) VALUES ({placeholders})", list(data.values()))
        return item

    def update(self, item_id: str, data: Dict[str, Any]) -> Optional[T]:
        if not data:
            return self.get_by_id(item_id)
        data = self._serialize_json(data.copy())
        set_clause = ", ".join(f"{k}=?" for k in data)
        params = list(data.values()) + [item_id]
        self.cursor().execute(f"UPDATE {self.table} SET {set_clause} WHERE id = ?", params)
        return self.get_by_id(item_id)

    def delete(self, item_id: str) -> bool:
        self.cursor().execute(f"DELETE FROM {self.table} WHERE id = ?", [item_id])
        return True

# ---------------------------------------------------------------------------
# concrete repositories
# ---------------------------------------------------------------------------

class DuckDBRepoMetadataRepo(_DuckDBBaseRepo[RepoMetadata], AbstractRepoMetadataRepository):
    table = "repos"
    model = RepoMetadata

    def get_by_path(self, root_path: str) -> Optional[RepoMetadata]:
        rows = _row_to_dict(self.cursor().execute("SELECT * FROM repos WHERE root_path = ?", [root_path]))
        return RepoMetadata(**rows[0]) if rows else None


class DuckDBPackageMetadataRepo(_DuckDBBaseRepo[PackageMetadata], AbstractPackageMetadataRepository):
    table = "packages"
    model = PackageMetadata

    def __init__(self, conn, file_repo: "DuckDBFileMetadataRepo"):  # type: ignore
        super().__init__(conn)
        self._file_repo = file_repo

    def get_by_physical_path(self, path: str) -> Optional[PackageMetadata]:
        rows = _row_to_dict(self.cursor().execute("SELECT * FROM packages WHERE physical_path = ?", [path]))
        return PackageMetadata(**rows[0]) if rows else None

    def get_by_virtual_path(self, path: str) -> Optional[PackageMetadata]:
        rows = _row_to_dict(self.cursor().execute("SELECT * FROM packages WHERE virtual_path = ?", [path]))
        return PackageMetadata(**rows[0]) if rows else None

    def get_list(self, flt: PackageFilter) -> list[PackageMetadata]:
        """
        Return all PackageMetadata rows matching *flt*.
        Currently supports filtering by repo_id only.
        """
        if flt.repo_id:
            rows = _row_to_dict(
                self.cursor().execute(
                    "SELECT * FROM packages WHERE repo_id = ?", [flt.repo_id]
                )
            )
        else:
            rows = _row_to_dict(self.cursor().execute("SELECT * FROM packages"))
        return [PackageMetadata(**r) for r in rows]

    def delete_orphaned(self) -> int:
        used_pkg_ids = {row["package_id"] for row in
                        _row_to_dict(self.cursor().execute("SELECT DISTINCT package_id FROM files WHERE package_id IS NOT NULL"))}
        rows = _row_to_dict(self.cursor().execute("SELECT id FROM packages"))
        orphan_ids = [r["id"] for r in rows if r["id"] not in used_pkg_ids]
        for oid in orphan_ids:
            self.delete(oid)
        return len(orphan_ids)


from know.data import FileFilter      # already present – keep / ensure

class DuckDBFileMetadataRepo(_DuckDBBaseRepo[FileMetadata], AbstractFileMetadataRepository):
    table = "files"
    model = FileMetadata

    def get_by_path(self, path: str) -> Optional[FileMetadata]:
        rows = _row_to_dict(self.cursor().execute("SELECT * FROM files WHERE path = ?", [path]))
        return FileMetadata(**rows[0]) if rows else None

    def get_list(self, flt: FileFilter) -> list[FileMetadata]:
        """
        Return all FileMetadata rows that satisfy *flt*.
        Supports filtering by repo_id and/or package_id.
        """
        where, params = [], []
        if flt.repo_id:
            where.append("repo_id = ?")
            params.append(flt.repo_id)
        if flt.package_id:
            where.append("package_id = ?")
            params.append(flt.package_id)

        sql = "SELECT * FROM files"
        if where:
            sql += " WHERE " + " AND ".join(where)

        rows = _row_to_dict(self.cursor().execute(sql, params))
        return [FileMetadata(**r) for r in rows]


class DuckDBSymbolMetadataRepo(_DuckDBBaseRepo[SymbolMetadata], AbstractSymbolMetadataRepository):
    table = "symbols"
    model = SymbolMetadata
    _json_fields = {"signature", "modifiers"}
    _json_parsers = {
        "signature": lambda v: SymbolSignature(**v) if v is not None else None,
        "modifiers": lambda v: [Modifier(m) for m in v] if v is not None else [],
    }

    RRF_K: int = 60          # tuning-parameter k (see RRF paper)
    RRF_CODE_WEIGHT: float = 0.7
    RRF_FTS_WEIGHT:  float = 0.3

    def search(self, repo_id: str, query: SymbolSearchQuery) -> list[SymbolMetadata]:
        # ---- FROM / JOIN clause to filter by repo_id via files table ----
        base_where, base_params = ["f.repo_id = ?"], [repo_id]

        # ---------- scalar filters ----------
        if query.symbol_name:
            base_where.append("LOWER(s.name) = ?")
            base_params.append(query.symbol_name.lower())

        if query.symbol_fqn:
            base_where.append("LOWER(s.fqn) LIKE ?")
            base_params.append(f"%{query.symbol_fqn.lower()}%")

        if query.symbol_kind:
            base_where.append("s.kind = ?")
            base_params.append(getattr(query.symbol_kind, "value", query.symbol_kind))

        if query.symbol_visibility:
            base_where.append("s.visibility = ?")
            base_params.append(getattr(query.symbol_visibility, "value", query.symbol_visibility))

        if query.top_level_only:
            base_where.append("s.parent_symbol_id IS NULL")

        if query.embedding is True:          # only symbols WITH an embedding
            base_where.append("s.embedding_code_vec IS NOT NULL")
        elif query.embedding is False:       # only symbols WITHOUT an embedding
            base_where.append("s.embedding_code_vec IS NULL")

        # Compose base WHERE clause string
        base_where_clause = " AND ".join(base_where)

        # Determine which search dimensions are provided
        has_fts = bool(query.doc_needle)
        has_embedding = bool(query.embedding_query)

        # ------------------------------------------------------------------ #
        # unified CTE / query construction                                   #
        # ------------------------------------------------------------------ #
        cte_parts: list[str] = [f"""
WITH candidates AS (
    SELECT s.*
    FROM symbols s
    JOIN files f ON s.file_id = f.id
    WHERE {base_where_clause}
)"""]

        params: list[Any] = base_params[:]      # start with scalar-filter params

        # ---------- optional rank CTEs ---------- #
        if has_fts:
            cte_parts.append("""
, rank_fts_scores AS (
    SELECT id,
           fts_main_symbols.match_bm25(id, ?) as score
    FROM candidates
), rank_fts AS (
    SELECT id,
           row_number() OVER (ORDER BY score DESC) AS fts_rank
    FROM rank_fts_scores
    WHERE score IS NOT NULL
)""")
            params.append(query.doc_needle)

        if has_embedding:
            cte_parts.append("""
, rank_code_scores AS (
    SELECT id,
           array_cosine_similarity(embedding_code_vec,
                          CAST(? AS FLOAT[1024])) AS dist
    FROM candidates
), rank_code AS (
    SELECT id,
           row_number() OVER (ORDER BY dist DESC) AS code_rank
    FROM rank_code_scores
    WHERE dist IS NOT NULL
      AND dist >= 0.4
)""")
            params.append(query.embedding_query)

        has_ranking = has_fts or has_embedding
        if has_ranking:
            union_parts: list[str] = []

            if has_embedding:
                union_parts.append(
                    "SELECT id, ? / (? + code_rank) AS score FROM rank_code"
                )
                params.extend([self.RRF_CODE_WEIGHT, self.RRF_K])

            if has_fts:
                union_parts.append(
                    "SELECT id, ? / (? + fts_rank) AS score FROM rank_fts"
                )
                params.extend([self.RRF_FTS_WEIGHT, self.RRF_K])

            cte_parts.append(f"""
, rrf_scores AS (
    {' UNION ALL '.join(union_parts)}
), fused AS (
    SELECT c.*,
           COALESCE(rs.rrf_score, 0) AS rrf_score
    FROM candidates c
    INNER JOIN (
        SELECT id, SUM(score) AS rrf_score
        FROM rrf_scores
        GROUP BY id
    ) rs USING(id)
)""")

        limit  = query.limit  if query.limit  is not None else 20
        offset = query.offset if query.offset is not None else 0

        if has_ranking:
            final_select = """
SELECT * FROM fused
ORDER BY rrf_score DESC, name ASC
LIMIT ? OFFSET ?
"""
        else:
            final_select = """
SELECT * FROM candidates
ORDER BY name ASC
LIMIT ? OFFSET ?
"""
        params.extend([limit, offset])

        sql = "".join(cte_parts) + final_select

        rows = _row_to_dict(self.cursor().execute(sql, params))
        syms = [self.model(**self._deserialize_row(r)) for r in rows]
        syms = include_direct_descendants(self, syms)
        return syms

    def delete_by_file_id(self, file_id: str) -> int:
        rows = self.cursor().execute(
            "DELETE FROM symbols WHERE file_id = ? RETURNING id", [file_id]
        ).fetchall()
        return len(rows)

    def get_list(self, flt: SymbolFilter) -> list[SymbolMetadata]:
        """
        Generic selector that supersedes the old specialised helpers.
        Supports filtering by ids, parent_ids, file_id and/or package_id.
        """
        where, params = [], []

        if flt.parent_ids:
            where.append(
                f"parent_symbol_id IN ({', '.join('?' for _ in flt.parent_ids)})"
            )
            params.extend(flt.parent_ids)
        if flt.file_id:
            where.append("file_id = ?")
            params.append(flt.file_id)
        if flt.package_id:
            where.append("package_id = ?")
            params.append(flt.package_id)

        sql = "SELECT * FROM symbols" + (" WHERE " + " AND ".join(where) if where else "")
        rows = _row_to_dict(self.cursor().execute(sql, params))
        syms = [self.model(**self._deserialize_row(r)) for r in rows]
        SymbolMetadata.resolve_symbol_hierarchy(syms)
        return syms


class DuckDBImportEdgeRepo(_DuckDBBaseRepo[ImportEdge], AbstractImportEdgeRepository):
    table = "import_edges"
    model = ImportEdge

    def get_list_by_source_package_id(self, package_id: str) -> list[ImportEdge]:
        rows = _row_to_dict(self.cursor().execute("SELECT * FROM import_edges WHERE from_package_id = ?", [package_id]))
        return [ImportEdge(**r) for r in rows]

    def get_list_by_source_file_id(self, file_id: str) -> list[ImportEdge]:
        rows = _row_to_dict(self.cursor().execute(f"SELECT * FROM {self.table} WHERE from_file_id = ?", [file_id]))
        return [ImportEdge(**r) for r in rows]

    def get_list_by_repo_id(self, repo_id: str) -> list[ImportEdge]:
        rows = _row_to_dict(self.cursor().execute(
            "SELECT * FROM import_edges WHERE repo_id = ?", [repo_id]))
        return [ImportEdge(**r) for r in rows]

    # generic selector
    def get_list(self, flt: ImportFilter) -> list[ImportEdge]:      # NEW
        where, params = [], []
        if flt.source_package_id:
            where.append("from_package_id = ?")
            params.append(flt.source_package_id)
        if flt.source_file_id:
            where.append("from_file_id = ?")
            params.append(flt.source_file_id)
        if flt.repo_id:
            where.append("repo_id = ?")
            params.append(flt.repo_id)

        sql = f"SELECT * FROM {self.table}" + (" WHERE " + " AND ".join(where) if where else "")
        rows = _row_to_dict(self.cursor().execute(sql, params))
        return [ImportEdge(**r) for r in rows]


class DuckDBSymbolRefRepo(_DuckDBBaseRepo[SymbolRef], AbstractSymbolRefRepository):
    table = "symbol_refs"
    model = SymbolRef

    def get_list_by_file_id(self, file_id: str) -> list[SymbolRef]:
        rows = _row_to_dict(self.cursor().execute(
            "SELECT * FROM symbol_refs WHERE file_id = ?", [file_id]))
        return [SymbolRef(**r) for r in rows]

    def get_list_by_package_id(self, package_id: str) -> list[SymbolRef]:
        rows = _row_to_dict(self.cursor().execute(
            "SELECT * FROM symbol_refs WHERE package_id = ?", [package_id]))
        return [SymbolRef(**r) for r in rows]

    def get_list_by_repo_id(self, repo_id: str) -> list[SymbolRef]:
        rows = _row_to_dict(self.cursor().execute(
            "SELECT * FROM symbol_refs WHERE repo_id = ?", [repo_id]))
        return [SymbolRef(**r) for r in rows]

    # NEW ---------------------------------------------------------------
    def delete_by_file_id(self, file_id: str) -> int:
        """
        Bulk-delete refs belonging to *file_id*.
        DuckDB ≥0.8.0 supports RETURNING; we use that to count rows.
        """
        rows = self.cursor().execute(
            "DELETE FROM symbol_refs WHERE file_id = ? RETURNING id", [file_id]
        ).fetchall()
        return len(rows)

# ---------------------------------------------------------------------------
# Migration logic
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Data-repository façade
# ---------------------------------------------------------------------------

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

        self._conn = duckdb.connect()

        # --- enable vector-similarity search extension --------------------------
        try:
            self._conn.execute("INSTALL vss")
            self._conn.execute("LOAD vss")
        except Exception:          # extension already installed / not available
            pass

        try:
            self._conn.execute("INSTALL fts")
            self._conn.execute("LOAD fts")
        except Exception:
            pass          # already installed / unavailable

        # TODO: SQL injection?
        self._conn.execute(f"ATTACH '{db_path}' as db")
        self._conn.execute("USE db")

        _apply_migrations(self._conn)

        # build repositories (some need cross-references)
        self._file_repo    = DuckDBFileMetadataRepo(self._conn)
        self._package_repo = DuckDBPackageMetadataRepo(self._conn, self._file_repo)
        self._repo_repo    = DuckDBRepoMetadataRepo(self._conn)
        self._symbol_repo  = DuckDBSymbolMetadataRepo(self._conn)
        self._edge_repo    = DuckDBImportEdgeRepo(self._conn)
        self._symbolref_repo = DuckDBSymbolRefRepo(self._conn)

    def close(self):
        pass

    # ---------- interface impl ----------
    @property
    def repo(self) -> AbstractRepoMetadataRepository:     # type: ignore[override]
        return self._repo_repo

    @property
    def package(self) -> AbstractPackageMetadataRepository:  # type: ignore[override]
        return self._package_repo

    @property
    def file(self) -> AbstractFileMetadataRepository:     # type: ignore[override]
        return self._file_repo

    @property
    def symbol(self) -> AbstractSymbolMetadataRepository:  # type: ignore[override]
        return self._symbol_repo

    @property
    def importedge(self) -> AbstractImportEdgeRepository:  # type: ignore[override]
        return self._edge_repo

    @property
    def symbolref(self) -> AbstractSymbolRefRepository:  # type: ignore[override]
        return self._symbolref_repo

    def refresh_full_text_indexes(self) -> None:               # NEW
        try:
            self._conn.execute("PRAGMA drop_fts_index('symbols');")
            self._conn.execute(
                "PRAGMA create_fts_index('symbols', "
                "'id', 'name', 'fqn', 'docstring', 'comment');"
            )
        except Exception as ex:
            # index absent or extension unavailable – ignore
            pass
