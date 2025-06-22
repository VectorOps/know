from __future__ import annotations
import os
import duckdb
import json
import pandas as pd   # new – required for .df() conversion
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
    Modifier,
)
from know.data import (
    AbstractRepoMetadataRepository,
    AbstractPackageMetadataRepository,
    AbstractFileMetadataRepository,
    AbstractSymbolMetadataRepository,
    AbstractImportEdgeRepository,
    AbstractDataRepository,
)

T = TypeVar("T")

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _row_to_dict(rel) -> list[dict[str, Any]]:
    """
    Convert a DuckDB relation to List[Dict] via a pandas DataFrame.
    Using DataFrame avoids the manual column-name handling and is faster.
    """
    df = rel.df()              # pandas DataFrame
    records = df.to_dict(orient="records")  # [] when df is empty
    cleaned: list[dict[str, Any]] = []
    for rec in records:
        cleaned.append(
            {k: (None if pd.isna(v) else v) for k, v in rec.items()}
        )
    return cleaned


# ---------------------------------------------------------------------------
# generic base repository
# ---------------------------------------------------------------------------

class _DuckDBBaseRepo(Generic[T]):
    table: str
    model: Type[T]

    _json_fields: set[str] = set()                    # columns stored as JSON
    _json_parsers: dict[str, Callable[[Any], Any]] = {}   # field → decode-fn

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

    # ---------- CRUD ----------
    def get_by_id(self, item_id: str) -> Optional[T]:
        rows = _row_to_dict(self.conn.execute(f"SELECT * FROM {self.table} WHERE id = ?", [item_id]))
        return self.model(**self._deserialize_row(rows[0])) if rows else None

    def get_list_by_ids(self, item_ids: list[str]) -> list[T]:
        if not item_ids:
            return []
        placeholders = ", ".join("?" for _ in item_ids)
        rows = _row_to_dict(self.conn.execute(f"SELECT * FROM {self.table} WHERE id IN ({placeholders})", item_ids))
        return [self.model(**self._deserialize_row(r)) for r in rows]

    def create(self, item: T) -> T:
        data = self._serialize_json(item.model_dump(exclude_none=True))
        cols = ", ".join(data.keys())
        placeholders = ", ".join("?" for _ in data)
        self.conn.execute(f"INSERT INTO {self.table} ({cols}) VALUES ({placeholders})", list(data.values()))
        return item

    def update(self, item_id: str, data: Dict[str, Any]) -> Optional[T]:
        if not data:
            return self.get_by_id(item_id)
        data = self._serialize_json(data.copy())
        set_clause = ", ".join(f"{k}=?" for k in data)
        params = list(data.values()) + [item_id]
        self.conn.execute(f"UPDATE {self.table} SET {set_clause} WHERE id = ?", params)
        return self.get_by_id(item_id)

    def delete(self, item_id: str) -> bool:
        self.conn.execute(f"DELETE FROM {self.table} WHERE id = ?", [item_id])
        return True
# ---------------------------------------------------------------------------
# concrete repositories
# ---------------------------------------------------------------------------

class DuckDBRepoMetadataRepo(_DuckDBBaseRepo[RepoMetadata], AbstractRepoMetadataRepository):
    table = "repos"
    model = RepoMetadata

    def get_by_path(self, root_path: str) -> Optional[RepoMetadata]:
        rows = _row_to_dict(self.conn.execute("SELECT * FROM repos WHERE root_path = ?", [root_path]))
        return RepoMetadata(**rows[0]) if rows else None


class DuckDBPackageMetadataRepo(_DuckDBBaseRepo[PackageMetadata], AbstractPackageMetadataRepository):
    table = "packages"
    model = PackageMetadata

    def __init__(self, conn, file_repo: "DuckDBFileMetadataRepo"):  # type: ignore
        super().__init__(conn)
        self._file_repo = file_repo

    def get_by_path(self, path: str) -> Optional[PackageMetadata]:
        rows = _row_to_dict(self.conn.execute("SELECT * FROM packages WHERE physical_path = ?", [path]))
        return PackageMetadata(**rows[0]) if rows else None

    def delete_orphaned(self) -> int:
        used_pkg_ids = {row["package_id"] for row in
                        _row_to_dict(self.conn.execute("SELECT DISTINCT package_id FROM files WHERE package_id IS NOT NULL"))}
        rows = _row_to_dict(self.conn.execute("SELECT id FROM packages"))
        orphan_ids = [r["id"] for r in rows if r["id"] not in used_pkg_ids]
        for oid in orphan_ids:
            self.delete(oid)
        return len(orphan_ids)


class DuckDBFileMetadataRepo(_DuckDBBaseRepo[FileMetadata], AbstractFileMetadataRepository):
    table = "files"
    model = FileMetadata

    def get_by_path(self, path: str) -> Optional[FileMetadata]:
        rows = _row_to_dict(self.conn.execute("SELECT * FROM files WHERE path = ?", [path]))
        return FileMetadata(**rows[0]) if rows else None

    def get_list_by_repo_id(self, repo_id: str) -> list[FileMetadata]:
        rows = _row_to_dict(self.conn.execute("SELECT * FROM files WHERE repo_id = ?", [repo_id]))
        return [FileMetadata(**r) for r in rows]

    def get_list_by_package_id(self, package_id: str) -> list[FileMetadata]:
        rows = _row_to_dict(self.conn.execute("SELECT * FROM files WHERE package_id = ?", [package_id]))
        return [FileMetadata(**r) for r in rows]


class DuckDBSymbolMetadataRepo(_DuckDBBaseRepo[SymbolMetadata], AbstractSymbolMetadataRepository):
    table = "symbols"
    model = SymbolMetadata
    _json_fields = {"signature", "score_security_flags", "modifiers"}
    _json_parsers = {
        "signature": lambda v: SymbolSignature(**v) if v is not None else None,
        "score_security_flags": lambda v: v,   # returns list[str]
        "modifiers": lambda v: [Modifier(m) for m in v] if v is not None else [],
    }

    def get_list_by_file_id(self, file_id: str) -> list[SymbolMetadata]:
        rows = _row_to_dict(self.conn.execute("SELECT * FROM symbols WHERE file_id = ?", [file_id]))
        return [self.model(**self._deserialize_row(r)) for r in rows]


class DuckDBImportEdgeRepo(_DuckDBBaseRepo[ImportEdge], AbstractImportEdgeRepository):
    table = "import_edges"
    model = ImportEdge

    def get_list_by_source_package_id(self, package_id: str) -> list[ImportEdge]:
        rows = _row_to_dict(self.conn.execute("SELECT * FROM import_edges WHERE from_package_id = ?", [package_id]))
        return [ImportEdge(**r) for r in rows]

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

        self._conn = duckdb.connect(db_path)
        _apply_migrations(self._conn)

        # build repositories (some need cross-references)
        self._file_repo    = DuckDBFileMetadataRepo(self._conn)
        self._package_repo = DuckDBPackageMetadataRepo(self._conn, self._file_repo)
        self._repo_repo    = DuckDBRepoMetadataRepo(self._conn)
        self._symbol_repo  = DuckDBSymbolMetadataRepo(self._conn)
        self._edge_repo    = DuckDBImportEdgeRepo(self._conn)

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
