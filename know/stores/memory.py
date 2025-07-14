import re      # for tokenisation
import threading
from typing import Optional, Dict, Any, List, TypeVar, Generic
from know.models import (
    RepoMetadata,
    PackageMetadata,
    FileMetadata,
    SymbolMetadata,
    ImportEdge,
    SymbolRef,
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
    FileFilter,
    SymbolFilter,
    include_direct_descendants,
)
from dataclasses import dataclass, field
import math
import numpy as np
from know.embeddings.interface import EMBEDDING_DIM

def _cosine(a: list[float], b: list[float]) -> float:
    """
    Return cosine similarity of two equal-length numeric vectors.
    If either vector has zero norm, –1.0 is returned (so it ranks last).
    """
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else -1.0

@dataclass
class _MemoryTables:
    repos:      dict[str, RepoMetadata]   = field(default_factory=dict)
    packages:   dict[str, PackageMetadata]= field(default_factory=dict)
    files:      dict[str, FileMetadata]   = field(default_factory=dict)
    symbols:    dict[str, SymbolMetadata] = field(default_factory=dict)
    edges:      dict[str, ImportEdge]     = field(default_factory=dict)
    symbolrefs: dict[str, SymbolRef]      = field(default_factory=dict)
    lock:       threading.RLock           = field(
        default_factory=threading.RLock, repr=False, compare=False
    )

T = TypeVar("T")

class InMemoryBaseRepository(Generic[T]):
    def __init__(self, table: Dict[str, T], lock: threading.RLock):
        self._items = table
        self._lock  = lock

    def get_by_id(self, item_id: str) -> Optional[T]:
        """Get an item by its ID."""
        with self._lock:
            return self._items.get(item_id)

    def get_list_by_ids(self, item_ids: list[str]) -> list[T]:
        """Get a list of items by their IDs."""
        with self._lock:
            return [self._items[iid] for iid in item_ids if iid in self._items]

    def create(self, item: T) -> T:
        """Create a new item entry."""
        with self._lock:
            self._items[item.id] = item
        return item

    def update(self, item_id: str, data: Dict[str, Any]) -> Optional[T]:
        """Update an item by its ID."""
        with self._lock:
            item = self._items.get(item_id)
            if not item:
                return None
            # Pydantic v2: use `model_copy`, fall back to legacy `copy`.
            if hasattr(item, "model_copy"):              # Pydantic >= 2
                updated = item.model_copy(update=data)
                self._items[item_id] = updated
                return updated
            if hasattr(item, "copy"):                    # Pydantic < 2
                updated = item.copy(update=data)
                self._items[item_id] = updated
                return updated
            # For dataclasses, update fields in place
            for k, v in data.items():
                setattr(item, k, v)
            return item

    def delete(self, item_id: str) -> bool:
        """Delete an item by its ID."""
        with self._lock:
            return self._items.pop(item_id, None) is not None

class InMemoryRepoMetadataRepository(InMemoryBaseRepository[RepoMetadata], AbstractRepoMetadataRepository):
    def __init__(self, tables: _MemoryTables):
        super().__init__(tables.repos, tables.lock)

    def get_by_path(self, root_path: str) -> Optional[RepoMetadata]:
        """Get a repo by its root path."""
        with self._lock:
            for repo in self._items.values():
                if repo.root_path == root_path:
                    return repo
        return None

class InMemoryPackageMetadataRepository(InMemoryBaseRepository[PackageMetadata], AbstractPackageMetadataRepository):
    def __init__(self, tables: _MemoryTables):
        super().__init__(tables.packages, tables.lock)
        self._file_items = tables.files

    def get_by_physical_path(self, path: str) -> Optional[PackageMetadata]:
        """
        Return the first PackageMetadata whose physical_path equals *path*.
        """
        with self._lock:
            for pkg in self._items.values():
                if pkg.physical_path == path:
                    return pkg
        return None

    def get_by_virtual_path(self, path: str) -> Optional[PackageMetadata]:
        """
        Return the first PackageMetadata whose physical_path equals *path*.
        """
        with self._lock:
            for pkg in self._items.values():
                if pkg.virtual_path == path:
                    return pkg
        return None

    def get_list_by_repo_id(self, repo_id: str) -> list[PackageMetadata]:
        """Return all packages that belong to *repo_id*."""
        with self._lock:
            return [pkg for pkg in self._items.values() if pkg.repo_id == repo_id]

    def get_list(self, flt: PackageFilter) -> list[PackageMetadata]:
        """
        Return all PackageMetadata objects that satisfy *flt*.
        Currently only repo_id is supported.
        """
        with self._lock:
            if flt.repo_id:
                return [pkg for pkg in self._items.values()
                        if pkg.repo_id == flt.repo_id]
            # no filter → return every package
            return list(self._items.values())

    def delete_orphaned(
        self,
    ) -> int:
        """
        Delete every PackageMetadata that is not referenced by any
        FileMetadata in *file_repo*.  Returns the number of deletions.
        """
        with self._lock:
            used_pkg_ids = {f.package_id for f in self._file_items.values() if f.package_id}
            removed = 0
            for pkg_id in list(self._items):
                if pkg_id not in used_pkg_ids:
                    self.delete(pkg_id)
                    removed += 1
            return removed


class InMemoryFileMetadataRepository(InMemoryBaseRepository[FileMetadata],
                                     AbstractFileMetadataRepository):
    def __init__(self, tables: _MemoryTables):
        super().__init__(tables.files, tables.lock)

    def get_by_path(self, path: str) -> Optional[FileMetadata]:
        """Get a file by its project-relative path."""
        with self._lock:
            for file in self._items.values():
                if file.path == path:
                    return file
        return None

    def get_list(self, flt: FileFilter) -> list[FileMetadata]:
        """
        Return all FileMetadata objects that satisfy *flt*.
        Supports filtering by repo_id and/or package_id.
        """
        with self._lock:
            return [
                f for f in self._items.values()
                if (not flt.repo_id   or f.repo_id   == flt.repo_id)
                and (not flt.package_id or f.package_id == flt.package_id)
            ]

class InMemorySymbolMetadataRepository(InMemoryBaseRepository[SymbolMetadata], AbstractSymbolMetadataRepository):
    RRF_K: int = 60          # tuning-parameter k (Reciprocal-Rank-Fusion)
    RRF_CODE_WEIGHT: float = 0.7
    RRF_FTS_WEIGHT:  float = 0.3
    # minimum cosine similarity for an embedding to participate in ranking
    EMBEDDING_SIM_THRESHOLD: float = 0.4
    EMBEDDING_TOP_K: int = 1000   # how many neighbours to ask FAISS for

    # ---------- BM25 params ----------
    BM25_K1: float = 1.5
    BM25_B:  float = 0.75

    def __init__(self, tables: _MemoryTables):
        super().__init__(tables.symbols, tables.lock)
        self._file_items = tables.files          # needed for package lookup

        self._embeddings: dict[str, np.ndarray] = {}

    @staticmethod
    def _norm(v: list[float]) -> np.ndarray:
        a = np.asarray(v, dtype="float32")
        n = np.linalg.norm(a)
        return a / n if n else a

    def _index_upsert(self, sym: SymbolMetadata) -> None:
        if not sym.embedding_code_vec:
            return

        with self._lock:
            self._embeddings[sym.id] = self._norm(sym.embedding_code_vec)

    def _index_remove(self, sid: str) -> None:
        with self._lock:
            self._embeddings.pop(sid, None)

    def create(self, item: SymbolMetadata) -> SymbolMetadata:
        res = super().create(item)
        self._index_upsert(item)
        return res

    def update(self, item_id: str, data: Dict[str, Any]) -> Optional[SymbolMetadata]:
        res = super().update(item_id, data)
        if res:
            if res.embedding_code_vec:
                self._index_upsert(res)
            else:
                self._index_remove(item_id)
        return res

    def delete(self, item_id: str) -> bool:
        ok = super().delete(item_id)
        if ok:
            self._index_remove(item_id)
        return ok

    def get_list_by_ids(self, symbol_ids: list[str]) -> list[SymbolMetadata]:  # NEW
        syms = super().get_list_by_ids(symbol_ids)
        SymbolMetadata.resolve_symbol_hierarchy(syms)
        return syms

    def get_list(self, flt: SymbolFilter) -> list[SymbolMetadata]:
        """
        Return all SymbolMetadata objects that satisfy *flt*.
        Supports ids, parent_ids, file_id and/or package_id.
        """
        with self._lock:
            syms = [
                s for s in self._items.values()
                and (not flt.parent_ids or s.parent_symbol_id in flt.parent_ids)
                and (not flt.file_id    or s.file_id == flt.file_id)
                and (not flt.package_id or s.package_id == flt.package_id)
            ]
        SymbolMetadata.resolve_symbol_hierarchy(syms)
        return syms

class InMemoryImportEdgeRepository(InMemoryBaseRepository[ImportEdge], AbstractImportEdgeRepository):
    def __init__(self, tables: _MemoryTables):
        super().__init__(tables.edges, tables.lock)

    def get_list_by_source_package_id(self, package_id: str) -> list[ImportEdge]:
        """Return all import-edges whose *from_package_id* equals *package_id*."""
        with self._lock:
            return [edge for edge in self._items.values() if edge.from_package_id == package_id]

    def get_list_by_source_file_id(self, file_id: str) -> list[ImportEdge]:
        with self._lock:
            return [edge for edge in self._items.values() if edge.from_file_id == file_id]

    def get_list_by_repo_id(self, repo_id: str) -> list[ImportEdge]:
        """Return all import-edges whose ``repo_id`` matches *repo_id*."""
        with self._lock:
            return [edge for edge in self._items.values() if edge.repo_id == repo_id]

    def get_list(self, flt: ImportFilter) -> list[ImportEdge]:      # NEW
        with self._lock:
            return [
                edge for edge in self._items.values()
                if (not flt.source_package_id or edge.from_package_id == flt.source_package_id)
                and (not flt.source_file_id  or edge.from_file_id    == flt.source_file_id)
                and (not flt.repo_id         or edge.repo_id         == flt.repo_id)
            ]


class InMemorySymbolRefRepository(InMemoryBaseRepository[SymbolRef],
                                  AbstractSymbolRefRepository):
    def __init__(self, tables: _MemoryTables):
        super().__init__(tables.symbolrefs, tables.lock)

    def get_list_by_file_id(self, file_id: str) -> list[SymbolRef]:
        with self._lock:
            return [r for r in self._items.values() if r.file_id == file_id]

    def get_list_by_package_id(self, package_id: str) -> list[SymbolRef]:
        with self._lock:
            return [r for r in self._items.values() if r.package_id == package_id]

    def get_list_by_repo_id(self, repo_id: str) -> list[SymbolRef]:
        with self._lock:
            return [r for r in self._items.values() if r.repo_id == repo_id]

    # NEW ---------------------------------------------------------------
    def delete_by_file_id(self, file_id: str) -> int:
        with self._lock:
            to_delete = [rid for rid, ref in self._items.items() if ref.file_id == file_id]
            for rid in to_delete:
                self._items.pop(rid, None)
        return len(to_delete)

class InMemoryDataRepository(AbstractDataRepository):
    def __init__(self):
        tables = _MemoryTables()
        self._repo = InMemoryRepoMetadataRepository(tables)
        self._file = InMemoryFileMetadataRepository(tables)
        self._package = InMemoryPackageMetadataRepository(tables)
        self._symbol = InMemorySymbolMetadataRepository(tables)
        self._importedge = InMemoryImportEdgeRepository(tables)
        self._symbolref  = InMemorySymbolRefRepository(tables)

    def close(self):
        pass

    @property
    def repo(self) -> AbstractRepoMetadataRepository:
        """Access the repo metadata repository."""
        return self._repo

    @property
    def package(self) -> AbstractPackageMetadataRepository:
        """Access the package metadata repository."""
        return self._package

    @property
    def file(self) -> AbstractFileMetadataRepository:
        """Access the file metadata repository."""
        return self._file

    @property
    def symbol(self) -> AbstractSymbolMetadataRepository:
        """Access the symbol metadata repository."""
        return self._symbol

    @property
    def importedge(self) -> AbstractImportEdgeRepository:
        """Access the import edge repository."""
        return self._importedge

    @property
    def symbolref(self) -> AbstractSymbolRefRepository:
        return self._symbolref

    def refresh_full_text_indexes(self) -> None:  # new – required by interface
        return
