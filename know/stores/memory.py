from typing import Optional, Dict, Any, List, TypeVar, Generic
from know.models import (
    RepoMetadata,
    PackageMetadata,
    FileMetadata,
    SymbolMetadata,
    ImportEdge,
)
from know.data import (
    AbstractRepoMetadataRepository,
    AbstractPackageMetadataRepository,
    AbstractFileMetadataRepository,
    AbstractSymbolMetadataRepository,
    AbstractImportEdgeRepository,
    AbstractDataRepository,
    SymbolSearchQuery,
)
from dataclasses import dataclass, field
import math

def _cosine(a: list[float], b: list[float]) -> float:
    """
    Return cosine similarity of two equal-length numeric vectors.
    If either vector has zero norm, –1.0 is returned (so it ranks last).
    """
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else -1.0

T = TypeVar("T")

@dataclass
class _MemoryTables:
    repos:   dict[str, RepoMetadata]   = field(default_factory=dict)
    packages:dict[str, PackageMetadata]= field(default_factory=dict)
    files:   dict[str, FileMetadata]   = field(default_factory=dict)
    symbols: dict[str, SymbolMetadata] = field(default_factory=dict)
    edges:   dict[str, ImportEdge]     = field(default_factory=dict)

class InMemoryBaseRepository(Generic[T]):
    def __init__(self, table: Dict[str, T]):
        self._items = table

    def get_by_id(self, item_id: str) -> Optional[T]:
        """Get an item by its ID."""
        return self._items.get(item_id)

    def get_list_by_ids(self, item_ids: list[str]) -> list[T]:
        """Get a list of items by their IDs."""
        return [self._items[iid] for iid in item_ids if iid in self._items]

    def create(self, item: T) -> T:
        """Create a new item entry."""
        self._items[item.id] = item
        return item

    def update(self, item_id: str, data: Dict[str, Any]) -> Optional[T]:
        """Update an item by its ID."""
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
        return self._items.pop(item_id, None) is not None

class InMemoryRepoMetadataRepository(InMemoryBaseRepository[RepoMetadata], AbstractRepoMetadataRepository):
    def __init__(self, tables: _MemoryTables):
        super().__init__(tables.repos)

    def get_by_path(self, root_path: str) -> Optional[RepoMetadata]:
        """Get a repo by its root path."""
        for repo in self._items.values():
            if repo.root_path == root_path:
                return repo
        return None

class InMemoryPackageMetadataRepository(InMemoryBaseRepository[PackageMetadata], AbstractPackageMetadataRepository):
    def __init__(self, tables: _MemoryTables):
        super().__init__(tables.packages)
        self._file_items = tables.files

    def get_by_path(self, path: str) -> Optional[PackageMetadata]:
        """
        Return the first PackageMetadata whose physical_path equals *path*.
        """
        for pkg in self._items.values():
            if pkg.physical_path == path:
                return pkg
        return None

    def get_list_by_repo_id(self, repo_id: str) -> list[PackageMetadata]:
        """Return all packages that belong to *repo_id*."""
        return [pkg for pkg in self._items.values() if pkg.repo_id == repo_id]

    def delete_orphaned(
        self,
    ) -> int:
        """
        Delete every PackageMetadata that is not referenced by any
        FileMetadata in *file_repo*.  Returns the number of deletions.
        """
        used_pkg_ids = {f.package_id for f in self._file_items.values() if f.package_id}
        removed = 0
        for pkg_id in list(self._items):
            if pkg_id not in used_pkg_ids:
                self.delete(pkg_id)
                removed += 1
        return removed

class InMemoryFileMetadataRepository(
    InMemoryBaseRepository[FileMetadata],
    AbstractFileMetadataRepository,
):
    def __init__(self, tables: _MemoryTables):
        super().__init__(tables.files)

    def get_by_path(self, path: str) -> Optional[FileMetadata]:
        """Get a file by its project-relative path."""
        for file in self._items.values():
            if file.path == path:
                return file
        return None

    def get_list_by_repo_id(self, repo_id: str) -> list[FileMetadata]:
        """Return all files whose ``repo_id`` matches *repo_id*."""
        return [f for f in self._items.values() if f.repo_id == repo_id]

    def get_list_by_package_id(self, package_id: str) -> list[FileMetadata]:
        """Return all files whose ``package_id`` matches *package_id*."""
        return [f for f in self._items.values() if f.package_id == package_id]

class InMemorySymbolMetadataRepository(InMemoryBaseRepository[SymbolMetadata], AbstractSymbolMetadataRepository):
    def __init__(self, tables: _MemoryTables):
        super().__init__(tables.symbols)

    def get_list_by_file_id(self, file_id: str) -> list[SymbolMetadata]:
        """Return all symbols that belong to the given *file_id*."""
        return [sym for sym in self._items.values() if sym.file_id == file_id]

    def search(self, repo_id: str, query: SymbolSearchQuery) -> list[SymbolMetadata]:
        # --- initial candidate set: symbols that belong to the requested repo
        res: list[SymbolMetadata] = [
            s for s in self._items.values() if getattr(s, "repo_id", None) == repo_id
        ]

        # ---------- scalar filters ----------
        if query.symbol_name:
            needle = query.symbol_name.lower()
            res = [s for s in res if needle in (s.name or "").lower()]

        if query.symbol_kind:
            res = [s for s in res if s.kind == query.symbol_kind]

        if query.symbol_visibility:
            res = [s for s in res if s.visibility == query.symbol_visibility]

        # ---------- doc / comment full-text search ----------
        if query.doc_needle:
            needles = [n.lower() for n in query.doc_needle]

            def _matches(s: SymbolMetadata) -> bool:
                haystack = f"{s.docstring or ''} {s.comment or ''}".lower()
                return all(n in haystack for n in needles)

            res = [s for s in res if _matches(s)]

        # ---------- embedding similarity ----------
        if query.embedding_query:
            qvec = query.embedding_query

            scores = {
                s.id: _cosine(qvec, s.embedding_code_vec)      # type: ignore[arg-type]
                for s in res if s.embedding_code_vec
            }
            res.sort(key=lambda s: scores.get(s.id, -1.0), reverse=True)
        else:
            res.sort(key=lambda s: s.name or "")

        # ---------- pagination ----------
        offset = query.offset or 0
        limit = query.limit or 20
        return res[offset: offset + limit]

class InMemoryImportEdgeRepository(InMemoryBaseRepository[ImportEdge], AbstractImportEdgeRepository):
    def __init__(self, tables: _MemoryTables):
        super().__init__(tables.edges)

    def get_list_by_source_package_id(self, package_id: str) -> list[ImportEdge]:
        """Return all import-edges whose *from_package_id* equals *package_id*."""
        return [edge for edge in self._items.values() if edge.from_package_id == package_id]

    def get_list_by_repo_id(self, repo_id: str) -> list[ImportEdge]:
        """Return all import-edges whose ``repo_id`` matches *repo_id*."""
        return [edge for edge in self._items.values() if edge.repo_id == repo_id]

class InMemoryDataRepository(AbstractDataRepository):
    def __init__(self):
        tables = _MemoryTables()
        self._repo = InMemoryRepoMetadataRepository(tables)
        self._file = InMemoryFileMetadataRepository(tables)
        self._package = InMemoryPackageMetadataRepository(tables)
        self._symbol = InMemorySymbolMetadataRepository(tables)
        self._importedge = InMemoryImportEdgeRepository(tables)

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
