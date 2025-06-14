from typing import Optional, Dict, Any, List, TypeVar, Generic
from know.models import (
    RepoMetadata,
    PackageMetadata,
    FileMetadata,
    SymbolMetadata,
    SymbolEdge,
    ImportEdge,
)
from know.data import (
    AbstractRepoMetadataRepository,
    AbstractPackageMetadataRepository,
    AbstractFileMetadataRepository,
    AbstractSymbolMetadataRepository,
    AbstractSymbolEdgeRepository,
    AbstractImportEdgeRepository,
    AbstractDataRepository,
)

T = TypeVar("T")

class InMemoryBaseRepository(Generic[T], AbstractRepoMetadataRepository):
    def __init__(self):
        """Initialize the in-memory base repository."""
        self._items: Dict[str, T] = {}

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
        # For pydantic models, use .copy(update=...)
        if hasattr(item, "copy"):
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

class InMemoryRepoMetadataRepository(InMemoryBaseRepository[RepoMetadata]):
    def get_by_path(self, root_path: str) -> Optional[RepoMetadata]:
        """Get a repo by its root path."""
        for repo in self._items.values():
            if repo.root_path == root_path:
                return repo
        return None

class InMemoryPackageMetadataRepository(InMemoryBaseRepository[PackageMetadata], AbstractPackageMetadataRepository):
    pass

class InMemoryFileMetadataRepository(InMemoryBaseRepository[FileMetadata]):
    def get_by_path(self, path: str) -> Optional[FileMetadata]:
        """Get a file by its project-relative path."""
        for file in self._items.values():
            if file.path == path:
                return file
        return None

class InMemorySymbolMetadataRepository(InMemoryBaseRepository[SymbolMetadata], AbstractSymbolMetadataRepository):
    pass

class InMemorySymbolEdgeRepository(InMemoryBaseRepository[SymbolEdge]):
    def update(self, edge_id: str, data: Dict[str, Any]) -> Optional[SymbolEdge]:
        """Update a symbol edge by its ID."""
        edge = self._items.get(edge_id)
        if not edge:
            return None
        for k, v in data.items():
            setattr(edge, k, v)
        return edge

class InMemoryImportEdgeRepository(InMemoryBaseRepository[ImportEdge]):
    def update(self, edge_id: str, data: Dict[str, Any]) -> Optional[ImportEdge]:
        """Update an import edge by its ID."""
        edge = self._items.get(edge_id)
        if not edge:
            return None
        for k, v in data.items():
            setattr(edge, k, v)
        return edge

class InMemoryDataRepository(AbstractDataRepository):
    def __init__(self):
        """Initialize the in-memory data repository."""
        self._repo = InMemoryRepoMetadataRepository()
        self._package = InMemoryPackageMetadataRepository()
        self._file = InMemoryFileMetadataRepository()
        self._symbol = InMemorySymbolMetadataRepository()
        self._symboledge = InMemorySymbolEdgeRepository()
        self._importedge = InMemoryImportEdgeRepository()

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
    def symboledge(self) -> AbstractSymbolEdgeRepository:
        """Access the symbol edge repository."""
        return self._symboledge

    @property
    def importedge(self) -> AbstractImportEdgeRepository:
        """Access the import edge repository."""
        return self._importedge
