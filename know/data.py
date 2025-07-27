from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod
from know.models import (
    RepoMetadata,
    PackageMetadata,
    FileMetadata,
    Node,
    ImportEdge,
    SymbolRef,
    NodeKind,
    Visibility,
    Vector,
)
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

# ---------------------------------------------------------------------------
# helper: enrich result-set with direct descendants
# ---------------------------------------------------------------------------
class AbstractRepoMetadataRepository(ABC):
    @abstractmethod
    def get_by_id(self, repo_id: str) -> Optional[RepoMetadata]:
        pass

    @abstractmethod
    def get_list_by_ids(self, repo_ids: List[str]) -> List[RepoMetadata]:
        pass

    @abstractmethod
    def create(self, repo: RepoMetadata) -> RepoMetadata:
        pass

    @abstractmethod
    def update(self, repo_id: str, data: Dict[str, Any]) -> Optional[RepoMetadata]:
        pass

    @abstractmethod
    def delete(self, repo_id: str) -> bool:
        pass

    @abstractmethod
    def get_by_path(self, root_path: str) -> Optional[RepoMetadata]:
        """Get a repo by its root path."""
        pass


@dataclass
class PackageFilter:
    repo_id: Optional[str] = None


class AbstractPackageMetadataRepository(ABC):
    @abstractmethod
    def get_by_id(self, package_id: str) -> Optional[PackageMetadata]:
        pass

    @abstractmethod
    def get_list_by_ids(self, package_ids: List[str]) -> List[PackageMetadata]:
        pass

    @abstractmethod
    def get_list(self, flt: PackageFilter) -> List[PackageMetadata]:
        pass

    @abstractmethod
    def get_by_physical_path(self, root_path: str) -> Optional[PackageMetadata]:
        """Get a repo by its root path."""
        pass

    @abstractmethod
    def get_by_virtual_path(self, root_path: str) -> Optional[PackageMetadata]:
        """Get a repo by its root path."""
        pass

    @abstractmethod
    def delete_orphaned(self) -> None:
        pass

    @abstractmethod
    def create(self, pkg: PackageMetadata) -> PackageMetadata:
        pass

    @abstractmethod
    def update(self, package_id: str, data: Dict[str, Any]) -> Optional[PackageMetadata]:
        pass

    @abstractmethod
    def delete(self, package_id: str) -> bool:
        pass


@dataclass
class FileFilter:
    repo_id: Optional[str] = None
    package_id: Optional[str] = None


class AbstractFileMetadataRepository(ABC):
    @abstractmethod
    def get_by_id(self, file_id: str) -> Optional[FileMetadata]:
        pass

    @abstractmethod
    def get_list_by_ids(self, file_ids: List[str]) -> List[FileMetadata]:
        pass

    @abstractmethod
    def create(self, file: FileMetadata) -> FileMetadata:
        pass

    @abstractmethod
    def update(self, file_id: str, data: Dict[str, Any]) -> Optional[FileMetadata]:
        pass

    @abstractmethod
    def delete(self, file_id: str) -> bool:
        pass

    @abstractmethod
    def get_by_path(self, path: str) -> Optional[FileMetadata]:
        """Get a file by its project-relative path."""
        pass

    @abstractmethod
    def get_list(self, flt: FileFilter) -> List[FileMetadata]:
        pass


# Symbols
@dataclass
class SymbolFilter:
    parent_ids: Optional[List[str]] = None
    repo_id: Optional[str] = None
    file_id: Optional[str] = None
    package_id: Optional[str] = None
    symbol_kind: Optional[NodeKind] = None
    symbol_visibility: Optional[Visibility] = None
    has_embedding: Optional[bool] = None
    top_level_only: Optional[bool] = False
    limit: Optional[int] = None
    offset: Optional[int] = None


@dataclass
class SymbolSearchQuery:
    # Filter by symbol name
    symbol_name: Optional[str] = None
    # Filter by symbol fully qualified name
    symbol_fqn: Optional[str] = None
    # Filter by symbol kind
    symbol_kind: Optional[NodeKind] = None
    # Filter by symbol visiblity
    symbol_visibility: Optional[Visibility] = None
    # Full-text search on symbol documentation or comment
    doc_needle: Optional[str] = None
    # Embedding similarity search
    embedding_query: Optional[Vector] = None
    # Number of records to return. If None is passed, no limit will be applied.
    limit: Optional[int] = None
    # Zero-based offset
    offset: Optional[int] = None


class AbstractNodeRepository(ABC):
    @abstractmethod
    def get_by_id(self, symbol_id: str) -> Optional[Node]:
        pass

    @abstractmethod
    def get_list_by_ids(self, item_ids: List[str]) -> List[Node]:
        pass

    @abstractmethod
    def delete_by_file_id(self, file_id: str) -> None:
        """
        Delete every Node whose ``file_id`` equals *file_id*.
        Return number of deleted rows.
        """
        pass

    @abstractmethod
    def get_list(self, flt: SymbolFilter) -> List[Node]:
        pass

    @abstractmethod
    def search(self, repo_id: str, query: SymbolSearchQuery) -> List[Node]:
        pass

    @abstractmethod
    def create(self, symbol: Node) -> Node:
        pass

    @abstractmethod
    def update(self, symbol_id: str, data: Dict[str, Any]) -> Optional[Node]:
        pass

    @abstractmethod
    def delete(self, symbol_id: str) -> bool:
        pass


@dataclass
class ImportEdgeFilter:
    source_package_id: Optional[str] = None
    source_file_id: Optional[str] = None
    repo_id: Optional[str] = None


class AbstractImportEdgeRepository(ABC):
    @abstractmethod
    def get_by_id(self, edge_id: str) -> Optional[ImportEdge]:
        pass

    @abstractmethod
    def get_list_by_ids(self, edge_ids: List[str]) -> List[ImportEdge]:
        pass

    @abstractmethod
    def get_list(self, flt: ImportEdgeFilter) -> List[ImportEdge]:
        pass

    @abstractmethod
    def create(self, edge: ImportEdge) -> ImportEdge:
        pass

    @abstractmethod
    def update(self, edge_id: str, data: Dict[str, Any]) -> Optional[ImportEdge]:
        pass

    @abstractmethod
    def delete(self, edge_id: str) -> bool:
        pass


@dataclass
class SymbolRefFilter:
    file_id: Optional[str] = None
    package_id: Optional[str] = None
    repo_id: Optional[str] = None


class AbstractSymbolRefRepository(ABC):
    @abstractmethod
    def get_by_id(self, ref_id: str) -> Optional[SymbolRef]:
        pass

    @abstractmethod
    def get_list_by_ids(self, ref_ids: List[str]) -> List[SymbolRef]:
        pass

    @abstractmethod
    def get_list(self, flt: SymbolRefFilter) -> List[SymbolRef]:
        pass

    @abstractmethod
    def create(self, ref: SymbolRef) -> SymbolRef:
        pass

    @abstractmethod
    def update(self, ref_id: str, data: Dict[str, Any]) -> Optional[SymbolRef]:
        pass

    @abstractmethod
    def delete(self, ref_id: str) -> bool:
        pass

    @abstractmethod
    def delete_by_file_id(self, file_id: str) -> None:
        """
        Delete every SymbolRef whose ``file_id`` equals *file_id*.
        Returns the number of deleted rows.
        """
        pass


class AbstractDataRepository(ABC):
    @abstractmethod
    def close(self) -> None:
        pass

    @property
    @abstractmethod
    def repo(self) -> AbstractRepoMetadataRepository:
        pass

    @property
    @abstractmethod
    def package(self) -> AbstractPackageMetadataRepository:
        pass

    @property
    @abstractmethod
    def file(self) -> AbstractFileMetadataRepository:
        pass

    @property
    @abstractmethod
    def symbol(self) -> AbstractNodeRepository:
        pass

    @property
    @abstractmethod
    def importedge(self) -> AbstractImportEdgeRepository:
        pass

    @property
    @abstractmethod
    def symbolref(self) -> AbstractSymbolRefRepository:
        pass

    @abstractmethod
    def refresh_full_text_indexes(self) -> None:
        """Refresh full-text search indexes (noop in back-ends that don’t need it)."""
        pass


# Helpers
def resolve_symbol_hierarchy(symbols: List[Node]) -> None:
    """
    Populate in-memory parent/child links inside *symbols* **in-place**.

    • parent_ref   ↔  points to the parent Node instance
    • children     ↔  list with direct child Node instances

    Function is no-op when list is empty.
    """
    if not symbols:
        return

    id_map: dict[str | None, Node] = {s.id: s for s in symbols if s.id}
    # clear any previous links to avoid duplicates on repeated invocations
    for s in symbols:
        s.children.clear()
        s.parent_ref = None

    for s in symbols:
        pid = s.parent_symbol_id
        if pid and (parent := id_map.get(pid)):
            s.parent_ref = parent
            parent.children.append(s)


def include_parents(
    repo: AbstractNodeRepository,
    symbols: List[Node],
) -> List[Node]:
    """
    Traverse all symbol parents and include them to in the tree.
    """
    source = symbols

    while source:
        parent_ids = {s.parent_symbol_id for s in source if s.parent_symbol_id}
        if not parent_ids:
            break

        parents = {s.id: s for s in repo.get_list_by_ids(list(parent_ids))}

        for s in source:
            if s.parent_symbol_id:
                parent = parents[s.parent_symbol_id]
                s.parent_ref = parent

                for i, c in enumerate(parent.children):
                    if c.id == s.id:
                        parent.children[i] = s
                        break
                else:
                    parent.children.append(s)

        source = list(parents.values())

    return symbols


def include_direct_descendants(
    repo: AbstractNodeRepository,    # repository to fetch children
    symbols: List[Node],             # initial search results
) -> List[Node]:
    """
    Ensure every symbol in *symbols* has its direct descendants attached.
    After resolving the hierarchy, any symbol that became a child of another
    returned symbol is dropped from the top-level list (to avoid duplicates).
    The original order of the parent symbols is preserved.
    """
    if not symbols:
        return symbols

    parent_ids = [s.id for s in symbols if s.id]
    if parent_ids:
        children = repo.get_list(SymbolFilter(parent_ids=parent_ids))
        seen_ids = {s.id for s in symbols}
        for c in children:
            if c.id not in seen_ids:
                symbols.append(c)
                seen_ids.add(c.id)

    resolve_symbol_hierarchy(symbols)

    parent_id_set = set(parent_ids)
    result = [s for s in symbols if s.parent_symbol_id not in parent_id_set]

    result = include_parents(repo, result)

    return result
