from typing import Optional, Dict, Any, List, TypeVar, Generic, Tuple
from abc import ABC, abstractmethod
from know.models import (
    Project,
    ProjectRepo,
    Repo,
    Package,
    File,
    Node,
    ImportEdge,
    NodeRef,
    NodeKind,
    Visibility,
    Vector,
)
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from know.settings import ProjectSettings


T = TypeVar("T")


class AbstractCRUDRepository(ABC, Generic[T]):
    @abstractmethod
    def get_by_id(self, item_id: str) -> Optional[T]:
        pass

    @abstractmethod
    def get_list_by_ids(self, item_ids: List[str]) -> List[T]:
        pass

    @abstractmethod
    def create(self, item: T) -> T:
        pass

    @abstractmethod
    def update(self, item_id: str, data: Dict[str, Any]) -> Optional[T]:
        pass

    @abstractmethod
    def delete(self, item_id: str) -> bool:
        pass

    @abstractmethod
    def delete_many(self, item_ids: List[str]) -> bool:
        pass

    @abstractmethod
    def create_many(self, items: List[T]) -> List[T]:
        pass

    @abstractmethod
    def update_many(self, updates: List[Tuple[str, Dict[str, Any]]]) -> List[T]:
        pass


class AbstractProjectRepository(AbstractCRUDRepository[Project]):
    @abstractmethod
    def get_by_name(self, name) -> Optional[Project]:
        pass


class AbstractProjectRepoRepository(ABC):
    @abstractmethod
    def get_repo_ids(self, project_id) -> List[str]:
        pass

    @abstractmethod
    def add_repo_id(self, project_id, repo_id):
        pass


@dataclass
class RepoFilter:
    project_id: Optional[str] = None


class AbstractRepoRepository(AbstractCRUDRepository[Repo]):
    @abstractmethod
    def get_list(self, flt: RepoFilter) -> List[Repo]:
        pass

    @abstractmethod
    def get_by_name(self, name: str) -> Optional[Repo]:
        """Get a repo by its name."""
        pass

    @abstractmethod
    def get_by_path(self, root_path: str) -> Optional[Repo]:
        """Get a repo by its root path."""
        pass


@dataclass
class PackageFilter:
    repo_ids: Optional[List[str]] = None


class AbstractPackageRepository(AbstractCRUDRepository[Package]):
    @abstractmethod
    def get_list(self, flt: PackageFilter) -> List[Package]:
        pass

    @abstractmethod
    def get_by_physical_path(self, repo_id: str, root_path: str) -> Optional[Package]:
        """Get a repo by its root path."""
        pass

    @abstractmethod
    def get_by_virtual_path(self, repo_id: str, root_path: str) -> Optional[Package]:
        """Get a repo by its root path."""
        pass

    @abstractmethod
    def delete_orphaned(self) -> None:
        pass


@dataclass
class FileFilter:
    repo_ids: Optional[List[str]] = None
    package_id: Optional[str] = None


class AbstractFileRepository(AbstractCRUDRepository[File]):
    @abstractmethod
    def get_by_path(self, repo_id: str, path: str) -> Optional[File]:
        """Get a file by its project-relative path."""
        pass

    @abstractmethod
    def get_list(self, flt: FileFilter) -> List[File]:
        pass


# Nodes
@dataclass
class NodeFilter:
    parent_ids: Optional[List[str]] = None
    repo_ids: Optional[List[str]] = None
    file_id: Optional[str] = None
    package_id: Optional[str] = None
    kind: Optional[NodeKind] = None
    visibility: Optional[Visibility] = None
    has_embedding: Optional[bool] = None
    top_level_only: Optional[bool] = False
    limit: Optional[int] = None
    offset: Optional[int] = None


@dataclass
class NodeSearchQuery:
    # Repo filter
    repo_ids: Optional[List[str]] = None
    # Filter by symbol name
    symbol_name: Optional[str] = None
    # Filter by symbol kind
    kind: Optional[NodeKind] = None
    # Filter by symbol visiblity
    visibility: Optional[Visibility] = None
    # Full-text search on symbol documentation or comment
    doc_needle: Optional[str] = None
    # Embedding similarity search
    embedding_query: Optional[Vector] = None
    # ID of a repo whose symbols should be boosted in search results
    boost_repo_id: Optional[str] = None
    # Boost factor to apply. Only used if boost_repo_id is also provided.
    # Values > 1.0 will boost, < 1.0 will penalize. Default is 1.0 (no change).
    repo_boost_factor: float = 1.0
    # Number of records to return. If None is passed, no limit will be applied.
    limit: Optional[int] = None
    # Zero-based offset
    offset: Optional[int] = None


class AbstractNodeRepository(AbstractCRUDRepository[Node]):
    @property
    @abstractmethod
    def settings(self) -> "ProjectSettings":
        pass

    @abstractmethod
    def delete_by_file_id(self, file_id: str) -> None:
        pass

    @abstractmethod
    def get_list(self, flt: NodeFilter) -> List[Node]:
        pass

    @abstractmethod
    def search(self, query: NodeSearchQuery) -> List[Node]:
        pass


@dataclass
class ImportEdgeFilter:
    repo_ids: Optional[List[str]] = None
    source_package_id: Optional[str] = None
    source_file_id: Optional[str] = None


class AbstractImportEdgeRepository(AbstractCRUDRepository[ImportEdge]):
    @abstractmethod
    def get_list(self, flt: ImportEdgeFilter) -> List[ImportEdge]:
        pass


@dataclass
class NodeRefFilter:
    repo_ids: Optional[List[str]] = None
    file_id: Optional[str] = None
    package_id: Optional[str] = None


class AbstractNodeRefRepository(AbstractCRUDRepository[NodeRef]):
    @abstractmethod
    def get_list(self, flt: NodeRefFilter) -> List[NodeRef]:
        pass

    @abstractmethod
    def delete_by_file_id(self, file_id: str) -> None:
        """
        Delete every NodeRef whose ``file_id`` equals *file_id*.
        Returns the number of deleted rows.
        """
        pass


class AbstractDataRepository(ABC):
    @abstractmethod
    def close(self) -> None:
        pass

    @property
    @abstractmethod
    def settings(self) -> "ProjectSettings":
        pass

    @property
    @abstractmethod
    def project(self) -> AbstractProjectRepository:
        pass

    @property
    @abstractmethod
    def prj_repo(self) -> AbstractProjectRepoRepository:
        pass

    @property
    @abstractmethod
    def repo(self) -> AbstractRepoRepository:
        pass

    @property
    @abstractmethod
    def package(self) -> AbstractPackageRepository:
        pass

    @property
    @abstractmethod
    def file(self) -> AbstractFileRepository:
        pass

    @property
    @abstractmethod
    def node(self) -> AbstractNodeRepository:
        pass

    @property
    @abstractmethod
    def importedge(self) -> AbstractImportEdgeRepository:
        pass

    @property
    @abstractmethod
    def symbolref(self) -> AbstractNodeRefRepository:
        pass

    @abstractmethod
    def refresh_full_text_indexes(self) -> None:
        """Refresh full-text search indexes (noop in back-ends that don’t need it)."""
        pass
