from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
from know.models import (
    RepoMetadata,
    PackageMetadata,
    FileMetadata,
    SymbolMetadata,
    ImportEdge,
)
from typing import TYPE_CHECKING
if TYPE_CHECKING:                      # new
    from .data import AbstractFileMetadataRepository  # type: ignore  # noqa: E402

class AbstractRepoMetadataRepository(ABC):
    @abstractmethod
    def get_by_id(self, repo_id: str) -> Optional[RepoMetadata]:
        pass

    @abstractmethod
    def get_list_by_ids(self, repo_ids: list[str]) -> list[RepoMetadata]:
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

class AbstractPackageMetadataRepository(ABC):
    @abstractmethod
    def get_by_id(self, package_id: str) -> Optional[PackageMetadata]:
        pass

    @abstractmethod
    def get_list_by_ids(self, package_ids: list[str]) -> list[PackageMetadata]:
        pass

    @abstractmethod
    def get_list_by_repo_id(self, repo_id: str) -> list[PackageMetadata]:
        pass

    @abstractmethod
    def get_by_path(self, root_path: str) -> Optional[RepoMetadata]:
        """Get a repo by its root path."""
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

    @abstractmethod
    def delete_orphaned(
        self,
    ) -> int:
        pass

class AbstractFileMetadataRepository(ABC):
    @abstractmethod
    def get_by_id(self, file_id: str) -> Optional[FileMetadata]:
        pass

    @abstractmethod
    def get_list_by_ids(self, file_ids: list[str]) -> list[FileMetadata]:
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
    def get_list_by_repo_id(self, repo_id: str) -> list[FileMetadata]:
        """Return **all** FileMetadata instances that belong to *repo_id*."""
        pass

    @abstractmethod
    def get_list_by_package_id(self, package_id: str) -> list[FileMetadata]:
        """Return **all** FileMetadata instances that belong to *package_id*."""
        pass

class AbstractSymbolMetadataRepository(ABC):
    @abstractmethod
    def get_by_id(self, symbol_id: str) -> Optional[SymbolMetadata]:
        pass

    @abstractmethod
    def get_list_by_ids(self, symbol_ids: list[str]) -> list[SymbolMetadata]:
        pass

    @abstractmethod
    def get_list_by_file_id(self, file_id: str) -> list[SymbolMetadata]:
        """Get symbols for a file"""
        pass

    @abstractmethod
    def create(self, symbol: SymbolMetadata) -> SymbolMetadata:
        pass

    @abstractmethod
    def update(self, symbol_id: str, data: Dict[str, Any]) -> Optional[SymbolMetadata]:
        pass

    @abstractmethod
    def delete(self, symbol_id: str) -> bool:
        pass

class AbstractImportEdgeRepository(ABC):
    @abstractmethod
    def get_by_id(self, edge_id: str) -> Optional[ImportEdge]:
        pass

    @abstractmethod
    def get_list_by_ids(self, edge_ids: list[str]) -> list[ImportEdge]:
        pass

    @abstractmethod
    def get_list_by_source_package_id(self, package_id: str) -> list[ImportEdge]:
        pass

    @abstractmethod
    def get_list_by_repo_id(self, repo_id: str) -> list[ImportEdge]:
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


class AbstractDataRepository(ABC):
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
    def symbol(self) -> AbstractSymbolMetadataRepository:
        pass

    @property
    @abstractmethod
    def importedge(self) -> AbstractImportEdgeRepository:
        pass
