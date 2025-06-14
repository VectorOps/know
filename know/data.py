from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
from know.models import (
    RepoMetadata,
    PackageMetadata,
    FileMetadata,
    SymbolMetadata,
    SymbolEdge,
    ImportEdge,
)

class AbstractRepoMetadataRepository(ABC):
    @abstractmethod
    def get_one(self, repo_id: str) -> Optional[RepoMetadata]:
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

class AbstractPackageMetadataRepository(ABC):
    @abstractmethod
    def get_one(self, package_id: str) -> Optional[PackageMetadata]:
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

class AbstractFileMetadataRepository(ABC):
    @abstractmethod
    def get_one(self, file_id: str) -> Optional[FileMetadata]:
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

class AbstractSymbolMetadataRepository(ABC):
    @abstractmethod
    def get_one(self, symbol_id: str) -> Optional[SymbolMetadata]:
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

class AbstractSymbolEdgeRepository(ABC):
    @abstractmethod
    def get_one(self, edge_id: str) -> Optional[SymbolEdge]:
        pass

    @abstractmethod
    def create(self, edge: SymbolEdge) -> SymbolEdge:
        pass

    @abstractmethod
    def update(self, edge_id: str, data: Dict[str, Any]) -> Optional[SymbolEdge]:
        pass

    @abstractmethod
    def delete(self, edge_id: str) -> bool:
        pass

class AbstractImportEdgeRepository(ABC):
    @abstractmethod
    def get_one(self, edge_id: str) -> Optional[ImportEdge]:
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
    def symboledge(self) -> AbstractSymbolEdgeRepository:
        pass

    @property
    @abstractmethod
    def importedge(self) -> AbstractImportEdgeRepository:
        pass
