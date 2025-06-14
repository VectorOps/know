from typing import Optional, Dict, Any, List
from know.models import (
    RepoMetadata,
    PackageMetadata,
    FileMetadata,
    SymbolMetadata,
    SymbolEdge,
    ImportEdge,
)

# In-memory storage for demonstration purposes.
# In a real implementation, this would be replaced with persistent storage.
_repo_store: Dict[str, RepoMetadata] = {}
_package_store: Dict[str, PackageMetadata] = {}
_file_store: Dict[str, FileMetadata] = {}
_symbol_store: Dict[str, SymbolMetadata] = {}
_symboledge_store: Dict[str, SymbolEdge] = {}
_importedge_store: Dict[str, ImportEdge] = {}

class RepoMetadataRepository:
    @staticmethod
    def get_one(repo_id: str) -> Optional[RepoMetadata]:
        return _repo_store.get(repo_id)

    @staticmethod
    def create(repo: RepoMetadata) -> RepoMetadata:
        _repo_store[repo.id] = repo
        return repo

    @staticmethod
    def update(repo_id: str, data: Dict[str, Any]) -> Optional[RepoMetadata]:
        repo = _repo_store.get(repo_id)
        if not repo:
            return None
        updated = repo.copy(update=data)
        _repo_store[repo_id] = updated
        return updated

    @staticmethod
    def delete(repo_id: str) -> bool:
        return _repo_store.pop(repo_id, None) is not None

class PackageMetadataRepository:
    @staticmethod
    def get_one(package_id: str) -> Optional[PackageMetadata]:
        return _package_store.get(package_id)

    @staticmethod
    def create(pkg: PackageMetadata) -> PackageMetadata:
        _package_store[pkg.id] = pkg
        return pkg

    @staticmethod
    def update(package_id: str, data: Dict[str, Any]) -> Optional[PackageMetadata]:
        pkg = _package_store.get(package_id)
        if not pkg:
            return None
        updated = pkg.copy(update=data)
        _package_store[package_id] = updated
        return updated

    @staticmethod
    def delete(package_id: str) -> bool:
        return _package_store.pop(package_id, None) is not None

class FileMetadataRepository:
    @staticmethod
    def get_one(file_id: str) -> Optional[FileMetadata]:
        return _file_store.get(file_id)

    @staticmethod
    def create(file: FileMetadata) -> FileMetadata:
        _file_store[file.id] = file
        return file

    @staticmethod
    def update(file_id: str, data: Dict[str, Any]) -> Optional[FileMetadata]:
        file = _file_store.get(file_id)
        if not file:
            return None
        updated = file.copy(update=data)
        _file_store[file_id] = updated
        return updated

    @staticmethod
    def delete(file_id: str) -> bool:
        return _file_store.pop(file_id, None) is not None

class SymbolMetadataRepository:
    @staticmethod
    def get_one(symbol_id: str) -> Optional[SymbolMetadata]:
        return _symbol_store.get(symbol_id)

    @staticmethod
    def create(symbol: SymbolMetadata) -> SymbolMetadata:
        _symbol_store[symbol.id] = symbol
        return symbol

    @staticmethod
    def update(symbol_id: str, data: Dict[str, Any]) -> Optional[SymbolMetadata]:
        symbol = _symbol_store.get(symbol_id)
        if not symbol:
            return None
        updated = symbol.copy(update=data)
        _symbol_store[symbol_id] = updated
        return updated

    @staticmethod
    def delete(symbol_id: str) -> bool:
        return _symbol_store.pop(symbol_id, None) is not None

class SymbolEdgeRepository:
    @staticmethod
    def get_one(edge_id: str) -> Optional[SymbolEdge]:
        return _symboledge_store.get(edge_id)

    @staticmethod
    def create(edge: SymbolEdge) -> SymbolEdge:
        _symboledge_store[edge.id] = edge
        return edge

    @staticmethod
    def update(edge_id: str, data: Dict[str, Any]) -> Optional[SymbolEdge]:
        edge = _symboledge_store.get(edge_id)
        if not edge:
            return None
        for k, v in data.items():
            setattr(edge, k, v)
        _symboledge_store[edge_id] = edge
        return edge

    @staticmethod
    def delete(edge_id: str) -> bool:
        return _symboledge_store.pop(edge_id, None) is not None

class ImportEdgeRepository:
    @staticmethod
    def get_one(edge_id: str) -> Optional[ImportEdge]:
        return _importedge_store.get(edge_id)

    @staticmethod
    def create(edge: ImportEdge) -> ImportEdge:
        _importedge_store[edge.id] = edge
        return edge

    @staticmethod
    def update(edge_id: str, data: Dict[str, Any]) -> Optional[ImportEdge]:
        edge = _importedge_store.get(edge_id)
        if not edge:
            return None
        for k, v in data.items():
            setattr(edge, k, v)
        _importedge_store[edge_id] = edge
        return edge

    @staticmethod
    def delete(edge_id: str) -> bool:
        return _importedge_store.pop(edge_id, None) is not None

# Optionally, a facade for all repositories
class DataRepository:
    repo = RepoMetadataRepository
    package = PackageMetadataRepository
    file = FileMetadataRepository
    symbol = SymbolMetadataRepository
    symboledge = SymbolEdgeRepository
    importedge = ImportEdgeRepository
