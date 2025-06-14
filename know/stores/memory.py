from typing import Optional, Dict, Any, List
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

class InMemoryRepoMetadataRepository(AbstractRepoMetadataRepository):
    def __init__(self):
        self._repos: Dict[str, RepoMetadata] = {}

    def get_by_id(self, repo_id: str) -> Optional[RepoMetadata]:
        return self._repos.get(repo_id)

    def get_list_by_ids(self, repo_ids: list[str]) -> list[RepoMetadata]:
        return [self._repos[rid] for rid in repo_ids if rid in self._repos]

    def create(self, repo: RepoMetadata) -> RepoMetadata:
        self._repos[repo.id] = repo
        return repo

    def update(self, repo_id: str, data: Dict[str, Any]) -> Optional[RepoMetadata]:
        repo = self._repos.get(repo_id)
        if not repo:
            return None
        updated = repo.copy(update=data)
        self._repos[repo_id] = updated
        return updated

    def delete(self, repo_id: str) -> bool:
        return self._repos.pop(repo_id, None) is not None

class InMemoryPackageMetadataRepository(AbstractPackageMetadataRepository):
    def __init__(self):
        self._pkgs: Dict[str, PackageMetadata] = {}

    def get_by_id(self, package_id: str) -> Optional[PackageMetadata]:
        return self._pkgs.get(package_id)

    def get_list_by_ids(self, package_ids: list[str]) -> list[PackageMetadata]:
        return [self._pkgs[pid] for pid in package_ids if pid in self._pkgs]

    def create(self, pkg: PackageMetadata) -> PackageMetadata:
        self._pkgs[pkg.id] = pkg
        return pkg

    def update(self, package_id: str, data: Dict[str, Any]) -> Optional[PackageMetadata]:
        pkg = self._pkgs.get(package_id)
        if not pkg:
            return None
        updated = pkg.copy(update=data)
        self._pkgs[package_id] = updated
        return updated

    def delete(self, package_id: str) -> bool:
        return self._pkgs.pop(package_id, None) is not None

class InMemoryFileMetadataRepository(AbstractFileMetadataRepository):
    def __init__(self):
        self._files: Dict[str, FileMetadata] = {}

    def get_by_id(self, file_id: str) -> Optional[FileMetadata]:
        return self._files.get(file_id)

    def get_list_by_ids(self, file_ids: list[str]) -> list[FileMetadata]:
        return [self._files[fid] for fid in file_ids if fid in self._files]

    def create(self, file: FileMetadata) -> FileMetadata:
        self._files[file.id] = file
        return file

    def update(self, file_id: str, data: Dict[str, Any]) -> Optional[FileMetadata]:
        file = self._files.get(file_id)
        if not file:
            return None
        updated = file.copy(update=data)
        self._files[file_id] = updated
        return updated

    def delete(self, file_id: str) -> bool:
        return self._files.pop(file_id, None) is not None

class InMemorySymbolMetadataRepository(AbstractSymbolMetadataRepository):
    def __init__(self):
        self._symbols: Dict[str, SymbolMetadata] = {}

    def get_by_id(self, symbol_id: str) -> Optional[SymbolMetadata]:
        return self._symbols.get(symbol_id)

    def get_list_by_ids(self, symbol_ids: list[str]) -> list[SymbolMetadata]:
        return [self._symbols[sid] for sid in symbol_ids if sid in self._symbols]

    def create(self, symbol: SymbolMetadata) -> SymbolMetadata:
        self._symbols[symbol.id] = symbol
        return symbol

    def update(self, symbol_id: str, data: Dict[str, Any]) -> Optional[SymbolMetadata]:
        symbol = self._symbols.get(symbol_id)
        if not symbol:
            return None
        updated = symbol.copy(update=data)
        self._symbols[symbol_id] = updated
        return updated

    def delete(self, symbol_id: str) -> bool:
        return self._symbols.pop(symbol_id, None) is not None

class InMemorySymbolEdgeRepository(AbstractSymbolEdgeRepository):
    def __init__(self):
        self._edges: Dict[str, SymbolEdge] = {}

    def get_by_id(self, edge_id: str) -> Optional[SymbolEdge]:
        return self._edges.get(edge_id)

    def get_list_by_ids(self, edge_ids: list[str]) -> list[SymbolEdge]:
        return [self._edges[eid] for eid in edge_ids if eid in self._edges]

    def create(self, edge: SymbolEdge) -> SymbolEdge:
        self._edges[edge.id] = edge
        return edge

    def update(self, edge_id: str, data: Dict[str, Any]) -> Optional[SymbolEdge]:
        edge = self._edges.get(edge_id)
        if not edge:
            return None
        for k, v in data.items():
            setattr(edge, k, v)
        return edge

    def delete(self, edge_id: str) -> bool:
        return self._edges.pop(edge_id, None) is not None

class InMemoryImportEdgeRepository(AbstractImportEdgeRepository):
    def __init__(self):
        self._edges: Dict[str, ImportEdge] = {}

    def get_by_id(self, edge_id: str) -> Optional[ImportEdge]:
        return self._edges.get(edge_id)

    def get_list_by_ids(self, edge_ids: list[str]) -> list[ImportEdge]:
        return [self._edges[eid] for eid in edge_ids if eid in self._edges]

    def create(self, edge: ImportEdge) -> ImportEdge:
        self._edges[edge.id] = edge
        return edge

    def update(self, edge_id: str, data: Dict[str, Any]) -> Optional[ImportEdge]:
        edge = self._edges.get(edge_id)
        if not edge:
            return None
        for k, v in data.items():
            setattr(edge, k, v)
        return edge

    def delete(self, edge_id: str) -> bool:
        return self._edges.pop(edge_id, None) is not None

class InMemoryDataRepository(AbstractDataRepository):
    def __init__(self):
        self._repo = InMemoryRepoMetadataRepository()
        self._package = InMemoryPackageMetadataRepository()
        self._file = InMemoryFileMetadataRepository()
        self._symbol = InMemorySymbolMetadataRepository()
        self._symboledge = InMemorySymbolEdgeRepository()
        self._importedge = InMemoryImportEdgeRepository()

    @property
    def repo(self) -> AbstractRepoMetadataRepository:
        return self._repo

    @property
    def package(self) -> AbstractPackageMetadataRepository:
        return self._package

    @property
    def file(self) -> AbstractFileMetadataRepository:
        return self._file

    @property
    def symbol(self) -> AbstractSymbolMetadataRepository:
        return self._symbol

    @property
    def symboledge(self) -> AbstractSymbolEdgeRepository:
        return self._symboledge

    @property
    def importedge(self) -> AbstractImportEdgeRepository:
        return self._importedge
