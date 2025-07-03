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

    def get_by_physical_path(self, path: str) -> Optional[PackageMetadata]:
        """
        Return the first PackageMetadata whose physical_path equals *path*.
        """
        for pkg in self._items.values():
            if pkg.physical_path == path:
                return pkg
        return None

    def get_by_virtual_path(self, path: str) -> Optional[PackageMetadata]:
        """
        Return the first PackageMetadata whose physical_path equals *path*.
        """
        for pkg in self._items.values():
            if pkg.virtual_path == path:
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
    RRF_K: int = 60          # tuning-parameter k (Reciprocal-Rank-Fusion)
    # minimum cosine similarity for an embedding to participate in ranking
    EMBEDDING_SIM_THRESHOLD: float = 0.4

    def __init__(self, tables: _MemoryTables):
        super().__init__(tables.symbols)
        self._file_items = tables.files          # needed for package lookup

    def get_list_by_ids(self, symbol_ids: list[str]) -> list[SymbolMetadata]:  # NEW
        syms = super().get_list_by_ids(symbol_ids)
        SymbolMetadata.resolve_symbol_hierarchy(syms)
        return syms

    def get_list_by_file_id(self, file_id: str) -> list[SymbolMetadata]:
        """Return all symbols that belong to the given *file_id*."""
        res = [sym for sym in self._items.values() if sym.file_id == file_id]
        SymbolMetadata.resolve_symbol_hierarchy(res)
        return res

    def get_list_by_package_id(self, package_id: str) -> list[SymbolMetadata]:
        """
        Return all symbols that belong to *package_id* (prefer the symbol’s
        own package_id field.
        """
        res: list[SymbolMetadata] = [
            s for s in self._items.values() if s.package_id == package_id
        ]
        SymbolMetadata.resolve_symbol_hierarchy(res)
        return res

    def search(self, repo_id: str, query: SymbolSearchQuery) -> list[SymbolMetadata]:
        # ---------- candidate set: repo + scalar filters ----------
        candidates: list[SymbolMetadata] = [
            s for s in self._items.values() if getattr(s, "repo_id", None) == repo_id
        ]

        # scalar filters ......................................................
        if query.symbol_fqn:
            needle_fqn = query.symbol_fqn.lower()
            candidates = [s for s in candidates if (s.fqn or "").lower() == needle_fqn]

        if query.symbol_name:
            needle = query.symbol_name.lower()
            candidates = [s for s in candidates if needle in (s.name or "").lower()]

        if query.symbol_kind:
            candidates = [s for s in candidates if s.kind == query.symbol_kind]

        if query.symbol_visibility:
            candidates = [s for s in candidates if s.visibility == query.symbol_visibility]

        if query.top_level_only:
            candidates = [s for s in candidates if s.parent_symbol_id is None]

        has_fts       = bool(query.doc_needle)
        has_embedding = bool(query.embedding_query)

        # ---------------------------------------------------------------
        # unified ranking  (RRF over optional FTS / embedding signals)
        # ---------------------------------------------------------------

        fts_rank : dict[str, int] = {}
        code_rank: dict[str, int] = {}
        doc_rank : dict[str, int] = {}

        # ----- FTS ranks ------------------------------------------------
        if has_fts:
            def _fts_match_pos(sym: SymbolMetadata) -> int:
                haystack = f"{sym.docstring or ''} {sym.comment or ''}".lower()
                return haystack.find(query.doc_needle.lower())

            fts_matches = [s for s in candidates if _fts_match_pos(s) >= 0]
            fts_matches.sort(key=_fts_match_pos)                 # best → rank 1
            fts_rank = {s.id: i + 1 for i, s in enumerate(fts_matches)}

        # ----- embedding ranks -----------------------------------------
        if has_embedding:
            qvec = query.embedding_query          # type: ignore[arg-type]

            code_sims: list[tuple[SymbolMetadata, float]] = []
            for s in candidates:
                if s.embedding_code_vec:
                    sim = _cosine(qvec, s.embedding_code_vec)     # type: ignore[arg-type]
                    if sim >= self.EMBEDDING_SIM_THRESHOLD:
                        code_sims.append((s, sim))
            code_sims.sort(key=lambda t: t[1], reverse=True)
            code_rank = {s.id: i + 1 for i, (s, _) in enumerate(code_sims)}

            doc_sims: list[tuple[SymbolMetadata, float]] = []
            for s in candidates:
                vec = getattr(s, "embedding_doc_vec", None)
                if vec:
                    sim = _cosine(qvec, vec)                      # type: ignore[arg-type]
                    if sim >= self.EMBEDDING_SIM_THRESHOLD:
                        doc_sims.append((s, sim))
            doc_sims.sort(key=lambda t: t[1], reverse=True)
            doc_rank = {s.id: i + 1 for i, (s, _) in enumerate(doc_sims)}

        # ----- fuse with Reciprocal-Rank Fusion ------------------------
        fused_score: dict[str, float] = {}
        for s in candidates:
            score = 0.0
            if s.id in code_rank:
                score += 1.0 / (self.RRF_K + code_rank[s.id])
            if s.id in doc_rank:
                score += 1.0 / (self.RRF_K + doc_rank[s.id])
            if s.id in fts_rank:
                score += 1.0 / (self.RRF_K + fts_rank[s.id])
            fused_score[s.id] = score

        if has_fts or has_embedding:
            ranked_candidates = [
                s for s in candidates
                if s.id in code_rank or s.id in doc_rank or s.id in fts_rank
            ]
            ranked_candidates.sort(
                key=lambda s: (-fused_score.get(s.id, 0.0), s.name or "")
            )
            candidates = ranked_candidates
        else:
            candidates.sort(key=lambda s: s.name or "")

        results = candidates

        # ---------- pagination + hierarchy resolution -------------------------
        offset = query.offset or 0
        limit  = query.limit  or 20
        results = results[offset: offset + limit]
        SymbolMetadata.resolve_symbol_hierarchy(results)
        return results


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

    def refresh_full_text_indexes(self) -> None:  # new – required by interface
        return
