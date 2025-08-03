import re      # for tokenisatio/
import threading
from typing import Optional, Dict, Any, List, TypeVar, Generic
from know.models import (
    Repo,
    Package,
    File,
    Node,
    ImportEdge,
    NodeRef,
    Project,
)
from know.data import (
    AbstractRepoRepository,
    AbstractPackageRepository,
    AbstractFileRepository,
    AbstractNodeRepository,
    AbstractImportEdgeRepository,
    AbstractNodeRefRepository,
    AbstractDataRepository,
    NodeSearchQuery,
    PackageFilter,
    FileFilter,
    NodeFilter,
    ImportEdgeFilter,
    AbstractProjectRepository,
    AbstractProjectRepoRepository,
    RepoFilter,
    NodeRefFilter,
)
from know.data_helpers import (
    include_direct_descendants,
    resolve_node_hierarchy,
    post_process_search_results,
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
    projects:   dict[str, Project] = field(default_factory=dict)
    project_repos: dict[str, set[str]] = field(default_factory=dict)
    repos:      dict[str, Repo]   = field(default_factory=dict)
    packages:   dict[str, Package]= field(default_factory=dict)
    files:      dict[str, File]   = field(default_factory=dict)
    symbols:    dict[str, Node] = field(default_factory=dict)
    edges:      dict[str, ImportEdge]     = field(default_factory=dict)
    symbolrefs: dict[str, NodeRef]      = field(default_factory=dict)
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
            self._items[item.id] = item # type: ignore
        return item

    def update(self, item_id: str, data: Dict[str, Any]) -> Optional[T]:
        """Update an item by its ID."""
        with self._lock:
            item = self._items.get(item_id)
            if not item:
                return None
            if hasattr(item, "model_copy"):
                updated = item.model_copy(update=data)
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

class InMemoryProjectRepository(InMemoryBaseRepository[Project], AbstractProjectRepository):
    def __init__(self, tables: _MemoryTables):
        super().__init__(tables.projects, tables.lock)

    def get_by_name(self, name: str) -> Optional[Project]:
        with self._lock:
            for project in self._items.values():
                if project.name == name:
                    return project
        return None

    def create(self, item: Project) -> Project:
        """Create a new project, enforcing name uniqueness."""
        with self._lock:
            if self.get_by_name(item.name):
                raise ValueError(f"A project with name '{item.name}' already exists.")
            return super().create(item)

    def update(self, item_id: str, data: Dict[str, Any]) -> Optional[Project]:
        """Update a project, enforcing name uniqueness."""
        with self._lock:
            if "name" in data:
                existing = self.get_by_name(data["name"])
                if existing and existing.id != item_id:
                    raise ValueError(f"A project with name '{data['name']}' already exists.")
            return super().update(item_id, data)


class InMemoryProjectRepoRepository(AbstractProjectRepoRepository):
    def __init__(self, tables: _MemoryTables):
        self._project_repos = tables.project_repos
        self._lock = tables.lock

    def get_repo_ids(self, project_id: str) -> List[str]:
        with self._lock:
            return list(self._project_repos.get(project_id, set()))

    def add_repo_id(self, project_id: str, repo_id: str):
        with self._lock:
            if project_id not in self._project_repos:
                self._project_repos[project_id] = set()
            self._project_repos[project_id].add(repo_id)


class InMemoryRepoRepository(InMemoryBaseRepository[Repo], AbstractRepoRepository):
    def __init__(self, tables: _MemoryTables):
        super().__init__(tables.repos, tables.lock)
        self._project_repos = tables.project_repos

    def get_by_name(self, name: str) -> Optional[Repo]:
        """Get a repo by its name."""
        with self._lock:
            for repo in self._items.values():
                if repo.name == name:
                    return repo
        return None

    def get_by_path(self, root_path: str) -> Optional[Repo]:
        """Get a repo by its root path."""
        with self._lock:
            for repo in self._items.values():
                if repo.root_path == root_path:
                    return repo
        return None

    def get_list(self, flt: RepoFilter) -> list[Repo]:
        with self._lock:
            repos = list(self._items.values())
            if flt.project_id:
                repo_ids = self._project_repos.get(flt.project_id, set())
                repos = [r for r in repos if r.id in repo_ids]
            return repos

    def create(self, item: Repo) -> Repo:
        """Create a new repo, enforcing name uniqueness."""
        with self._lock:
            if self.get_by_name(item.name):
                raise ValueError(f"A repo with name '{item.name}' already exists.")
            return super().create(item)

    def update(self, item_id: str, data: Dict[str, Any]) -> Optional[Repo]:
        """Update a repo, enforcing name uniqueness."""
        with self._lock:
            if "name" in data:
                existing = self.get_by_name(data["name"])
                if existing and existing.id != item_id:
                    raise ValueError(f"A repo with name '{data['name']}' already exists.")
            return super().update(item_id, data)

class InMemoryPackageRepository(InMemoryBaseRepository[Package], AbstractPackageRepository):
    def __init__(self, tables: _MemoryTables):
        super().__init__(tables.packages, tables.lock)
        self._file_items = tables.files

    def get_by_physical_path(self, repo_id: str, root_path: str) -> Optional[Package]:
        """
        Return the first Package whose physical_path equals *path*.
        """
        with self._lock:
            for pkg in self._items.values():
                if pkg.repo_id == repo_id and pkg.physical_path == root_path:
                    return pkg
        return None

    def get_by_virtual_path(self, repo_id: str, root_path: str) -> Optional[Package]:
        """
        Return the first Package whose physical_path equals *path*.
        """
        with self._lock:
            for pkg in self._items.values():
                if pkg.repo_id == repo_id and pkg.virtual_path == root_path:
                    return pkg
        return None

    def get_list(self, flt: PackageFilter) -> list[Package]:
        """
        Return all Package objects that satisfy *flt*.
        Currently only repo_id is supported.
        """
        with self._lock:
            if flt.repo_ids:
                return [pkg for pkg in self._items.values()
                        if pkg.repo_id in flt.repo_ids]
            # no filter → return every package
            return list(self._items.values())

    def delete_orphaned(
        self,
    ):
        """
        Delete every Package that is not referenced by any
        File in *file_repo*.  Returns the number of deletions.
        """
        with self._lock:
            used_pkg_ids = {f.package_id for f in self._file_items.values() if f.package_id}
            for pkg_id in list(self._items):
                if pkg_id not in used_pkg_ids:
                    self.delete(pkg_id)


class InMemoryFileRepository(InMemoryBaseRepository[File],
                                     AbstractFileRepository):
    def __init__(self, tables: _MemoryTables):
        super().__init__(tables.files, tables.lock)

    def get_by_path(self, repo_id: str, path: str) -> Optional[File]:
        """Get a file by its project-relative path."""
        with self._lock:
            for file in self._items.values():
                if file.repo_id == repo_id and file.path == path:
                    return file
        return None

    def get_list(self, flt: FileFilter) -> list[File]:
        """
        Return all File objects that satisfy *flt*.
        Supports filtering by repo_id and/or package_id.
        """
        with self._lock:
            return [
                f for f in self._items.values()
                if (not flt.repo_ids   or f.repo_id in flt.repo_ids)
                and (not flt.package_id or f.package_id == flt.package_id)
            ]

class InMemoryNodeRepository(InMemoryBaseRepository[Node], AbstractNodeRepository):
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

    def _index_upsert(self, sym: Node) -> None:
        if not sym.embedding_code_vec:
            return

        with self._lock:
            self._embeddings[sym.id] = self._norm(sym.embedding_code_vec)

    def _index_remove(self, sid: str) -> None:
        with self._lock:
            self._embeddings.pop(sid, None)

    def create(self, item: Node) -> Node:
        res = super().create(item)
        self._index_upsert(item)
        return res

    def update(self, item_id: str, data: Dict[str, Any]) -> Optional[Node]:
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

    def get_list_by_ids(self, symbol_ids: list[str]) -> list[Node]:
        syms = super().get_list_by_ids(symbol_ids)
        resolve_node_hierarchy(syms)
        return syms

    def get_list(self, flt: NodeFilter) -> list[Node]:
        """
        Return all Node objects that satisfy *flt*.
        Supports ids, parent_ids, file_id and/or package_id.
        """
        with self._lock:
            syms = [
                s for s in self._items.values()
                if (not flt.parent_ids or s.parent_node_id in flt.parent_ids)
                and (not flt.repo_ids    or s.repo_id in flt.repo_ids)
                and (not flt.file_id    or s.file_id == flt.file_id)
                and (not flt.package_id or s.package_id == flt.package_id)
                and (not flt.kind or s.kind == flt.kind)
                and (not flt.visibility or s.visibility == flt.visibility)
                and (not flt.top_level_only or s.parent_node_id is None)
                and (
                    flt.has_embedding is None
                    or (flt.has_embedding is True  and s.embedding_code_vec is not None)
                    or (flt.has_embedding is False and s.embedding_code_vec is None)
                )
            ]
        offset = flt.offset or 0
        limit  = flt.limit  or len(syms)
        syms   = syms[offset : offset + limit]

        resolve_node_hierarchy(syms)

        return syms

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Lower-case word tokenisation used by BM25 search."""
        return re.findall(r"\w+", text.lower())

    def _bm25_ranks(
        self,
        docs: list[tuple[Node, list[str]]],
        query_tokens: list[str],
    ) -> dict[str, int]:
        """
        Return {symbol_id -> rank (1-based)} for *docs* against *query_tokens*
        using BM25 (same formula as DuckDB’s match_bm25).
        Symbols with score 0 are omitted.
        """
        N = len(docs)
        if N == 0 or not query_tokens:
            return {}

        avg_dl = sum(len(toks) for _, toks in docs) / N
        # idf per query token
        idf: dict[str, float] = {}
        for tok in query_tokens:
            n_q = sum(1 for _, toks in docs if tok in toks)
            idf[tok] = math.log((N - n_q + 0.5) / (n_q + 0.5) + 1)

        scored: list[tuple[str, float]] = []
        for sym, toks in docs:
            dl  = len(toks)
            tfc = {t: toks.count(t) for t in query_tokens if t in toks}
            score = 0.0
            for t, tf in tfc.items():
                denom  = tf + self.BM25_K1 * (1 - self.BM25_B + self.BM25_B * dl / avg_dl)
                score += idf[t] * tf * (self.BM25_K1 + 1) / denom
            if score > 0:
                scored.append((sym.id, score))

        scored.sort(key=lambda p: p[1], reverse=True)            # best first
        return {sid: rank + 1 for rank, (sid, _) in enumerate(scored)}

    def search(self, query: NodeSearchQuery) -> list[Node]:
        with self._lock:
            # candidate set: repo + scalar filters
            candidates: list[Node] = [
                s for s in self._items.values()
            ]

            # scalar filters
            if query.repo_ids:
                candidates = [s for s in candidates if s.repo_id in query.repo_ids]

            if query.symbol_name:
                needle = query.symbol_name.lower()
                candidates = [s for s in candidates if (s.name or "").lower() == needle]

            if query.kind:
                candidates = [s for s in candidates if s.kind == query.kind]

            if query.visibility:
                candidates = [s for s in candidates if s.visibility == query.visibility]

            has_fts       = bool(query.doc_needle)
            has_embedding = bool(query.embedding_query)

            # unified ranking  (RRF over optional FTS / embedding signals)
            fts_rank : dict[str, int] = {}
            code_rank: dict[str, int] = {}

            # FTS ranks
            if has_fts:
                q_tokens = self._tokenize(query.doc_needle) # type: ignore
                docs = [
                    (
                        s,
                        self._tokenize(
                            f"{s.name or ''} {s.fqn or ''} {s.docstring or ''} {s.comment or ''}"
                        ),
                    )
                    for s in candidates
                ]
                fts_rank = self._bm25_ranks(docs, q_tokens)

            # embedding ranks
            if has_embedding:
                qvec = self._norm(query.embedding_query) # type: ignore
                id_candidates = {c.id for c in candidates}
                sims: list[tuple[Node, float]] = []
                for sid in id_candidates:
                    vec = self._embeddings.get(sid)
                    if vec is None:
                        continue
                    sim = float(np.dot(qvec, vec))  # cosine (both normalised)
                    if sim < self.EMBEDDING_SIM_THRESHOLD:
                        continue
                    sims.append((self._items[sid], sim))
                sims.sort(key=lambda p: p[1], reverse=True)
                sims = sims[: self.EMBEDDING_TOP_K]
                code_rank = {s.id: i + 1 for i, (s, _) in enumerate(sims)}

            # fuse with Reciprocal-Rank Fusion
            fused_score: dict[str, float] = {}
            for s in candidates:
                score = 0.0
                if s.id in code_rank:
                    score += self.RRF_CODE_WEIGHT / (self.RRF_K + code_rank[s.id])
                if s.id in fts_rank:
                    score += self.RRF_FTS_WEIGHT / (self.RRF_K + fts_rank[s.id])

                if (
                    score > 0
                    and query.boost_repo_id
                    and query.repo_boost_factor != 1.0
                    and s.repo_id == query.boost_repo_id
                ):
                    score *= query.repo_boost_factor

                fused_score[s.id] = score

            if has_fts or has_embedding:
                ranked_candidates = [
                    s for s in candidates
                    if s.id in code_rank or s.id in fts_rank
                ]
                ranked_candidates.sort(
                    key=lambda s: (-fused_score.get(s.id, 0.0), s.name or "")
                )
                candidates = ranked_candidates
            else:
                candidates.sort(key=lambda s: s.name or "")

            results = candidates

            offset = query.offset or 0
            limit = query.limit if query.limit is not None else 20
            offset = query.offset if query.offset is not None else 0
            fetch_limit = limit * 2

            results = candidates[offset : offset + fetch_limit]

        limit = query.limit if query.limit is not None else 20
        return post_process_search_results(self, results, limit)

    def delete_by_file_id(self, file_id: str) -> None:
        with self._lock:
            to_delete = [sid for sid, sym in self._items.items() if sym.file_id == file_id]
            for sid in to_delete:
                self._items.pop(sid, None)
                self._index_remove(sid)


class InMemoryImportEdgeRepository(InMemoryBaseRepository[ImportEdge], AbstractImportEdgeRepository):
    def __init__(self, tables: _MemoryTables):
        super().__init__(tables.edges, tables.lock)

    def get_list(self, flt: ImportEdgeFilter) -> list[ImportEdge]:
        with self._lock:
            return [
                edge for edge in self._items.values()
                if (not flt.source_package_id or edge.from_package_id == flt.source_package_id)
                and (not flt.source_file_id  or edge.from_file_id    == flt.source_file_id)
                and (not flt.repo_ids         or edge.repo_id in flt.repo_ids)
            ]


class InMemoryNodeRefRepository(InMemoryBaseRepository[NodeRef],
                                  AbstractNodeRefRepository):
    def __init__(self, tables: _MemoryTables):
        super().__init__(tables.symbolrefs, tables.lock)

    def get_list(self, flt: NodeRefFilter) -> list[NodeRef]:
        with self._lock:
            return [
                r for r in self._items.values()
                if (not flt.file_id    or r.file_id    == flt.file_id)
                and (not flt.package_id or r.package_id == flt.package_id)
                and (not flt.repo_ids    or r.repo_id in flt.repo_ids)
            ]

    def delete_by_file_id(self, file_id: str) -> None:
        with self._lock:
            to_delete = [rid for rid, ref in self._items.items() if ref.file_id == file_id]
            for rid in to_delete:
                self._items.pop(rid, None)


class InMemoryDataRepository(AbstractDataRepository):
    def __init__(self):
        tables = _MemoryTables()
        self._project = InMemoryProjectRepository(tables)
        self._prj_repo = InMemoryProjectRepoRepository(tables)
        self._repo = InMemoryRepoRepository(tables)
        self._file = InMemoryFileRepository(tables)
        self._package = InMemoryPackageRepository(tables)
        self._symbol = InMemoryNodeRepository(tables)
        self._importedge = InMemoryImportEdgeRepository(tables)
        self._symbolref  = InMemoryNodeRefRepository(tables)

    def close(self):
        pass

    @property
    def project(self) -> AbstractProjectRepository:
        """Access the project metadata repository."""
        return self._project

    @property
    def prj_repo(self) -> AbstractProjectRepoRepository:
        """Access the project repo link repository."""
        return self._prj_repo

    @property
    def repo(self) -> AbstractRepoRepository:
        """Access the repo metadata repository."""
        return self._repo

    @property
    def package(self) -> AbstractPackageRepository:
        """Access the package metadata repository"""
        return self._package

    @property
    def file(self) -> AbstractFileRepository:
        """Access the file metadata repository"""
        return self._file

    @property
    def symbol(self) -> AbstractNodeRepository:
        """Access the symbol metadata repository."""
        return self._symbol

    @property
    def importedge(self) -> AbstractImportEdgeRepository:
        """Access the import edge repository."""
        return self._importedge

    @property
    def symbolref(self) -> AbstractNodeRefRepository:
        return self._symbolref

    def refresh_full_text_indexes(self) -> None:  # new – required by interface
        return
