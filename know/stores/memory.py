import re      # for tokenisatio/
import threading
from typing import Optional, Dict, Any, List, TypeVar, Generic
from know.models import (
    RepoMetadata,
    PackageMetadata,
    FileMetadata,
    Node,
    ImportEdge,
    NodeRef,
)
from know.data import (
    AbstractRepoMetadataRepository,
    AbstractPackageMetadataRepository,
    AbstractFileMetadataRepository,
    AbstractNodeRepository,
    AbstractImportEdgeRepository,
    AbstractNodeRefRepository,
    AbstractDataRepository,
    NodeSearchQuery,
    PackageFilter,
    FileFilter,
    NodeFilter,
    ImportEdgeFilter,
    include_direct_descendants,
    resolve_symbol_hierarchy,
)
from know.data import NodeRefFilter
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
    repos:      dict[str, RepoMetadata]   = field(default_factory=dict)
    packages:   dict[str, PackageMetadata]= field(default_factory=dict)
    files:      dict[str, FileMetadata]   = field(default_factory=dict)
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

class InMemoryRepoMetadataRepository(InMemoryBaseRepository[RepoMetadata], AbstractRepoMetadataRepository):
    def __init__(self, tables: _MemoryTables):
        super().__init__(tables.repos, tables.lock)

    def get_by_path(self, root_path: str) -> Optional[RepoMetadata]:
        """Get a repo by its root path."""
        with self._lock:
            for repo in self._items.values():
                if repo.root_path == root_path:
                    return repo
        return None

class InMemoryPackageMetadataRepository(InMemoryBaseRepository[PackageMetadata], AbstractPackageMetadataRepository):
    def __init__(self, tables: _MemoryTables):
        super().__init__(tables.packages, tables.lock)
        self._file_items = tables.files

    def get_by_physical_path(self, path: str) -> Optional[PackageMetadata]:
        """
        Return the first PackageMetadata whose physical_path equals *path*.
        """
        with self._lock:
            for pkg in self._items.values():
                if pkg.physical_path == path:
                    return pkg
        return None

    def get_by_virtual_path(self, path: str) -> Optional[PackageMetadata]:
        """
        Return the first PackageMetadata whose physical_path equals *path*.
        """
        with self._lock:
            for pkg in self._items.values():
                if pkg.virtual_path == path:
                    return pkg
        return None

    def get_list(self, flt: PackageFilter) -> list[PackageMetadata]:
        """
        Return all PackageMetadata objects that satisfy *flt*.
        Currently only repo_id is supported.
        """
        with self._lock:
            if flt.repo_id:
                return [pkg for pkg in self._items.values()
                        if pkg.repo_id == flt.repo_id]
            # no filter → return every package
            return list(self._items.values())

    def delete_orphaned(
        self,
    ):
        """
        Delete every PackageMetadata that is not referenced by any
        FileMetadata in *file_repo*.  Returns the number of deletions.
        """
        with self._lock:
            used_pkg_ids = {f.package_id for f in self._file_items.values() if f.package_id}
            for pkg_id in list(self._items):
                if pkg_id not in used_pkg_ids:
                    self.delete(pkg_id)


class InMemoryFileMetadataRepository(InMemoryBaseRepository[FileMetadata],
                                     AbstractFileMetadataRepository):
    def __init__(self, tables: _MemoryTables):
        super().__init__(tables.files, tables.lock)

    def get_by_path(self, path: str) -> Optional[FileMetadata]:
        """Get a file by its project-relative path."""
        with self._lock:
            for file in self._items.values():
                if file.path == path:
                    return file
        return None

    def get_list(self, flt: FileFilter) -> list[FileMetadata]:
        """
        Return all FileMetadata objects that satisfy *flt*.
        Supports filtering by repo_id and/or package_id.
        """
        with self._lock:
            return [
                f for f in self._items.values()
                if (not flt.repo_id   or f.repo_id   == flt.repo_id)
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
        resolve_symbol_hierarchy(syms)
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
                and (not flt.repo_id    or s.repo_id == flt.repo_id)
                and (not flt.file_id    or s.file_id == flt.file_id)
                and (not flt.package_id or s.package_id == flt.package_id)
                and (not flt.symbol_kind or s.kind == flt.symbol_kind)
                and (not flt.symbol_visibility or s.visibility == flt.symbol_visibility)
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

        resolve_symbol_hierarchy(syms)

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

    def search(self, repo_id: str, query: NodeSearchQuery) -> list[Node]:
        with self._lock:
            # candidate set: repo + scalar filters
            candidates: list[Node] = [
                s for s in self._items.values() if getattr(s, "repo_id", None) == repo_id
            ]

            # scalar filters
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
            limit  = query.limit  or 20
            results = results[offset: offset + limit]

        results = include_direct_descendants(self, results)

        return results

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
                and (not flt.repo_id         or edge.repo_id         == flt.repo_id)
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
                and (not flt.repo_id    or r.repo_id    == flt.repo_id)
            ]

    def delete_by_file_id(self, file_id: str) -> None:
        with self._lock:
            to_delete = [rid for rid, ref in self._items.items() if ref.file_id == file_id]
            for rid in to_delete:
                self._items.pop(rid, None)


class InMemoryDataRepository(AbstractDataRepository):
    def __init__(self):
        tables = _MemoryTables()
        self._repo = InMemoryRepoMetadataRepository(tables)
        self._file = InMemoryFileMetadataRepository(tables)
        self._package = InMemoryPackageMetadataRepository(tables)
        self._symbol = InMemoryNodeRepository(tables)
        self._importedge = InMemoryImportEdgeRepository(tables)
        self._symbolref  = InMemoryNodeRefRepository(tables)

    def close(self):
        pass

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
