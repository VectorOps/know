import re
import math
from dataclasses import dataclass
from typing import Any, Optional
from collections import defaultdict
from typing import Dict, Set, List
import networkx as nx

from know.logger import KnowLogger as logger
from know.models import (
    FileMetadata, SymbolMetadata, SymbolRef, Visibility
)
from know.project import Project, ScanResult


@dataclass(slots=True)
class NameProps:                # cache entry for a single identifier
    name: str
    visibility: Optional[str]   # "public" / "private" / "protected" / …
    descriptiveness: float      # 0.0 … 1.0


class RepoMap:
    """Keeps an up-to-date call/reference graph (file-level granularity)."""

    DESCRIPTIVENESS_THRESHOLD = 0.5           # ≥0.5 → “descriptive”
    DESCRIPTIVE_MULTIPLIER    = 10.0
    PRIVATE_PROTECTED_MULT    = 0.1
    POLYDEF_THRESHOLD         = 6             # ≥6 distinct definition files
    POLYDEF_MULTIPLIER        = 0.1
    ISOLATED_SELF_WEIGHT      = 0.1           # rule 7
    BOOST_FACTOR_DEFAULT      = 10.0          # for personalization helper

    def __init__(self, project: Project):
        self.project = project
        self.G = nx.MultiDiGraph()
        self._defs: Dict[str, Set[str]] = defaultdict(set)   # was Dict[str, str]
        self._refs: Dict[str, Set[str]] = defaultdict(set)
        self._path_to_fid: Dict[str, str] = {}
        self._name_props: Dict[str, NameProps] = {}      # NEW  name → NameProps

    # ------------------------------------------------------------------  
    #  Debug helpers
    # ------------------------------------------------------------------

    def get_definition_count(self, name: str) -> int:
        return len(self._defs.get(name, []))
    def debug_summary(self) -> str:
        """Return short ‘N-nodes / M-edges’ summary, compatible with all NX ≥2.0."""
        if hasattr(nx, "info"):               # NetworkX < 3.0
            return nx.info(self.G)
        # NetworkX ≥3.0 dropped top-level info()
        return f"{self.G.__class__.__name__} with " \
               f"{self.G.number_of_nodes()} nodes and " \
               f"{self.G.number_of_edges()} edges"

    def debug_edges(self) -> list[str]:
        """Return human-readable list of all edges with optional *name* attr."""
        return [f"{u} -> {v} ({d.get('name', '')})" for u, v, d in self.G.edges(data=True)]

    def to_dot(self) -> str:
        """
        Render the graph as a GraphViz DOT string using NetworkX’s
        ``nx_pydot.to_pydot`` helper.  Falls back to edge list if pydot
        is unavailable.
        """
        try:
            from networkx.drawing.nx_pydot import to_pydot
            return to_pydot(self.G).to_string()
        except Exception as exc:       # ImportError or other pydot issues
            logger.debug("DOT rendering failed: %s – falling back to edge list", exc)
            return "\n".join(self.debug_edges())

    # ------------------------------------------------------------------
    #  Name-property helpers                                # NEW
    # ------------------------------------------------------------------
    @staticmethod
    def _calc_descriptiveness(name: str) -> float:
        """
        Very light heuristic: split camel- / snake-case into words,
        clamp (#words / 5) to [0,1].
        """
        if not name:
            return 0.0
        # camelCase → snake_case
        snake = re.sub(r'(?<!^)(?=[A-Z])', '_', name)
        words = [w for w in snake.split('_') if w]
        return round(min(len(words) / 5.0, 1.0), 3)

    def _make_props(self, sym: SymbolMetadata) -> NameProps:
        """Build NameProps instance from a SymbolMetadata object."""
        vis = sym.visibility.value if isinstance(sym.visibility, Visibility) else (
            str(sym.visibility) if sym.visibility else None
        )
        return NameProps(
            name=sym.name,
            visibility=vis,
            descriptiveness=self._calc_descriptiveness(sym.name),
        )

    def get_name_properties(self, name: str) -> Optional[NameProps]:
        """External accessor used by recommendation engine."""
        return self._name_props.get(name)

    # ------------------------------------------------------------------
    #  Edge-weighting, self-loop, and personalization helpers
    # ------------------------------------------------------------------
    def _compute_edge_weight(self, name: str) -> float:
        weight = 1.0
        props = self._name_props.get(name)
        if props and props.descriptiveness >= self.DESCRIPTIVENESS_THRESHOLD:
            weight *= self.DESCRIPTIVE_MULTIPLIER
        if props and props.visibility in ("private", "protected"):
            weight *= self.PRIVATE_PROTECTED_MULT
        if len(self._defs.get(name, [])) >= self.POLYDEF_THRESHOLD:
            weight *= self.POLYDEF_MULTIPLIER
        refs_cnt = len(self._refs.get(name, [])) or 1
        weight *= math.sqrt(refs_cnt)          # rule 5
        return weight

    def _ensure_self_loop(self, fid: str) -> None:
        """Guarantee rule 7 for *fid*."""
        if self.G.has_edge(fid, fid):
            return
        if self.G.degree(fid) == 0:
            self.G.add_edge(fid, fid, name="__self__", weight=self.ISOLATED_SELF_WEIGHT)

    def build_pagerank_personalization(
            self,
            *,
            boost_paths: list[str] | None = None,
            boost_symbols: list[str] | None = None,
            boost_factor: float = BOOST_FACTOR_DEFAULT,
        ) -> dict[str, float]:
        """Return NX PageRank personalization vector."""
        boost_paths = boost_paths or []
        boost_symbols = boost_symbols or []
        weights: dict[str, float] = {}
        for n in self.G.nodes:
            # base weight = weighted degree; fall back to 1.0 when the node is isolated
            w = float(self.G.degree(n, weight="weight")) or 1.0
            weights[n] = w
        for p in boost_paths:
            fid = self._path_to_fid.get(p)
            if fid:
                weights[fid] *= boost_factor
        for s in boost_symbols:
            for fid in self._defs.get(s, set()) | self._refs.get(s, set()):
                weights[fid] *= boost_factor
        total = sum(weights.values()) or 1.0
        return {n: w / total for n, w in weights.items()}

    # ------------------------------------------------------------------  
    #  Initial full build
    # ------------------------------------------------------------------
    def build_initial_graph(self) -> None:
        repo_id = self.project.get_repo().id
        file_repo     = self.project.data_repository.file
        symbol_repo   = self.project.data_repository.symbol
        symbolref_repo = self.project.data_repository.symbolref

        for fm in file_repo.get_list_by_repo_id(repo_id):
            self._path_to_fid[fm.path] = fm.id
            # collect defs
            for sym in symbol_repo.get_list_by_file_id(fm.id):
                if sym.name:
                    self._defs[sym.name].add(fm.id)
                    self._name_props[sym.name] = self._make_props(sym)   # NEW
            # collect refs
            for ref in symbolref_repo.get_list_by_file_id(fm.id):
                if ref.name:
                    self._refs[ref.name].add(fm.id)

        # materialise edges
        for name, def_files in self._defs.items():
            for ref_file in self._refs.get(name, []):
                for def_file in def_files:
                    w = self._compute_edge_weight(name)
                    self.G.add_edge(ref_file, def_file, name=name, weight=w)
        # Ensure self-loops for all nodes (rule 7)
        for fid in self._path_to_fid.values():
            self._ensure_self_loop(fid)

    # ------------------------------------------------------------------  
    #  Incremental refresh
    # ------------------------------------------------------------------
    def refresh(self, scan: ScanResult) -> None:
        """Update graph after running `scanner.scan_project_directory`."""
        file_repo     = self.project.data_repository.file
        symbol_repo   = self.project.data_repository.symbol
        symbolref_repo = self.project.data_repository.symbolref

        # ----- handle deleted files ------------------------------------
        for rel_path in scan.files_deleted:
            fid = self._path_to_fid.pop(rel_path, None)
            if fid is None:
                continue
            # drop node & incident edges
            if self.G.has_node(fid):
                self.G.remove_node(fid)
            # purge defs / refs caches
            for name, def_files in list(self._defs.items()):
                if fid in def_files:
                    def_files.remove(fid)
                    if not def_files:
                        del self._defs[name]
                        self._name_props.pop(name, None)
            for refs in self._refs.values():
                refs.discard(fid)

        # ----- handle added & updated files ----------------------------
        changed = list(scan.files_added) + list(scan.files_updated)
        for rel_path in changed:
            fm = file_repo.get_by_path(rel_path)
            if fm is None:
                continue
            fid = fm.id
            self._path_to_fid[rel_path] = fid

            # remove existing node & caches for this file
            if self.G.has_node(fid):
                self.G.remove_node(fid)
            for name, def_files in list(self._defs.items()):
                if fid in def_files:
                    def_files.remove(fid)
                    if not def_files:
                        del self._defs[name]
                        self._name_props.pop(name, None)
            for refs in self._refs.values():
                refs.discard(fid)

            # ----- rebuild caches for this file -----------------------
            new_def_names: Set[str] = set()
            new_ref_names: Set[str] = set()

            for sym in symbol_repo.get_list_by_file_id(fid):
                if sym.name:
                    self._defs[sym.name].add(fid)
                    self._name_props[sym.name] = self._make_props(sym)   # NEW
                    new_def_names.add(sym.name)

            for ref in symbolref_repo.get_list_by_file_id(fid):
                if ref.name:
                    self._refs[ref.name].add(fid)
                    new_ref_names.add(ref.name)

            # ensure node exists
            self.G.add_node(fid)

            # edges from this file’s refs → all its targets
            for name in new_ref_names:
                w = self._compute_edge_weight(name)
                for def_fid in self._defs.get(name, []):
                    self.G.add_edge(fid, def_fid, name=name, weight=w)

            # edges from other refs → new defs in this file
            for name in new_def_names:
                w = self._compute_edge_weight(name)
                for ref_fid in self._refs.get(name, []):
                    self.G.add_edge(ref_fid, fid, name=name, weight=w)

            # ensure self-loop for this node if needed
            self._ensure_self_loop(fid)

        # Ensure self-loops for all nodes (rule 7)
        for node in list(self.G.nodes):
            self._ensure_self_loop(node)
