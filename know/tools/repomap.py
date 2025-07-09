import re
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Dict, Set, List, Sequence

import networkx as nx

from know.tools.file_summary_helper import build_file_summary

from pydantic import BaseModel

from know.logger import KnowLogger as logger
from know.models import SymbolMetadata, Visibility
from know.project import Project, ScanResult, ProjectComponent
from know.tools.base import BaseTool          # ← new


# ------------------------------------------------------------------
#  Module-level constants
# ------------------------------------------------------------------
SYMBOL_EDGE_BOOST: float  = 15.0       # stronger multiplier – must be > FILE_EDGE_BOOST (10)
FILE_EDGE_BOOST:   float  = 10.0      # multiplier for edges incident to requested files
DESCRIPTIVE_MULTIPLIER    = 4.0
PRIVATE_PROTECTED_MULT    = 0.1
POLYDEF_THRESHOLD         = 6             # >= 6 distinct definition files
POLYDEF_MULTIPLIER        = 0.1
ISOLATED_SELF_WEIGHT      = 0.3           # rule 7
BOOST_FACTOR_DEFAULT      = 10.0          # for personalization helper


@dataclass(slots=True)
class NameProps:
    name: str
    visibility: Optional[str]   # "public" / "private" / "protected" / …
    descriptiveness: float      # 0.0 ... 1.0


class RepoMap(ProjectComponent):
    """Keeps an up-to-date call/reference graph (file-level granularity)."""
    component_name = "repomap"

    def __init__(self, project: Project):
        super().__init__(project)
        self.G = nx.MultiDiGraph()
        self._defs: Dict[str, Set[str]] = defaultdict(set)   # now stores file paths
        self._refs: Dict[str, Set[str]] = defaultdict(set)   # now stores file paths
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
        Very light heuristic: split camel / snake-case into words,
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
    def _compute_edge_data(self, name: str) -> dict[str, float]:
        base = 1.0
        props = self._name_props.get(name)
        if props:
            base *= props.descriptiveness * DESCRIPTIVE_MULTIPLIER
            if props.visibility in (Visibility.PRIVATE.value,
                                    Visibility.PROTECTED.value):
                base *= PRIVATE_PROTECTED_MULT

        defs_cnt = len(self._defs.get(name, []))
        if defs_cnt >= POLYDEF_THRESHOLD:
            base *= POLYDEF_MULTIPLIER

        refs_cnt = len(self._refs.get(name, []))
        weight   = base * math.log1p(refs_cnt)

        return {
            "base_weight": base,
            "defs_cnt":   defs_cnt,
            "refs_cnt":   refs_cnt,
            "weight":     weight,
        }

    def _ensure_self_loop(self, fid: str) -> None:
        if self.G.has_edge(fid, fid):
            return

        #if self.G.degree(fid) == 0:
        self.G.add_edge(fid, fid, name="", weight=ISOLATED_SELF_WEIGHT)

    # ------------------------------------------------------------------
    #  Initial full build
    # ------------------------------------------------------------------
    def initialize(self) -> None:
        """Build the initial call/reference graph."""
        repo_id = self.project.get_repo().id
        file_repo     = self.project.data_repository.file
        symbol_repo   = self.project.data_repository.symbol
        symbolref_repo = self.project.data_repository.symbolref

        for fm in file_repo.get_list_by_repo_id(repo_id):
            path = fm.path
            fid  = fm.id
            self._path_to_fid[path] = fid          # keep helper mapping
            # collect defs
            for sym in symbol_repo.get_list_by_file_id(fid):
                if sym.name:
                    self._defs[sym.name].add(path)         # definitions
                    self._name_props[sym.name] = self._make_props(sym)
            # collect refs
            for ref in symbolref_repo.get_list_by_file_id(fid):
                if ref.name:
                    self._refs[ref.name].add(path)         # references

        # materialise edges
        for name, def_files in self._defs.items():
            for ref_path in self._refs.get(name, []):
                for def_path in def_files:
                    edge_data = self._compute_edge_data(name)
                    w = edge_data.pop("weight")
                    self.G.add_edge(ref_path, def_path, name=name, weight=w, **edge_data)

        # Ensure self-loops for all nodes
        for path in self._path_to_fid.keys():
            self._ensure_self_loop(path)

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
            path = rel_path
            self._path_to_fid.pop(path, None)
            # drop node & incident edges
            if self.G.has_node(path):
                self.G.remove_node(path)
            # purge defs / refs caches
            for name, def_files in list(self._defs.items()):
                if path in def_files:
                    def_files.remove(path)
                    if not def_files:
                        del self._defs[name]
                        self._name_props.pop(name, None)
            for refs in self._refs.values():
                refs.discard(path)

        # ----- handle added & updated files ----------------------------
        changed = list(scan.files_added) + list(scan.files_updated)
        for rel_path in changed:
            fm = file_repo.get_by_path(rel_path)
            if fm is None:
                continue
            path, fid = fm.path, fm.id
            self._path_to_fid[path] = fid

            # remove existing node & caches for this file
            if self.G.has_node(path):
                self.G.remove_node(path)
            for name, def_files in list(self._defs.items()):
                if path in def_files:
                    def_files.remove(path)
                    if not def_files:
                        del self._defs[name]
                        self._name_props.pop(name, None)
            for refs in self._refs.values():
                refs.discard(path)

            # ----- rebuild caches for this file -----------------------
            new_def_names: Set[str] = set()
            new_ref_names: Set[str] = set()

            for sym in symbol_repo.get_list_by_file_id(fid):
                if sym.name:
                    self._defs[sym.name].add(path)
                    self._name_props[sym.name] = self._make_props(sym)   # NEW
                    new_def_names.add(sym.name)

            for ref in symbolref_repo.get_list_by_file_id(fid):
                if ref.name:
                    self._refs[ref.name].add(path)
                    new_ref_names.add(ref.name)

            # ensure node exists
            self.G.add_node(path)

            # edges from this file’s refs → all its targets
            for name in new_ref_names:
                edge_data = self._compute_edge_data(name)
                w = edge_data.pop("weight")
                for def_path in self._defs.get(name, []):
                    self.G.add_edge(path, def_path, name=name, weight=w, **edge_data)

            # edges from other refs → new defs in this file
            for name in new_def_names:
                edge_data = self._compute_edge_data(name)
                w = edge_data.pop("weight")
                for ref_path in self._refs.get(name, []):
                    self.G.add_edge(ref_path, path, name=name, weight=w, **edge_data)

            # ensure self-loop for this node if needed
            self._ensure_self_loop(path)

        # Ensure self-loops for all nodes (rule 7)
        for node in list(self.G.nodes):
            self._ensure_self_loop(node)


# Tool implementation
class RepoMapScore(BaseModel):
    file_path: str
    score:     float
    summary:   str | None = None


class RepoMapTool(BaseTool):
    """
    Produce PageRank-based importance scores for project files using
    the RepoMap graph. The original graph is never mutated.
    Accepts lists of symbol names and/or file paths to boost related edges.
    """
    tool_name = "vectorops_repomap"

    def __init__(self, *a, **kw):
        # TODO: fix me. Local import avoids circularity
        from know.project import Project
        Project.register_component(RepoMap)
        super().__init__(*a, **kw)

    def execute(
        self,
        project: Project,
        *,
        symbol_names: Optional[Sequence[str]] = None,
        file_paths: Optional[Sequence[str]] = None,
        limit: int = 20,
        mentioned_summary: bool = False,
    ) -> List[RepoMapScore]:
        """
        Run PageRank on the file-level reference graph, optionally boosting
        edges whose `name` matches any symbol in `symbol_names` and/or
        edges outgoing from any file in `file_paths`.
        Returns a list of RepoMapScore objects for the top files.

        By default, summaries for files explicitly listed in `file_paths`
        are omitted unless `include_summary_for_mentioned` is True.
        """

        repomap: RepoMap | None = project.get_component("repomap")
        if repomap is None:
            raise RuntimeError("RepoMap component is not available.")

        # shallow-copy graph
        G = repomap.G.copy()

        # Prepare sets for boosting
        sym_set: set[str]  = set(symbol_names or [])
        path_set: set[str] = set(file_paths or [])

        # NEW ────────────────────────────────────────────────────────────
        # if a symbol was mentioned, treat all its definition files as
        # “mentioned files” as well (needed for proper boosting / bias)
        if sym_set:
            for name in sym_set:
                path_set.update(repomap._defs.get(name, ()))

        # Adjust edge weights based on input parameters
        for u, v, _k, d in G.edges(keys=True, data=True):
            base     = d.get("base_weight", d.get("weight", 1.0))
            refs_cnt = d.get("refs_cnt", 0)

            # already present: boost by symbol name
            if sym_set and d.get("name") in sym_set:
                base *= SYMBOL_EDGE_BOOST

            # NEW: boost every edge that leaves a mentioned file
            if path_set and u in path_set and u != v:
                base *= FILE_EDGE_BOOST

            d["weight"] = base * math.sqrt(refs_cnt)

        # ── new: teleport-bias for selected files ─────────────────────────
        personalization: dict[str, float] | None = None
        # only build a personalised teleport vector when the user
        # actually mentioned symbols (sym_set ≠ ∅).  Pure file-based
        # boosting should rely on edge weights alone.
        if sym_set:
            boost = BOOST_FACTOR_DEFAULT
            pers = {n: boost for n in path_set if n in G}   # keep only existing nodes
            if pers:
                tot = sum(pers.values())
                personalization = {k: v / tot for k, v in pers.items()}

        # PageRank
        pr = nx.pagerank(
            G,
            weight="weight",
            personalization=personalization,
        )

        # collect & sort
        top = sorted(pr.items(), key=lambda t: t[1], reverse=True)[: limit]

        results: list[RepoMapScore] = []
        for path, score in top:
            include_summary = mentioned_summary or path not in path_set

            summary_val: str | None = None
            if include_summary:
                fs = build_file_summary(project, path)   # generate only when needed
                summary_val = fs.definitions if fs else None

            results.append(
                RepoMapScore(
                    file_path=path,
                    score=score,
                    summary=summary_val,
                )
            )
        return self.to_python(results)

    def get_openai_schema(self) -> dict:
        return {
            "name": self.tool_name,
            "description": (
                "Run PageRank over the repository file-level reference graph "
                "and return the most important files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Boost edges whose `name` attribute matches any of these symbols."
                    },
                    "file_paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Boost edges outgoing from any of the given file paths."
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "default": 20,
                        "description": "Number of top files to return.",
                    },
                    "mentioned_summary": {
                        "type": "boolean",
                        "default": False,
                        "description": "If true, include summaries for files explicitly listed in `file_paths`; otherwise omit them."
                    },
                },
            },
        }
