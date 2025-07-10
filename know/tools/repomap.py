# ---------------------------------------------------------------------
#  Repo-level Random-Walk-with-Restart graph component
# ---------------------------------------------------------------------
from __future__ import annotations

import math, re
from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, Optional, Sequence, Set

import networkx as nx
from pydantic import BaseModel

from know.project import Project, ScanResult, ProjectComponent
from know.models import SymbolMetadata, Visibility
from know.tools.base import BaseTool
from know.tools.file_summary_helper import build_file_summary
from know.logger import KnowLogger as logger


# ---------------------------------------------------------------------
#  Tunables that are *still* used
# ---------------------------------------------------------------------
RESTART_PROB           = 0.15
EDGE_W_DEF             = 3.0        # file ⇆ symbol definition
EDGE_W_REF             = 1.0        # file → symbol reference
EDGE_W_IMPORT          = 0.5        # (optional) file → file imports
MIN_WEIGHT             = 1e-9
LIMIT_DEFAULT          = 20

DESCRIPTIVE_MULTIPLIER = 4.0
PRIVATE_PROTECTED_MULT = 0.1
POLYDEF_THRESHOLD      = 6
POLYDEF_MULTIPLIER     = 0.1
ISOLATED_SELF_WEIGHT   = 0.3


# ---------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------
def _sym_node(name: str) -> str:
    """Canonical node id for a symbol."""
    return f"sym::{name}"


@dataclass(slots=True)
class NameProps:
    name: str
    visibility: Optional[str]
    descriptiveness: float


@dataclass(slots=True)
class _NodeAttr:
    kind: str  # "file" | "sym"


# ---------------------------------------------------------------------
#  Component that **maintains** the heterograph
# ---------------------------------------------------------------------
class RepoMap(ProjectComponent):
    component_name = "repomap"

    # ---------- lifecycle ------------------------------------------------
    def __init__(self, project: Project):
        super().__init__(project)
        self.G = nx.MultiDiGraph()
        self._path_to_fid: Dict[str, str] = {}

        # caches for fast weight computations
        self._defs: Dict[str, Set[str]]  = defaultdict(set)   # symbol → {file}
        self._refs: Dict[str, Set[str]]  = defaultdict(set)   # symbol → {file}
        self._name_props: Dict[str, NameProps] = {}

    # ---------- public helpers ------------------------------------------
    def sym_node(self, name: str) -> str:
        return _sym_node(name)

    def get_name_properties(self, name: str) -> Optional[NameProps]:
        return self._name_props.get(name)

    # ---------- edge-weight helpers -------------------------------------
    @staticmethod
    def _calc_descriptiveness(name: str) -> float:
        if not name:
            return 0.0
        snake = re.sub(r'(?<!^)(?=[A-Z])', '_', name)
        words = [w for w in snake.split('_') if w]
        return round(min(len(words) / 5.0, 1.0), 3)

    def _make_props(self, sym: SymbolMetadata) -> NameProps:
        vis = (
            sym.visibility.value
            if isinstance(sym.visibility, Visibility)
            else (str(sym.visibility) if sym.visibility else None)
        )
        return NameProps(
            name=sym.name,
            visibility=vis,
            descriptiveness=self._calc_descriptiveness(sym.name),
        )

    def _compute_edge_data(self, name: str) -> float:
        base = 1.0

        props = self._name_props.get(name)
        if props:
            base *= props.descriptiveness * DESCRIPTIVE_MULTIPLIER
            if props.visibility in (Visibility.PRIVATE.value, Visibility.PROTECTED.value):
                base *= PRIVATE_PROTECTED_MULT

        defs_cnt = len(self._defs.get(name, []))
        if defs_cnt >= POLYDEF_THRESHOLD:
            base *= POLYDEF_MULTIPLIER

        refs_cnt = len(self._refs.get(name, []))
        weight_factor = math.log1p(refs_cnt) if refs_cnt > 0 else 1.0

        weight = base * weight_factor

        if weight <= 0.0:
            weight = MIN_WEIGHT

        return weight

    # ---------- graph construction --------------------------------------
    def _rebuild_graph(self) -> None:
        """(Re)build the 2-layer heterograph from the current caches."""
        G = nx.MultiDiGraph()

        # -- add nodes ----------------------------------------------------
        for path in self._path_to_fid.keys():
            G.add_node(path, kind="file")
        for name in self._defs.keys():
            G.add_node(_sym_node(name), kind="sym")

        # -- add edges ----------------------------------------------------
        for name, ref_files in self._refs.items():
            sym_n = _sym_node(name)
            w_ref  = self._compute_edge_data(name) * EDGE_W_REF
            for f in ref_files:
                G.add_edge(f, sym_n, etype="ref", weight=w_ref, name=name)

        for name, def_files in self._defs.items():
            sym_n = _sym_node(name)
            w_def  = self._compute_edge_data(name) * EDGE_W_DEF
            for f in def_files:
                # file -> symbol (defines)
                G.add_edge(f, sym_n, etype="def_out", weight=w_def, name=name)
                # symbol -> file (defined_by)
                G.add_edge(sym_n, f, etype="def_in",  weight=w_def, name=name)

        # -- keep every node ergodic -------------------------------------
        for n in G.nodes:
            if G.out_degree(n) == 0:
                G.add_edge(n, n, etype="self", weight=ISOLATED_SELF_WEIGHT)

        self.G = G  # swap in the freshly built graph

    # ---------- initial build -------------------------------------------
    def initialize(self) -> None:
        self._collect_caches_full()
        self._rebuild_graph()

    def _collect_caches_full(self) -> None:
        """Populate _defs / _refs / _name_props / _path_to_fid from scratch."""
        repo_id        = self.project.get_repo().id
        file_repo      = self.project.data_repository.file
        symbol_repo    = self.project.data_repository.symbol
        symbolref_repo = self.project.data_repository.symbolref

        # clear previous state
        self._defs.clear()
        self._refs.clear()
        self._name_props.clear()
        self._path_to_fid.clear()

        for fm in file_repo.get_list_by_repo_id(repo_id):
            path, fid = fm.path, fm.id
            self._path_to_fid[path] = fid

            # defs
            for sym in symbol_repo.get_list_by_file_id(fid):
                if sym.name:
                    self._defs[sym.name].add(path)
                    self._name_props[sym.name] = self._make_props(sym)

            # refs
            for ref in symbolref_repo.get_list_by_file_id(fid):
                if ref.name:
                    self._refs[ref.name].add(path)

    # ---------- incremental refresh -------------------------------------
    def refresh(self, scan: ScanResult) -> None:
        """
        Update caches, then rebuild the graph.  The granular cache logic
        stays exactly the same as before; we simply call `_rebuild_graph()`
        at the end so that the heterograph is always current.
        """
        file_repo      = self.project.data_repository.file
        symbol_repo    = self.project.data_repository.symbol
        symbolref_repo = self.project.data_repository.symbolref

        # ----- deletions -------------------------------------------------
        for rel_path in scan.files_deleted:
            path = rel_path
            self._path_to_fid.pop(path, None)
            for name in list(self._defs.keys()):
                self._defs[name].discard(path)
                if not self._defs[name]:
                    self._defs.pop(name)
                    self._name_props.pop(name, None)
            for refs in self._refs.values():
                refs.discard(path)

        # ----- additions / updates --------------------------------------
        changed = list(scan.files_added) + list(scan.files_updated)
        for rel_path in changed:
            fm = file_repo.get_by_path(rel_path)
            if fm is None:
                continue
            path, fid = fm.path, fm.id
            self._path_to_fid[path] = fid

            # purge old caches for this path
            for name in list(self._defs.keys()):
                self._defs[name].discard(path)
                if not self._defs[name]:
                    self._defs.pop(name)
                    self._name_props.pop(name, None)
            for refs in self._refs.values():
                refs.discard(path)

            # rebuild caches for this path
            for sym in symbol_repo.get_list_by_file_id(fid):
                if sym.name:
                    self._defs[sym.name].add(path)
                    self._name_props[sym.name] = self._make_props(sym)

            for ref in symbolref_repo.get_list_by_file_id(fid):
                if ref.name:
                    self._refs[ref.name].add(path)

        # finally, rebuild the graph from the updated caches
        self._rebuild_graph()


# ---------------------------------------------------------------------
#  Score DTO
# ---------------------------------------------------------------------
class RepoMapScore(BaseModel):
    file_path: str
    score:     float
    summary:   Optional[str] = None


# ---------------------------------------------------------------------
#  Tool that *uses* the graph and builds the boosting vector
# ---------------------------------------------------------------------
class RepoMapTool(BaseTool):
    """
    Rank repository files with a heterogeneous Random-Walk-with-Restart.
    Only builds a personalization vector; the heavy graph is maintained
    by the `RepoMap` component.
    """
    tool_name = "vectorops_repomap_rwr"

    def __init__(self, *a, **kw):
        from know.project import Project
        Project.register_component(RepoMap)   # one-time registration
        super().__init__(*a, **kw)

    # ------------- public ‘execute’ ----------------------------------
    def execute(
        self,
        project: Project,
        *,
        symbol_names: Optional[Sequence[str]] = None,
        file_paths:   Optional[Sequence[str]] = None,
        limit:        int   = LIMIT_DEFAULT,
        restart_prob: float = RESTART_PROB,
        include_summary_for_mentioned: bool = False,
    ) -> list[RepoMapScore]:

        repomap: RepoMap = project.get_component("repomap")  # type: ignore
        if repomap is None:
            raise RuntimeError(
                "RepoMap component is missing.  Call "
                "`project.get_component('repomap')` (or ensure it is "
                "initialised) before using this tool."
            )

        G = repomap.G
        sym_node = repomap.sym_node

        # -----------------------------------------------------------------
        #  1. Personalisation / restart vector (a.k.a. «boosting»)
        # -----------------------------------------------------------------
        if symbol_names or file_paths:
            seeds: Dict[str, float] = {}

            for s in symbol_names or []:
                n = sym_node(s)
                if G.has_node(n):
                    seeds[n] = seeds.get(n, 0.0) + 1.0

            for p in file_paths or []:
                if G.has_node(p):
                    seeds[p] = seeds.get(p, 0.0) + 1.0

            if not seeds:
                raise ValueError("None of the provided symbols/files exist in the repository.")

            tot = float(sum(seeds.values()))
            personalization = {k: v / tot for k, v in seeds.items()}
        else:
            personalization = None  # vanilla PageRank == fully random restart

        #print(personalization)
        #for k in [f"{u} -> {v} ({d.get('name', '')}) ({d.get('weight')})" for u, v, d in G.edges(data=True)]:
        #    print(k)

        # -----------------------------------------------------------------
        #  2. Run Random-Walk-with-Restart (= personalised PageRank)
        # -----------------------------------------------------------------
        pr = nx.pagerank(
            G,
            alpha=(1.0 - restart_prob),
            personalization=personalization,
            weight="weight",
        )

        # -----------------------------------------------------------------
        #  3. Collect top-k *file* nodes
        # -----------------------------------------------------------------
        is_file = lambda n: G.nodes[n].get("kind") == "file"
        ranked = [(p, sc) for p, sc in pr.items() if is_file(p)]
        ranked.sort(key=lambda t: t[1], reverse=True)
        ranked = ranked[: max(1, limit)]

        # -----------------------------------------------------------------
        #  4. Build response objects
        # -----------------------------------------------------------------
        mentioned = set(file_paths or [])
        results: list[RepoMapScore] = []

        for path, score in ranked:
            need_summary = include_summary_for_mentioned or path not in mentioned
            summary = None
            if need_summary:
                fs = build_file_summary(project, path)
                summary = fs.definitions if fs else None

            results.append(
                RepoMapScore(file_path=path, score=score, summary=summary)
            )

        return self.to_python(results)

    # ---------- OpenAI schema (unchanged aside from defaults) ----------
    def get_openai_schema(self) -> dict:
        return {
            "name": self.tool_name,
            "description": (
                "Rank repository files by running a heterogeneous Random-Walk-with-Restart "
                "starting from the given symbols and/or files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Symbol names mentioned in the user request.",
                    },
                    "file_paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "File paths explicitly mentioned in the user request.",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "default": LIMIT_DEFAULT,
                        "description": "Number of top files to return.",
                    },
                    "restart_prob": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "default": RESTART_PROB,
                        "description": (
                            "Probability of restarting the walk at the seed set "
                            "(typical values 0.1 – 0.2)."
                        ),
                    },
                    "include_summary_for_mentioned": {
                        "type": "boolean",
                        "default": False,
                        "description": (
                            "If true, attach summaries even to the files that the "
                            "user explicitly mentioned."
                        ),
                    },
                },
            },
        }
