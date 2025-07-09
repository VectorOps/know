"""
Heterogeneous Random-Walk-with-Restart (RWR) recommender
========================================================
Returns the files that receive the highest steady-state
probability when a random walker repeatedly

    • starts from the user-mentioned symbols / files,
    • follows typed edges on a 2-layer (files + symbols) graph, and
    • restarts with probability *r* back to the same seed set.

Compared with the original PageRank-plus-boost this directly
models both goals:

    1.  symbol → file (defined_by) edges pull probability to
        symbol-defining files;
    2.  file → symbol (reference) + symbol → file (defined_by)
        paths pull probability from a mentioned file to the files
        that actually define the referenced symbols.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Optional, List, Dict

import networkx as nx
from pydantic import BaseModel

from know.project import Project
from know.tools.base import BaseTool
from know.tools.repomap import RepoMap
from know.tools.file_summary_helper import build_file_summary

# ---------------------------------------------------------------------
#  Tunables
# ---------------------------------------------------------------------
RESTART_PROB           = 0.15       # probability of jumping back to seeds
EDGE_W_DEF             = 3.0        # file ⇆ symbol definition edges
EDGE_W_REF             = 1.0        # file → symbol reference edges
EDGE_W_IMPORT          = 0.5        # (optional) file → file import edges
MIN_WEIGHT             = 1e-9       # guard against zero-degree nodes
LIMIT_DEFAULT          = 20


@dataclass(slots=True)
class _NodeAttr:
    kind: str          # "file" | "sym"
    # other attributes can be added later (language, size, …)


# ---------------------------------------------------------------------
#  Return type
# ---------------------------------------------------------------------
class RepoMapRWRScore(BaseModel):
    file_path: str
    score:     float
    summary:   Optional[str] = None


# ---------------------------------------------------------------------
#  The tool
# ---------------------------------------------------------------------
class RepoMapRWRTool(BaseTool):
    """
    Rank repository files with a heterogeneous Random-Walk-with-Restart.
    """
    tool_name = "vectorops_repomap_rwr"

    # ------------- public ‘execute’ ----------------------------------
    def execute(
        self,
        project: Project,
        *,
        symbol_names: Optional[Sequence[str]] = None,
        file_paths:   Optional[Sequence[str]] = None,
        limit:        int = LIMIT_DEFAULT,
        restart_prob: float = RESTART_PROB,
        include_summary_for_mentioned: bool = False,
    ) -> List[RepoMapRWRScore]:

        repomap = project.get_component("repomap")
        if repomap is None:
            raise RuntimeError("RepoMap component is not available. Call project.get_component('repomap') first.")

        # helper: quality factor derived from symbol properties
        _sym_quality = lambda n: repomap._compute_edge_data(n)["base_weight"]

        # -----------------------------------------------------------------
        #  1.  Build the 2-layer “heterograph”
        # -----------------------------------------------------------------
        G: nx.MultiDiGraph = nx.MultiDiGraph()

        # -- helper lambdas ------------------------------------------------
        sym_node = lambda s: f"sym::{s}"

        def _add_node(node: str, kind: str):
            if not G.has_node(node):
                G.add_node(node, data=_NodeAttr(kind=kind))

        # ---- add file & symbol nodes  ------------------------------------
        for path in repomap._path_to_fid.keys():
            _add_node(path, "file")

        for name in repomap._defs.keys():
            _add_node(sym_node(name), "sym")

        # ---- add edges ---------------------------------------------------
        # file  →  referenced-symbol
        for name, ref_files in repomap._refs.items():
            sym_n = sym_node(name)
            for f in ref_files:
                quality = _sym_quality(name)
                G.add_edge(f, sym_n, etype="ref",
                           weight=EDGE_W_REF * quality)

        # file  ↔  symbol it defines
        for name, def_files in repomap._defs.items():
            sym_n = sym_node(name)
            for f in def_files:
                quality = _sym_quality(name)
                # file → sym  (“defines”)
                G.add_edge(f,      sym_n, etype="def_out",
                           weight=EDGE_W_DEF * quality)
                # sym  → file (“defined_by”)
                G.add_edge(sym_n,  f,     etype="def_in",
                           weight=EDGE_W_DEF * quality)

        # (Optional) import edges – if you store them elsewhere you can add
        # them here with G.add_edge(importer, importee, etype="import", weight=EDGE_W_IMPORT)

        # ---- ensure no isolated node has zero outgoing weight ------------
        for n in G.nodes:
            if G.out_degree(n) == 0:
                G.add_edge(n, n, etype="self", weight=MIN_WEIGHT)

        # -----------------------------------------------------------------
        #  2.  Prepare the restart / personalization vector
        # -----------------------------------------------------------------
        personalization: Dict[str, float] | None = None

        if symbol_names or file_paths:
            seeds: Dict[str, float] = {}

            # seed on mentioned symbols
            for s in symbol_names or []:
                n = sym_node(s)
                if G.has_node(n):
                    seeds[n] = seeds.get(n, 0.0) + 1.0

            # seed on mentioned files
            for p in file_paths or []:
                if G.has_node(p):
                    seeds[p] = seeds.get(p, 0.0) + 1.0

            if not seeds:
                raise ValueError("None of the provided symbols/files exist in the graph.")

            # normalise to probability distribution
            total_seed = float(sum(seeds.values()))
            personalization = {k: v / total_seed for k, v in seeds.items()}

        # -----------------------------------------------------------------
        #  3.  Run Random-Walk-with-Restart (via PageRank)
        # -----------------------------------------------------------------
        pr = nx.pagerank(
            G,
            alpha=(1.0 - restart_prob),
            personalization=personalization,
            weight="weight",
        )

        print(pr)

        # -----------------------------------------------------------------
        #  4.  Collect top-k files only
        # -----------------------------------------------------------------
        is_file = lambda n: isinstance(G.nodes[n]["data"], _NodeAttr) and G.nodes[n]["data"].kind == "file"

        ranked_files = [ (p, sc) for p, sc in pr.items() if is_file(p) ]
        ranked_files.sort(key=lambda t: t[1], reverse=True)
        ranked_files = ranked_files[: max(1, limit) ]

        # -----------------------------------------------------------------
        #  5.  Build response
        # -----------------------------------------------------------------
        results: list[RepoMapRWRScore] = []
        mentioned_file_set = set(file_paths or [])

        for path, score in ranked_files:
            need_summary = include_summary_for_mentioned or path not in mentioned_file_set
            summary: Optional[str] = None
            if need_summary:
                fs = build_file_summary(project, path)
                summary = fs.definitions if fs else None

            results.append(
                RepoMapRWRScore(
                    file_path=path,
                    score=score,
                    summary=summary,
                )
            )

        return self.to_python(results)

    # ------------- OpenAI function-calling schema ------------------------
    def get_openai_schema(self) -> dict:
        return {
            "name": self.tool_name,
            "description": (
                "Rank repository files by running a heterogeneous Random-Walk-with-Restart "
                "starting from the given symbols/files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Names of symbols that are mentioned in the user request."
                    },
                    "file_paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "File paths explicitly mentioned in the user request."
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "default": LIMIT_DEFAULT,
                        "description": "Number of top files to return."
                    },
                    "restart_prob": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "default": RESTART_PROB,
                        "description": "Probability of restarting the walk at the seed set (typical values 0.1 – 0.2)."
                    },
                    "include_summary_for_mentioned": {
                        "type": "boolean",
                        "default": False,
                        "description": "If true, also attach summaries to files that the user explicitly mentioned."
                    },
                },
            },
        }
