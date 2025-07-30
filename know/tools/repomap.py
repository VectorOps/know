import math, re, os
import time
from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, Optional, Sequence, Set, List
from litellm import token_counter
import networkx as nx

from .base import BaseTool, MCPToolDefinition
from pydantic import BaseModel, Field
from know.logger import logger
from know.project import ProjectManager, ScanResult, ProjectComponent
from know.models import Node, Visibility
from know.file_summary import SummaryMode, build_file_summary
from know.data import FileFilter, NodeFilter, NodeRefFilter


EDGE_W_DEF             = 3.0
EDGE_W_REF             = 1.0
EDGE_W_IMPORT          = 0.5
MIN_WEIGHT             = 1e-9
LIMIT_DEFAULT          = 20

DESCRIPTIVE_MULTIPLIER = 4.0
PRIVATE_PROTECTED_MULT = 0.1
POLYDEF_THRESHOLD      = 5
POLYDEF_MULTIPLIER     = 0.1
ISOLATED_SELF_WEIGHT   = 0.3


def _sym_node(name: str) -> str:
    """Return the canonical node ID for a symbol."""
    return f"sym::{name}"


def _count_tokens(text: str, model: str) -> int:
    """Count tokens in a string using a specified model."""
    return token_counter(text=text, model=model)


@dataclass(slots=True)
class NameProps:
    """Properties of a symbol name relevant for weighting."""
    name: str
    visibility: Optional[str]
    descriptiveness: float


@dataclass(slots=True)
class _NodeAttr:
    """Graph node attributes."""
    kind: str  # "file" | "sym"


class RepoMap(ProjectComponent):
    """Component that maintains the heterograph of files and symbols."""
    component_name = "repomap"

    def __init__(self, pm: ProjectManager):
        """Initialize the RepoMap component."""
        super().__init__(pm)
        self.G: nx.MultiDiGraph = nx.MultiDiGraph()
        self._path_to_fid: Dict[str, str] = {}

        # caches for fast weight computations
        self._defs: Dict[str, Set[str]]  = defaultdict(set)   # symbol -> {file}
        self._refs: Dict[str, Set[str]]  = defaultdict(set)   # symbol -> {file}
        self._name_props: Dict[str, NameProps] = {}

    def destroy(self):
        """Clean up resources used by the component."""
        pass

    def sym_node(self, name: str) -> str:
        """Return the canonical node ID for a symbol."""
        return _sym_node(name)

    def get_name_properties(self, name: str) -> Optional[NameProps]:
        """Return the cached properties for a symbol name."""
        return self._name_props.get(name)

    @staticmethod
    def _calc_descriptiveness(name: str) -> float:
        """
        Descriptiveness grows with
          • the number of “words” in the identifier, and
          • the overall character length.
        Both sub-scores are clamped to 1.0 and averaged, then rounded to 3 decimals.
        """
        if not name:
            return 0.0

        # word-based score
        snake = re.sub(r'(?<!^)(?=[A-Z])', '_', name)
        words = [w for w in snake.split('_') if w]
        word_score = min(len(words) / 6.0, 1.0)          # >=6 words - cap at 1.0

        # length-based score
        length_score = min(len(name) / 30.0, 1.0)        # >=30 chars - cap at 1.0

        #print(name, word_score, length_score)

        # final descriptiveness
        return round((word_score + length_score) / 2.0, 4)

    def _make_props(self, sym: Node) -> NameProps:
        assert sym.name is not None
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
        G: nx.MultiDiGraph = nx.MultiDiGraph()

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
        repo_id        = self.pm.default_repo.id
        file_repo      = self.pm.data.file
        symbol_repo    = self.pm.data.symbol
        symbolref_repo = self.pm.data.symbolref

        # clear previous state
        self._defs.clear()
        self._refs.clear()
        self._name_props.clear()
        self._path_to_fid.clear()

        for fm in file_repo.get_list(FileFilter(repo_id=[repo_id])):
            path, fid = fm.path, fm.id
            self._path_to_fid[path] = fid

            # defs
            for sym in symbol_repo.get_list(NodeFilter(file_id=fid)):
                if sym.name:
                    self._defs[sym.name].add(path)
                    self._name_props[sym.name] = self._make_props(sym)

            # refs
            for ref in symbolref_repo.get_list(NodeRefFilter(file_id=fid)):
                if ref.name:
                    self._refs[ref.name].add(path)

    # ---------- incremental refresh -------------------------------------
    def refresh(self, scan: ScanResult) -> None:
        """
        Update caches, then rebuild the graph.  The granular cache logic
        stays exactly the same as before; we simply call `_rebuild_graph()`
        at the end so that the heterograph is always current.
        """
        file_repo      = self.pm.data.file
        symbol_repo    = self.pm.data.symbol
        symbolref_repo = self.pm.data.symbolref

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
            fm = file_repo.get_by_path(self.pm.default_repo.id, rel_path)
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
            for sym in symbol_repo.get_list(NodeFilter(file_id=fid)):
                if sym.name:
                    self._defs[sym.name].add(path)
                    self._name_props[sym.name] = self._make_props(sym)

            for ref in symbolref_repo.get_list(NodeRefFilter(file_id=fid)):
                if ref.name:
                    self._refs[ref.name].add(path)

        # finally, rebuild the graph from the updated caches
        self._rebuild_graph()


#  Tool
class RepoMapReq(BaseModel):
    symbol_names: Optional[Sequence[str]] = Field(
        default=None,
        description="Symbol names to use as high-priority seeds for the random walk.",
    )
    file_paths: Optional[Sequence[str]] = Field(
        default=None,
        description="File paths to use as high-priority seeds for the random walk.",
    )
    prompt: Optional[str] = Field(
        default=None,
        description="A free-text prompt from which to extract additional symbol and file names to use as seeds.",
    )
    limit: int = Field(
        default=LIMIT_DEFAULT, description="The maximum number of files to return."
    )
    summary_mode: SummaryMode | str = Field(
        default=SummaryMode.ShortSummary,
        description="The level of detail for the summary of each file.",
    )
    skip_mentioned_summary: bool = Field(
        default=False,
        description="If true, do not generate summaries for files that were explicitly provided in `file_paths`.",
    )
    token_limit_count: Optional[int] = Field(
        default=None,
        description="A token budget for the total size of all returned summaries. The tool will stop adding summaries once this limit is exceeded.",
    )
    token_limit_model: Optional[str] = Field(
        default=None,
        description="The model to use for counting tokens for `token_limit_count`. Required if `token_limit_count` is set.",
    )


class RepoMapScore(BaseModel):
    file_path: str = Field(
        description="The path of the ranked file, relative to the project root."
    )
    score: float = Field(
        description="The relevance score of the file, as determined by the random walk."
    )
    summary: Optional[str] = Field(
        default=None, description="The generated summary of the file, if one was requested."
    )


class RepoMapTool(BaseTool):
    """
    Rank repository files with a heterogeneous Random-Walk-with-Restart.
    Only builds a personalization vector; the heavy graph is maintained
    by the `RepoMap` component.
    """
    tool_name = "vectorops_repomap"
    tool_input = RepoMapReq
    tool_output = List[RepoMapScore]

    def __init__(self, *a, **kw):
        from know.project import ProjectManager
        ProjectManager.register_component(RepoMap)
        super().__init__(*a, **kw)

    # ------------- public ‘execute’ ----------------------------------
    def execute(
        self,
        pm: ProjectManager,
        req: RepoMapReq,
    ) -> List[RepoMapScore]:
        summary_mode = req.summary_mode
        if isinstance(summary_mode, str):
            summary_mode = SummaryMode(summary_mode)

        _t_start = time.perf_counter()

        repomap = pm.get_component("repomap")
        if not isinstance(repomap, RepoMap):
            raise RuntimeError(
                "RepoMap component is missing. Call "
                "`pm.get_component('repomap')` (or ensure it is "
                "initialised) before using this tool."
            )

        # 0.  Extract symbol / file hints from a free-text *prompt*
        symbol_names_set: Set[str] = set(req.symbol_names or [])
        file_paths_set:   Set[str] = set(req.file_paths or [])

        if req.prompt:
            txt = req.prompt.lower()

            # ----- file names / paths ----------------------------------
            for path in repomap._path_to_fid.keys():
                if path.lower() in txt or os.path.basename(path).lower() in txt:
                    file_paths_set.add(path)

            # ----- symbol names ----------------------------------------
            known_syms = set(repomap._defs.keys()) | set(repomap._refs.keys())
            known_syms_lower_map = defaultdict(list)
            for s in known_syms:
                known_syms_lower_map[s.lower()].append(s)

            tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_\.]*", txt)
            for tok in tokens:
                if tok.endswith('.'):
                    tok = tok[:-1]
                if not tok:
                    continue
                if len(tok) < pm.settings.repomap.min_symbol_len:
                    continue

                last = tok.rsplit(".", 1)[-1]

                if tok in known_syms_lower_map:
                    for s in known_syms_lower_map[tok]:
                        symbol_names_set.add(s)

                if len(last) >= pm.settings.repomap.min_symbol_len and last in known_syms_lower_map:
                    for s in known_syms_lower_map[last]:
                        symbol_names_set.add(s)

        # convert back to the lists consumed below
        symbol_names = list(symbol_names_set) if symbol_names_set else None
        file_paths   = list(file_paths_set)   if file_paths_set   else None

        G = repomap.G
        sym_node = repomap.sym_node

        #  1. Personalisation / restart vector (a.k.a. «boosting»)
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

        #  2. Run Random-Walk-with-Restart (= personalised PageRank)
        pr = nx.pagerank(
            G,
            alpha=(1.0 - pm.settings.repomap.restart_prob),
            personalization=personalization,
            weight="weight",
        )

        #  3. Collect top-k *file* nodes
        is_file = lambda n: G.nodes[n].get("kind") == "file"
        ranked = [(p, sc) for p, sc in pr.items() if is_file(p)]
        ranked.sort(key=lambda t: t[1], reverse=True)
        ranked = ranked[: max(1, req.limit)]

        #  4. Build response objects
        mentioned = set(file_paths or [])
        results: list[RepoMapScore] = []

        tokens_used = 0
        for path, score in ranked:
            skip_summary = req.skip_mentioned_summary and path in mentioned
            summary = None
            summary_tokens = 0

            effective_mode = SummaryMode.Skip if skip_summary else summary_mode
            fs = build_file_summary(
                pm,
                pm.default_repo,
                path,
                effective_mode,
            )
            summary = fs.content if fs else None
            if summary and req.token_limit_count and req.token_limit_model:
                summary_tokens = _count_tokens(summary, req.token_limit_model)

            # -- enforce token budget ---------------------------------------
            if req.token_limit_count and req.token_limit_model:
                if tokens_used + summary_tokens > req.token_limit_count:
                    break
                tokens_used += summary_tokens

            results.append(
                RepoMapScore(file_path=path, score=score, summary=summary)
            )

        _elapsed = time.perf_counter() - _t_start
        logger.debug("RepoMapTool.execute finished",
                     duration_sec=round(_elapsed, 4),
                     results=len(results))

        return results

    # ---------- OpenAI schema ----------
    def get_openai_schema(self) -> dict:
        summary_enum = [m.value for m in SummaryMode]

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
                        "description": "Symbol names to use as high-priority seeds for the random walk.",
                    },
                    "file_paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "File paths to use as high-priority seeds for the random walk.",
                    },
                    "prompt": {
                        "type": "string",
                        "description": "A free-text prompt from which to extract additional symbol and file names to use as seeds.",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "default": LIMIT_DEFAULT,
                        "description": "The maximum number of files to return.",
                    },
                    "summary_mode": {
                        "type": "string",
                        "enum": summary_enum,
                        "default": SummaryMode.ShortSummary.value,
                        "description": (
                            "The level of detail for the summary of each file "
                            "(`skip`/`summary_short`/`summary_full`/`full`)."
                        ),
                    },
                    "skip_mentioned_summary": {
                        "type": "boolean",
                        "default": False,
                        "description": "If true, do not generate summaries for files that were explicitly provided in `file_paths`.",
                    },
                    "token_limit_count": {
                        "type": "integer",
                        "description": "A token budget for the total size of all returned summaries. The tool will stop adding summaries once this limit is exceeded.",
                    },
                    "token_limit_model": {
                        "type": "string",
                        "description": "The model to use for counting tokens for `token_limit_count`. Required if `token_limit_count` is set.",
                    },
                },
            },
        }

    def get_mcp_definition(self, pm: ProjectManager) -> MCPToolDefinition:
        def repomap(req: RepoMapReq) -> List[RepoMapScore]:
            return self.execute(pm, req)

        schema = self.get_openai_schema()
        return MCPToolDefinition(
            fn=repomap,
            name=self.tool_name,
            description=schema.get("description"),
        )
