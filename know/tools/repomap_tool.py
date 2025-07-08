from __future__ import annotations

from typing import List, Optional
import networkx as nx

from know.project import Project
from know.tools.base import BaseTool
from know.tools.repomap import RepoMap
from know.models import FileMetadata


class RepoMapScore(BaseTool._pyd_base_model):  # helper through BaseTool
    file_path: str
    score:     float


class RepoMapTool(BaseTool):
    """
    Produce PageRank-based importance scores for project files using
    the RepoMap graph.  The original graph is never mutated.
    """
    tool_name = "vectorops_repomap"

    def execute(
        self,
        project: Project,
        *,
        symbol_name: Optional[str] = None,
        file_path:   Optional[str] = None,
        limit: int = 20,
    ) -> List[RepoMapScore]:

        repomap: RepoMap | None = project.get_component("repomap")
        if repomap is None:
            raise RuntimeError("RepoMap component is not available.")

        # 1) shallow-copy graph
        G = repomap.G.copy()

        # 2–3) adjust edge weights
        if symbol_name:
            for _u, _v, _k, d in G.edges(keys=True, data=True):
                if d.get("name") == symbol_name:
                    d["weight"] = d.get("weight", 1.0) * 10.0
        if file_path:
            target_fid = repomap._path_to_fid.get(file_path)
            if target_fid:
                for u, v, _k, d in G.edges(keys=True, data=True):
                    if u == target_fid or v == target_fid:
                        d["weight"] = d.get("weight", 1.0) * 50.0

        # 4) PageRank
        pr = nx.pagerank(G, weight="weight")

        # helper to convert fid → path
        fid_to_path = {fid: path for path, fid in repomap._path_to_fid.items()}

        # 5) collect & sort
        top = sorted(pr.items(), key=lambda t: t[1], reverse=True)[: limit]
        results = [
            RepoMapScore(file_path=fid_to_path.get(fid, fid), score=score)
            for fid, score in top
        ]
        return self.to_python(results)

    # 6) OpenAI / JSON schema
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
                    "symbol_name": {
                        "type": "string",
                        "description": (
                            "Boost all edges that represent references to this symbol "
                            "(weight ×10)."
                        ),
                    },
                    "file_path": {
                        "type": "string",
                        "description": (
                            "Boost all edges incident to the given file path (weight ×50)."
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "default": 20,
                        "description": "Number of top files to return.",
                    },
                },
            },
        }
