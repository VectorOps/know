from __future__ import annotations

from typing import List, Sequence, Optional

from pydantic import BaseModel

from know.data import SymbolSearchQuery
from know.models import SymbolKind, Visibility
from know.project import Project
from know.tools.base import BaseTool


class SymbolSearchResult(BaseModel):
    symbol_id: str
    fqn:       Optional[str] = None
    name:      Optional[str] = None
    kind:      Optional[str] = None
    visibility:Optional[str] = None
    file_path: Optional[str] = None


class SearchSymbolsTool(BaseTool):
    tool_name = "vectorops_search_symbols"

    def execute(
        self,
        project: Project,
        *,
        symbol_name: Optional[str] = None,
        symbol_fqn: Optional[str] = None,
        symbol_kind: Optional[str] = None,
        symbol_visibility: Optional[str] = None,
        doc_needle: Optional[Sequence[str]] = None,
        embedding_text: Optional[str] = None,
        limit: int | None = 20,
        offset: int | None = 0,
    ) -> List[SymbolSearchResult]:

        # translate enums if string values are given
        kind  = SymbolKind(symbol_kind)           if symbol_kind else None
        vis   = Visibility(symbol_visibility)     if symbol_visibility else None

        # ------------------------------------------------------------
        # transform free-text query → embedding vector (if requested)
        # ------------------------------------------------------------
        embedding_vec = None
        if embedding_text:
            # treat query text as plain language (not code)
            embedding_vec = project.compute_embedding(embedding_text, is_code=False)

        repo_id = project.get_repo().id
        query   = SymbolSearchQuery(
            symbol_name       = symbol_name,
            symbol_fqn        = symbol_fqn,
            symbol_kind       = kind,
            symbol_visibility = vis,
            doc_needle        = list(doc_needle) if doc_needle else None,
            embedding_query   = embedding_vec,
            limit             = limit,
            offset            = offset,
        )
        syms = project.data_repository.symbol.search(repo_id, query)
        file_repo = project.data_repository.file

        results: list[SymbolSearchResult] = []
        for s in syms:
            file_path = None
            if s.file_id:
                fm = file_repo.get_by_id(s.file_id)
                file_path = fm.path if fm else None
            results.append(
                SymbolSearchResult(
                    symbol_id  = s.id,
                    fqn        = s.fqn,
                    name       = s.name,
                    kind       = getattr(s.kind, "value", s.kind) if s.kind else None,
                    visibility = getattr(s.visibility, "value", s.visibility) if s.visibility else None,
                    file_path  = file_path,
                )
            )
        return results

    def get_openai_schema(self) -> dict:
        return {
            "name": self.tool_name,
            "description": "Search symbols in the current project repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol_name":       {"type": "string", "description": "Substring match on symbol name"},
                    "symbol_fqn":        {"type": "string", "description": "Exact fully-qualified name"},
                    "symbol_kind":       {"type": "string", "description": "Filter by kind (class, function, …)"},
                    "symbol_visibility": {"type": "string", "description": "public / protected / private"},
                    "doc_needle":        {"type": "array",  "items": {"type": "string"}, "description": "Full-text search tokens"},
                    "embedding_text": {
                        "type": "string",
                        "description": (
                            "Natural-language search string to be embedded and used "
                            "for semantic similarity search."
                        )
                    },
                    "limit":             {"type": "integer", "minimum": 1, "default": 20},
                    "offset":            {"type": "integer", "minimum": 0, "default": 0},
                },
            },
        }
