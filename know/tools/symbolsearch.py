from __future__ import annotations

from typing import List, Sequence, Optional

from pydantic import BaseModel

from know.data import SymbolSearchQuery
from know.models import SymbolKind, Visibility
from know.project import Project
from know.tools.base import BaseTool
from know.parsers import CodeParserRegistry, AbstractCodeParser


class SymbolSearchResult(BaseModel):
    symbol_id: str
    fqn:       Optional[str] = None
    name:      Optional[str] = None
    kind:      Optional[str] = None
    visibility:Optional[str] = None
    file_path: Optional[str] = None
    summary:    Optional[str] = None
    body:       Optional[str] = None


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
        query: Optional[str] = None,
        limit: int | None = 20,
        offset: int | None = 0,
        exclude_summary: bool = False,
    ) -> List[SymbolSearchResult]:

        # translate enums if string values are given
        kind  = SymbolKind(symbol_kind)           if symbol_kind else None
        vis   = Visibility(symbol_visibility)     if symbol_visibility else None

        # ------------------------------------------------------------
        # transform free-text query → embedding vector (if requested)
        # ------------------------------------------------------------
        embedding_vec = None
        if query:
            embedding_vec = project.compute_embedding(query, is_code=True)

        repo_id = project.get_repo().id
        query   = SymbolSearchQuery(
            symbol_name       = symbol_name,
            symbol_fqn        = symbol_fqn,
            symbol_kind       = kind,
            symbol_visibility = vis,
            doc_needle        = list(doc_needle) if doc_needle else None,
            embedding_query   = embedding_vec,
            limit             = limit or 20,
            offset            = offset,
        )
        syms = project.data_repository.symbol.search(repo_id, query)
        file_repo = project.data_repository.file

        results: list[SymbolSearchResult] = []
        for s in syms:
            parser: AbstractCodeParser | None = None

            # TODO: Optimize to dedupe and get a list of files by ids
            fm: FileMetadata | None = None
            file_path = None
            if s.file_id:
                fm = file_repo.get_by_id(s.file_id)
                file_path = fm.path if fm else None

                if fm.language:
                    parser = CodeParserRegistry.get_language(fm.language)

            sym_summary = None
            if parser and not exclude_summary:
                sym_summary = parser.get_symbol_summary(s)

            results.append(
                SymbolSearchResult(
                    symbol_id  = s.id,
                    fqn        = s.fqn,
                    name       = s.name,
                    kind       = s.kind,
                    visibility = s.visibility,
                    file_path  = file_path,
                    summary    = sym_summary,
                )
            )
        return self.to_python(results)

    def get_openai_schema(self) -> dict:
        kind_enum        = [k.value for k in SymbolKind]
        visibility_enum  = [v.value for v in Visibility]

        return {
            "name": self.tool_name,
            "description": "Search symbols in the current project repository. Use the tool "
            "to do project discovery and find relevant symbol definitions that can be used to "
            "solve users request. All paramers use the AND logical condition. "
            "Prefer semantic code similarity search over documentation search.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol_name":       {"type": "string", "description": "Exact match on symbol name"},
                    "symbol_fqn":        {"type": "string", "description": "Substring match on fully-qualified name"},
                    "symbol_kind": {
                        "type": "string",
                        "enum": kind_enum,
                        "description": "Filter by symbol kind."
                    },
                    "symbol_visibility": {
                        "type": "string",
                        "enum": visibility_enum,
                        "description": "Filter by visibility (public / protected / private)."
                    },
                    "doc_needle":        {"type": "array",  "items": {"type": "string"}, "description": "Full-text search tokens"},
                    "query": {
                        "type": "string",
                        "description": (
                            "Natural-language search string using embedding vector and used "
                            "for semantic code similarity search. If nothing is found, try doc_needle search."
                        )
                    },
                    "limit":             {"type": "integer", "minimum": 1, "default": 20},
                    "offset":            {"type": "integer", "minimum": 0, "default": 0},
                    "exclude_summary": {
                        "type": "boolean",
                        "description": "If true, disables symbol summary output.",
                    },
                },
            },
        }
