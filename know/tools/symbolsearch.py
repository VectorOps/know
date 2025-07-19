from __future__ import annotations

from typing import List, Sequence, Optional

from pydantic import BaseModel

from know.data import SymbolSearchQuery
from know.models import SymbolKind, Visibility
from know.project import Project
from know.tools.base import BaseTool, SummaryMode
from know.parsers import CodeParserRegistry, AbstractCodeParser, AbstractLanguageHelper
from know.models import FileMetadata


class SymbolSearchResult(BaseModel):
    symbol_id: str
    fqn:       Optional[str] = None
    name:      Optional[str] = None
    kind:      Optional[str] = None
    visibility:Optional[str] = None
    file_path: Optional[str] = None
    body:      Optional[str] = None


class SearchSymbolsTool(BaseTool):
    tool_name = "vectorops_search_symbols"

    def execute(
        self,
        project: Project,
        *,
        symbol_name: Optional[str] = None,
        symbol_fqn: Optional[str] = None,
        symbol_kind: Optional[SymbolKind | str] = None,
        symbol_visibility: Optional[Visibility | str] = None,
        query: Optional[str] = None,
        limit: int | None = 20,
        offset: int | None = 0,
        summary_mode: SummaryMode | str = SummaryMode.ShortSummary,
    ) -> List[SymbolSearchResult]:
        # normalise string / enum inputs
        if symbol_kind is None:
            kind: SymbolKind | None = None
        elif isinstance(symbol_kind, SymbolKind):
            kind = symbol_kind
        else:
            kind = SymbolKind(symbol_kind)

        # symbol_visibility
        if symbol_visibility is None:
            vis: Visibility | None = Visibility.PUBLIC
        elif isinstance(symbol_visibility, Visibility):
            vis = symbol_visibility
        elif str(symbol_visibility).lower() == "all":
            vis = None
        else:
            vis = Visibility(symbol_visibility)

        # summary_mode
        if isinstance(summary_mode, str):
            summary_mode = SummaryMode(summary_mode)

        # transform free-text query → embedding vector (if requested)
        embedding_vec = None
        if query:
            embedding_vec = project.compute_embedding(query)

        repo_id = project.get_repo().id
        query   = SymbolSearchQuery(
            symbol_name       = symbol_name,
            symbol_fqn        = symbol_fqn,
            symbol_kind       = kind,
            symbol_visibility = vis,
            doc_needle        = query,
            embedding_query   = embedding_vec,
            limit             = limit or 20,
            offset            = offset,
        )
        syms = project.data_repository.symbol.search(repo_id, query)
        file_repo = project.data_repository.file

        results: list[SymbolSearchResult] = []
        for s in syms:
            helper: AbstractLanguageHelper | None = None

            # TODO: Optimize to dedupe and get a list of files by ids
            fm: FileMetadata | None = None
            file_path = None
            if s.file_id:
                fm = file_repo.get_by_id(s.file_id)
                file_path = fm.path if fm else None

                if fm.language:
                    helper = CodeParserRegistry.get_helper(fm.language)

            sym_summary: Optional[str] = None
            sym_body:    Optional[str] = None

            sym_body = None
            if summary_mode != SummaryMode.Skip:
                if summary_mode == SummaryMode.Full:
                    sym_body = s.body
                elif helper is not None:
                    include_docs = summary_mode == SummaryMode.FullSummary
                    include_comments = summary_mode == SummaryMode.FullSummary
                    sym_body  = helper.get_symbol_summary(s,
                                                          include_comments=include_comments,
                                                          include_docs=include_docs,
                                                          include_parents=True)

            results.append(
                SymbolSearchResult(
                    symbol_id  = s.id,
                    fqn        = s.fqn,
                    name       = s.name,
                    kind       = s.kind,
                    visibility = s.visibility,
                    file_path  = file_path,
                    body       = sym_body,
                )
            )
        return self.to_python(results)

    def get_openai_schema(self) -> dict:
        kind_enum        = [k.value for k in SymbolKind]
        visibility_enum  = [v.value for v in Visibility] + ["all"]
        summary_enum = [m.value for m in SummaryMode]

        return {
            "name": self.tool_name,
            "description": (
                "Search for symbols (functions, classes, variables, etc.) in the current repository. "
                "All supplied filters are combined with logical **AND**. When both keyword and semantic "
                "search are available, semantic (vector-based) search is preferred."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol_name": {
                        "type": "string",
                        "description": "Exact, case-sensitive match on the symbol’s short name."
                    },
                    "symbol_fqn": {
                        "type": "string",
                        "description": (
                            "Substring match against the fully-qualified name "
                            "(e.g. `package.module.Class.method`)."
                        )
                    },
                    "symbol_kind": {
                        "type": "string",
                        "enum": kind_enum,
                        "description": "Restrict results to a specific kind of symbol."
                    },
                    "symbol_visibility": {
                        "type": "string",
                        "enum": visibility_enum,
                        "description": (
                            "Restrict by visibility modifier (`public`, `protected`, `private`) "
                            "or use `all` to include every symbol. Defaults to `public`."
                        )
                    },
                    "query": {
                        "type": "string",
                        "description": (
                            "Natural-language search string evaluated against docstrings, comments, and code "
                            "with both full-text and vector search. Use when you don’t know the exact name."
                        )
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "default": 20,
                        "description": "Maximum number of results to return."
                    },
                    "summary_mode": {
                        "type": "string",
                        "enum": summary_enum,
                        "default": SummaryMode.ShortSummary.value,
                        "description": (
                            "Amount of source code to include with each match"
                        ),
                    },
                },
            },
        }
