from __future__ import annotations

from typing import List, Sequence, Optional

from pydantic import BaseModel

from know.data import SymbolSearchQuery
from know.models import SymbolKind, Visibility
from know.project import Project
from know.tools.base import BaseTool
from know.parsers import AbstractCodeParser      # top-level import section


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
        include_summary: bool = False,
        include_body:    bool = False,
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
            limit             = limit,
            offset            = offset,
        )
        syms = project.data_repository.symbol.search(repo_id, query)
        file_repo = project.data_repository.file

        # --- eager load parser for summary/body only if required ---
        parser: AbstractCodeParser | None = None
        if include_summary or include_body:
            # project.get_parser() returns a language specific parser; falls back to None
            parser = project.get_parser()

        results: list[SymbolSearchResult] = []
        for s in syms:
            file_path = None
            if s.file_id:
                fm = file_repo.get_by_id(s.file_id)
                file_path = fm.path if fm else None

            sym_summary = None
            sym_body    = None
            if parser and (include_summary or include_body):
                try:
                    if include_summary:
                        sym_summary = parser.get_symbol_summary(s)
                    if include_body and s.file_id:
                        fm = file_repo.get_by_id(s.file_id)
                        if fm and fm.content:
                            # assume FileMetadata has raw source in .content
                            src_lines = fm.content.splitlines()
                            if getattr(s, "start_line", None) is not None and getattr(s, "end_line", None) is not None:
                                start = s.start_line - 1
                                end   = s.end_line
                                sym_body = "\n".join(src_lines[start:end])
                except Exception:
                    pass    # graceful degradation

            results.append(
                SymbolSearchResult(
                    symbol_id  = s.id,
                    fqn        = s.fqn,
                    name       = s.name,
                    kind       = s.kind,
                    visibility = s.visibility,
                    file_path  = file_path,
                    summary    = sym_summary,
                    body       = sym_body,
                )
            )
        return results

    def get_openai_schema(self) -> dict:
        kind_enum        = [k.value for k in SymbolKind]
        visibility_enum  = [v.value for v in Visibility]

        return {
            "name": self.tool_name,
            "description": "Search symbols in the current project repository. Use the tool "
            "to do project discovery and find relevant public symbols that can be used to "
            "solve users request.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol_name":       {"type": "string", "description": "Substring match on symbol name"},
                    "symbol_fqn":        {"type": "string", "description": "Exact fully-qualified name"},
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
                    #"doc_needle":        {"type": "array",  "items": {"type": "string"}, "description": "Full-text search tokens"},
                    "query": {
                        "type": "string",
                        "description": (
                            "Natural-language search string to be embedded and used "
                            "for semantic similarity search."
                        )
                    },
                    "limit":             {"type": "integer", "minimum": 1, "default": 20},
                    "offset":            {"type": "integer", "minimum": 0, "default": 0},
                    "include_summary": {
                        "type": "boolean",
                        "description": "If true, include a natural-language summary of each symbol."
                    },
                    "include_body": {
                        "type": "boolean",
                        "description": "If true, include full source code body of each symbol."
                    },
                },
            },
        }
