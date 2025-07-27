from typing import List, Sequence, Optional

from pydantic import BaseModel

from know.data import SymbolSearchQuery
from know.models import SymbolKind, Visibility
from know.project import Project
from .base import BaseTool, MCPToolDefinition
from know.file_summary import SummaryMode
from know.parsers import CodeParserRegistry, AbstractCodeParser, AbstractLanguageHelper
from know.models import FileMetadata


class SymbolSearchReq(BaseModel):
    symbol_name: Optional[str] = None
    symbol_fqn: Optional[str] = None
    symbol_kind: Optional[SymbolKind | str] = None
    symbol_visibility: Optional[Visibility | str] = None
    query: Optional[str] = None
    limit: int | None = 20
    offset: int | None = 0
    summary_mode: SummaryMode | str = SummaryMode.ShortSummary


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
    tool_input = SymbolSearchReq
    tool_output = List[SymbolSearchResult]

    def execute(
        self,
        project: Project,
        req: SymbolSearchReq,
    ) -> List[SymbolSearchResult]:
        # normalise string / enum inputs
        kind: SymbolKind | None = None
        if req.symbol_kind is None:
            pass
        elif isinstance(req.symbol_kind, SymbolKind):
            kind = req.symbol_kind
        else:
            kind = SymbolKind(req.symbol_kind)

        # symbol_visibility
        vis = None
        if isinstance(req.symbol_visibility, Visibility):
            vis = symbol_visibility
        elif isinstance(req.symbol_visibility, str):
            if req.symbol_visibility.lower() == "all":
                vis = None
            else:
                vis = Visibility(req.symbol_visibility)

        # summary_mode
        summary_mode = req.summary_mode
        if isinstance(summary_mode, str):
            summary_mode = SummaryMode(summary_mode)

        # transform free-text query -> embedding vector (if requested)
        embedding_vec = None
        if req.query:
            embedding_vec = project.compute_embedding(req.query)

        repo_id = project.get_repo().id
        query   = SymbolSearchQuery(
            symbol_name       = req.symbol_name,
            symbol_fqn        = req.symbol_fqn,
            symbol_kind       = kind,
            symbol_visibility = vis,
            doc_needle        = req.query,
            embedding_query   = embedding_vec,
            limit             = req.limit or 20,
            offset            = req.offset,
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
        return results

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

    def get_mcp_definition(self, project: Project) -> MCPToolDefinition:
        def symbolsearch(req: SymbolSearchReq) -> List[SymbolSearchResult]:
            return self.execute(project, req)

        schema = self.get_openai_schema()
        return MCPToolDefinition(
            fn=symbolsearch,
            name=self.tool_name,
            description=schema.get("description"),
        )
