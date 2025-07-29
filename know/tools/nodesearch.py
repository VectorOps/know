from typing import List, Sequence, Optional

from pydantic import BaseModel, Field

from know.data import NodeSearchQuery
from know.models import NodeKind, Visibility
from know.project import ProjectManager
from .base import BaseTool, MCPToolDefinition
from know.file_summary import SummaryMode
from know.parsers import CodeParserRegistry, AbstractCodeParser, AbstractLanguageHelper
from know.models import File


class NodeSearchReq(BaseModel):
    symbol_name: Optional[str] = Field(
        default=None, description="Exact, case-sensitive match on the symbol’s short name."
    )
    symbol_fqn: Optional[str] = Field(
        default=None,
        description="Substring match against the fully-qualified name (e.g. `package.module.Class.method`).",
    )
    kind: Optional[NodeKind | str] = Field(
        default=None, description="Restrict results to a specific kind of node or a symbol."
    )
    visibility: Optional[Visibility | str] = Field(
        default=None,
        description=(
            "Restrict by visibility modifier (`public`, `protected`, `private`) or use `all` to include "
            "every symbol. Defaults to `public`."
        ),
    )
    query: Optional[str] = Field(
        default=None,
        description=(
            "Natural-language search string evaluated against docstrings, comments, and code with both "
            "full-text and vector search. Use when you don’t know the exact name."
        ),
    )
    limit: int | None = Field(default=20, description="Maximum number of results to return.")
    offset: int | None = Field(default=0, description="Number of results to skip. Used for pagination.")
    summary_mode: SummaryMode | str = Field(
        default=SummaryMode.ShortSummary,
        description="Amount of source code to include with each match",
    )


class NodeSearchResult(BaseModel):
    symbol_id: str = Field(description="Unique identifier for the symbol.")
    fqn: Optional[str] = Field(default=None, description="The fully-qualified name of the symbol.")
    name: Optional[str] = Field(default=None, description="The short name of the symbol.")
    kind: Optional[str] = Field(
        default=None, description="The kind of symbol (e.g., 'function', 'class')."
    )
    visibility: Optional[str] = Field(
        default=None, description="The visibility of the symbol (e.g., 'public', 'private')."
    )
    file_path: Optional[str] = Field(
        default=None, description="The path to the file containing the symbol."
    )
    body: Optional[str] = Field(
        default=None, description="The summary or body of the symbol, depending on the summary_mode."
    )


class NodeSearchTool(BaseTool):
    tool_name = "vectorops_search"
    tool_input = NodeSearchReq
    tool_output = List[NodeSearchResult]

    def execute(
        self,
        pm: ProjectManager,
        req: NodeSearchReq,
    ) -> List[NodeSearchResult]:
        # normalise string / enum inputs
        kind: NodeKind | None = None
        if req.kind is None:
            pass
        elif isinstance(req.kind, NodeKind):
            kind = req.kind
        else:
            try:
                kind = NodeKind(req.kind)
            except ValueError:
                valid_kinds = [k.value for k in NodeKind]
                raise ValueError(f"Invalid kind '{req.kind}'. Valid values are: {valid_kinds}")

        # visibility
        vis = None
        if isinstance(req.visibility, Visibility):
            vis = req.visibility
        elif isinstance(req.visibility, str):
            if req.visibility.lower() == "all":
                vis = None
            else:
                try:
                    vis = Visibility(req.visibility)
                except ValueError:
                    valid_vis = [v.value for v in Visibility] + ["all"]
                    raise ValueError(f"Invalid visibility '{req.visibility}'. Valid values are: {valid_vis}")

        # summary_mode
        summary_mode = req.summary_mode
        if isinstance(summary_mode, str):
            try:
                summary_mode = SummaryMode(summary_mode)
            except ValueError:
                valid_modes = [m.value for m in SummaryMode]
                raise ValueError(f"Invalid summary_mode '{req.summary_mode}'. Valid values are: {valid_modes}")

        # transform free-text query -> embedding vector (if requested)
        embedding_vec = None
        if req.query:
            embedding_vec = pm.compute_embedding(req.query)

        repo_id = pm.default_repo.id
        query   = NodeSearchQuery(
            symbol_name = req.symbol_name,
            symbol_fqn = req.symbol_fqn,
            kind = kind,
            visibility = vis,
            doc_needle = req.query,
            embedding_query = embedding_vec,
            limit = req.limit or 20,
            offset = req.offset,
        )
        syms = pm.data.symbol.search(repo_id, query)

        file_repo = pm.data.file

        results: list[NodeSearchResult] = []
        for s in syms:
            helper: AbstractLanguageHelper | None = None

            # TODO: Optimize to dedupe and get a list of files by ids
            fm: File | None = None
            file_path = None
            if s.file_id:
                fm = file_repo.get_by_id(s.file_id)
                file_path = fm.path if fm else None

                if fm and fm.language:
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
                                                          include_parents=True)  # type: ignore[call-arg]

            results.append(
                NodeSearchResult(
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
        kind_enum        = [k.value for k in NodeKind]
        visibility_enum  = [v.value for v in Visibility] + ["all"]
        summary_enum = [m.value for m in SummaryMode]

        return {
            "name": self.tool_name,
            "description": (
                "Search for code blocks (functions, classes, variables, etc.) in the current repository. "
                "All supplied filters are combined with logical **AND**."
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
                    "kind": {
                        "type": "string",
                        "enum": kind_enum,
                        "description": "Restrict results to a specific kind of symbol."
                    },
                    "visibility": {
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
                    "offset": {
                        "type": "integer",
                        "minimum": 0,
                        "default": 0,
                        "description": "Number of results to skip. Used for pagination."
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

    def get_mcp_definition(self, pm: ProjectManager) -> MCPToolDefinition:
        def symbolsearch(req: NodeSearchReq) -> List[NodeSearchResult]:
            return self.execute(pm, req)

        schema = self.get_openai_schema()
        return MCPToolDefinition(
            fn=symbolsearch,
            name=self.tool_name,
            description=schema.get("description"),
        )
