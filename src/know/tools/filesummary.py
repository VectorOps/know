
from typing import Sequence, List
from .base import BaseTool, MCPToolDefinition
from pydantic import BaseModel

from know.file_summary import FileSummary, SummaryMode, build_file_summary

from know.project import ProjectManager
from know.models import Visibility


class SummarizeFilesReq(BaseModel):
    """Request model for the SummarizeFilesTool."""
    paths: Sequence[str]
    summary_mode: SummaryMode | str = SummaryMode.Definition


class SummarizeFilesTool(BaseTool):
    """Tool to generate summaries for a list of files."""
    tool_name = "vectorops_summarize_files"
    tool_input = SummarizeFilesReq
    tool_output = List[FileSummary]

    def execute(
        self,
        pm: ProjectManager,
        req: SummarizeFilesReq,
    ) -> List[FileSummary]:
        """Generate summaries for the requested files."""
        pm.maybe_refresh()

        summary_mode = req.summary_mode
        if isinstance(summary_mode, str):
            summary_mode = SummaryMode(summary_mode)

        summaries: list[FileSummary] = []
        for path in req.paths:
            deconstructed = pm.deconstruct_virtual_path(path)
            if not deconstructed:
                continue

            repo, rel_path = deconstructed
            fs = build_file_summary(
                pm, repo, rel_path, summary_mode=summary_mode
            )
            if fs:
                fs.path = path
                summaries.append(fs)

        return summaries

    def get_openai_schema(self) -> dict:
        """Return the OpenAI schema for the tool."""
        summary_enum = [m.value for m in SummaryMode]

        return {
            "name": self.tool_name,
            "description": (
                "Return a summary for each supplied file, consisting of its "
                "import statements and top-level symbol definitions. Use this tool "
                "to get an overview of interesting files. Prefer the default "
                "`summary_short` mode, but request `full` if needed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Relative paths of the files to be summarized.",
                    },
                    "summary_mode": {
                        "type": "string",
                        "enum": summary_enum,
                        "default": SummaryMode.Definition.value,
                        "description": (
                            "Level of detail for the generated summary "
                            "(`definition`, `documentation`). "
                        ),
                    },
                },
                "required": ["paths"],
            },
        }

    def get_mcp_definition(self, pm: ProjectManager) -> MCPToolDefinition:
        """Return the MCP definition for the tool."""
        def filesummary(req: SummarizeFilesReq) -> List[FileSummary]:
            return self.execute(pm, req)

        schema = self.get_openai_schema()
        return MCPToolDefinition(
            fn=filesummary,
            name=self.tool_name,
            description=schema.get("description"),
        )
