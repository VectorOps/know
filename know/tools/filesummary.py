
from typing import Sequence, List
from .base import BaseTool, MCPToolDefinition
from pydantic import BaseModel

from know.file_summary import FileSummary, SummaryMode, build_file_summary

from know.project import Project
from know.models import Visibility


class SummarizeFilesReq(BaseModel):
    """Request model for the SummarizeFilesTool."""
    paths: Sequence[str]
    summary_mode: SummaryMode | str = SummaryMode.ShortSummary


class SummarizeFilesTool(BaseTool):
    """Tool to generate summaries for a list of files."""
    tool_name = "vectorops_summarize_files"
    tool_input = SummarizeFilesReq
    tool_output = List[FileSummary]

    def execute(
        self,
        project: Project,
        req: SummarizeFilesReq,
    ) -> List[FileSummary]:
        """Generate summaries for the requested files."""
        summary_mode = req.summary_mode
        if isinstance(summary_mode, str):
            summary_mode = SummaryMode(summary_mode)

        summaries: list[FileSummary] = []
        for rel_path in req.paths:
            fs = build_file_summary(project, rel_path, summary_mode=summary_mode)
            if fs:
                summaries.append(fs)

        return summaries

    def get_openai_schema(self) -> dict:
        """Return the OpenAI schema for the tool."""
        summary_enum = [m.value for m in SummaryMode]

        return {
            "name": self.tool_name,
            "description": (
                "Return a text summary for each supplied file, consisting of its "
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
                        "description": "Project-relative paths of the files to be summarized.",
                    },
                    "summary_mode": {
                        "type": "string",
                        "enum": summary_enum,
                        "default": SummaryMode.ShortSummary.value,
                        "description": (
                            "Level of detail for the generated summary "
                            "(`skip`, `summary_short`, or `full`)."
                        ),
                    },
                },
                "required": ["paths"],
            },
        }

    def get_mcp_definition(self, project: Project) -> MCPToolDefinition:
        """Return the MCP definition for the tool."""
        def filesummary(req: SummarizeFilesReq) -> List[FileSummary]:
            return self.execute(project, req)

        schema = self.get_openai_schema()
        return MCPToolDefinition(
            fn=filesummary,
            name=self.tool_name,
            description=schema.get("description"),
        )
