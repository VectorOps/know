
from __future__ import annotations

from typing import Sequence, List
from .base import BaseTool

from know.tools.file_summary_helper import FileSummary, build_file_summary
from know.tools.base import SummaryMode

from know.project import Project
from know.models import Visibility

class SummarizeFilesTool(BaseTool):
    tool_name = "vectorops_summarize_files"

    def execute(
        self,
        project: Project,
        paths: Sequence[str],
        symbol_visibility: str | None = None,
        summary_mode: SummaryMode = SummaryMode.ShortSummary,
    ) -> List[FileSummary]:
        summaries: list[FileSummary] = []
        for rel_path in paths:
            mode = summary_mode
            fs = build_file_summary(project, rel_path, mode)
            if fs:
                summaries.append(fs)
        return self.to_python(summaries)

    def get_openai_schema(self) -> dict:
        visibility_enum = [v.value for v in Visibility] + ["all"]
        summary_enum = [m.value for m in SummaryMode]

        return {
            "name": self.tool_name,
            "description": (
                "Return a text summary for each supplied file consisting "
                "of its import statements and top-level symbol definitions."
                "Use this tool to find overview of the interesting files. Prefer "
                "the default summary_short mode, but request full file if needed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Project-relative paths of the files to be "
                            "summarized."
                        ),
                    },
                    "symbol_visibility": {
                        "type": "string",
                        "enum": visibility_enum,
                        "description": (
                            "Restrict by visibility modifier (`public`, `protected`, `private`) "
                            "or use `all` to include every symbol. Defaults to `public`."
                        ),
                    },
                    "summary_mode": {
                        "type": "string",
                        "enum": summary_enum,
                        "default": SummaryMode.ShortSummary.value,
                        "description": (
                            "Level of detail for the generated summary "
                            "(`skip`/`summary_short`/`summary_full`/`full`)."
                        ),
                    },
                },
                "required": ["paths"],
            },
        }
