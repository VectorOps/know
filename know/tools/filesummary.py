
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
        skip_docs: bool = False,
    ) -> List[FileSummary]:
        """
        Given a *project* and an iterable of project-relative *paths*,
        return a list of FileSummary objects where *definitions* contains
        the concatenated textual representation of every symbol found in
        the corresponding file.

        If *skip_docs* is true, omit preceding comments and docstrings
        from the summary.
        """
        summaries: list[FileSummary] = []
        for rel_path in paths:
            mode = SummaryMode.ShortSummary if skip_docs else SummaryMode.FullSummary
            fs = build_file_summary(project, rel_path, mode)
            if fs:
                summaries.append(fs)
        return self.to_python(summaries)

    def get_openai_schema(self) -> dict:
        visibility_enum = [v.value for v in Visibility] + ["all"]

        return {
            "name": self.tool_name,
            "description": (
                "Generate a text summary for each supplied file consisting "
                "of its import statements and top-level symbol definitions."
                "Use this tool to find overview of the interesting files."
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
                    "skip_docs": {
                        "type": "boolean",
                        "default": False,
                        "description": (
                            "If true, omit preceding comments and docstrings "
                            "from the summary."
                        ),
                    },
                },
                "required": ["paths"],
            },
        }
