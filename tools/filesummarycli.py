from __future__ import annotations

import sys
from typing import List

from pydantic import Field, AliasChoices
from pydantic_settings import SettingsConfigDict

from know.settings import ProjectSettings, print_help
from know.project import init_project
from know.tools.base import ToolRegistry
from know.file_summary import SummaryMode
from know.logger import logger

class Settings(ProjectSettings):
    """filesummarycli specific settings."""

    model_config = SettingsConfigDict(
        cli_parse_args=True,
        cli_kebab_case=True,
        cli_enforce_required=True,
        env_prefix="KNOW_",
        env_nested_delimiter="_",
    )

    project_path: str = Field(
        description="Project root directory.",
        validation_alias=AliasChoices("project-path", "p", "path"),
    )

    files: List[str] = Field(
        description="Project-relative file paths to summarise.",
    )

    summary_mode: SummaryMode = Field(
        default=SummaryMode.ShortSummary,
        description="Detail level for summaries.",
        validation_alias=AliasChoices("summary-mode", "m"),
    )


# Main method
def main() -> None:
    # Custom help handler using iter_settings
    if "--help" in sys.argv or "-h" in sys.argv:
        print_help(Settings, "filesummarycli.py")
        sys.exit(0)

    try:
        settings = Settings()
    except Exception as e:
        print(f"Error: Invalid settings.\n{e}", file=sys.stderr)
        sys.exit(1)

    project = init_project(settings)

    summarize_tool = ToolRegistry.get("vectorops_summarize_files")

    summaries: List[dict] = summarize_tool.execute(
        project,
        paths=settings.files,
        summary_mode=settings.summary_mode,
    )

    for s in summaries:
        print(f"── {s['path']}")
        print(s["content"])
        print()

if __name__ == "__main__":
    main()
