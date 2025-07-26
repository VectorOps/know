from __future__ import annotations

import sys
from typing import List, Dict, Any

from pydantic import Field, AliasChoices
from pydantic_settings import SettingsConfigDict
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.patch_stdout import patch_stdout

from know.settings import ProjectSettings, print_help
from know.project import init_project
from know.tools.base import ToolRegistry
from know.logger import logger

import logging
logging.basicConfig(
    level=logging.DEBUG,
)


class Settings(ProjectSettings):
    """Search-CLI specific settings, extending project settings."""
    model_config = SettingsConfigDict(
        cli_parse_args=True,
        cli_kebab_case=True,
        cli_enforce_required=True,
        env_prefix="KNOW_",
        env_nested_delimiter="_",
    )

    project_path: str = Field(
        ...,
        description="Root directory of the project to analyse/assist with.",
        validation_alias=AliasChoices("project-path", "p", "path"),
    )

    limit: int = Field(20, description="Maximum number of search results to return.")
    offset: int = Field(0, description="Offset for paginating search results.")


def _print_results(results: List[Dict[str, Any]]) -> None:
    if not results:
        print("No symbols found.")
        return
    for r in results:
        print("-" * 80)
        print(f"{r.get('name') or '<unnamed>'}   ({r.get('kind')}) ({r.get('symbol_id')})")
        if r.get("fqn"):
            print(f"FQN:  {r['fqn']}")
        if r.get("file_path"):
            print(f"File: {r['file_path']}")
        if r.get("body"):
            print("\n" + r["body"])
    print("-" * 80)
    print(f"{len(results)} result(s).")


def main() -> None:
    if "--help" in sys.argv or "-h" in sys.argv:
        print_help(Settings, "searchcli.py")
        sys.exit(0)

    try:
        settings = Settings()
    except Exception as e:
        print(f"Error: Invalid settings.\n{e}", file=sys.stderr)
        sys.exit(1)

    project = init_project(settings)

    # obtain tool instance from registry
    search_tool = ToolRegistry.get("vectorops_search_symbols")

    print("Interactive symbol search.  Type '/exit' or Ctrl-D to quit.")
    session = PromptSession(history=FileHistory(".symbol_search_history"))
    with patch_stdout():
        while True:
            try:
                query = session.prompt("> ")
            except (EOFError, KeyboardInterrupt):
                import threading
                import traceback
                for th in threading.enumerate():
                    print(th)
                    traceback.print_stack(sys._current_frames()[th.ident])
                    print()
                break
            query = query.strip()
            if not query:
                continue
            if query.lower() in {"/exit", "/quit"}:
                break

            try:
                results = search_tool.execute(
                    project,
                    query=query,
                    limit=settings.limit,
                    offset=settings.offset,
                )
                _print_results(results)
            except Exception as exc:        # noqa: BLE001
                logger.error("Search failed: %s", exc)


if __name__ == "__main__":
    main()
