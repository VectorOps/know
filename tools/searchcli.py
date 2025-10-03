import sys
import json
from typing import List, Dict, Any

from pydantic import Field, AliasChoices
from pydantic_settings import SettingsConfigDict
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.patch_stdout import patch_stdout

from know.settings import ProjectSettings, print_help, ToolOutput
from know import init_project
from know.tools.base import ToolRegistry
from know.tools.nodesearch import NodeSearchResult
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

    limit: int = Field(default=20, description="Maximum number of search results to return.")
    offset: int = Field(default=0, description="Offset for paginating search results.")


def _print_results(results: List[NodeSearchResult]) -> None:
    if not results:
        print("No symbols found.")
        return
    for r in results:
        print("-" * 80)
        print(f"{r.name or '<unnamed>'}   ({r.kind}) ({r.symbol_id})")
        if r.fqn:
            print(f"FQN:  {r.fqn}")
        if r.file_path:
            print(f"File: {r.file_path}")
        if r.body:
            print("\n" + r.body)
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

    # obtain tool instance from registry
    search_tool = ToolRegistry.get("vectorops_search")

    # Force JSON output for the CLI tool
    settings.tools.outputs[search_tool.tool_name] = ToolOutput.JSON

    project = init_project(settings)

    print("Interactive symbol search.  Type '/exit' or Ctrl-D to quit.")
    session: PromptSession = PromptSession(history=FileHistory(".symbol_search_history"))
    with patch_stdout():
        while True:
            try:
                query = session.prompt("> ")
            except (EOFError, KeyboardInterrupt):
                break
            query = query.strip()
            if not query:
                continue
            if query.lower() in {"/exit", "/quit"}:
                break

            try:
                results_str = search_tool.execute(project, {
                    "query": query,
                    "limit": settings.limit,
                    "offset": settings.offset,
                })
                results_data = json.loads(results_str)
                results = [NodeSearchResult(**r) for r in results_data]
                _print_results(results)
            except Exception as exc:        # noqa: BLE001
                logger.error("Search failed: %s", exc)


if __name__ == "__main__":
    main()
