import sys
from typing import List, Dict, Any, Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.patch_stdout import patch_stdout
from pydantic import Field, AliasChoices
from pydantic_settings import SettingsConfigDict

from know.settings import ProjectSettings, EmbeddingSettings, print_help
from know.project import init_project
from know.file_summary import SummaryMode
from know.tools.base import ToolRegistry
from know.data import FileFilter
from know.logger import logger

import logging
logging.basicConfig(
    level=logging.DEBUG,
)


class RepomapCliEmbeddingSettings(EmbeddingSettings):
    """Custom embedding settings for repomapcli with different defaults."""
    cache_path: Optional[str] = Field(
        "cache.duckdb",
        description=(
            "The file path or connection string for the embedding cache backend. "
            'This is ignored if `cache_backend` is "none".'
        ),
    )


class Settings(ProjectSettings):
    """RepoMap-CLI specific settings, extending project settings."""
    model_config = SettingsConfigDict(
        cli_parse_args=True,
        cli_kebab_case=True,
        cli_enforce_required=True,
        env_prefix="KNOW_",
        env_nested_delimiter="_",
    )

    project_path: str = Field(
        ...,
        description="Project root directory.",
        validation_alias=AliasChoices("project-path", "p", "path"),
    )
    limit: int = Field(20, description="Default result limit for RepoMap runs.")
    repository_backend: str = Field(
        "duckdb",
        description='The backend to use for storing metadata. Options are "memory" or "duckdb".',
    )

    # To override nested defaults, we must override the parent field.
    embedding: RepomapCliEmbeddingSettings = Field(default_factory=RepomapCliEmbeddingSettings)


def _print_scores(scores: List[Dict[str, Any]]) -> None:
    if not scores:
        print("No results.")
        return
    for r in scores:
        print(f"{r['file_path']:60}  {r['score']:.6f}")
        if r.get("summary"):
            print(f"   {r['summary']}")
    print(f"{len(scores)} file(s).")


def main() -> None:
    # Custom help handler using iter_settings
    if "--help" in sys.argv or "-h" in sys.argv:
        print_help(Settings, "repomapcli.py")
        sys.exit(0)

    try:
        settings = Settings()
    except Exception as e:
        print(f"Error: Invalid settings.\n{e}", file=sys.stderr)
        sys.exit(1)

    project = init_project(settings)

    repomap_tool = ToolRegistry.get("vectorops_repomap")

    symbol_seeds: list[str] = []
    file_seeds:   list[str] = []
    prompt_text:  str | None = None
    token_limit_count: int | None = None
    token_limit_model: str | None = None
    summary_mode: SummaryMode = SummaryMode.ShortSummary

    print("RepoMap interactive CLI.  Type '/help' for commands, '/exit' to quit.")
    session = PromptSession(history=FileHistory(".repomap_history"))
    with patch_stdout():
        while True:
            try:
                line = session.prompt("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not line:
                continue
            if line.lower() in {"/exit", "/quit"}:
                break

            # ------------------ slash-commands -------------------------
            if line.startswith("/"):
                cmd, *rest = line.split(maxsplit=1)
                arg = rest[0] if rest else ""

                if cmd == "/help":
                    print(
                        "Slash-commands:\n"
                        "  /files                – list project files\n"
                        "  /sym  <name>          – add symbol seed\n"
                        "  /file <rel/path.py>   – add file-path seed\n"
                        "  /prompt <text>       – set/replace free-text prompt\n"
                        "  /list                 – show current seeds\n"
                        "  /run                  – run RepoMap and show scores\n"
                        "  /clear                – clear all seeds\n"
                        "  /summary [on|off]        – toggle / force summaries\n"
                        "  /tokens  <count> <model> – set summary-token budget\n"
                        "  /tokens  clear           – remove token budget\n"
                        "  /exit                 – quit"
                    )

                elif cmd == "/files":
                    repo_id = project.get_repo().id
                    paths = sorted(
                        f.path for f in project.data_repository.file.get_list(
                            FileFilter(repo_id=repo_id)
                        )
                    )
                    for p in paths: print(p)
                    print(f"{len(paths)} file(s)")

                elif cmd == "/sym":
                    if arg:
                        symbol_seeds.append(arg)
                        print(f"added symbol: {arg!r}")

                elif cmd == "/file":
                    if arg:
                        file_seeds.append(arg)
                        print(f"added file:   {arg!r}")

                elif cmd == "/prompt":
                    if arg:
                        prompt_text = arg
                        print(f"prompt set to: {prompt_text!r}")
                    else:
                        prompt_text = None
                        print("prompt cleared.")

                elif cmd == "/list":
                    print("Symbol seeds:", symbol_seeds or "—")
                    print("File   seeds:", file_seeds or "—")
                    print("Prompt text :", repr(prompt_text) if prompt_text else "—")
                    print("Summaries    :", "on" if include_summaries else "off")
                    print("Token budget :", (
                        f"{token_limit_count} @ {token_limit_model}"
                        if token_limit_count and token_limit_model else "—"
                    ))

                elif cmd == "/clear":
                    symbol_seeds.clear()
                    file_seeds.clear()
                    prompt_text = None
                    print("Seeds cleared.")

                elif cmd == "/summary":
                    if arg.lower() in {"on", "yes"}:
                        summary_mode = SummaryMode.ShortSummary
                    elif arg.lower() in {"off", "no"}:
                        summary_mode = SummaryMode.Skip
                    print(f"Summaries {summary_mode}.")

                elif cmd == "/tokens":                 # NEW
                    if not arg:
                        if token_limit_count and token_limit_model:
                            print(f"Token limit: {token_limit_count} @ {token_limit_model}")
                        else:
                            print("No token limit set.")
                    elif arg.strip().lower() == "clear":
                        token_limit_count = token_limit_model = None
                        print("Token limit cleared.")
                    else:
                        parts = arg.split(maxsplit=1)
                        if len(parts) == 2 and parts[0].isdigit():
                            token_limit_count = int(parts[0])
                            token_limit_model = parts[1]
                            print(f"Token limit set: {token_limit_count} @ {token_limit_model}")
                        else:
                            print("Usage: /tokens <count> <model>  or  /tokens clear")

                elif cmd == "/run":
                    try:
                        res = repomap_tool.execute(
                            project,
                            repomap_tool.tool_input(
                                symbol_names=symbol_seeds or None,
                                file_paths=file_seeds   or None,
                                prompt=prompt_text,
                                limit=settings.limit,
                                summary_mode=summary_mode,
                                token_limit_count=token_limit_count,
                                token_limit_model=token_limit_model,
                            )
                        )
                        _print_scores(res)
                    except Exception as exc:             # noqa: BLE001
                        logger.error("RepoMap run failed: %s", exc)

                else:
                    print("Unknown command.  Type /help.")
                continue

            print("Input ignored – start commands with '/'")

if __name__ == "__main__":
    main()
