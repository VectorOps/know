from __future__ import annotations

import argparse
from typing import List, Dict, Any

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.patch_stdout import patch_stdout

from know.settings import ProjectSettings, EmbeddingSettings
from know.project import init_project
from know.tools.base import ToolRegistry
from know.logger import logger


def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog = "repomapcli",
        description = "Interactive test-CLI for RepoMapTool.",
    )
    p.add_argument("-p", "--path", required=True,
                   help="Project root directory")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--repo-backend", choices=["memory", "duckdb"],
                   default="duckdb")
    p.add_argument("--repo-connection",
                   default=None)
    p.add_argument("--enable-embeddings", action="store_true")
    p.add_argument("--embedding-model",
                   default="all-MiniLM-L6-v2")
    p.add_argument("--embedding-cache-backend",
                   choices=["duckdb", "sqlite", "none"],
                   default="duckdb")
    p.add_argument("--embedding-cache-path",
                   default="cache.duckdb")
    return p.parse_args()


def _print_scores(scores: List[Dict[str, Any]]) -> None:
    if not scores:
        print("No results.")
        return
    for r in scores:
        print(f"{r['file_path']:60}  {r['score']:.6f}")
    print(f"{len(scores)} file(s).")


def main() -> None:
    args = _parse_cli()

    ps_kwargs = {
        "project_path": args.path,
        "repository_backend": args.repo_backend,
        "repository_connection": args.repo_connection,
    }
    if args.enable_embeddings:
        ps_kwargs["embedding"] = EmbeddingSettings(
            enabled=True,
            model_name=args.embedding_model,
            cache_backend=args.embedding_cache_backend,
            cache_path=args.embedding_cache_path,
        )
    project = init_project(ProjectSettings(**ps_kwargs))

    repomap_tool = ToolRegistry.get("vectorops_repomap")

    symbol_seeds: list[str] = []
    file_seeds:   list[str] = []
    prompt_text:  str | None = None          # NEW – free-text prompt seed

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
                        "  /exit                 – quit"
                    )

                elif cmd == "/files":
                    repo_id = project.get_repo().id
                    paths = sorted(
                        f.path for f in project.data_repository.file.get_list_by_repo_id(repo_id)
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

                elif cmd == "/clear":
                    symbol_seeds.clear()
                    file_seeds.clear()
                    prompt_text = None             # NEW
                    print("Seeds cleared.")

                elif cmd == "/run":
                    try:
                        res = repomap_tool.execute(
                            project,
                            symbol_names=symbol_seeds or None,
                            file_paths=file_seeds   or None,
                            prompt=prompt_text,              # NEW
                            limit=args.limit,
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
