

from __future__ import annotations

import argparse
import os
from typing import List, Dict, Any

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.patch_stdout import patch_stdout

from know.settings import ProjectSettings, EmbeddingSettings
from know.project import init_project
from know.tools.base import ToolRegistry
from know.tools.symbolsearch import IncludeBody   # enum
from know.logger import logger   # optional (use same logger as chatcli)


def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Interactive symbol search CLI.")
    p.add_argument("-p", "--path", "--project-path",
                   required=True,
                   help="Root directory of the project to analyse/assist with")
    p.add_argument(
        "--enable-embeddings",
        action="store_true",
        help="Load the embeddings engine so semantic-search works.",
    )
    p.add_argument(
        "--embedding-model",
        default=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
    )
    p.add_argument(
        "--embedding-cache-backend",
        choices=["duckdb", "sqlite", "none"],
        default=os.getenv("EMBEDDING_CACHE_BACKEND", "duckdb"),
    )
    p.add_argument(
        "--embedding-cache-path",
        default=os.getenv("EMBEDDING_CACHE_PATH", "cache.duckdb"),
    )
    p.add_argument(
        "--repo-backend",
        choices=["memory", "duckdb"],
        default=os.getenv("REPO_BACKEND", "duckdb"),
    )
    p.add_argument(
        "--repo-connection",
        default=os.getenv("REPO_CONNECTION"),
    )
    p.add_argument(
        "--include-body", choices=[e.value for e in IncludeBody],
        default=IncludeBody.SUMMARY.value,
        help="Controls presence of code in the response."
    )
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--offset", type=int, default=0)
    return p.parse_args()


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

    # obtain tool instance from registry
    search_tool = ToolRegistry.get("vectorops_search_symbols")

    print("Interactive symbol search.  Type '/exit' or Ctrl-D to quit.")
    session = PromptSession(history=FileHistory(".symbol_search_history"))
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
                results = search_tool.execute(
                    project,
                    query=query,
                    limit=args.limit,
                    offset=args.offset,
                    include_body=IncludeBody(args.include_body),
                )
                _print_results(results)
            except Exception as exc:        # noqa: BLE001
                logger.error("Search failed: %s", exc)


if __name__ == "__main__":
    main()
