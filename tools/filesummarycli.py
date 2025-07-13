from __future__ import annotations

import argparse
from typing import List
from know.settings import ProjectSettings, EmbeddingSettings
from know.project import init_project
from know.tools.base import ToolRegistry, SummaryMode
from know.logger import logger

def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="filesummarycli",
        description="CLI for SummarizeFilesTool.",
    )
    p.add_argument("-p", "--path",      required=True, help="Project root directory")
    p.add_argument("files",             nargs="+",     help="Project-relative file paths to summarise")
    p.add_argument("-m", "--summary-mode",
                   choices=[m.value for m in SummaryMode],
                   default=SummaryMode.ShortSummary.value,
                   help="Detail level for summaries")
    p.add_argument("--visibility",
                   choices=["public", "protected", "private", "all"],
                   default="public",
                   help="Visibility filter passed to the tool")
    p.add_argument("--repo-backend",    choices=["memory", "duckdb"], default="duckdb")
    p.add_argument("--repo-connection", default=None)
    p.add_argument("--enable-embeddings", action="store_true")
    p.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    p.add_argument("--embedding-cache-backend",
                   choices=["duckdb", "sqlite", "none"], default="duckdb")
    p.add_argument("--embedding-cache-path", default="cache.duckdb")
    return p.parse_args()


def main() -> None:
    args = _parse_cli()

    ps_kwargs = {
        "project_path":         args.path,
        "repository_backend":   args.repo_backend,
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

    summarize_tool = ToolRegistry.get("vectorops_summarize_files")

    summaries: List[dict] = summarize_tool.execute(
        project,
        paths=args.files,
        symbol_visibility=args.visibility,
        summary_mode=SummaryMode(args.summary_mode),
    )

    for s in summaries:
        print(f"── {s['path']}")
        print(s["content"])
        print()

if __name__ == "__main__":
    main()
