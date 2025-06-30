import os
import asyncio
import argparse
import json
from typing import List, Dict

import litellm
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.patch_stdout import patch_stdout

from know.logger import logger
from know.settings import ProjectSettings, EmbeddingSettings
from know.project import init_project
from know.tools.base import ToolRegistry
from devtools import pformat


SYSTEM_PROMPT = """
Please resolve the user's task by editing and testing the code files in your current code
execution session. You are a deployed coding agent. The repo(s) are already cloned
in your working directory, and you must fully solve the problem for your answer to be
considered correct. You also have access to special tools that allow quick navigation
on a codebase. Use these tools to navigate codebase or search for relevant code needed
to solve users question.
"""


def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simple interactive chat CLI.")
    p.add_argument("-m", "--model", default=os.getenv("OPENAI_MODEL", "gpt-4.1"))
    p.add_argument("-s", "--system", default=SYSTEM_PROMPT)
    p.add_argument("-p", "--path", "--project-path",
                   required=True,
                   help="Root directory of the project to analyse/assist with")
    p.add_argument(
        "--enable-embeddings",
        action="store_true",
        help="Load the embeddings engine so semantic-search tools work.",
    )
    p.add_argument(
        "--embedding-model",
        default=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        help=(
            "HuggingFace hub model name or local path to use for the embeddings "
            "engine (only used when --enable-embeddings is given)."
        ),
    )
    p.add_argument(
        "--repo-backend",
        choices=["memory", "duckdb"],
        default=os.getenv("REPO_BACKEND", "memory"),
        help="Metadata store backend to use ('memory' or 'duckdb').",
    )
    p.add_argument(
        "--repo-connection",
        default=os.getenv("REPO_CONNECTION"),
        help="Connection string / path for the selected backend "
             "(e.g. DuckDB file path).",
    )
    return p.parse_args()


async def _chat(model: str, system_msg: str, project):
    session = PromptSession()
    messages: List[Dict] = [{"role": "system", "content": system_msg}]

    tools = [t.get_openai_schema() for t in ToolRegistry._tools.values()]

    print(f"Loaded model '{model}'.   Type '/new' to start a fresh session, "
          "'/exit' or Ctrl-D to quit.")
    while True:
        try:
            user_input = await session.prompt_async("> ")
        except (EOFError, KeyboardInterrupt):
            break
        cmd = user_input.strip().lower()
        if cmd in {"/exit", "/quit"}:
            break
        if cmd in {"/new", "/restart", "/reset"}:
            # start a fresh chat session – keep the original system prompt
            messages[:] = [{"role": "system", "content": system_msg}]
            print("  Started new session.")
            continue        # ask for the next user prompt
        if not user_input.strip():
            continue

        messages.append({"role": "user", "content": user_input})

        # -------- LLM / tool loop -----------------------------------------
        while True:
            response = litellm.completion(
                model=model,
                messages=messages,
                tools=[
                    {                       # OpenAI / litellm tool-definition format
                        "type": "function",
                        "function": t.get_openai_schema(),
                    }
                    for t in ToolRegistry._tools.values()
                ],          # expose tools
            )
            usage = response.usage or {}
            prompt_toks     = usage.get("prompt_tokens", 0)
            completion_toks = usage.get("completion_tokens", 0)
            total_toks      = usage.get("total_tokens", 0)

            # LiteLLM surfaces the estimated cost in the hidden-params field
            hidden          = getattr(response, "_hidden_params", {}) or {}
            cost_usd        = hidden.get("response_cost", 0.0)
            msg = response.choices[0].message
            # If the assistant wants to call a tool …
            if getattr(msg, "tool_calls", None):
                messages.append(msg)  # add assistant tool request
                for call in msg.tool_calls:
                    tool_name = call.function.name.split("/", 1)[-1]
                    tool      = ToolRegistry.get(tool_name)
                    args      = json.loads(call.function.arguments or "{}")

                    # TODO: Logger
                    print(f"Tool call request {call.function.name}")
                    print(pformat(args))

                    result = tool.execute(project, **args)

                    print(f"Tool call response {call.function.name}")
                    print(pformat(result))

                    messages.append({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": call.function.name,
                        "content": json.dumps(result),
                    })
                # run another completion to let the model consume the tool output
                continue
            # otherwise, regular assistant answer
            print(f"\nAssistant: {msg.content}")
            print(f"[usage] prompt={prompt_toks}  completion={completion_toks} "
                  f"total={total_toks}  -  est. cost ${cost_usd:.6f}")
            messages.append({"role": "assistant", "content": msg.content})
            break


def main() -> None:
    args = _parse_cli()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY env var not set.")
    ps_kwargs = {
        "project_path": args.path,
        "repository_backend": args.repo_backend,
        "repository_connection": args.repo_connection,
    }
    if args.enable_embeddings:
        ps_kwargs["embedding"] = EmbeddingSettings(
            enabled=True,
            model_name=args.embedding_model,
        )
    settings = ProjectSettings(**ps_kwargs)
    project  = init_project(settings)
    with patch_stdout():
        asyncio.run(_chat(args.model, args.system, project))


if __name__ == "__main__":
    main()
