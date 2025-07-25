import os
import asyncio
import sys
import json
from typing import List, Dict

import litellm
from pydantic import Field
from pydantic_settings import SettingsConfigDict
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.patch_stdout import patch_stdout

from know.logger import logger
from know.settings import ProjectSettings
from know.project import init_project
from know.tools.base import ToolRegistry
from devtools import pformat
from .helper import print_help


SYSTEM_PROMPT = """
Please resolve the user's task by editing and testing the code files in your current code
execution session. You are a deployed coding agent. The repo is already cloned
in your working directory, and you must fully solve the problem for your answer to be
considered correct. You also have access to special tools that allow quick navigation
in the codebase. Use these tools to search for relevant code needed to solve users question.
Design and architect the best solution and provide code diffs with explanation of the proposed
changes to the user. Avoid reading excessive number of files - get *minimum* required set
of file summaries to answer the question. Use symbol search to find relevant code snippets.
To do overall discovery, use the repo map and pass original user prompt as an input.
"""


class Settings(ProjectSettings):
    """Chat-CLI specific settings, extending project settings."""
    model_config = SettingsConfigDict(
        cli_parse_args=True,
        cli_kebab_case=True,
        env_prefix="KNOW_",
        env_nested_delimiter="_",
    )

    model: str = Field(
        default=os.getenv("OPENAI_MODEL", "gpt-4.1"),
        description="Name of the LLM to use for chat.",
        cli_alias="m",
    )
    system: str = Field(
        default=SYSTEM_PROMPT,
        description="System prompt for the chat.",
        cli_alias="s",
    )
    project_path: str = Field(
        description="Root directory of the project to analyse/assist with.",
        cli_aliases=["p", "path"],
    )


async def _chat(model: str, system_msg: str, project):
    session = PromptSession()
    messages: List[Dict] = [{"role": "system", "content": system_msg}]

    tools = [t.get_openai_schema() for t in ToolRegistry._tools.values()]
    session_cost_usd: float = 0.0      # running total for the whole chat session

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
            session_cost_usd += cost_usd
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
                print(f"[usage] prompt={prompt_toks}  completion={completion_toks} "
                      f"total={total_toks}  -  est. cost ${cost_usd:.6f}  "
                      f"session ${session_cost_usd:.6f}")
                continue
            # otherwise, regular assistant answer
            print(f"\nAssistant: {msg.content}")
            print(f"[usage] prompt={prompt_toks}  completion={completion_toks} "
                  f"total={total_toks}  -  est. cost ${cost_usd:.6f}  "
                  f"session ${session_cost_usd:.6f}")
            messages.append({"role": "assistant", "content": msg.content})
            break


def main() -> None:
    # Custom help handler using iter_settings
    if "--help" in sys.argv or "-h" in sys.argv:
        print_help(Settings, "chatcli.py")
        sys.exit(0)

    try:
        settings = Settings()
    except Exception as e:
        print(f"Error: Invalid settings.\n{e}", file=sys.stderr)
        sys.exit(1)


    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY env var not set.")

    project = init_project(settings)
    with patch_stdout():
        asyncio.run(_chat(settings.model, settings.system, project))


if __name__ == "__main__":
    main()
