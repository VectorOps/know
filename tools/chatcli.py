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
from know.settings import ProjectSettings
from know.project import init_project
from know.tools.base import ToolRegistry


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
    return p.parse_args()


def _serialize_tool_result(res):
    # pydantic BaseModel or list[BaseModel] → python primitives
    if isinstance(res, list):
        return [r.model_dump() if hasattr(r, "model_dump") else r.dict()
                for r in res]
    if hasattr(res, "model_dump"):
        return res.model_dump()
    if hasattr(res, "dict"):
        return res.dict()
    return res


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
            msg = response.choices[0].message
            # If the assistant wants to call a tool …
            if getattr(msg, "tool_calls", None):
                messages.append(msg)  # add assistant tool request
                for call in msg.tool_calls:
                    tool_name = call.function.name.split("/", 1)[-1]
                    tool      = ToolRegistry.get(tool_name)
                    args      = json.loads(call.function.arguments or "{}")

                    # TODO: Logger
                    print("Tool call request", call.function.name, args)

                    result = tool.execute(project, **args)

                    print("Tool call response", call.function.name, result)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": call.function.name,
                        "content": json.dumps(_serialize_tool_result(result)),
                    })
                # run another completion to let the model consume the tool output
                continue
            # otherwise, regular assistant answer
            print(f"\nAssistant: {msg.content}")
            messages.append({"role": "assistant", "content": msg.content})
            break


def main() -> None:
    args = _parse_cli()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY env var not set.")
    settings = ProjectSettings(project_path=args.path)
    project  = init_project(settings)
    with patch_stdout():
        asyncio.run(_chat(args.model, args.system, project))


if __name__ == "__main__":
    main()
