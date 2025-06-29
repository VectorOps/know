import os
import asyncio
import argparse
from typing import List, Dict

import litellm
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.patch_stdout import patch_stdout


def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simple interactive chat CLI.")
    p.add_argument("-m", "--model", default=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"))
    p.add_argument("-s", "--system", default="You are a helpful assistant.")
    return p.parse_args()


async def _stream_completion(messages: List[Dict], model: str):
    resp = litellm.completion(model=model, messages=messages, stream=True)
    for chunk in resp:
        delta = chunk.choices[0].delta.get("content", "")
        if delta:
            yield delta


async def _gather_tokens(messages: List[Dict], model: str) -> List[str]:
    tokens: List[str] = []
    async for t in _stream_completion(messages, model):
        tokens.append(t)
    return tokens


async def _chat(model: str, system_msg: str):
    session = PromptSession(history=FileHistory("~/.chatcli-history"))
    messages: List[Dict] = [{"role": "system", "content": system_msg}]

    print(f"Loaded model '{model}'.   Type '/exit' or Ctrl-D to quit.")
    while True:
        try:
            user_input = await session.prompt_async("> ")
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.strip().lower() in {"/exit", "/quit"}:
            break
        if not user_input.strip():
            continue

        messages.append({"role": "user", "content": user_input})

        # stream assistant answer
        print("\nAssistant: ", end="", flush=True)
        async for token in _stream_completion(messages, model):
            print(token, end="", flush=True)
        print()

        # append full assistant message to history
        assistant_text = "".join(await _gather_tokens(messages, model))
        messages.append({"role": "assistant", "content": assistant_text})


def main() -> None:
    args = _parse_cli()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY env var not set.")
    with patch_stdout():
        asyncio.run(_chat(args.model, args.system))


if __name__ == "__main__":
    main()
