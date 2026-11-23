"""Text chat with Achoose in the terminal.  Run:  python -m achoose"""

from __future__ import annotations

import sys
from typing import Any

from .agent import Agent
from .brain import BrainError, OllamaBrain, make_brain
from .config import CONFIG_PATH, load_config
from .tools import Risk, Tool, ToolRegistry, builtin_tools

HELP = """Commands:
  /help    show this help
  /tools   list what Achoose can do
  /stats   speed of the last answer
  /reset   forget this conversation
  /quit    leave (or press Ctrl+D)"""


def ask_user_to_confirm(tool: Tool, arguments: dict[str, Any]) -> bool:
    print("\n  ⚠  Achoose wants to do this:")
    for line in tool.describe(arguments).splitlines():
        print(f"     {line}")
    try:
        answer = input("  Allow? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    allowed = answer in ("y", "yes")
    print("  ✓ allowed" if allowed else "  ✗ declined")
    return allowed


def show_tool_step(name: str, arguments: dict[str, Any], result: str) -> None:
    args = ", ".join(f"{k}={v!r}" for k, v in arguments.items())
    first_line = result.splitlines()[0] if result else ""
    if len(first_line) > 80:
        first_line = first_line[:77] + "..."
    print(f"  · {name}({args}) → {first_line}")


def main() -> int:
    config = load_config()
    brain = make_brain(config, "local")

    if isinstance(brain, OllamaBrain):
        if not brain.is_running():
            print(f"Ollama is not running at {config.ollama_host}.")
            print("Start it with:  sudo systemctl start ollama   (or: ollama serve)")
            return 1
        installed = brain.installed_models()
        if config.model not in installed and f"{config.model}:latest" not in installed:
            print(f"The model '{config.model}' is not downloaded yet.")
            print(f"Download it with:  ollama pull {config.model}")
            if installed:
                print(f"Models you have: {', '.join(installed)}")
            return 1

    tools = ToolRegistry(builtin_tools(config))
    agent = Agent(
        brain,
        tools,
        confirm=ask_user_to_confirm,
        on_tool=show_tool_step,
        history_messages=config.history_messages,
        max_tool_rounds=config.max_tool_rounds,
    )

    print(f"Achoose is ready (brain: {brain.name}). Type /help for commands.")
    if not CONFIG_PATH.exists():
        print(f"(Using default settings. Your own settings file would go in {CONFIG_PATH})")

    while True:
        try:
            text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            return 0
        if not text:
            continue
        if text.startswith("/"):
            cmd = text.lower()
            if cmd in ("/quit", "/exit", "/q"):
                print("Bye!")
                return 0
            if cmd == "/help":
                print(HELP)
            elif cmd == "/reset":
                agent.reset()
                print("Conversation forgotten.")
            elif cmd == "/tools":
                for t in tools:
                    tag = "asks first" if t.risk is not Risk.READ else "automatic"
                    print(f"  {t.name:18} ({tag})  {t.description}")
            elif cmd == "/stats":
                print_stats(agent)
            else:
                print("Unknown command. Type /help.")
            continue

        print("  (thinking...)", flush=True)
        try:
            answer = agent.ask(text)
        except BrainError as err:
            print(f"\n[problem] {err}")
            continue
        except KeyboardInterrupt:
            print("\n(stopped)")
            continue
        print(f"Achoose: {answer or '(no answer)'}")


def print_stats(agent: Agent) -> None:
    if not agent.last_stats:
        print("No answer yet.")
        return
    total = sum(s.total_seconds for s in agent.last_stats)
    out_tokens = sum(s.output_tokens for s in agent.last_stats)
    out_secs = sum(s.output_seconds for s in agent.last_stats)
    speed = out_tokens / out_secs if out_secs else 0
    print(
        f"Last answer: {total:.1f}s total, {len(agent.last_stats)} brain call(s), "
        f"{speed:.1f} tokens per second"
    )


if __name__ == "__main__":
    sys.exit(main())
