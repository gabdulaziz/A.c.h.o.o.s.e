"""Text chat with Achoose in the terminal.  Run:  python -m achoose"""

from __future__ import annotations

import sys
import time
from typing import Any

from .agent import Agent
from .brain import BrainError, OllamaBrain, make_brain
from .config import CONFIG_PATH, load_config
from .memory import open_memory
from .tools import Risk, Tool, ToolRegistry, builtin_tools
from .tools.memory_tool import memory_tools

HELP = """Commands:
  /help    show this help
  /tools   list what Achoose can do
  /stats   speed of the last answer
  /reset   forget this conversation
  /index   read new or changed files in your memory folders
  /quit    leave (or press Ctrl+D)"""


def ask_user_to_confirm(tool: Tool, arguments: dict[str, Any]) -> bool:
    stream.end_line()
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


class Stream:
    """Prints the answer as it is written, and keeps tool lines tidy."""

    def __init__(self) -> None:
        self.streaming = False  # are we in the middle of printing an answer?

    def text(self, piece: str) -> None:
        if not self.streaming:
            print("Achoose: ", end="")
            self.streaming = True
        print(piece, end="", flush=True)

    def end_line(self) -> bool:
        """Finish the current answer line. Returns True if anything was printed."""
        was = self.streaming
        if was:
            print()
        self.streaming = False
        return was


stream = Stream()


def show_tool_step(name: str, arguments: dict[str, Any], result: str) -> None:
    stream.end_line()
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
    memory, why_not = open_memory(config)
    if memory:
        for tool in memory_tools(memory):
            tools.add(tool)
    agent = Agent(
        brain,
        tools,
        confirm=ask_user_to_confirm,
        on_tool=show_tool_step,
        history_messages=config.history_messages,
        max_tool_rounds=config.max_tool_rounds,
    )

    print(f"Achoose is ready (brain: {brain.name}). Type /help for commands.")
    print(memory_status(memory, why_not))
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
            elif cmd == "/index":
                if memory:
                    run_index(memory)
                else:
                    print(why_not)
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
        started = time.perf_counter()
        try:
            answer = agent.ask(text, on_text=stream.text)
        except BrainError as err:
            stream.end_line()
            print(f"[problem] {err}")
            continue
        except KeyboardInterrupt:
            stream.end_line()
            print("(stopped)")
            continue
        if not stream.end_line():  # nothing was shown live, so show it now
            print(f"Achoose: {answer or '(no answer)'}")
        print(f"  ({time.perf_counter() - started:.1f} s)")


def memory_status(memory, why_not: str) -> str:
    if not memory:
        return f"(Note: {why_not})"
    folders = ", ".join(_short(p) for p in memory.folders)
    files, pieces = memory.store.counts()
    if files == 0:
        return f"Memory: nothing indexed yet. Type /index to read {folders}."
    return f"Memory: {files} files ({pieces} pieces) from {folders}. Type /index to refresh."


def run_index(memory) -> None:
    folders = ", ".join(_short(p) for p in memory.folders)
    missing = [p for p in memory.folders if not p.is_dir()]
    for p in missing:
        print(f"  (folder {_short(p)} does not exist, skipped)")
    print(f"Indexing {folders}. The first time can take a while; later runs only read changes.")
    print("Press Ctrl+C to stop. Finished files are kept, and the next /index continues.")
    started = time.perf_counter()

    def progress(done: int, total: int, path: str) -> None:
        if total == 0:
            return
        elapsed = time.perf_counter() - started
        eta = ""
        if done >= 3:
            left = elapsed / done * (total - done)
            eta = f", about {left / 60:.0f} min left" if left >= 90 else f", about {left:.0f} s left"
        name = path.rsplit("/", 1)[-1][:40]
        print(f"\r  {done}/{total} files{eta}  {name:<40}", end="", flush=True)

    try:
        report = memory.update(progress)
    except KeyboardInterrupt:
        print("\n  Stopped. Type /index to continue later.")
        return
    except BrainError as err:
        print(f"\n[problem] {err}")
        return
    print()
    print(
        f"Done in {report.seconds:.0f} s: {report.added} new, {report.updated} changed, "
        f"{report.removed} removed, {report.unchanged} unchanged."
    )
    if report.missing_packages:
        print(f"  Skipped some documents: missing Python package(s) {', '.join(sorted(report.missing_packages))}.")
    if report.failed:
        print(f"  {len(report.failed)} file(s) could not be read, for example: {report.failed[0]}")
    print(memory_status(memory, ""))


def _short(path) -> str:
    from pathlib import Path

    try:
        return "~/" + str(Path(path).relative_to(Path.home()))
    except ValueError:
        return str(path)


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
