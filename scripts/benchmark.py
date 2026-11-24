#!/usr/bin/env python3
"""Compare small local models on THIS computer.

For each model it measures:
  * load time      how long the model takes to wake up
  * speed          how fast it writes (tokens per second; ~1 token = 3/4 of a word)
  * tool calling   whether it picks the right tool (or no tool) for typical requests
  * memory         how much RAM the loaded model uses

Run from the project folder:
    python3 scripts/benchmark.py                      # the default candidates
    python3 scripts/benchmark.py llama3.2:3b qwen3.5:4b  # your own choice

Models must be downloaded first (ollama pull <name>). Nothing is run on your
files: tool requests are only checked, never carried out.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from achoose.agent import build_system_prompt  # noqa: E402
from achoose.brain import BrainError, Message, OllamaBrain  # noqa: E402
from achoose.config import load_config  # noqa: E402
from achoose.tools import ToolRegistry, builtin_tools  # noqa: E402
from achoose.tools.memory_tool import memory_tools  # noqa: E402

DEFAULT_MODELS = ["qwen3:4b-instruct-2507-q4_K_M", "llama3.2:3b"]

SPEED_PROMPT = "In about 120 words, explain why making regular backups of your files is important."


@dataclass
class Case:
    prompt: str
    expect_tool: str | None  # None means: should answer without any tool
    check_args: Callable[[dict[str, Any]], bool] = lambda a: True
    # Some questions may also be answered correctly without a tool
    ok_without_tool: Callable[[str], bool] | None = None


def _text(args: dict[str, Any], key: str) -> str:
    return str(args.get(key, "")).lower()


CASES = [
    Case("What time is it right now?", "get_current_time"),
    # Achoose tells the model today's date, so a correct direct answer is fine too.
    Case(
        "Which day of the week is it today?",
        "get_current_time",
        ok_without_tool=lambda text: datetime.now().strftime("%A").lower() in text.lower(),
    ),
    Case(
        "What files are in my Downloads folder?",
        "list_folder",
        lambda a: "download" in _text(a, "path"),
    ),
    Case(
        "Open ~/notes/todo.txt and tell me what it says.",
        "read_text_file",
        lambda a: _text(a, "path").endswith("todo.txt"),
    ),
    Case(
        "Please write down a note called Shopping: milk, eggs and bread.",
        "save_note",
        # Putting everything in the title is a fair reading too.
        lambda a: "shop" in _text(a, "title") and "milk" in _text(a, "title") + _text(a, "text"),
    ),
    Case(
        "What did my Java midterm cover?",
        "search_my_files",
        lambda a: "java" in _text(a, "query"),
    ),
    Case("Hi! How are you today?", None),
    Case("What is the capital of France?", None),
    Case("Write a short, kind birthday wish for my sister.", None),
]


@dataclass
class Result:
    model: str
    ok: bool = True
    error: str = ""
    load_seconds: float = 0.0
    output_tokens_per_second: float = 0.0
    prompt_tokens_per_second: float = 0.0
    speed_test_seconds: float = 0.0
    tool_score: int = 0
    tool_total: int = 0
    tool_avg_seconds: float = 0.0
    ram_gb: float = 0.0
    # Where the time goes, averaged over the tool tests
    avg_read_tokens: float = 0.0
    avg_read_seconds: float = 0.0
    avg_write_tokens: float = 0.0
    avg_write_seconds: float = 0.0
    thinking_count: int = 0  # answers where the model "thought" first (slow)
    text_rescues: int = 0  # tool requests rescued from plain text
    failures: list[str] = field(default_factory=list)
    cases: list[dict[str, Any]] = field(default_factory=list)


def unload(brain: OllamaBrain) -> None:
    try:
        brain._post("/api/generate", {"model": brain.model, "keep_alive": 0})
    except BrainError:
        pass


def loaded_ram_gb(brain: OllamaBrain) -> float:
    try:
        want = _full_name(brain.model)
        for m in brain._get("/api/ps").get("models", []):
            if want in (_full_name(m.get("name", "")), _full_name(m.get("model", ""))):
                return round(m.get("size", 0) / 1024**3, 1)
    except BrainError:
        pass
    return 0.0


def _full_name(name: str) -> str:
    return name if ":" in name else f"{name}:latest"


def bench_model(model: str, host: str, schemas: list[dict[str, Any]]) -> Result:
    res = Result(model=model, tool_total=len(CASES))
    brain = OllamaBrain(model, host=host, context_tokens=4096, temperature=0.0, think=False)
    system = Message("system", build_system_prompt("~/Documents, ~/Downloads"))

    try:
        # 1. Wake the model up (measures load time). This also lets Ollama
        #    remember the instructions + tool list, as it does in a real chat.
        print("  loading...", flush=True)
        warm = brain.chat([system, Message("user", "Say OK.")], schemas)
        res.load_seconds = round(warm.stats.load_seconds, 1)
        res.ram_gb = loaded_ram_gb(brain)

        # 2. Writing speed (average of 2 runs)
        print("  speed test...", flush=True)
        out_tok = out_sec = in_tok = in_sec = wall = 0.0
        for _ in range(2):
            t0 = time.perf_counter()
            r = brain.chat([Message("user", SPEED_PROMPT)], [])
            wall += time.perf_counter() - t0
            out_tok += r.stats.output_tokens
            out_sec += r.stats.output_seconds
            in_tok += r.stats.prompt_tokens
            in_sec += r.stats.prompt_seconds
        res.output_tokens_per_second = round(out_tok / out_sec, 1) if out_sec else 0.0
        res.prompt_tokens_per_second = round(in_tok / in_sec, 1) if in_sec else 0.0
        res.speed_test_seconds = round(wall / 2, 1)

        # 3. Tool calling
        total_time = 0.0
        for i, case in enumerate(CASES, 1):
            print(f"  tool test {i}/{len(CASES)}: {case.prompt}", flush=True)
            t0 = time.perf_counter()
            r = brain.chat([system, Message("user", case.prompt)], schemas)
            seconds = time.perf_counter() - t0
            total_time += seconds
            st = r.stats
            res.cases.append(
                {
                    "prompt": case.prompt,
                    "seconds": round(seconds, 1),
                    "read_tokens": st.prompt_tokens,
                    "read_seconds": round(st.prompt_seconds, 1),
                    "write_tokens": st.output_tokens,
                    "write_seconds": round(st.output_seconds, 1),
                    "thinking_chars": len(r.thinking),
                    "tool_from_text": r.tool_calls_from_text,
                }
            )
            res.thinking_count += bool(r.thinking)
            res.text_rescues += r.tool_calls_from_text
            called = r.tool_calls[0] if r.tool_calls else None
            if case.expect_tool is None:
                passed = called is None
                got = f"called {called.name}" if called else "answered"
            else:
                passed = (
                    called is not None
                    and called.name == case.expect_tool
                    and case.check_args(called.arguments)
                ) or (
                    called is None
                    and case.ok_without_tool is not None
                    and case.ok_without_tool(r.content)
                )
                got = f"{called.name}({called.arguments})" if called else f"no tool: {r.content[:60]!r}"
            if passed:
                res.tool_score += 1
            else:
                res.failures.append(f"{case.prompt} -> wanted {case.expect_tool or 'no tool'}, got {got}")
        res.tool_avg_seconds = round(total_time / len(CASES), 1)
        n = len(res.cases)
        res.avg_read_tokens = round(sum(c["read_tokens"] for c in res.cases) / n)
        res.avg_read_seconds = round(sum(c["read_seconds"] for c in res.cases) / n, 1)
        res.avg_write_tokens = round(sum(c["write_tokens"] for c in res.cases) / n)
        res.avg_write_seconds = round(sum(c["write_seconds"] for c in res.cases) / n, 1)
    except BrainError as err:
        res.ok = False
        res.error = str(err)
    finally:
        unload(brain)  # free the RAM before the next model
    return res


def print_table(results: list[Result]) -> None:
    head = f"{'model':<30}{'load s':>8}{'tok/s':>8}{'read tok/s':>12}{'tools':>8}{'s/request':>11}{'RAM GB':>8}"
    print("\n" + head)
    print("-" * len(head))
    for r in results:
        if not r.ok:
            print(f"{r.model:<30}  FAILED: {r.error}")
            continue
        print(
            f"{r.model:<30}{r.load_seconds:>8}{r.output_tokens_per_second:>8}"
            f"{r.prompt_tokens_per_second:>12}{f'{r.tool_score}/{r.tool_total}':>8}"
            f"{r.tool_avg_seconds:>11}{r.ram_gb:>8}"
        )
    print("\nWhere the time goes (average per tool test):")
    for r in results:
        if not r.ok:
            continue
        line = (
            f"  {r.model:<30} reading {r.avg_read_tokens} tokens took {r.avg_read_seconds}s, "
            f"writing {r.avg_write_tokens} tokens took {r.avg_write_seconds}s"
        )
        if r.thinking_count:
            line += f"  <- 'thinking' in {r.thinking_count}/{r.tool_total} answers (slow)"
        if r.text_rescues:
            line += f"  ({r.text_rescues} tool requests rescued from plain text)"
        print(line)
    for r in results:
        if r.ok and r.failures:
            print(f"\n{r.model} missed:")
            for f in r.failures:
                print(f"  - {f}")


def recommend(results: list[Result]) -> str:
    good = [r for r in results if r.ok]
    if not good:
        return "No model finished the test."
    # Tool accuracy matters most, then speed.
    best = max(good, key=lambda r: (r.tool_score, r.output_tokens_per_second))
    note = ""
    if best.output_tokens_per_second < 5:
        note = " (it is slow on this computer; answers will take a while)"
    return f"Best overall here: {best.model} ({best.tool_score}/{best.tool_total} tool tests, {best.output_tokens_per_second} tok/s){note}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark local models for Achoose.")
    parser.add_argument("models", nargs="*", default=DEFAULT_MODELS)
    args = parser.parse_args()

    config = load_config()
    probe = OllamaBrain(args.models[0], host=config.ollama_host)
    if not probe.is_running():
        print(f"Ollama is not running at {config.ollama_host}. Start it with: sudo systemctl start ollama")
        return 1
    installed = set(probe.installed_models())
    installed |= {n.removesuffix(":latest") for n in installed}

    missing = [m for m in args.models if m not in installed]
    if missing:
        print("These models are not downloaded yet, so they are skipped:")
        for m in missing:
            print(f"  ollama pull {m}")
    models = [m for m in args.models if m in installed]
    if not models:
        return 1

    # Tools are only described to the model here, never run.
    schemas = ToolRegistry(builtin_tools(config) + memory_tools(None)).schemas()
    results = []
    for m in models:
        print(f"\n=== {m} ===")
        results.append(bench_model(m, config.ollama_host, schemas))

    print_table(results)
    print("\n" + recommend(results))

    out_dir = ROOT / "benchmark-results"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"benchmark-{datetime.now():%Y%m%d-%H%M}.json"
    out_file.write_text(json.dumps([asdict(r) for r in results], indent=2))
    print(f"\nFull results saved to {out_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
