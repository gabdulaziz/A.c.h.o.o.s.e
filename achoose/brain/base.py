"""The brain interface.

Achoose talks to every AI model through this one small interface, so the
model behind it can be swapped (local Ollama today, a cloud model on demand
later) without touching the rest of the code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ToolCall:
    """The brain asking Achoose to run one tool."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class Message:
    """One message in a conversation.

    role is "system", "user", "assistant" or "tool".
    A "tool" message carries a tool's result back to the brain, and
    tool_name says which tool it came from.
    """

    role: str
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_name: str | None = None


@dataclass
class BrainStats:
    """Timing numbers from one brain call, used for benchmarks and /stats."""

    prompt_tokens: int = 0
    output_tokens: int = 0
    prompt_seconds: float = 0.0
    output_seconds: float = 0.0
    load_seconds: float = 0.0
    total_seconds: float = 0.0

    @property
    def output_tokens_per_second(self) -> float:
        if self.output_seconds <= 0:
            return 0.0
        return self.output_tokens / self.output_seconds

    @property
    def prompt_tokens_per_second(self) -> float:
        if self.prompt_seconds <= 0:
            return 0.0
        return self.prompt_tokens / self.prompt_seconds


@dataclass
class BrainReply:
    """What the brain answered: text, tool requests, or both."""

    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    stats: BrainStats = field(default_factory=BrainStats)


class BrainError(RuntimeError):
    """The brain could not answer (not running, model missing, and so on)."""


class Brain(Protocol):
    """Anything that can chat and ask for tools.

    tools uses the common JSON "function" format that Ollama, OpenAI-style
    and Anthropic-style APIs all understand (with small conversions).
    """

    name: str

    def chat(self, messages: list[Message], tools: list[dict[str, Any]]) -> BrainReply:
        ...
