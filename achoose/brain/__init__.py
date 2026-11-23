"""Brains Achoose can think with. Pick one with make_brain()."""

from __future__ import annotations

from ..config import Config
from .base import Brain, BrainError, BrainReply, BrainStats, Message, ToolCall
from .ollama import OllamaBrain

__all__ = [
    "Brain",
    "BrainError",
    "BrainReply",
    "BrainStats",
    "Message",
    "OllamaBrain",
    "ToolCall",
    "make_brain",
]


def make_brain(config: Config, kind: str = "local") -> Brain:
    """Build the brain named by kind.

    "local" is the default and the only one that exists today.
    "cloud" is reserved for a cloud model used only when you ask for it
    (for hard jobs such as big coding tasks). It is added in a later step.
    """
    if kind == "local":
        return OllamaBrain(
            model=config.model,
            host=config.ollama_host,
            context_tokens=config.context_tokens,
            temperature=config.temperature,
            max_answer_tokens=config.max_answer_tokens,
            think=config.think,
        )
    if kind == "cloud":
        raise BrainError("The cloud brain is not set up yet. It comes in a later step.")
    raise BrainError(f"Unknown brain kind: {kind}")
