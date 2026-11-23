"""What a tool is, and the list of tools the brain may use."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class Risk(str, Enum):
    """How careful Achoose must be before running a tool.

    READ:   only looks at things. Runs on its own.
    CHANGE: sends, saves, deletes, installs or runs something.
            Always asks you first.
    """

    READ = "read"
    CHANGE = "change"


class ToolError(Exception):
    """A tool failed in an expected way. The message is shown to the brain."""


@dataclass
class Tool:
    name: str
    description: str
    run: Callable[..., str]
    risk: Risk = Risk.READ
    # JSON Schema "properties" for the arguments, and which are required
    parameters: dict[str, Any] = field(default_factory=dict)
    required: list[str] = field(default_factory=list)
    # Optional: turns arguments into a plain sentence for the confirmation question
    explain: Callable[..., str] | None = None

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": self.required,
                },
            },
        }

    def describe(self, arguments: dict[str, Any]) -> str:
        if self.explain:
            try:
                return self.explain(**arguments)
            except TypeError:
                pass
        args = ", ".join(f"{k}={v!r}" for k, v in arguments.items())
        return f"{self.name}({args})"


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.add(tool)

    def add(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self._tools.values()]

    def __iter__(self):
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)
