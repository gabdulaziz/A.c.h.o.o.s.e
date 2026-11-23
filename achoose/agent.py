"""The thinking loop: question -> brain -> tools -> brain -> answer.

The confirmation gate lives here. Every tool marked Risk.CHANGE goes through
`confirm` before it runs, and nothing else can skip that check.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable

from .brain import Brain, BrainStats, Message, ToolCall
from .tools import Risk, Tool, ToolError, ToolRegistry

# Asked before any CHANGE tool runs. Return True only if the user said yes.
ConfirmFn = Callable[[Tool, dict[str, Any]], bool]
# Told about every tool step, so the screen can show what is happening.
ToolEventFn = Callable[[str, dict[str, Any], str], None]

SYSTEM_PROMPT = """You are Achoose, a private personal assistant running on the user's own computer.
Be helpful, warm and brief. Answer in the user's language.
Use a tool whenever it helps you give a correct answer about the user's computer, files or the time.
Never invent file contents or tool results. If a tool fails or the user declines an action, say so plainly.
Today is {today}."""


def deny_all(tool: Tool, arguments: dict[str, Any]) -> bool:
    """Safe default when there is no one to ask."""
    return False


class Agent:
    def __init__(
        self,
        brain: Brain,
        tools: ToolRegistry,
        confirm: ConfirmFn = deny_all,
        on_tool: ToolEventFn | None = None,
        history_messages: int = 20,
        max_tool_rounds: int = 5,
    ) -> None:
        self.brain = brain
        self.tools = tools
        self.confirm = confirm
        self.on_tool = on_tool
        self.history_messages = history_messages
        self.max_tool_rounds = max_tool_rounds
        self.history: list[Message] = []
        self.last_stats: list[BrainStats] = []

    def reset(self) -> None:
        self.history.clear()

    def ask(self, text: str) -> str:
        self.history.append(Message("user", text))
        self._trim_history()
        self.last_stats = []

        for _ in range(self.max_tool_rounds + 1):
            reply = self.brain.chat(self._messages(), self.tools.schemas())
            self.last_stats.append(reply.stats)
            self.history.append(Message("assistant", reply.content, tool_calls=reply.tool_calls))
            if not reply.tool_calls:
                return reply.content
            for call in reply.tool_calls:
                result = self._run_tool(call)
                self.history.append(Message("tool", result, tool_name=call.name))

        answer = "I stopped because that took too many steps. Could you ask in a simpler way?"
        self.history.append(Message("assistant", answer))
        return answer

    # ---- internals ----------------------------------------------------

    def _run_tool(self, call: ToolCall) -> str:
        tool = self.tools.get(call.name)
        if tool is None:
            result = f"Error: there is no tool called '{call.name}'."
        elif tool.risk is not Risk.READ and not self.confirm(tool, call.arguments):
            result = "The user declined this action. It was NOT done."
        else:
            try:
                result = tool.run(**call.arguments)
            except ToolError as err:
                result = f"Error: {err}"
            except TypeError as err:  # wrong or missing arguments from the model
                result = f"Error: wrong arguments for {call.name}: {err}"
            except OSError as err:
                result = f"Error: {err.strerror or err}"
        if self.on_tool:
            self.on_tool(call.name, call.arguments, result)
        return result

    def _messages(self) -> list[Message]:
        system = SYSTEM_PROMPT.format(today=datetime.now().strftime("%A %d %B %Y"))
        return [Message("system", system), *self.history]

    def _trim_history(self) -> None:
        """Keep the last N messages, always starting at a user message."""
        if len(self.history) <= self.history_messages:
            return
        cut = len(self.history) - self.history_messages
        while cut < len(self.history) and self.history[cut].role != "user":
            cut += 1
        self.history = self.history[cut:]
