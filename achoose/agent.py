"""The thinking loop: question -> brain -> tools -> brain -> answer.

The confirmation gate lives here. Every tool marked Risk.CHANGE goes through
`confirm` before it runs, and nothing else can skip that check.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Callable

from .brain import Brain, BrainStats, Message, ToolCall
from .tools import Risk, Tool, ToolError, ToolRegistry

# Asked before any CHANGE tool runs. Return True only if the user said yes.
ConfirmFn = Callable[[Tool, dict[str, Any]], bool]
# Told about every tool step, so the screen can show what is happening.
ToolEventFn = Callable[[str, dict[str, Any], str], None]

SYSTEM_PROMPT = """You are Achoose, a private assistant on the user's own computer. Be warm and brief. Reply in the user's language.
Use a tool when the request needs facts from this computer or the user's own files (the time, a folder, a file, their documents), or the user clearly asks to save a note.
For greetings, general knowledge, advice or writing text, answer directly with no tool.
Never announce a tool ("Let me check..."); just call it.
Keep answers to a few sentences unless the user asks for more. Never repeat lists or file contents the user can already see.
Never invent file contents or results. If a tool fails or the user declines, say so.{memory}
Today is {today}."""

MEMORY_PROMPT = """
You CAN search the user's own documents and notes ({folders}) with search_my_files.
For any question about the user's own files, notes, school work, projects or past work, call search_my_files first.
To count or list ALL files of some kind ("how many labs", "all my PDFs about X"), use find_my_files instead.
Answer from what it finds in 2 to 3 plain sentences (no headings or lists) and name the file. Give details only if the user asks.
Never say you cannot access the user's files."""

# Small models sometimes forget they can search. This spots that kind of answer.
NO_ACCESS = re.compile(
    r"\b(don'?t|do not|can'?t|cannot|can not|unable to|no way to)\s+(have\s+)?(direct\s+)?"
    r"(access|see|view|read|browse|look)",
    re.IGNORECASE,
)


def build_system_prompt(memory_folders: str = "") -> str:
    memory = MEMORY_PROMPT.format(folders=memory_folders) if memory_folders else ""
    return SYSTEM_PROMPT.format(memory=memory, today=datetime.now().strftime("%A %d %B %Y"))


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
        memory_folders: str = "",
    ) -> None:
        """memory_folders: e.g. "~/Documents, ~/Downloads" when search_my_files is available."""
        self.brain = brain
        self.tools = tools
        self.confirm = confirm
        self.on_tool = on_tool
        self.history_messages = history_messages
        self.max_tool_rounds = max_tool_rounds
        self.memory_folders = memory_folders
        self.history: list[Message] = []
        self.last_stats: list[BrainStats] = []

    def reset(self) -> None:
        self.history.clear()

    def ask(self, text: str, on_text: Callable[[str], None] | None = None) -> str:
        """Answer one message. on_text, if given, receives the answer as it is written."""
        self.history.append(Message("user", text))
        self._trim_history()
        self.last_stats = []

        safety_net = bool(self.memory_folders) and self.tools.get("search_my_files") is not None
        for round_no in range(self.max_tool_rounds + 1):
            # While the safety net may still trigger, hold back the start of the answer.
            hold = _HoldBack(on_text) if (on_text and safety_net and round_no == 0) else None
            sink = hold.feed if hold else on_text
            extra = {"on_text": sink} if sink else {}
            reply = self.brain.chat(self._messages(), self.tools.schemas(), **extra)
            self.last_stats.append(reply.stats)

            if round_no == 0 and safety_net and not reply.tool_calls and NO_ACCESS.search(reply.content):
                # The brain claimed it can't see the user's files. It can: search for it.
                call = ToolCall("search_my_files", {"query": text})
                self.history.append(Message("assistant", "", tool_calls=[call]))
                result, _ = self._run_tool(call)
                self.history.append(Message("tool", result, tool_name=call.name))
                continue
            if hold:
                hold.flush()

            self.history.append(Message("assistant", reply.content, tool_calls=reply.tool_calls))
            if not reply.tool_calls:
                return reply.content
            shown: list[str] = []
            for call in reply.tool_calls:
                result, ok = self._run_tool(call)
                self.history.append(Message("tool", result, tool_name=call.name))
                tool = self.tools.get(call.name)
                shown.append(result if ok and tool and tool.show_directly else "")
            if all(shown):
                # Every result can be shown as it is: skip the slow re-writing step.
                # The brain still has the results in the history for follow-ups.
                self.history.append(Message("assistant", "(The result above was shown to the user.)"))
                return "\n\n".join(shown)

        answer = "I stopped because that took too many steps. Could you ask in a simpler way?"
        self.history.append(Message("assistant", answer))
        return answer

    # ---- internals ----------------------------------------------------

    def _run_tool(self, call: ToolCall) -> tuple[str, bool]:
        """Run one tool through the safety gate. Returns (result, worked)."""
        ok = False
        tool = self.tools.get(call.name)
        if tool is None:
            result = f"Error: there is no tool called '{call.name}'."
        elif tool.risk is not Risk.READ and not self.confirm(tool, call.arguments):
            result = "The user declined this action. It was NOT done."
        else:
            try:
                result = tool.run(**call.arguments)
                ok = True
            except ToolError as err:
                result = f"Error: {err}"
            except TypeError as err:  # wrong or missing arguments from the model
                result = f"Error: wrong arguments for {call.name}: {err}"
            except OSError as err:
                result = f"Error: {err.strerror or err}"
        if self.on_tool:
            self.on_tool(call.name, call.arguments, result)
        return result, ok

    def _messages(self) -> list[Message]:
        return [Message("system", build_system_prompt(self.memory_folders)), *self.history]

    def _trim_history(self) -> None:
        """Keep the last N messages, always starting at a user message."""
        if len(self.history) <= self.history_messages:
            return
        cut = len(self.history) - self.history_messages
        while cut < len(self.history) and self.history[cut].role != "user":
            cut += 1
        self.history = self.history[cut:]


class _HoldBack:
    """Holds the first part of a streamed answer until we know it is not an
    "I can't access your files" reply that the safety net will replace."""

    LIMIT = 160

    def __init__(self, on_text: Callable[[str], None]) -> None:
        self.on_text = on_text
        self.buffer = ""
        self.released = False

    def feed(self, piece: str) -> None:
        if self.released:
            self.on_text(piece)
            return
        self.buffer += piece
        if NO_ACCESS.search(self.buffer):
            return  # keep holding; the answer will probably be replaced
        if len(self.buffer) >= self.LIMIT:
            self.flush()

    def flush(self) -> None:
        if not self.released:
            self.released = True
            if self.buffer:
                self.on_text(self.buffer)
