"""The thinking loop: question -> brain -> tools -> brain -> answer.

The confirmation gate lives here. Every tool marked Risk.CHANGE goes through
`confirm` before it runs, and nothing else can skip that check. READ and ACT
(harmless, like opening an app) tools run on their own.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Callable

from .brain import Brain, BrainStats, Message, ToolCall
from .tools import Risk, Tool, ToolError, ToolRegistry, Unsure

# Asked before any CHANGE tool runs. Return True only if the user said yes.
ConfirmFn = Callable[[Tool, dict[str, Any]], bool]
# Asks the user a plain yes/no question ("Did you mean Terminal?").
AskFn = Callable[[str], bool]
# Told about every tool step, so the screen can show what is happening.
ToolEventFn = Callable[[str, dict[str, Any], str], None]

SYSTEM_PROMPT = """You are Achoose, a private assistant on the user's own computer. Be warm and brief. Reply in the user's language.
Use a tool when the request needs facts from this computer or the user's own files (the time, a folder, a file, their documents), or the user clearly asks to save a note.
For greetings, general knowledge, advice or writing text, answer directly with no tool.
Never announce a tool ("Let me check..."); just call it.
Keep answers to a few sentences unless the user asks for more. Never repeat lists or file contents the user can already see.
Never invent file contents or results. If a tool fails or the user declines, say so.
You do not know the current time: for any question about the time, call get_current_time.{memory}
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


COMPUTER_PROMPT = """
You can also control this computer: open and close apps, open files, folders and websites, delete (move to the Trash), lock, sleep, shut down or restart.
When the user asks for one of these, call the tool at once. The user is asked to confirm risky steps automatically, so never ask "are you sure" yourself.
To delete, call move_to_trash with the full path or just the name the user said. To close an open folder, close the app "Files".
To open a file you only know by name, find its full path with find_my_files first."""


VOICE_PROMPT = """
Your answers are SPOKEN aloud: use at most 3 short sentences, no lists, no markdown, no file paths."""


# Questions about the current time/date. Small models guess the time, or misread
# 22:45 when turning it into words, so a plain "what time is it" is answered
# straight from the clock (CLOCK_QUESTION), and any other question that mentions
# the time gets the clock reading handed to the brain first (TIME_QUESTION).
CLOCK_QUESTION = re.compile(
    r"^\W*(?:(?:hey|hi|ok|okay|so|and|jarvis|achoose|please|tell me|can you tell me|do you know)\W+)*"
    r"(?:what(?:'s| is)? the (?P<t1>time|date)(?: (?:now|right now|today))?"
    r"|what (?P<t2>time) (?:is it|it is)(?: now| right now)?"
    r"|(?:the )?(?P<t4>time|date)(?: now| right now| please)?"
    r"|what (?P<d1>day|date) is (?:it|today)(?: today)?"
    r"|what(?:'s| is) today'?s (?P<d2>date)"
    r"|(?:the )?current (?P<t3>time|date))"
    r"\W*(?:please|sir)?\W*$",
    re.IGNORECASE,
)
TIME_QUESTION = re.compile(
    r"\bwhat(?:'s| is)? the (?:time|date)\b|\bwhat time\b|\btime is it\b|\bcurrent (?:time|date)\b"
    r"|\bwhat day is (?:it|today)\b|\bwhat(?:'s| is) today'?s date\b",
    re.IGNORECASE,
)


def build_system_prompt(
    memory_folders: str = "", call_user: str = "", voice: bool = False, computer: bool = False
) -> str:
    extra = MEMORY_PROMPT.format(folders=memory_folders) if memory_folders else ""
    if computer:
        extra += COMPUTER_PROMPT
    if call_user:
        extra += f"\nYou have the calm, polite, formal manner of a British butler. Address the user as '{call_user}' now and then. Never invent facts, times, places or offers: say only what you know or what a tool returned."
    if voice:
        extra += VOICE_PROMPT
    return SYSTEM_PROMPT.format(memory=extra, today=datetime.now().strftime("%A %d %B %Y"))


def clock_answer(now: datetime, asks_time: bool, call_user: str = "") -> str:
    """A spoken-style answer made from the clock, e.g. "It's 10:45 in the evening, sir. Today is Tuesday, 24 November."."""
    sir = f", {call_user}" if call_user else ""
    day = f"{now:%A}, {now.day} {now:%B}"
    if not asks_time:
        return f"Today is {day} {now.year}{sir}."
    hour12 = now.hour % 12 or 12
    if 5 <= now.hour < 12:
        part = "in the morning"
    elif 12 <= now.hour < 17:
        part = "in the afternoon"
    elif 17 <= now.hour < 21:
        part = "in the evening"
    else:
        part = "at night"
    if now.hour == 12 and now.minute == 0:
        when = "exactly noon"
    elif now.hour == 0 and now.minute == 0:
        when = "exactly midnight"
    else:
        when = f"{hour12}:{now.minute:02d} {part}"
    return f"It's {when}{sir}. Today is {day}."


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
        call_user: str = "",
        ask: AskFn | None = None,
    ) -> None:
        """memory_folders: e.g. "~/Documents, ~/Downloads" when search_my_files is available."""
        self.brain = brain
        self.tools = tools
        self.confirm = confirm
        self.on_tool = on_tool
        self.history_messages = history_messages
        self.max_tool_rounds = max_tool_rounds
        self.memory_folders = memory_folders
        self.call_user = call_user
        self.ask_user = ask or (lambda question: False)
        self.voice = False  # set True in voice mode: answers are spoken
        self.history: list[Message] = []
        self.last_stats: list[BrainStats] = []

    def reset(self) -> None:
        self.history.clear()

    def ask(self, text: str, on_text: Callable[[str], None] | None = None) -> str:
        """Answer one message. on_text, if given, receives the answer as it is written."""
        self.history.append(Message("user", text))
        self._trim_history()
        self.last_stats = []
        clock = CLOCK_QUESTION.search(text.strip())
        if clock:
            asks_time = any(clock.group(g) and clock.group(g).lower() == "time" for g in ("t1", "t2", "t3", "t4"))
            answer = clock_answer(datetime.now(), asks_time, self.call_user)
            if on_text:
                on_text(answer)
            self.history.append(Message("assistant", answer))
            return answer
        if TIME_QUESTION.search(text) and self.tools.get("get_current_time"):
            call = ToolCall("get_current_time", {})
            self.history.append(Message("assistant", "", tool_calls=[call]))
            result, _ = self._run_tool(call)
            self.history.append(Message("tool", result, tool_name=call.name))

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

    def _run_tool(self, call: ToolCall, guessed: bool = False) -> tuple[str, bool]:
        """Run one tool through the safety gate. Returns (result, worked)."""
        ok = False
        tool = self.tools.get(call.name)
        if tool is None:
            result = f"Error: there is no tool called '{call.name}'."
        else:
            try:
                if tool.check:
                    tool.check(**call.arguments)
                if tool.risk.asks_first and not self.confirm(tool, call.arguments):
                    result = "The user declined this action. It was NOT done."
                else:
                    result = tool.run(**call.arguments)
                    ok = True
            except Unsure as unsure:
                # Close but not exact (an accent or a mis-heard word): ask, don't fail.
                if not guessed and self.ask_user(unsure.question):
                    return self._run_tool(ToolCall(call.name, unsure.arguments), guessed=True)
                result = f"Error: {unsure} The user said no, that is not what they meant."
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
        computer = self.tools.get("open_app") is not None
        system = build_system_prompt(self.memory_folders, self.call_user, self.voice, computer)
        return [Message("system", system), *self.history]

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
