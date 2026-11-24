"""Voice mode: "Hey Jarvis" -> listen -> think -> speak, again and again."""

from __future__ import annotations

import time
from typing import Any, Callable

from ..agent import Agent
from ..brain import BrainError
from ..tools import Tool
from .engine import Ears, Microphone, Mouth, WakeWord, record_until_silence
from .text import heard_yes

STOP_PHRASES = ("goodbye", "good bye", "stop listening", "go to sleep", "shut down", "exit voice")


def run_voice(
    agent: Agent,
    mic: Microphone,
    ears: Ears,
    mouth: Mouth,
    wake: WakeWord | None,
    show_text: Callable[[str], None],
    end_line: Callable[[], bool],
    call_user: str = "",
) -> None:
    """Runs until you say "goodbye" or press Ctrl+C.

    wake=None means push-to-talk: press Enter, then speak.
    """
    sir = f", {call_user}" if call_user else ""

    def listen() -> str:
        frames = mic.frames()
        try:
            audio = record_until_silence(frames)
        finally:
            frames.close()  # microphone off while Achoose thinks and talks
        return ears.transcribe(audio)

    def confirm_by_voice(tool: Tool, arguments: dict[str, Any]) -> bool:
        end_line()
        question = tool.describe(arguments).splitlines()[0]
        print(f"\n  ⚠  Achoose wants to do this: {tool.describe(arguments)}")
        mouth.say(f"Shall I do this{sir}? {question}. Please say yes or no.")
        mouth.wait()
        mouth.beep()
        answer = listen()
        allowed = heard_yes(answer)
        print(f"  You said: {answer or '(nothing)'}  →  {'✓ allowed' if allowed else '✗ declined'}")
        return allowed

    agent.confirm = confirm_by_voice  # the safety rule still applies, now by voice
    agent.voice = True  # answers are spoken: the brain keeps them short

    if wake:
        mouth.say(f"Online and ready{sir}. Say hey Jarvis when you need me.")
    else:
        mouth.say(f"Online and ready{sir}. Press Enter, then speak.")
    mouth.wait()

    while True:
        try:
            if wake:
                print("\n(Listening for “Hey Jarvis”... Ctrl+C to stop)")
                time.sleep(0.3)  # let the speaker go quiet so we don't hear ourselves
                frames = mic.frames()
                try:
                    wake.wait(
                        frames,
                        on_near_miss=lambda s: print(
                            f"  (heard something close to “Hey Jarvis”: {s:.2f}, needs {wake.threshold})",
                            flush=True,
                        ),
                    )
                finally:
                    frames.close()
            else:
                input("\n(Press Enter and speak. Ctrl+C to stop) ")
            mouth.beep()
            text = listen()
            if not text:
                mouth.say(f"Sorry{sir}, I didn't catch that.")
                mouth.wait()
                continue
            print(f"You (spoken): {text}")
            if any(p in text.lower() for p in STOP_PHRASES):
                mouth.say(f"Goodbye{sir}.")
                mouth.wait()
                return

            print("  (thinking...)", flush=True)

            def on_text(piece: str) -> None:
                show_text(piece)
                mouth.feed(piece)

            try:
                answer = agent.ask(text, on_text=on_text)
            except BrainError as err:
                end_line()
                print(f"[problem] {err}")
                mouth.say(f"I'm afraid something went wrong{sir}.")
                mouth.wait()
                continue
            if not end_line():
                # Nothing was streamed (for example a folder list shown directly):
                # print all of it, but only say the first line.
                print(f"Achoose: {answer}")
                mouth.say(answer.splitlines()[0] if answer else f"Done{sir}.")
            mouth.finish()
        except KeyboardInterrupt:
            end_line()
            print("\nVoice mode stopped.")
            return
