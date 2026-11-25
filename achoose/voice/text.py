"""Small text helpers for speaking. Pure Python, no audio needed (easy to test)."""

from __future__ import annotations

import re

# A sentence ends at . ! ? followed by a space (so "3.5" is not cut), or at a new line.
_SENTENCE_END = re.compile(r"[.!?]+\s+|\n+")


def clean_for_speech(text: str) -> str:
    """Remove things that sound silly when read aloud (markdown, paths, emoji)."""
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)  # code blocks
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\*\*|__|\*|#+\s*|^\s*[-•]\s+", " ", text, flags=re.MULTILINE)
    text = re.sub(r"(~|/home/\w+)/(\S+/)*", "", text)  # folder parts of paths
    text = re.sub(r"[\U0001F300-\U0001FAFF☀-➿]", "", text)  # emoji
    return re.sub(r"\s+", " ", text).strip()


class SentenceSplitter:
    """Collects streamed text and hands back whole sentences as they complete,
    so speaking can start before the full answer is written."""

    MIN_CHARS = 25  # join tiny fragments like "Yes." with the next sentence

    def __init__(self) -> None:
        self.buffer = ""

    def feed(self, piece: str) -> list[str]:
        self.buffer += piece
        out = []
        while True:
            cut = None
            for m in _SENTENCE_END.finditer(self.buffer):
                if m.end() >= self.MIN_CHARS or "\n" in m.group(0):
                    cut = m.end()
                    break
            if cut is None:
                break
            sentence, self.buffer = self.buffer[:cut].strip(), self.buffer[cut:]
            if sentence:
                out.append(sentence)
        return out

    def flush(self) -> list[str]:
        rest, self.buffer = self.buffer.strip(), ""
        return [rest] if rest else []


_YES = {
    "yes", "yeah", "yea", "yah", "ya", "yep", "yup", "yess", "yesh", "sure", "ok", "okay", "alright",
    "affirmative", "proceed", "confirm", "confirmed", "certainly", "absolutely", "correct", "indeed", "please",
}
_YES_PHRASES = ("go ahead", "do it", "please do", "of course", "all right", "close it", "delete it", "you may")
_NO = {"no", "nope", "don't", "dont", "stop", "cancel", "not", "never", "wait"}


def heard_yes(text: str) -> bool:
    """True only for a clear yes. Anything unclear (or containing a no) is a no."""
    words = re.findall(r"[a-z']+", text.lower())
    if not words or any(w in _NO for w in words):
        return False
    joined = " ".join(words)
    return words[0] in _YES or any(p in joined for p in _YES_PHRASES)


def drop_repeats(text: str, keep: int = 1) -> str:
    """Remove a phrase the speech model repeated over and over:
    "Hey, I choose. Hey, I choose. Hey, I choose." -> "Hey, I choose."
    A phrase (1 to 8 words) said 3 or more times in a row is kept once."""
    words = text.split()
    if len(words) < 3:
        return text
    out: list[str] = []
    i = 0
    while i < len(words):
        cut = False
        for size in range(1, 9):
            chunk = [w.lower().strip(".,!?") for w in words[i : i + size]]
            if len(chunk) < size:
                break
            n = 1
            while [w.lower().strip(".,!?") for w in words[i + n * size : i + (n + 1) * size]] == chunk:
                n += 1
            if n >= 3:
                out.extend(words[i : i + size * keep])
                i += n * size
                cut = True
                break
        if not cut:
            out.append(words[i])
            i += 1
    return " ".join(out)
