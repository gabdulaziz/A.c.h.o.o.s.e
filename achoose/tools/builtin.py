"""The first, simple tools.

Three READ tools (time, list a folder, read a text file) and one CHANGE tool
(save a note) that shows how the confirmation gate works.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from ..config import Config
from .base import Risk, Tool, ToolError

MAX_LIST_ENTRIES = 100
MAX_READ_BYTES = 20_000


def builtin_tools(config: Config) -> list[Tool]:
    def safe_path(raw: str) -> Path:
        """Turn what the brain wrote into a real path, inside allowed folders only."""
        raw = (raw or "").strip()
        if not raw:
            raise ToolError("No path given.")
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = Path.home() / path
        path = path.resolve()
        if not any(path == root or root in path.parents for root in config.read_folders):
            allowed = ", ".join(str(r) for r in config.read_folders)
            raise ToolError(f"Not allowed to read {path}. Allowed folders: {allowed}")
        return path

    # ---- READ tools ---------------------------------------------------

    def get_current_time() -> str:
        now = datetime.now().astimezone()
        return now.strftime("%A %d %B %Y, %H:%M (%Z)")

    def list_folder(path: str = "~") -> str:
        folder = safe_path(path)
        if not folder.exists():
            raise ToolError(f"{folder} does not exist.")
        if not folder.is_dir():
            raise ToolError(f"{folder} is a file, not a folder.")
        entries = sorted(
            (p for p in folder.iterdir() if not p.name.startswith(".")),
            key=lambda p: (not p.is_dir(), p.name.lower()),
        )
        lines = []
        for p in entries[:MAX_LIST_ENTRIES]:
            if p.is_dir():
                lines.append(f"[folder] {p.name}")
            else:
                try:
                    size = p.stat().st_size
                except OSError:
                    size = 0
                lines.append(f"[file]   {p.name} ({_human_size(size)})")
        extra = len(entries) - MAX_LIST_ENTRIES
        if extra > 0:
            lines.append(f"... and {extra} more")
        return f"{folder}:\n" + ("\n".join(lines) if lines else "(empty)")

    def read_text_file(path: str) -> str:
        file = safe_path(path)
        if not file.is_file():
            raise ToolError(f"{file} is not a file.")
        data = file.read_bytes()[:MAX_READ_BYTES]
        if b"\x00" in data:
            raise ToolError(f"{file.name} is not a text file.")
        text = data.decode("utf-8", errors="replace")
        if file.stat().st_size > MAX_READ_BYTES:
            text += f"\n\n[cut off: only the first {MAX_READ_BYTES // 1000} KB were read]"
        return text

    # ---- CHANGE tools (always confirmed by the user) -------------------

    notes_dir = config.data_dir / "notes"

    def save_note(title: str, text: str) -> str:
        notes_dir.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:50] or "note"
        stem = f"{datetime.now():%Y-%m-%d}-{slug}"
        target = notes_dir / f"{stem}.md"
        n = 2
        while target.exists():  # never overwrite an existing note
            target = notes_dir / f"{stem}-{n}.md"
            n += 1
        target.write_text(f"# {title}\n\n{text}\n", encoding="utf-8")
        return f"Saved note to {target}"

    def explain_save_note(title: str = "", text: str = "") -> str:
        preview = text if len(text) <= 200 else text[:200] + "..."
        return f"Save a note called '{title}' in {notes_dir}:\n  {preview}"

    return [
        Tool(
            name="get_current_time",
            description="Current date, weekday and time. Only when the user asks about the time or date.",
            run=get_current_time,
        ),
        Tool(
            name="list_folder",
            description="List what is inside a folder on this computer.",
            run=list_folder,
            parameters={
                "path": {
                    "type": "string",
                    "description": "Folder, e.g. ~/Downloads",
                }
            },
            required=["path"],
        ),
        Tool(
            name="read_text_file",
            description="Read a text file on this computer.",
            run=read_text_file,
            parameters={
                "path": {"type": "string", "description": "File, e.g. ~/notes/todo.txt"}
            },
            required=["path"],
        ),
        Tool(
            name="save_note",
            description="Save a note. Only when the user explicitly asks to save, note or write something down.",
            run=save_note,
            risk=Risk.CHANGE,
            parameters={
                "title": {"type": "string", "description": "Short title"},
                "text": {"type": "string", "description": "Note text"},
            },
            required=["title", "text"],
            explain=explain_save_note,
        ),
    ]


def _human_size(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{n} B"
