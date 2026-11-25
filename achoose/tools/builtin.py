"""The first, simple tools.

Three READ tools (time, list a folder, read a text file) and one CHANGE tool
(save a note) that shows how the confirmation gate works.
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path

from ..config import Config
from .base import Risk, Tool, ToolError
from .computer import NameFinder

# Kept small on purpose: the model must read every line before it answers,
# and on a laptop CPU that reading is slow.
MAX_LIST_ENTRIES = 25
MAX_READ_CHARS = 3_000
COUNT_LIMIT = 20_000  # stop counting inside huge folders

# What people call kinds of files -> file endings
CATEGORIES = {
    "video": {"mp4", "mkv", "avi", "mov", "webm", "wmv", "flv", "m4v", "3gp", "mpeg", "mpg"},
    "picture": {"jpg", "jpeg", "png", "gif", "webp", "bmp", "heic", "svg", "tiff"},
    "music": {"mp3", "wav", "ogg", "flac", "m4a", "aac", "opus", "wma"},
    "document": {"pdf", "doc", "docx", "odt", "txt", "rtf", "ppt", "pptx", "xls", "xlsx", "odp", "ods", "md"},
    "archive": {"zip", "rar", "7z", "tar", "gz", "xz"},
}
CATEGORY_WORDS = {
    "video": "video", "videos": "video", "movie": "video", "movies": "video", "film": "video", "films": "video",
    "picture": "picture", "pictures": "picture", "photo": "picture", "photos": "picture", "image": "picture",
    "images": "picture", "pic": "picture", "pics": "picture",
    "music": "music", "song": "music", "songs": "music", "audio": "music",
    "document": "document", "documents": "document", "docs": "document",
    "archive": "archive", "archives": "archive", "zip": "archive",
}


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
        hour12 = now.hour % 12 or 12
        return now.strftime(f"%A %d %B %Y, %H:%M ({hour12}:%M %p), time zone %Z")

    finder = NameFinder(Path.home().resolve())

    def list_folder(path: str = "~", only: str = "") -> str:
        raw = (path or "~").strip()
        folder = safe_path(raw)
        if not folder.exists():
            if "/" in raw or raw.startswith("~"):
                raise ToolError(f"{folder} does not exist.")
            # Just a name ("Alparslan"): look for that folder in the home folder.
            folder = safe_path(str(finder.find(raw + " folder", tool="list_folder", arg="path")))
        if not folder.is_dir():
            raise ToolError(f"{folder} is a file, not a folder.")
        entries = []
        for p in folder.iterdir():
            if p.name.startswith("."):
                continue
            try:
                st = p.stat()
            except OSError:
                continue
            entries.append((st.st_mtime, p.is_dir(), p.name, st.st_size))
        if not entries:
            return f"{folder} is empty."
        entries.sort(reverse=True)  # most recently changed first
        n_dirs = sum(1 for e in entries if e[1])
        lines = [
            f"{folder}: {len(entries)} items ({n_dirs} folders, {len(entries) - n_dirs} files)."
        ]
        kind = (only or "").strip().lower().lstrip(".")
        if kind in ("folder", "folders", "dir", "dirs", "directories"):
            entries = [e for e in entries if e[1]]
            lines.append(f"Folders ({len(entries)}):")
        elif kind in ("file", "files"):
            entries = [e for e in entries if not e[1]]
            lines.append(f"Files ({len(entries)}):")
        elif kind and kind not in ("all", "everything"):
            category = CATEGORY_WORDS.get(kind)
            endings = CATEGORIES[category] if category else {kind}
            label = f"{category}s" if category and category != "music" else (category or f".{kind} files")
            entries = [e for e in entries if not e[1] and e[2].lower().rsplit(".", 1)[-1] in endings]
            deeper = _count_inside(folder, endings) - len(entries)
            also = f", and {deeper} more inside its subfolders" if deeper > 0 else ""
            # The answer first (voice mode speaks only the first line).
            if not entries:
                deeper_part = f", but {deeper} inside its subfolders" if deeper > 0 else ""
                return f"No {label} right in {folder.name}{deeper_part}."
            one = label[:-1] if label.endswith("s") else label  # "1 video", "1 .pdf file"
            lines[0] = f"{folder.name} has {len(entries)} {one if len(entries) == 1 else label}{also}."
            lines.append(f"{label.capitalize()} ({len(entries)}):")
        if len(entries) > MAX_LIST_ENTRIES:
            lines.append(f"The {MAX_LIST_ENTRIES} most recently changed:")
        for mtime, is_dir, name, size in entries[:MAX_LIST_ENTRIES]:
            day = datetime.fromtimestamp(mtime).strftime("%d %b")
            lines.append(f"{name}/ ({day})" if is_dir else f"{name} ({_human_size(size)}, {day})")
        return "\n".join(lines)

    def read_text_file(path: str) -> str:
        file = safe_path(path)
        if not file.is_file():
            raise ToolError(f"{file} is not a file.")
        with file.open("rb") as fh:
            data = fh.read(MAX_READ_CHARS * 4)
        if b"\x00" in data:
            raise ToolError(f"{file.name} is not a text file.")
        text = data.decode("utf-8", errors="replace")
        if len(text) > MAX_READ_CHARS or file.stat().st_size > len(data):
            text = text[:MAX_READ_CHARS] + "\n\n[cut off: only the beginning of the file was read]"
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
            description="List or count what is inside a folder (e.g. how many videos are in it). A folder name is enough.",
            run=list_folder,
            show_directly=True,
            parameters={
                "path": {
                    "type": "string",
                    "description": "Folder path like ~/Downloads, or just its name like Alparslan",
                },
                "only": {
                    "type": "string",
                    "description": "Optional filter: folders, files, videos, pictures, music, documents, or a file type like pdf",
                },
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


def _count_inside(folder: Path, endings: set[str]) -> int:
    """How many files with these endings are in the folder and all its subfolders."""
    found = seen = 0
    for root, dirs, files in os.walk(folder):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for name in files:
            seen += 1
            if not name.startswith(".") and name.lower().rsplit(".", 1)[-1] in endings:
                found += 1
        if seen > COUNT_LIMIT:
            break
    return found


def _human_size(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{n} B"
