"""The search_my_files tool: lets the brain look inside indexed folders."""

from __future__ import annotations

from pathlib import Path

from .base import Tool, ToolError

SNIPPET_CHARS = 400  # per result; the brain must read these before answering
MAX_FOUND = 60  # longest list of file names shown by find_my_files


def memory_tools(memory) -> list[Tool]:
    def search_my_files(query: str) -> str:
        query = (query or "").strip()
        if not query:
            raise ToolError("No search words given.")
        files, _ = memory.store.counts()
        if files == 0:
            return "Nothing is indexed yet. Tell the user to type /index first."
        hits = memory.search(query, limit=3)
        if not hits:
            folders = ", ".join(_short(p) for p in memory.folders)
            return f"Nothing matching was found in the indexed folders ({folders})."
        parts = []
        for n, hit in enumerate(hits, 1):
            text = " ".join(hit.text.split())
            if len(text) > SNIPPET_CHARS:
                text = text[:SNIPPET_CHARS] + "..."
            parts.append(f"[{n}] {_short(Path(hit.path))}\n{text}")
        return "\n\n".join(parts)

    def find_my_files(name: str) -> str:
        words = [w for w in (name or "").replace("*", " ").split() if w]
        if not words:
            raise ToolError("No name given.")
        paths = memory.store.find_paths(words)
        if not paths:
            return f"No indexed file has '{' '.join(words)}' in its name or folder."
        lines = [f"{len(paths)} files with '{' '.join(words)}' in the name or folder:"]
        lines += [_short(Path(p)) for p in paths[:MAX_FOUND]]
        if len(paths) > MAX_FOUND:
            lines.append(f"... and {len(paths) - MAX_FOUND} more")
        return "\n".join(lines)

    return [
        Tool(
            name="find_my_files",
            description=(
                "Find ALL of the user's files whose name contains some words, e.g. every 'lab' file. "
                "Use for 'how many', 'list all' or 'find the file called' questions."
            ),
            run=find_my_files,
            show_directly=True,
            parameters={"name": {"type": "string", "description": "Words in the file name, e.g. java lab"}},
            required=["name"],
        ),
        Tool(
            name="search_my_files",
            description=(
                "Search inside the user's own documents and notes, by meaning and by words. "
                "Use for questions about their files, notes, school work or projects."
            ),
            run=search_my_files,
            parameters={"query": {"type": "string", "description": "What to look for"}},
            required=["query"],
        )
    ]


def _short(path: Path) -> str:
    try:
        return "~/" + str(Path(path).relative_to(Path.home()))
    except ValueError:
        return str(path)
