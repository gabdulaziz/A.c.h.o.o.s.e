"""The search_my_files tool: lets the brain look inside indexed folders."""

from __future__ import annotations

from pathlib import Path

from .base import Tool, ToolError

SNIPPET_CHARS = 600  # per result; the brain must read these before answering


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

    return [
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
