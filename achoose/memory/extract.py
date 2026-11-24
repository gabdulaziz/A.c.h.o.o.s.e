"""Turn a file into plain text, so it can be indexed.

Plain-text files need nothing extra. PDF, Word and PowerPoint need small
Python packages (pypdf, python-docx, python-pptx). If one is missing,
those files are simply skipped and Achoose says so.
"""

from __future__ import annotations

import html
import re
from pathlib import Path

TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".rst", ".csv", ".tsv", ".json", ".yaml", ".yml", ".toml",
    ".ini", ".cfg", ".log", ".tex", ".org",
    ".py", ".c", ".h", ".cpp", ".hpp", ".cc", ".java", ".kt", ".js", ".ts", ".jsx", ".tsx",
    ".go", ".rs", ".rb", ".php", ".cs", ".swift", ".sh", ".bash", ".sql", ".r", ".m", ".lua",
    ".css", ".scss", ".xml",
}
HTML_EXTENSIONS = {".html", ".htm"}
DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".pptx"}
SUPPORTED = TEXT_EXTENSIONS | HTML_EXTENSIONS | DOCUMENT_EXTENSIONS

MAX_TEXT_CHARS = 300_000  # about 150 pages; the rest of a huge file is ignored


class MissingPackage(Exception):
    """A package needed for this file type is not installed."""


def is_supported(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED


def extract_text(path: Path) -> str:
    """Return the text inside a file ('' if it has none)."""
    ext = path.suffix.lower()
    if ext in TEXT_EXTENSIONS:
        text = _read_text(path)
    elif ext in HTML_EXTENSIONS:
        text = _html_to_text(_read_text(path))
    elif ext == ".pdf":
        text = _pdf(path)
    elif ext == ".docx":
        text = _docx(path)
    elif ext == ".pptx":
        text = _pptx(path)
    else:
        return ""
    return text[:MAX_TEXT_CHARS]


def _read_text(path: Path) -> str:
    data = path.read_bytes()[: MAX_TEXT_CHARS * 2]
    if b"\x00" in data[:4096]:
        return ""  # binary file with a text-like name
    return data.decode("utf-8", errors="replace")


def _html_to_text(raw: str) -> str:
    raw = re.sub(r"(?is)<(script|style).*?</\1>", " ", raw)
    raw = re.sub(r"(?s)<[^>]+>", " ", raw)
    return html.unescape(re.sub(r"[ \t]+", " ", raw))


def _pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        raise MissingPackage("pypdf") from None
    reader = PdfReader(str(path))
    pages, total = [], 0
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
        total += len(text)
        if total > MAX_TEXT_CHARS:
            break
    return "\n\n".join(pages)


def _docx(path: Path) -> str:
    try:
        import docx
    except ImportError:
        raise MissingPackage("python-docx") from None
    document = docx.Document(str(path))
    parts = [p.text for p in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            parts.append(" | ".join(cell.text for cell in row.cells))
    return "\n".join(parts)


def _pptx(path: Path) -> str:
    try:
        from pptx import Presentation
    except ImportError:
        raise MissingPackage("python-pptx") from None
    slides = []
    for number, slide in enumerate(Presentation(str(path)).slides, 1):
        texts = [
            shape.text_frame.text
            for shape in slide.shapes
            if getattr(shape, "has_text_frame", False) and shape.text_frame.text.strip()
        ]
        if texts:
            slides.append(f"Slide {number}:\n" + "\n".join(texts))
    return "\n\n".join(slides)


def chunk_text(text: str, size: int = 800, overlap: int = 120) -> list[str]:
    """Cut text into pieces of about `size` characters, on paragraph or
    sentence boundaries where possible, with a little overlap so an idea
    split across two pieces can still be found."""
    text = re.sub(r"\n{3,}", "\n\n", text.replace("\r", "")).strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    pieces, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            window = text[start:end]
            cut = max(window.rfind("\n\n"), window.rfind(". "), window.rfind("\n"))
            if cut > size // 2:
                end = start + cut + 1
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return pieces
