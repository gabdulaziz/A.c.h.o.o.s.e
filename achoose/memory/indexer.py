"""Keep the search index in step with the chosen folders.

Only new or changed files are read again, so after the first run an update
is quick. Files that were deleted are removed from the index too.
"""

from __future__ import annotations

import fnmatch
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..brain import BrainError
from .extract import MissingPackage, chunk_text, extract_text, is_supported
from .store import Embedder, Hit, MemoryStore

SKIP_DIRS = {
    "node_modules", "__pycache__", "venv", ".venv", "site-packages", "dist", "build",
    "target", ".git", ".cache",
}
MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_PIECES_PER_FILE = 300

ProgressFn = Callable[[int, int, str], None]  # done, total, current file


@dataclass
class IndexReport:
    added: int = 0
    updated: int = 0
    removed: int = 0
    unchanged: int = 0
    failed: list[str] = field(default_factory=list)
    missing_packages: set[str] = field(default_factory=set)
    seconds: float = 0.0


class Memory:
    def __init__(
        self,
        store: MemoryStore,
        embedder: Embedder,
        folders: list[Path],
        skip: list[str] | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.folders = folders
        self.skip = [s.lower() for s in (skip or [])]

    def _skip_dir(self, parent: str, name: str) -> bool:
        if name.startswith(".") or name in SKIP_DIRS:
            return True
        if any(fnmatch.fnmatch(name.lower(), pattern) for pattern in self.skip):
            return True
        # A folder with both bin/ and lib/ is almost always an installed program
        # (a JDK, an SDK...): thousands of files that never answer questions.
        inside = os.path.join(parent, name)
        return os.path.isdir(os.path.join(inside, "bin")) and os.path.isdir(os.path.join(inside, "lib"))

    # ---- indexing -----------------------------------------------------

    def find_files(self) -> dict[str, tuple[float, int]]:
        found: dict[str, tuple[float, int]] = {}
        for root in self.folders:
            if not root.is_dir():
                continue
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if not self._skip_dir(dirpath, d)]
                for name in filenames:
                    path = Path(dirpath) / name
                    if name.startswith(".") or not is_supported(path):
                        continue
                    try:
                        st = path.stat()
                    except OSError:
                        continue
                    if 0 < st.st_size <= MAX_FILE_BYTES:
                        found[str(path)] = (st.st_mtime, st.st_size)
        return found

    def update(
        self,
        progress: ProgressFn | None = None,
        threads: int | None = None,
        pause_seconds: float = 0.0,
    ) -> IndexReport:
        """Bring the index up to date.

        threads / pause_seconds make it gentler: fewer processor cores and a
        short rest after each file, so the laptop stays cool (but it is slower).
        """
        started = time.perf_counter()
        report = IndexReport()
        if hasattr(self.embedder, "threads"):
            self.embedder.threads = threads
        try:
            return self._update(report, started, progress, pause_seconds)
        finally:
            if hasattr(self.embedder, "threads"):
                self.embedder.threads = None  # searches run at full speed

    def _update(self, report, started, progress, pause_seconds) -> IndexReport:
        # A different fingerprint model means old fingerprints can't be compared.
        if self.store.get_meta("embedder") not in (None, self.embedder.name):
            self.store.clear()
        self.store.set_meta("embedder", self.embedder.name)

        known = self.store.known_files()
        found = self.find_files()

        for path in set(known) - set(found):
            self.store.remove_file(path)
            report.removed += 1

        todo = [p for p, sig in found.items() if known.get(p) != sig]
        report.unchanged = len(found) - len(todo)
        todo.sort()
        for i, path in enumerate(todo):
            if progress:
                progress(i, len(todo), path)
            mtime, size = found[path]
            try:
                text = extract_text(Path(path))
                pieces = chunk_text(text)[:MAX_PIECES_PER_FILE]
                vectors = self.embedder.embed_documents(pieces, title=Path(path).name) if pieces else []
            except MissingPackage as err:
                report.missing_packages.add(str(err))
                continue
            except BrainError:
                raise  # Ollama problems affect every file: stop and say so
            except Exception as err:  # a broken file must never stop the whole run
                report.failed.append(f"{path}: {type(err).__name__}")
                continue
            # Files with no text are still recorded, so they are not re-read every time.
            self.store.put_file(path, mtime, size, pieces, vectors)
            if path in known:
                report.updated += 1
            else:
                report.added += 1
            if pause_seconds and pieces:
                time.sleep(pause_seconds)
        if progress and todo:
            progress(len(todo), len(todo), "")
        report.seconds = time.perf_counter() - started
        return report

    # ---- searching ----------------------------------------------------

    def search(self, query: str, limit: int = 3) -> list[Hit]:
        try:
            vector = self.embedder.embed_query(query)
        except Exception:
            vector = None  # still search by words if the fingerprint model is unavailable
        return self.store.search(query, vector, limit)
