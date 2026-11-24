"""The private search index: one SQLite file on this computer.

For every piece of every indexed file it keeps:
  * the text, in a word index (SQLite FTS5) for exact words and names
  * a "meaning fingerprint" (embedding vector) so similar ideas match too

A search uses both and merges the results.
"""

from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE NOT NULL,
    mtime REAL NOT NULL,
    size INTEGER NOT NULL,
    indexed_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    text TEXT NOT NULL,
    embedding BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_file ON chunks(file_id);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, content='chunks', content_rowid='id'
);
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


class Embedder(Protocol):
    """Anything that turns text into meaning fingerprints."""

    name: str

    def embed_documents(self, texts: list[str], title: str) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


@dataclass
class Hit:
    path: str
    text: str
    score: float


class MemoryStore:
    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(db_path))
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.executescript(SCHEMA)
        self._matrix: np.ndarray | None = None
        self._matrix_ids: np.ndarray | None = None

    # ---- bookkeeping --------------------------------------------------

    def get_meta(self, key: str) -> str | None:
        row = self.db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self.db.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (key, value))
        self.db.commit()

    def known_files(self) -> dict[str, tuple[float, int]]:
        return {p: (m, s) for p, m, s in self.db.execute("SELECT path, mtime, size FROM files")}

    def counts(self) -> tuple[int, int]:
        files = self.db.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        chunks = self.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return files, chunks

    def clear(self) -> None:
        self.db.execute("DELETE FROM chunks")
        self.db.execute("DELETE FROM files")
        self.db.commit()
        self._matrix = None

    # ---- changes ------------------------------------------------------

    def put_file(
        self, path: str, mtime: float, size: int, pieces: list[str], vectors: list[list[float]]
    ) -> None:
        """Replace everything stored for one file (pieces may be empty)."""
        with self.db:
            self.db.execute("DELETE FROM files WHERE path = ?", (path,))
            cur = self.db.execute(
                "INSERT INTO files(path, mtime, size, indexed_at) VALUES (?, ?, ?, ?)",
                (path, mtime, size, time.time()),
            )
            file_id = cur.lastrowid
            self.db.executemany(
                "INSERT INTO chunks(file_id, position, text, embedding) VALUES (?, ?, ?, ?)",
                [
                    (file_id, i, text, np.asarray(vec, dtype=np.float32).tobytes())
                    for i, (text, vec) in enumerate(zip(pieces, vectors))
                ],
            )
        self._matrix = None

    def remove_file(self, path: str) -> None:
        with self.db:
            self.db.execute("DELETE FROM files WHERE path = ?", (path,))
        self._matrix = None

    # ---- search -------------------------------------------------------

    def search(self, query: str, query_vector: list[float] | None, limit: int = 3) -> list[Hit]:
        """Best pieces for a question, mixing word matches and meaning matches."""
        ranked: dict[int, float] = {}
        # Reciprocal rank fusion: a piece near the top of either list scores high.
        for rank, chunk_id in enumerate(self._word_search(query, 20)):
            ranked[chunk_id] = ranked.get(chunk_id, 0.0) + 1.0 / (60 + rank)
        if query_vector is not None:
            for rank, chunk_id in enumerate(self._vector_search(query_vector, 20)):
                ranked[chunk_id] = ranked.get(chunk_id, 0.0) + 1.0 / (60 + rank)
        best = sorted(ranked.items(), key=lambda kv: kv[1], reverse=True)

        hits: list[Hit] = []
        per_file: dict[str, int] = {}
        for chunk_id, score in best:
            row = self.db.execute(
                "SELECT files.path, chunks.text FROM chunks JOIN files ON files.id = chunks.file_id"
                " WHERE chunks.id = ?",
                (chunk_id,),
            ).fetchone()
            if not row:
                continue
            if per_file.get(row[0], 0) >= 2:  # variety: at most 2 pieces from one file
                continue
            per_file[row[0]] = per_file.get(row[0], 0) + 1
            hits.append(Hit(path=row[0], text=row[1], score=score))
            if len(hits) >= limit:
                break
        return hits

    def _word_search(self, query: str, limit: int) -> list[int]:
        words = re.findall(r"\w+", query.lower())
        words = [w for w in words if len(w) > 2]
        if not words:
            return []
        fts_query = " OR ".join(f'"{w}"' for w in words)
        try:
            rows = self.db.execute(
                "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT ?",
                (fts_query, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [r[0] for r in rows]

    def _vector_search(self, vector: list[float], limit: int) -> list[int]:
        if self._matrix is None:
            rows = self.db.execute("SELECT id, embedding FROM chunks").fetchall()
            if not rows:
                return []
            self._matrix_ids = np.array([r[0] for r in rows])
            matrix = np.stack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            self._matrix = matrix / np.maximum(norms, 1e-9)
        q = np.asarray(vector, dtype=np.float32)
        if q.shape[0] != self._matrix.shape[1]:
            return []  # index was built with a different fingerprint model
        q = q / max(float(np.linalg.norm(q)), 1e-9)
        scores = self._matrix @ q
        top = np.argsort(-scores)[:limit]
        return [int(self._matrix_ids[i]) for i in top]
