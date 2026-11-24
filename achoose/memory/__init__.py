"""Achoose's memory: a private search index over folders you choose.

Needs numpy (and pypdf / python-docx / python-pptx for documents).
If numpy is missing, Achoose still runs, just without memory.
"""

from __future__ import annotations

from pathlib import Path

from ..config import Config


def open_memory(config: Config):
    """Return a Memory object, or None (with a reason) if it can't be used."""
    try:
        from .embed import OllamaEmbedder
        from .indexer import Memory
        from .store import MemoryStore
    except ImportError as err:
        return None, f"memory is off: missing Python package ({err.name}). See README, 'Memory setup'."
    store = MemoryStore(Path(config.data_dir) / "memory.db")
    embedder = OllamaEmbedder(config.embed_model, host=config.ollama_host)
    return Memory(store, embedder, config.memory_folders), ""
