"""Settings for Achoose.

Defaults live here. To change them, copy config.example.toml to
~/.config/achoose/config.toml and edit it. A few settings can also be set
for one run with environment variables, for example:

    ACHOOSE_MODEL=llama3.2:3b python -m achoose
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    tomllib = None  # type: ignore[assignment]

CONFIG_PATH = Path(os.environ.get("ACHOOSE_CONFIG", "~/.config/achoose/config.toml")).expanduser()


@dataclass
class Config:
    # Brain
    model: str = "qwen3:4b-instruct-2507-q4_K_M"
    ollama_host: str = "http://localhost:11434"
    context_tokens: int = 4096
    temperature: float = 0.2
    max_answer_tokens: int = 400  # about 1 minute of writing on a laptop CPU
    think: bool = False  # slow "thinking" mode of some models; off for speed

    # Where Achoose keeps its own files (notes, later memory)
    data_dir: Path = Path("~/Achoose")

    # Folders Achoose may read on its own. Reading anywhere else is refused.
    read_folders: list[Path] = field(default_factory=lambda: [Path("~")])

    # Memory: folders Achoose indexes so it can search inside your files
    memory_folders: list[Path] = field(
        default_factory=lambda: [Path("~/Documents"), Path("~/Downloads")]
    )
    # Folder names to leave out of memory (wildcards allowed, e.g. "old-*").
    # Program installs (folders with both bin/ and lib/ inside) are skipped anyway.
    memory_skip: list[str] = field(default_factory=list)
    # Gentle indexing (plain /index): fewer cores + a short rest per file = cooler laptop.
    # "/index fast" ignores these and uses every core.
    index_threads: int = 2
    index_pause_seconds: float = 0.5
    # Small model that makes the "meaning fingerprints" for search
    embed_model: str = "embeddinggemma"

    # Voice mode (python -m achoose --voice)
    voice_name: str = "en_GB-alan-medium"  # Piper voice: calm British male
    voice_speed: float = 1.0  # 1.2 = a bit faster, 0.9 = a bit slower
    stt_model: str = "base.en"  # speech-to-text model; "small.en" is more accurate but slower
    wake_word: str = "hey_jarvis"
    wake_threshold: float = 0.5  # higher = fewer false wake-ups, but may miss you
    call_user: str = "sir"  # how Achoose addresses you; "" for nothing

    # How many past messages to keep in the conversation (keeps it fast)
    history_messages: int = 20

    # Safety limit: how many tool rounds per question before giving up
    max_tool_rounds: int = 5

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir).expanduser()
        self.read_folders = [Path(p).expanduser().resolve() for p in self.read_folders]
        self.memory_folders = [Path(p).expanduser().resolve() for p in self.memory_folders]


_ENV = {
    "ACHOOSE_MODEL": "model",
    "ACHOOSE_OLLAMA_HOST": "ollama_host",
}


def load_config(path: Path | None = None) -> Config:
    values: dict[str, Any] = {}
    path = path or CONFIG_PATH
    if path.exists():
        if tomllib is None:
            print(f"[achoose] Python 3.11+ is needed to read {path}; using defaults.")
        else:
            with path.open("rb") as fh:
                values = tomllib.load(fh)
    known = {f.name for f in fields(Config)}
    unknown = sorted(set(values) - known)
    if unknown:
        print(f"[achoose] Ignoring unknown settings in {path}: {', '.join(unknown)}")
    values = {k: v for k, v in values.items() if k in known}
    for env, key in _ENV.items():
        if os.environ.get(env):
            values[key] = os.environ[env]
    return Config(**values)
