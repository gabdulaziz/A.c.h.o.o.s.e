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
    model: str = "qwen3.5:4b"
    ollama_host: str = "http://localhost:11434"
    context_tokens: int = 4096
    temperature: float = 0.2
    think: bool = False  # slow "thinking" mode of some models; off for speed

    # Where Achoose keeps its own files (notes, later memory)
    data_dir: Path = Path("~/Achoose")

    # Folders Achoose may read on its own. Reading anywhere else is refused.
    read_folders: list[Path] = field(default_factory=lambda: [Path("~")])

    # How many past messages to keep in the conversation (keeps it fast)
    history_messages: int = 20

    # Safety limit: how many tool rounds per question before giving up
    max_tool_rounds: int = 5

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir).expanduser()
        self.read_folders = [Path(p).expanduser().resolve() for p in self.read_folders]


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
