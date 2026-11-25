"""Computer control: open apps, files, folders and websites; move things to
the Trash; lock, sleep, shut down or restart.

Safety:
  * Opening things and locking/sleeping run on their own (Risk.ACT).
  * Moving to the Trash, shutting down and restarting always ask first (Risk.CHANGE).
  * Achoose never runs a command the brain made up. Apps are started only from
    the app list your app menu uses (.desktop files). Files go to the Trash,
    never deleted for good, and only inside your home folder.
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable
from urllib.parse import quote, urlparse

from ..config import Config
from .base import Risk, Tool, ToolError

# Folders that are never moved to the Trash themselves (things inside them can be).
PROTECTED_NAMES = {
    "Desktop", "Documents", "Downloads", "Music", "Pictures", "Videos",
    "Templates", "Public", "snap", "Achoose", "achoose",
}
POWER_DELAY_SECONDS = 5  # time to finish speaking before the computer goes off
COUNT_LIMIT = 20_000  # stop counting files in huge folders (keeps the question quick)

# What people say -> words in the app's real name
ALIASES = {
    "vs code": "visual studio code",
    "vscode": "visual studio code",
    "code": "visual studio code",
    "chrome": "google chrome",
    "browser": "web browser",
    "file manager": "files",
    "file explorer": "files",
    "explorer": "files",
    "nautilus": "files",
    "calculator": "calculator",
    "word": "libreoffice writer",
    "excel": "libreoffice calc",
    "powerpoint": "libreoffice impress",
    "settings": "settings",
    "terminal": "terminal",
    "console": "terminal",
}
FILLER = re.compile(r"\b(the|app|application|program|a|an|my|new|window)\b")


# ---- running things (swapped for a fake in tests) ------------------------


class Runner:
    """Starts programs. Tests replace this with a recorder."""

    def start(self, argv: list[str]) -> None:
        """Start a program in the background, detached from Achoose."""
        try:
            subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,  # keeps running if Achoose is closed
            )
        except FileNotFoundError:
            raise ToolError(f"The program '{argv[0]}' is not installed.") from None

    def run(self, argv: list[str]) -> tuple[int, str]:
        """Run a short command and wait for it. Returns (exit code, output)."""
        try:
            done = subprocess.run(argv, capture_output=True, text=True, timeout=15)
        except FileNotFoundError:
            return 127, f"{argv[0]} is not installed"
        except subprocess.TimeoutExpired:
            return 124, "it took too long"
        return done.returncode, (done.stderr or done.stdout).strip()

    def has(self, program: str) -> bool:
        return shutil.which(program) is not None


# ---- the installed apps ---------------------------------------------------


@dataclass
class App:
    id: str  # e.g. "code" for code.desktop
    name: str  # e.g. "Visual Studio Code"
    exec: str
    path: Path
    generic: str = ""
    keywords: list[str] = field(default_factory=list)
    actions: dict[str, tuple[str, str]] = field(default_factory=dict)  # id -> (name, exec)

    def new_window_exec(self) -> str | None:
        for action_id, (name, cmd) in self.actions.items():
            text = f"{action_id} {name}".lower()
            if "new" in text and "window" in text and "private" not in text and "incognito" not in text:
                return cmd
        return None


def app_folders() -> list[Path]:
    """Where Linux keeps the app list (same places the app menu looks)."""
    home_data = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local/share")
    data_dirs = os.environ.get("XDG_DATA_DIRS") or "/usr/local/share:/usr/share"
    folders = [home_data / "applications"]
    folders += [Path(d) / "applications" for d in data_dirs.split(":") if d]
    folders += [
        Path("/var/lib/snapd/desktop/applications"),
        Path("/var/lib/flatpak/exports/share/applications"),
        home_data / "flatpak/exports/share/applications",
    ]
    seen: list[Path] = []
    for f in folders:
        if f not in seen:
            seen.append(f)
    return seen


def parse_desktop_file(path: Path) -> App | None:
    """Read a .desktop file. Returns None for hidden entries and non-apps."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    sections: dict[str, dict[str, str]] = {}
    current: dict[str, str] | None = None
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = sections.setdefault(line[1:-1], {})
            continue
        if current is not None and "=" in line:
            key, value = line.split("=", 1)
            current.setdefault(key.strip(), value.strip())  # first one wins
    entry = sections.get("Desktop Entry")
    if not entry or entry.get("Type") != "Application" or not entry.get("Exec"):
        return None
    if entry.get("NoDisplay", "").lower() == "true" or entry.get("Hidden", "").lower() == "true":
        return None
    actions = {}
    for action_id in filter(None, entry.get("Actions", "").split(";")):
        sec = sections.get(f"Desktop Action {action_id}", {})
        if sec.get("Exec"):
            actions[action_id] = (sec.get("Name", action_id), sec["Exec"])
    return App(
        id=path.stem,
        name=entry.get("Name", path.stem),
        exec=entry["Exec"],
        path=path,
        generic=entry.get("GenericName", ""),
        keywords=[k for k in entry.get("Keywords", "").split(";") if k],
        actions=actions,
    )


def installed_apps(folders: list[Path] | None = None) -> list[App]:
    apps: dict[str, App] = {}
    for folder in folders if folders is not None else app_folders():
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.desktop")):
            if path.stem in apps:  # the first folder wins (your own overrides come first)
                continue
            app = parse_desktop_file(path)
            if app:
                apps[path.stem] = app
    return list(apps.values())


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def find_app(apps: list[App], wanted: str) -> tuple[App | None, list[App]]:
    """Best match for what the user said, plus close alternatives."""
    said = _norm(wanted)
    said = ALIASES.get(said, said)
    core = " ".join(FILLER.sub(" ", said).split()) or said
    core = ALIASES.get(core, core)

    def score(app: App) -> float:
        name, ident = _norm(app.name), _norm(app.id)
        if core in (name, ident) or core == ident.split()[-1]:
            return 1.0
        words = set(core.split())
        name_words = set(name.split())
        if words and words <= name_words:
            return 0.9
        if core and core in name:
            return 0.85
        if core and (core == _norm(app.generic) or core in [_norm(k) for k in app.keywords]):
            return 0.8
        if words and words <= set(_norm(app.generic).split()):
            return 0.75
        return SequenceMatcher(None, core, name).ratio() * 0.7

    ranked = sorted(apps, key=score, reverse=True)
    if not ranked:
        return None, []
    best = ranked[0]
    close = [a for a in ranked[1:4] if score(a) >= 0.5]
    return (best if score(best) >= 0.6 else None), ([best] + close if score(best) < 0.6 else close)


def exec_to_argv(exec_line: str, app: App | None = None) -> list[str]:
    """Turn a .desktop Exec line into a command, dropping the %f/%u placeholders."""
    try:
        parts = shlex.split(exec_line)
    except ValueError:
        parts = exec_line.split()
    argv = []
    for part in parts:
        if re.fullmatch(r"%[fFuUdDnNvmick]", part):
            continue  # file/URL/icon placeholders: we open the app with nothing
        argv.append(re.sub(r"%(.)", lambda m: "%" if m.group(1) == "%" else ("" if m.group(1) in "fFuUdDnNvmick" else m.group(0)), part))
    if not argv:
        raise ToolError(f"Can't start {app.name if app else 'this app'}: its start command is empty.")
    return argv


# ---- the tools ------------------------------------------------------------


def computer_tools(config: Config, runner: Runner | None = None, app_list: Callable[[], list[App]] = installed_apps) -> list[Tool]:
    runner = runner or Runner()
    home = Path.home().resolve()

    def _short(path: Path) -> str:
        try:
            return "~/" + str(path.relative_to(home)) if path != home else "~"
        except ValueError:
            return str(path)

    # ---- open an app ----

    def open_app(name: str, new_window: bool = False) -> str:
        if not (name or "").strip():
            raise ToolError("No app name given.")
        apps = app_list()
        app, close = find_app(apps, name)
        if app is None:
            hint = f" Did you mean: {', '.join(a.name for a in close)}?" if close else ""
            raise ToolError(f"No installed app called '{name}'.{hint}")
        if _truthy(new_window) and app.new_window_exec():
            runner.start(exec_to_argv(app.new_window_exec(), app))
            return f"Opened a new {app.name} window."
        if runner.has("gio"):
            # gio starts the app exactly the way the app menu does
            code, out = runner.run(["gio", "launch", str(app.path)])
            if code == 0:
                return f"Opened {app.name}."
        runner.start(exec_to_argv(app.exec, app))
        return f"Opened {app.name}."

    # ---- open a file, folder or website ----

    def _website(target: str) -> str | None:
        t = target.strip()
        if re.match(r"^https?://", t, re.IGNORECASE):
            return t
        if re.fullmatch(r"(www\.)?[a-z0-9-]+(\.[a-z0-9-]+)+(/\S*)?", t, re.IGNORECASE) and not t.lower().endswith(
            (".pdf", ".txt", ".docx", ".doc", ".pptx", ".xlsx", ".java", ".py", ".png", ".jpg", ".mp3", ".mp4", ".zip")
        ):
            return "https://" + t
        return None

    def _home_path(raw: str) -> Path:
        path = Path((raw or "").strip()).expanduser()
        if not path.is_absolute():
            path = home / path
        return path

    def open_file_or_website(target: str) -> str:
        target = (target or "").strip()
        if not target:
            raise ToolError("Nothing to open was given.")
        if re.match(r"^[a-z][a-z0-9+.-]*:", target, re.IGNORECASE) and not re.match(r"^https?://", target, re.IGNORECASE):
            raise ToolError("Only web links (http or https), files and folders can be opened.")
        path = _home_path(target)
        if not path.exists():
            url = _website(target)
            if url:
                runner.start(["xdg-open", url])
                return f"Opened {urlparse(url).netloc or url} in your browser."
            raise ToolError(f"{_short(path)} does not exist. If you only know the file's name, find its full path with find_my_files first.")
        path = path.resolve()
        if not any(path == root or root in path.parents for root in config.read_folders):
            raise ToolError(f"Not allowed to open {path}.")
        runner.start(["xdg-open", str(path)])
        what = "folder" if path.is_dir() else "file"
        return f"Opened the {what} {_short(path)}."

    # ---- move to the Trash ----

    def _trash_target(path: str) -> Path:
        raw = (path or "").strip()
        if not raw:
            raise ToolError("No file or folder given.")
        p = _home_path(raw)
        # Resolve the folder it is in, but not the item itself: a shortcut (link)
        # is trashed as a link, never the thing it points to.
        p = p.parent.resolve() / p.name
        if not p.exists() and not p.is_symlink():
            raise ToolError(f"{_short(p)} does not exist. If you only know the name, find its full path with find_my_files first.")
        if home not in p.parents:
            raise ToolError(f"Not allowed: {p} is outside your home folder.")
        rel = p.relative_to(home)
        if any(part.startswith(".") for part in rel.parts):
            raise ToolError("Not allowed: hidden folders and settings files are never deleted by Achoose.")
        if len(rel.parts) == 1 and rel.parts[0] in PROTECTED_NAMES:
            raise ToolError(f"Not allowed: {_short(p)} is one of your main folders. I can delete things inside it, not the folder itself.")
        program = Path(__file__).resolve().parents[2]  # Achoose's own folder
        if p == program or program in p.parents or p in program.parents:
            raise ToolError("Not allowed: that is where Achoose itself lives.")
        if p == config.data_dir.resolve() or p in config.data_dir.resolve().parents:
            raise ToolError("Not allowed: that is Achoose's own data folder.")
        return p

    def check_trash(path: str) -> None:
        _trash_target(path)

    def explain_trash(path: str = "") -> str:
        try:
            p = _trash_target(path)
        except ToolError as err:
            return f"Move {path} to the Trash ({err})"
        if p.is_dir() and not p.is_symlink():
            files, size, more = _count(p)
            count = f"{files}{'+' if more else ''} files, {_human_size(size)}"
            return f"Move the folder {p.name} ({count}) to the Trash.\n  Where: {_short(p)}\n  You can get it back from the Trash."
        size = p.lstat().st_size
        return f"Move {p.name} ({_human_size(size)}) to the Trash.\n  Where: {_short(p)}\n  You can get it back from the Trash."

    def move_to_trash(path: str) -> str:
        p = _trash_target(path)
        send_to_trash(p, trash_dir())
        return f"Moved {p.name} to the Trash. You can get it back from there."

    # ---- power ----

    def _power_action(action: str) -> str:
        a = _norm(action or "")
        if a in ("shutdown", "shut down", "power off", "poweroff", "turn off", "off", "switch off"):
            return "shutdown"
        if a in ("restart", "reboot", "reset"):
            return "restart"
        raise ToolError("Action must be 'shutdown' or 'restart'.")

    def check_power(action: str) -> None:
        _power_action(action)

    def explain_power(action: str = "") -> str:
        try:
            a = _power_action(action)
        except ToolError as err:
            return str(err)
        verb = "Shut down" if a == "shutdown" else "Restart"
        return f"{verb} the computer in {POWER_DELAY_SECONDS} seconds.\n  Save your work first: open apps are closed without asking."

    def shut_down_or_restart(action: str) -> str:
        a = _power_action(action)
        command = "poweroff" if a == "shutdown" else "reboot"
        # A short wait so Achoose can say it is going before the screen goes dark.
        runner.start(["sh", "-c", f"sleep {POWER_DELAY_SECONDS}; exec systemctl {command}"])
        return f"{'Shutting down' if a == 'shutdown' else 'Restarting'} in {POWER_DELAY_SECONDS} seconds."

    def lock_or_sleep(action: str) -> str:
        a = _norm(action or "")
        if a in ("lock", "lock screen", "lock the screen"):
            for argv in (["loginctl", "lock-session"], ["xdg-screensaver", "lock"]):
                code, _ = runner.run(argv)
                if code == 0:
                    return "Screen locked."
            raise ToolError("Could not lock the screen.")
        if a in ("sleep", "suspend"):
            runner.start(["sh", "-c", f"sleep 2; exec systemctl suspend"])
            return "Going to sleep in 2 seconds."
        raise ToolError("Action must be 'lock' or 'sleep'.")

    return [
        Tool(
            name="open_app",
            description="Open an installed app, e.g. VS Code, Firefox, Terminal, Files, Calculator.",
            run=open_app,
            risk=Risk.ACT,
            show_directly=True,
            parameters={
                "name": {"type": "string", "description": "App name, e.g. VS Code"},
                "new_window": {"type": "boolean", "description": "true if the user asks for a new window"},
            },
            required=["name"],
        ),
        Tool(
            name="open_file_or_website",
            description="Open a file or folder in its normal program, or a website in the browser.",
            run=open_file_or_website,
            risk=Risk.ACT,
            show_directly=True,
            parameters={
                "target": {"type": "string", "description": "Full path like ~/Downloads/lab5.pdf, or a website like youtube.com"}
            },
            required=["target"],
        ),
        Tool(
            name="move_to_trash",
            description="Delete a file or folder by moving it to the Trash. Needs the full path.",
            run=move_to_trash,
            risk=Risk.CHANGE,
            show_directly=True,
            check=check_trash,
            explain=explain_trash,
            parameters={"path": {"type": "string", "description": "Full path, e.g. ~/Downloads/old.zip"}},
            required=["path"],
        ),
        Tool(
            name="shut_down_or_restart",
            description="Shut down or restart the computer.",
            run=shut_down_or_restart,
            risk=Risk.CHANGE,
            show_directly=True,
            check=check_power,
            explain=explain_power,
            parameters={"action": {"type": "string", "enum": ["shutdown", "restart"]}},
            required=["action"],
        ),
        Tool(
            name="lock_or_sleep",
            description="Lock the screen or put the computer to sleep.",
            run=lock_or_sleep,
            risk=Risk.ACT,
            show_directly=True,
            parameters={"action": {"type": "string", "enum": ["lock", "sleep"]}},
            required=["action"],
        ),
    ]


# ---- the Trash (freedesktop.org standard, the same Trash the Files app shows) ----


def trash_dir() -> Path:
    data = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local/share")
    return data / "Trash"


def send_to_trash(path: Path, trash: Path) -> Path:
    """Move path into the Trash with a note of where it came from, so the
    Files app can put it back ("Restore"). Returns where it went."""
    files, info = trash / "files", trash / "info"
    files.mkdir(parents=True, exist_ok=True)
    info.mkdir(parents=True, exist_ok=True)
    base, n = path.name, 1
    while True:
        name = base if n == 1 else f"{Path(base).stem}.{n}{Path(base).suffix}"
        try:
            # Created exclusively, so two deletes can never pick the same name.
            fd = os.open(info / f"{name}.trashinfo", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            break
        except FileExistsError:
            n += 1
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(
            "[Trash Info]\n"
            f"Path={quote(str(path))}\n"
            f"DeletionDate={datetime.now():%Y-%m-%dT%H:%M:%S}\n"
        )
    target = files / name
    try:
        shutil.move(str(path), str(target))
    except BaseException:
        (info / f"{name}.trashinfo").unlink(missing_ok=True)
        raise
    return target


def _count(folder: Path) -> tuple[int, int, bool]:
    files = size = 0
    for root, dirs, names in os.walk(folder):
        for n in names:
            files += 1
            try:
                size += os.lstat(os.path.join(root, n)).st_size
            except OSError:
                pass
            if files >= COUNT_LIMIT:
                return files, size, True
    return files, size, False


def _truthy(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "1")
    return bool(value)


def _human_size(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{n} B"
