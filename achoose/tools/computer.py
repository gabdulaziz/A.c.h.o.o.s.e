"""Computer control: open and close apps; open files, folders and websites;
move things to the Trash; lock, sleep, shut down or restart.

Safety:
  * Opening things and locking/sleeping run on their own (Risk.ACT).
  * Closing apps, moving to the Trash, shutting down and restarting always
    ask first (Risk.CHANGE).
  * Achoose never runs a command the brain made up. Apps are started only from
    the app list your app menu uses (.desktop files). Files go to the Trash,
    never deleted for good, and only inside your home folder.
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import signal
import subprocess
import time
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
SEARCH_SECONDS = 4.0  # longest search of the home folder for a name to delete
SEARCH_DEPTH = 7  # how many folders deep that search goes
SKIP_SEARCH = {"node_modules", "__pycache__", "venv", ".venv", "site-packages", "snap"}
# Process names that differ from the app's start command
EXTRA_PROCESS_NAMES = {"gnome-terminal": ["gnome-terminal-server"]}

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

    def processes(self) -> list[tuple[int, int, set[str]]]:
        """Your own running programs: (process id, parent id, names)."""
        found = []
        uid = os.getuid()
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                if entry.stat().st_uid != uid:
                    continue
                stat = (entry / "stat").read_text()
                comm = stat[stat.index("(") + 1 : stat.rindex(")")]
                ppid = int(stat[stat.rindex(")") + 2 :].split()[1])
                names = {comm}
                try:
                    names.add(Path(os.readlink(entry / "exe")).name.removesuffix(" (deleted)"))
                except OSError:
                    pass
                cmd = (entry / "cmdline").read_bytes().split(b"\0")[0].decode(errors="replace")
                if cmd:
                    names.add(Path(cmd).name)
            except (OSError, ValueError):
                continue
            found.append((int(entry.name), ppid, names))
        return found

    def stop(self, pid: int) -> None:
        """Ask a program to quit politely (the same signal as logging out)."""
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


# ---- the installed apps ---------------------------------------------------


@dataclass
class App:
    id: str  # e.g. "code" for code.desktop
    name: str  # e.g. "Visual Studio Code"
    exec: str
    path: Path
    generic: str = ""
    wm_class: str = ""
    keywords: list[str] = field(default_factory=list)
    actions: dict[str, tuple[str, str]] = field(default_factory=dict)  # id -> (name, exec)

    def process_names(self) -> set[str]:
        """Names its running program is likely to have, e.g. {'code'}."""
        names = set()
        try:
            argv = shlex.split(self.exec)
        except ValueError:
            argv = self.exec.split()
        # skip "env", "VAR=value" and similar wrappers
        while argv and (Path(argv[0]).name in ("env", "flatpak") or "=" in argv[0] or argv[0].startswith("-")):
            if Path(argv[0]).name == "flatpak":
                return {self.id.split(".")[-1].lower(), self.wm_class.lower()} - {""}
            argv = argv[1:]
        if argv:
            names.add(Path(argv[0]).name)
        if self.wm_class:
            names.add(self.wm_class.lower())
        for n in list(names):
            names.update(EXTRA_PROCESS_NAMES.get(n, []))
        return {n[:15] for n in names} | names  # the kernel cuts names at 15 letters

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
        wm_class=entry.get("StartupWMClass", ""),
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

    # ---- close an app ----

    def _own_family() -> set[int]:
        """Achoose itself and the programs it runs inside (e.g. the Terminal)."""
        procs = {pid: ppid for pid, ppid, _ in runner.processes()}
        family, pid = set(), os.getpid()
        while pid and pid not in family:
            family.add(pid)
            pid = procs.get(pid, 0)
        return family

    def _app_processes(name: str) -> tuple[App, list[int]]:
        if not (name or "").strip():
            raise ToolError("No app name given.")
        app, close = find_app(app_list(), name)
        if app is None:
            hint = f" Did you mean: {', '.join(a.name for a in close)}?" if close else ""
            raise ToolError(f"No installed app called '{name}'.{hint}")
        wanted = app.process_names()
        procs = runner.processes()
        matching = {pid: ppid for pid, ppid, names in procs if names & wanted}
        if not matching:
            raise ToolError(f"{app.name} is not open.")
        if set(matching) & _own_family():
            raise ToolError(f"I am running inside {app.name}, so closing it would close me too. Close it yourself.")
        # Only the main program of each app: its helpers close along with it.
        tops = [pid for pid, ppid in matching.items() if ppid not in matching]
        return app, tops

    def check_close(name: str) -> None:
        _app_processes(name)

    def explain_close(name: str = "") -> str:
        try:
            app, _ = _app_processes(name)
        except ToolError as err:
            return str(err)
        return f"Close {app.name} (all its windows).\n  Unsaved work in it may be lost."

    def close_app(name: str) -> str:
        app, pids = _app_processes(name)
        for pid in pids:
            runner.stop(pid)
        return f"Closed {app.name}."

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

    found_cache: dict[str, tuple[float, Path]] = {}

    def _find_by_name(raw: str) -> Path:
        """Search the home folder for a file or folder called (roughly) raw."""
        cached = found_cache.get(raw)
        if cached and time.monotonic() - cached[0] < 120:
            return cached[1]
        words = [w for w in re.split(r"\s+", raw) if w.lower() not in ("the", "my", "folder", "file", "directory")]
        wanted = _squash(" ".join(words) or raw)
        if not wanted:
            raise ToolError("No file or folder name given.")
        exact: list[Path] = []
        close: list[tuple[float, Path]] = []
        deadline = time.monotonic() + SEARCH_SECONDS
        for root, dirs, files in os.walk(home):
            depth = len(Path(root).relative_to(home).parts)
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in SKIP_SEARCH and depth < SEARCH_DEPTH]
            for name in dirs + files:
                if name.startswith("."):
                    continue
                squashed, stem = _squash(name), _squash(Path(name).stem)
                path = Path(root) / name
                if wanted in (squashed, stem):
                    exact.append(path)
                elif not exact:
                    ratio = max(SequenceMatcher(None, wanted, squashed).ratio(), SequenceMatcher(None, wanted, stem).ratio())
                    if ratio >= 0.85:
                        close.append((ratio, path))
            if time.monotonic() > deadline:
                break
        matches = exact or [p for _, p in sorted(close, key=lambda c: -c[0])]
        if not matches:
            raise ToolError(f"Nothing called '{raw}' was found in your home folder. Ask the user to check the name.")
        if len(matches) > 1:
            shown = "\n".join(f"  {_short(m)}{'/' if m.is_dir() else ''}" for m in matches[:6])
            more = f"\n  ...and {len(matches) - 6} more" if len(matches) > 6 else ""
            raise ToolError(
                f"Several things are called '{raw}':\n{shown}{more}\n"
                "Ask the user which one, then call move_to_trash again with its full path."
            )
        found_cache[raw] = (time.monotonic(), matches[0])
        return matches[0]

    def _trash_target(path: str) -> Path:
        raw = (path or "").strip()
        if not raw:
            raise ToolError("No file or folder given.")
        p = _home_path(raw)
        if not p.exists() and not p.is_symlink() and "/" not in raw and not raw.startswith("~"):
            p = _find_by_name(raw)  # just a name: look for it
        # Resolve the folder it is in, but not the item itself: a shortcut (link)
        # is trashed as a link, never the thing it points to.
        p = p.parent.resolve() / p.name
        if not p.exists() and not p.is_symlink():
            raise ToolError(f"{_short(p)} does not exist.")
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
            return f"Move the folder {p.name} from {_place(p)} ({count}) to the Trash.\n  Where: {_short(p)}\n  You can get it back from the Trash."
        size = p.lstat().st_size
        return f"Move {p.name} from {_place(p)} ({_human_size(size)}) to the Trash.\n  Where: {_short(p)}\n  You can get it back from the Trash."

    def _place(p: Path) -> str:
        """Spoken-friendly folder name: 'Documents', 'your home folder'."""
        return "your home folder" if p.parent == home else p.parent.name

    def move_to_trash(path: str) -> str:
        p = _trash_target(path)
        send_to_trash(p, trash_dir())
        found_cache.clear()
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
            name="close_app",
            description="Close an app and all its windows. Open folders are windows of the Files app.",
            run=close_app,
            risk=Risk.CHANGE,
            show_directly=True,
            check=check_close,
            explain=explain_close,
            parameters={"name": {"type": "string", "description": "App name, e.g. VS Code, or Files for folders"}},
            required=["name"],
        ),
        Tool(
            name="move_to_trash",
            description="Delete a file or folder by moving it to the Trash. Give its full path, or just its name.",
            run=move_to_trash,
            risk=Risk.CHANGE,
            show_directly=True,
            check=check_trash,
            explain=explain_trash,
            parameters={"path": {"type": "string", "description": "Full path like ~/Downloads/old.zip, or just the name like Alparslan"}},
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


def _squash(text: str) -> str:
    """'Alp Arslan', 'alp-arslan' and 'Alparslan' all become 'alparslan'."""
    return re.sub(r"[^a-z0-9]+", "", text.lower())


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
