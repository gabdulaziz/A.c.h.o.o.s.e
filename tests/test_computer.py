"""Computer control tools, tested without touching the real computer.

Nothing here starts an app, opens a browser or shuts anything down: a fake
Runner records what would have been started instead.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from achoose.agent import Agent
from achoose.brain import BrainReply, ToolCall
from achoose.config import Config
from achoose.tools import Risk, ToolRegistry, builtin_tools, computer_tools
from achoose.tools.computer import exec_to_argv, find_app, installed_apps, parse_desktop_file

from .test_agent import ScriptedBrain, call


class FakeRunner:
    def __init__(self, gio: bool = False, run_code: int = 0) -> None:
        self.stopped: list[int] = []
        # (pid, parent pid, names): VS Code with a helper, and Achoose's own terminal
        me = os.getpid()
        self.procs = [
            (100, 1, {"code"}),
            (101, 100, {"code"}),
            (200, 1, {"gnome-terminal-", "gnome-terminal-server"}),
            (201, 200, {"bash"}),
            (me, 201, {"python3"}),
        ]
        self.started: list[list[str]] = []
        self.ran: list[list[str]] = []
        self.gio = gio
        self.run_code = run_code

    def start(self, argv):
        self.started.append(argv)

    def run(self, argv):
        self.ran.append(argv)
        return self.run_code, ""

    def has(self, program):
        return program == "gio" and self.gio

    def processes(self):
        return self.procs

    def stop(self, pid):
        self.stopped.append(pid)


DESKTOP = {
    "code.desktop": """[Desktop Entry]
Name=Visual Studio Code
GenericName=Text Editor
Exec=/usr/share/code/code %F
Type=Application
Actions=new-empty-window;

[Desktop Action new-empty-window]
Name=New Empty Window
Exec=/usr/share/code/code --new-window %F
""",
    "org.gnome.Terminal.desktop": """[Desktop Entry]
Name=Terminal
Keywords=shell;prompt;command;commandline;cmd;
Exec=gnome-terminal
Type=Application
Actions=new-window;preferences;

[Desktop Action new-window]
Name=New Window
Exec=gnome-terminal --window
""",
    "firefox_firefox.desktop": """[Desktop Entry]
Name=Firefox Web Browser
GenericName=Web Browser
Exec=firefox %u
Type=Application
Actions=new-window;new-private-window;

[Desktop Action new-private-window]
Name=New Private Window
Exec=firefox --private-window %u

[Desktop Action new-window]
Name=New Window
Exec=firefox --new-window %u
""",
    "org.gnome.Nautilus.desktop": """[Desktop Entry]
Name=Files
Exec=nautilus --new-window %U
Type=Application
""",
    "hidden-helper.desktop": """[Desktop Entry]
Name=Helper
Exec=helper
Type=Application
NoDisplay=true
""",
}


class ComputerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name).resolve()
        self.home = root / "home"
        (self.home / "Downloads" / "old").mkdir(parents=True)
        (self.home / "Downloads" / "old" / "a.txt").write_text("aaa")
        (self.home / "Downloads" / "old" / "b.txt").write_text("bbbb")
        (self.home / "Downloads" / "lab5.pdf").write_text("pdf")
        (self.home / ".ssh").mkdir()
        (self.home / ".ssh" / "id").write_text("secret")
        apps = root / "apps"
        apps.mkdir()
        for name, text in DESKTOP.items():
            (apps / name).write_text(text)
        self.apps = installed_apps([apps])
        env = {"HOME": str(self.home), "XDG_DATA_HOME": str(self.home / ".local/share")}
        self.env = mock.patch.dict(os.environ, env)
        self.env.start()
        self.config = Config(data_dir=self.home / "Achoose", read_folders=[self.home])
        self.runner = FakeRunner()
        self.tools = ToolRegistry(computer_tools(self.config, self.runner, app_list=lambda: self.apps))

    def tearDown(self) -> None:
        self.env.stop()
        self.tmp.cleanup()

    def run_tool(self, tool_name, /, **args):
        return self.tools.get(tool_name).run(**args)

    def agent(self, replies, confirm, ask=None):
        tools = ToolRegistry(builtin_tools(self.config) + list(self.tools))
        brain = ScriptedBrain(replies)
        return Agent(brain, tools, confirm=confirm, ask=ask), brain

    # ---- apps ----

    def test_reads_app_list_and_skips_hidden(self):
        names = sorted(a.name for a in self.apps)
        self.assertEqual(names, ["Files", "Firefox Web Browser", "Terminal", "Visual Studio Code"])

    def test_finds_apps_by_what_people_say(self):
        for said, expected in [
            ("VS Code", "Visual Studio Code"),
            ("vscode", "Visual Studio Code"),
            ("code", "Visual Studio Code"),
            ("firefox", "Firefox Web Browser"),
            ("the browser", "Firefox Web Browser"),
            ("terminal", "Terminal"),
            ("file manager", "Files"),
            ("Terminal app", "Terminal"),
        ]:
            app, _ = find_app(self.apps, said)
            self.assertIsNotNone(app, said)
            self.assertEqual(app.name, expected, said)

    def test_unknown_app_is_not_guessed(self):
        app, _ = find_app(self.apps, "Photoshop")
        self.assertIsNone(app)
        with self.assertRaisesRegex(Exception, "No installed app"):
            self.run_tool("open_app", name="Photoshop")
        self.assertEqual(self.runner.started, [])

    def test_open_app_uses_its_own_start_command(self):
        self.assertEqual(self.run_tool("open_app", name="VS Code"), "Opened Visual Studio Code.")
        self.assertEqual(self.runner.started, [["/usr/share/code/code"]])

    def test_open_app_prefers_gio(self):
        self.runner.gio = True
        self.run_tool("open_app", name="terminal")
        self.assertEqual(self.runner.ran[0][:2], ["gio", "launch"])
        self.assertTrue(self.runner.ran[0][2].endswith("org.gnome.Terminal.desktop"))
        self.assertEqual(self.runner.started, [])

    def test_new_window_uses_the_apps_new_window_action(self):
        self.assertEqual(self.run_tool("open_app", name="terminal", new_window=True), "Opened a new Terminal window.")
        self.run_tool("open_app", name="firefox", new_window="true")
        self.assertEqual(self.runner.started, [["gnome-terminal", "--window"], ["firefox", "--new-window"]])

    def test_exec_placeholders_are_dropped(self):
        self.assertEqual(exec_to_argv('"/opt/My App/run" --x %U %%done'), ["/opt/My App/run", "--x", "%done"])

    # ---- "Did you mean ...?" ----

    def test_misheard_app_asks_did_you_mean(self):
        for said, expected in [("Teratra", "Terminal"), ("Firefocks", "Firefox Web Browser")]:
            questions = []
            agent, _ = self.agent(
                [BrainReply("", tool_calls=[ToolCall("open_app", {"name": said})])],
                confirm=lambda t, a: self.fail("asked to confirm"),
                ask=lambda q: questions.append(q) or True,
            )
            self.assertEqual(agent.ask(f"open {said}"), f"Opened {expected}.", said)
            self.assertEqual(questions, [f"Did you mean {expected}?"])

    def test_did_you_mean_no_tells_the_brain(self):
        agent, brain = self.agent(
            [BrainReply("", tool_calls=[ToolCall("open_app", {"name": "Teratra"})]), BrainReply("Which app, sir?")],
            confirm=lambda t, a: False,
            ask=lambda q: False,
        )
        self.assertEqual(agent.ask("open teratra"), "Which app, sir?")
        result = [m for m in brain.seen[-1] if m.role == "tool"][-1].content
        self.assertIn("closest is Terminal", result)
        self.assertIn("said no", result)
        self.assertEqual(self.runner.started, [])

    def test_nonsense_is_not_guessed(self):
        questions = []
        agent, brain = self.agent(
            [BrainReply("", tool_calls=[ToolCall("open_app", {"name": "Photoshop"})]), BrainReply("Not installed.")],
            confirm=lambda t, a: False,
            ask=lambda q: questions.append(q) or True,
        )
        agent.ask("open photoshop")
        self.assertEqual(questions, [])

    def test_misheard_folder_guess_then_still_confirms_delete(self):
        (self.home / "Downloads" / "Alparslan").mkdir()
        questions, confirms = [], []
        agent, _ = self.agent(
            [call("move_to_trash", path="Arslan")],
            confirm=lambda t, a: confirms.append(t.describe(a)) or True,
            ask=lambda q: questions.append(q) or True,
        )
        self.assertEqual(agent.ask("delete arslan"), "Moved Alparslan to the Trash. You can get it back from there.")
        self.assertEqual(questions, ["Did you mean the folder Alparslan in Downloads?"])
        self.assertEqual(len(confirms), 1)  # deleting still asks "Shall I?"

    def test_speech_hints_list_apps_and_folders(self):
        from achoose.tools.computer import speech_hints

        hints = speech_hints(self.apps, self.home)
        self.assertIn("Terminal", hints)
        self.assertIn("Visual Studio Code", hints)
        self.assertIn("old", hints)  # a folder in Downloads
        self.assertNotIn(".ssh", hints)
        self.assertLessEqual(len(speech_hints(self.apps * 50, self.home, limit=300)), 300)

    # ---- folders by name ----

    def test_open_folder_by_spoken_name_goes_straight_to_the_folder(self):
        (self.home / "Documents").mkdir()
        (self.home / "Telegram Desktop" / "Downloads").mkdir(parents=True)
        for said, expected in [
            ("Open download folder.", "Downloads"),
            ("open my Downloads folder", "Downloads"),
            ("Open the old folder please", "Downloads/old"),
        ]:
            self.runner.started.clear()
            agent, brain = self.agent([], confirm=lambda t, a: self.fail("asked"))
            answer = agent.ask(said)
            self.assertEqual(self.runner.started, [["xdg-open", str(self.home / expected)]], said)
            self.assertTrue(answer.startswith("Opened the folder"), answer)
            self.assertEqual(brain.seen, [])  # the brain was not needed

    def test_misheard_folder_to_open_asks_did_you_mean(self):
        (self.home / "Documents").mkdir()
        questions = []
        agent, _ = self.agent([], confirm=lambda t, a: False, ask=lambda q: questions.append(q) or True)
        answer = agent.ask("Open Lockament folder.")
        self.assertEqual(questions, ["Did you mean the folder Documents in your home folder?"])
        self.assertEqual(answer, "Opened the folder ~/Documents.")

    def test_unknown_folder_falls_back_to_the_brain(self):
        agent, brain = self.agent([BrainReply("I couldn't find that folder, sir.")], confirm=lambda t, a: False)
        self.assertEqual(agent.ask("open the zebra folder"), "I couldn't find that folder, sir.")
        self.assertIn("Nothing called", [m for m in brain.seen[0] if m.role == "tool"][-1].content)

    def test_count_videos_in_a_folder_by_name(self):
        target = self.home / "Projects" / "Alparslan"
        (target / "part2").mkdir(parents=True)
        for name in ("a.mp4", "b.MKV", "notes.txt"):
            (target / name).write_text("x")
        (target / "part2" / "c.mov").write_text("x")
        tools = ToolRegistry(builtin_tools(self.config))
        result = tools.get("list_folder").run(path="Al-Parslan", only="videos")
        self.assertTrue(result.startswith("Alparslan has 2 videos, and 1 more inside its subfolders."), result)
        self.assertIn("Videos (2):", result)
        self.assertIn("a.mp4", result)
        self.assertNotIn("notes.txt", result)

    def test_spoken_folder_questions_go_straight_to_the_folder(self):
        target = self.home / "Downloads" / "Alparslan"
        target.mkdir()
        (target / "a.mp4").write_text("x")
        (target / "b.png").write_text("x")
        cases = [
            ("Tell me how many videos are inside the Alparslan folder inside the download folder.", "Alparslan has 1 video."),
            ("How many pictures are in the Alparslan folder?", "Alparslan has 1 picture."),
            ("What's in my alparslan folder?", "Alparslan: 2 items"),
        ]
        for said, start in cases:
            agent, brain = self.agent([], confirm=lambda t, a: False, ask=lambda q: self.fail(q))
            answer = agent.ask(said)
            self.assertIn(start, answer.splitlines()[0], f"{said} -> {answer}")
            self.assertEqual(brain.seen, [])

    def test_accented_folder_name_asks_did_you_mean(self):
        target = self.home / "Downloads" / "Alparslan"
        target.mkdir()
        (target / "a.mp4").write_text("x")
        for heard in ("alpac-land", "Alpaslan", "Al Parsalan"):
            questions = []
            agent, _ = self.agent([], confirm=lambda t, a: False, ask=lambda q: questions.append(q) or True)
            answer = agent.ask(f"How many videos are inside the {heard} folder?")
            self.assertTrue(answer.startswith("Alparslan has 1 video."), f"{heard} -> {answer}")
            self.assertLessEqual(len(questions), 1)
            if questions:
                self.assertEqual(questions, ["Did you mean the folder Alparslan in Downloads?"])

    # ---- files and websites ----

    def test_open_website_and_file(self):
        self.assertIn("youtube.com", self.run_tool("open_file_or_website", target="youtube.com"))
        self.run_tool("open_file_or_website", target="Downloads/lab5.pdf")
        self.run_tool("open_file_or_website", target="~/Downloads")
        self.assertEqual(
            self.runner.started,
            [
                ["xdg-open", "https://youtube.com"],
                ["xdg-open", str(self.home / "Downloads/lab5.pdf")],
                ["xdg-open", str(self.home / "Downloads")],
            ],
        )

    def test_open_refuses_odd_links_and_missing_files(self):
        for bad in ("file:///etc/passwd", "javascript:alert(1)", "~/Downloads/missing.pdf"):
            with self.assertRaises(Exception):
                self.run_tool("open_file_or_website", target=bad)
        self.assertEqual(self.runner.started, [])

    # ---- the Trash ----

    def test_trash_asks_first_and_can_be_restored(self):
        asked = []

        def confirm(tool, args):
            asked.append(tool.describe(args))
            return True

        agent, _ = self.agent(
            [call("move_to_trash", path="~/Downloads/old")], confirm
        )
        answer = agent.ask("delete my old folder")
        self.assertEqual(answer, "Moved old to the Trash. You can get it back from there.")
        self.assertIn("Move the folder old from Downloads (2 files, 7 B) to the Trash", asked[0])
        self.assertFalse((self.home / "Downloads/old").exists())
        trash = self.home / ".local/share/Trash"
        self.assertEqual((trash / "files/old/b.txt").read_text(), "bbbb")
        info = (trash / "info/old.trashinfo").read_text()
        self.assertIn(f"Path={self.home}/Downloads/old", info)
        self.assertIn("DeletionDate=", info)

    def test_trash_declined_does_nothing(self):
        agent, brain = self.agent(
            [call("move_to_trash", path="~/Downloads/lab5.pdf"), BrainReply("Okay, I left it.")],
            confirm=lambda t, a: False,
        )
        self.assertEqual(agent.ask("delete lab5"), "Okay, I left it.")
        self.assertTrue((self.home / "Downloads/lab5.pdf").exists())

    def test_same_name_twice_in_trash(self):
        tool = self.tools.get("move_to_trash")
        tool.run(path="~/Downloads/lab5.pdf")
        (self.home / "Downloads/lab5.pdf").write_text("again")
        tool.run(path="~/Downloads/lab5.pdf")
        files = sorted(p.name for p in (self.home / ".local/share/Trash/files").iterdir())
        self.assertEqual(files, ["lab5.2.pdf", "lab5.pdf"])

    def test_trash_refuses_dangerous_targets_without_asking(self):
        asked = []
        for bad in ("~", "~/Downloads", "~/.ssh/id", "/etc/hostname", "~/Downloads/missing.txt", "~/Achoose", "../.."):
            agent, brain = self.agent(
                [call("move_to_trash", path=bad), BrainReply("I can't.")],
                confirm=lambda t, a: asked.append(a) or True,
            )
            agent.ask("delete it")
            result = [m for m in brain.seen[-1] if m.role == "tool"][-1].content
            self.assertTrue(result.startswith("Error:"), f"{bad}: {result}")
        self.assertEqual(asked, [])  # never asked "Shall I?" about something refused
        self.assertTrue((self.home / ".ssh/id").exists())
        self.assertTrue((self.home / "Downloads").exists())

    def test_trash_of_a_link_keeps_what_it_points_to(self):
        link = self.home / "Downloads/shortcut"
        link.symlink_to(self.home / "Downloads/old")
        self.run_tool("move_to_trash", path="~/Downloads/shortcut")
        self.assertFalse(link.is_symlink())
        self.assertTrue((self.home / "Downloads/old/a.txt").exists())

    def test_trash_by_name_finds_it_anywhere_in_home(self):
        target = self.home / "Projects" / "history" / "Alparslan"
        target.mkdir(parents=True)
        (target / "notes.txt").write_text("x")
        asked = []
        for said in ("Alparslan", "Alp Arslan", "alparslan folder", "Alp Arslen"):
            tool = self.tools.get("move_to_trash")
            tool.check(path=said)
            asked.append(tool.describe({"path": said}).splitlines()[0])
        self.assertTrue(all(a.startswith("Move the folder Alparslan from history (1 files") for a in asked), asked)
        agent, _ = self.agent([call("move_to_trash", path="Alp Arslan")], confirm=lambda t, a: True)
        self.assertEqual(agent.ask("delete Alp Arslan"), "Moved Alparslan to the Trash. You can get it back from there.")
        self.assertFalse(target.exists())

    def test_trash_path_with_wrong_capitals_still_found(self):
        (self.home / "Downloads" / "DII").mkdir()
        (self.home / "Downloads" / "DII" / "x.txt").write_text("x")
        tool = self.tools.get("move_to_trash")
        tool.check(path="~/Downloads/dii")
        self.assertTrue(tool.describe({"path": "~/Downloads/dii"}).startswith("Move the folder DII from Downloads"))
        # wrong folder too: found elsewhere in home
        tool.check(path="~/Documents/dii")
        agent, _ = self.agent([call("move_to_trash", path="~/Downloads/dii")], confirm=lambda t, a: True)
        self.assertEqual(agent.ask("delete dii"), "Moved DII to the Trash. You can get it back from there.")
        self.assertFalse((self.home / "Downloads" / "DII").exists())
        with self.assertRaisesRegex(Exception, "Nothing called"):
            tool.check(path="~/Downloads/zebra")
        self.run_tool("open_file_or_website", target="~/downloads/LAB5.pdf".replace("downloads", "Downloads"))
        self.assertEqual(self.runner.started[-1], ["xdg-open", str(self.home / "Downloads/lab5.pdf")])

    def test_trash_by_name_never_guesses_between_several(self):
        (self.home / "Documents" / "old").mkdir(parents=True)
        agent, brain = self.agent(
            [call("move_to_trash", path="old"), BrainReply("Which one?")],
            confirm=lambda t, a: self.fail("asked"),
        )
        agent.ask("delete old")
        result = [m for m in brain.seen[-1] if m.role == "tool"][-1].content
        self.assertIn("Several things are called 'old'", result)
        self.assertIn("~/Documents/old", result)
        self.assertIn("~/Downloads/old", result)

    def test_trash_by_name_skips_hidden_folders_and_unknown_names(self):
        for said in ("id", "Zebra"):
            with self.assertRaisesRegex(Exception, "Nothing called"):
                self.tools.get("move_to_trash").check(path=said)

    # ---- closing apps ----

    def test_close_app_asks_first_and_stops_only_the_main_program(self):
        asked = []
        agent, _ = self.agent(
            [BrainReply("", tool_calls=[ToolCall("close_app", {"name": "VS Code"})])], confirm=lambda t, a: asked.append(t.describe(a)) or True
        )
        self.assertEqual(agent.ask("close vs code"), "Closed Visual Studio Code.")
        self.assertIn("Close Visual Studio Code (all its windows)", asked[0])
        self.assertEqual(self.runner.stopped, [100])

    def test_close_app_declined_or_not_open(self):
        agent, _ = self.agent([BrainReply("", tool_calls=[ToolCall("close_app", {"name": "code"})]), BrainReply("Left it open.")], confirm=lambda t, a: False)
        agent.ask("close code")
        self.assertEqual(self.runner.stopped, [])
        with self.assertRaisesRegex(Exception, "not open"):
            self.tools.get("close_app").check(name="firefox")

    def test_close_app_refuses_to_close_its_own_terminal(self):
        with self.assertRaisesRegex(Exception, "close me too"):
            self.tools.get("close_app").check(name="terminal")
        self.assertEqual(self.runner.stopped, [])

    def test_real_process_list_reads(self):
        from achoose.tools.computer import Runner

        procs = Runner().processes()
        self.assertIn(os.getpid(), [pid for pid, _, _ in procs])

    # ---- power ----

    def test_shutdown_asks_first_and_waits_a_moment(self):
        asked = []
        agent, _ = self.agent(
            [call("shut_down_or_restart", action="shutdown")],
            confirm=lambda t, a: asked.append(t.describe(a)) or True,
        )
        self.assertEqual(agent.ask("shut down"), "Shutting down in 5 seconds.")
        self.assertIn("Shut down the computer in 5 seconds", asked[0])
        self.assertEqual(self.runner.started, [["sh", "-c", "sleep 5; exec systemctl poweroff"]])

    def test_restart_declined_does_nothing(self):
        agent, _ = self.agent(
            [call("shut_down_or_restart", action="restart"), BrainReply("Very well.")],
            confirm=lambda t, a: False,
        )
        agent.ask("restart")
        self.assertEqual(self.runner.started, [])

    def test_lock_and_sleep_run_on_their_own(self):
        agent, _ = self.agent([call("lock_or_sleep", action="lock")], confirm=lambda t, a: self.fail("asked"))
        self.assertEqual(agent.ask("lock the screen"), "Screen locked.")
        self.assertEqual(self.runner.ran, [["loginctl", "lock-session"]])

    def test_which_tools_ask_first(self):
        risky = sorted(t.name for t in self.tools if t.risk.asks_first)
        self.assertEqual(risky, ["close_app", "move_to_trash", "shut_down_or_restart"])
        self.assertTrue(all(t.risk in (Risk.ACT, Risk.CHANGE) for t in self.tools))

    def test_system_prompt_mentions_computer_control_only_when_available(self):
        agent, brain = self.agent([BrainReply("Hello!")], confirm=lambda t, a: False)
        agent.ask("hi")
        self.assertIn("control this computer", brain.seen[0][0].content)
        plain = Agent(ScriptedBrain([BrainReply("Hi")]), ToolRegistry(builtin_tools(self.config)))
        plain.ask("hi")
        self.assertNotIn("control this computer", plain.brain.seen[0][0].content)


if __name__ == "__main__":
    unittest.main()
