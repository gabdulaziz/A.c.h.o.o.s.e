"""Tests that run without Ollama. Run:  python3 -m unittest -v"""

from __future__ import annotations

import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from achoose.agent import Agent
from achoose.brain import BrainError, BrainReply, Message, OllamaBrain, ToolCall
from achoose.config import Config
from achoose.tools import Risk, ToolRegistry, builtin_tools


class ScriptedBrain:
    """A fake brain that replies from a list, and records what it was sent."""

    name = "scripted"

    def __init__(self, replies: list[BrainReply]) -> None:
        self.replies = list(replies)
        self.seen: list[list[Message]] = []

    def chat(self, messages, tools):
        self.seen.append(list(messages))
        return self.replies.pop(0)


def call(name: str, **args) -> BrainReply:
    return BrainReply("", tool_calls=[ToolCall(name, args)])


class AgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        (root / "docs").mkdir()
        (root / "docs" / "todo.txt").write_text("buy milk")
        self.root = root.resolve()
        self.config = Config(data_dir=root / "data", read_folders=[root / "docs"])
        self.tools = ToolRegistry(builtin_tools(self.config))

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def agent(self, replies, confirm=None) -> tuple[Agent, ScriptedBrain]:
        brain = ScriptedBrain(replies)
        kwargs = {"confirm": confirm} if confirm else {}
        return Agent(brain, self.tools, **kwargs), brain

    def last_tool_result(self, brain: ScriptedBrain) -> str:
        return [m for m in brain.seen[-1] if m.role == "tool"][-1].content

    def test_plain_answer(self):
        agent, _ = self.agent([BrainReply("Hello!")])
        self.assertEqual(agent.ask("hi"), "Hello!")

    def test_read_tool_runs_without_asking(self):
        asked = []
        agent, brain = self.agent(
            [call("read_text_file", path=str(self.root / "docs/todo.txt")), BrainReply("It says buy milk.")],
            confirm=lambda t, a: asked.append(t.name) or False,
        )
        self.assertEqual(agent.ask("read my todo"), "It says buy milk.")
        self.assertEqual(asked, [])
        self.assertEqual(self.last_tool_result(brain), "buy milk")

    def test_change_tool_declined_does_nothing(self):
        agent, brain = self.agent(
            [call("save_note", title="x", text="y"), BrainReply("Okay, not saved.")],
            confirm=lambda t, a: False,
        )
        agent.ask("note this")
        self.assertIn("declined", self.last_tool_result(brain))
        self.assertFalse((self.root / "data" / "notes").exists())

    def test_change_tool_default_is_deny(self):
        agent, brain = self.agent([call("save_note", title="x", text="y"), BrainReply("ok")])
        agent.ask("note this")
        self.assertFalse((self.root / "data" / "notes").exists())

    def test_change_tool_allowed_runs(self):
        agent, brain = self.agent(
            [call("save_note", title="Shopping List", text="milk"), BrainReply("Saved.")],
            confirm=lambda t, a: True,
        )
        agent.ask("note this")
        notes = list((self.root / "data" / "notes").glob("*-shopping-list.md"))
        self.assertEqual(len(notes), 1)
        self.assertIn("milk", notes[0].read_text())

    def test_reading_outside_allowed_folders_is_refused(self):
        agent, brain = self.agent([call("read_text_file", path="/etc/passwd"), BrainReply("Can't.")])
        agent.ask("read passwd")
        self.assertIn("Not allowed", self.last_tool_result(brain))

    def test_path_trick_is_refused(self):
        sneaky = str(self.root / "docs" / ".." / "data")
        agent, brain = self.agent([call("list_folder", path=sneaky), BrainReply("no")])
        agent.ask("list")
        self.assertIn("Not allowed", self.last_tool_result(brain))

    def test_unknown_tool_and_bad_arguments(self):
        agent, brain = self.agent([call("format_disk"), call("list_folder", folder="x"), BrainReply("hm")])
        agent.ask("go")
        tool_msgs = [m.content for m in brain.seen[-1] if m.role == "tool"]
        self.assertIn("no tool called", tool_msgs[0])
        self.assertIn("wrong arguments", tool_msgs[1])

    def test_stops_after_too_many_tool_rounds(self):
        agent, _ = self.agent([call("get_current_time")] * 10)
        agent.max_tool_rounds = 2
        self.assertIn("too many steps", agent.ask("loop"))

    def test_history_is_trimmed_at_a_user_message(self):
        agent, _ = self.agent([BrainReply("a")] * 20)
        agent.history_messages = 5
        for i in range(10):
            agent.ask(f"q{i}")
        self.assertLessEqual(len(agent.history), 6)
        self.assertEqual(agent.history[0].role, "user")

    def test_folder_list_is_shown_directly_without_second_brain_call(self):
        agent, brain = self.agent([call("list_folder", path=str(self.root / "docs"))])
        answer = agent.ask("what's in docs?")
        self.assertIn("todo.txt", answer)
        self.assertEqual(len(brain.seen), 1)  # the brain was not asked to re-write the list
        self.assertEqual(agent.history[-1].role, "assistant")
        self.assertIn("todo.txt", agent.history[-2].content)  # still known for follow-ups

    def test_folder_list_is_short_and_newest_first(self):
        import os

        big = self.root / "docs" / "big"
        big.mkdir()
        for i in range(40):
            f = big / f"file{i:02}.txt"
            f.write_text("x")
            os.utime(f, (1_700_000_000 + i, 1_700_000_000 + i))
        out = self.tools.get("list_folder").run(path=str(big))
        lines = out.splitlines()
        self.assertIn("40 items", lines[0])
        self.assertEqual(len(lines), 2 + 25)
        self.assertTrue(lines[2].startswith("file39.txt"))

    def test_long_file_is_cut(self):
        long = self.root / "docs" / "long.txt"
        long.write_text("a" * 10_000)
        out = self.tools.get("read_text_file").run(path=str(long))
        self.assertLess(len(out), 3_200)
        self.assertIn("cut off", out)

    def test_only_save_note_needs_confirmation(self):
        risky = [t.name for t in self.tools if t.risk is not Risk.READ]
        self.assertEqual(risky, ["save_note"])


class FakeOllama(BaseHTTPRequestHandler):
    last_body: dict = {}

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeOllama.last_body = body
        if body["model"] == "texty":
            self._send(200, {"message": {"role": "assistant", "thinking": "hmm", "content":
                '[{"type":"function","function":{"name":"get_current_time","arguments":{}}}]'}})
            return
        if body["model"] in ("streamer", "jsonstreamer") and body["stream"]:
            if body["model"] == "streamer":
                pieces = ["Hel", "lo ", "there."]
            else:
                pieces = ['[{"type":"function",', '"function":{"name":"get_current_time","arguments":{}}}]']
            lines = [{"message": {"role": "assistant", "content": p}, "done": False} for p in pieces]
            lines.append({"message": {"role": "assistant", "content": ""}, "done": True, "eval_count": 3,
                          "eval_duration": 1_000_000_000})
            raw = "".join(json.dumps(x) + "\n" for x in lines).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        if body["model"] == "thinker":
            self._send(200, {"message": {"role": "assistant", "content": "Let me see...\n</think>\n\nParis."}})
            return
        if body["model"] == "missing":
            self._send(404, {"error": f"model '{body['model']}' not found"})
            return
        self._send(
            200,
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"function": {"name": "get_current_time", "arguments": {}}}],
                },
                "eval_count": 50,
                "eval_duration": 5_000_000_000,
                "prompt_eval_count": 100,
                "prompt_eval_duration": 1_000_000_000,
            },
        )

    def _send(self, code, data):
        raw = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args):
        pass


class OllamaBrainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), FakeOllama)
        cls.host = f"http://127.0.0.1:{cls.server.server_port}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def test_parses_tool_calls_and_speed(self):
        brain = OllamaBrain("test", host=self.host)
        reply = brain.chat([Message("user", "time?")], [{"type": "function"}])
        self.assertEqual(reply.tool_calls[0].name, "get_current_time")
        self.assertEqual(reply.stats.output_tokens_per_second, 10.0)
        self.assertEqual(FakeOllama.last_body["think"], False)
        self.assertFalse(FakeOllama.last_body["stream"])
        self.assertEqual(FakeOllama.last_body["options"]["num_predict"], 400)

    def test_sends_tool_results_back(self):
        brain = OllamaBrain("test", host=self.host)
        brain.chat(
            [
                Message("assistant", "", tool_calls=[ToolCall("get_current_time", {})]),
                Message("tool", "Monday", tool_name="get_current_time"),
            ],
            [],
        )
        msgs = FakeOllama.last_body["messages"]
        self.assertEqual(msgs[0]["tool_calls"][0]["function"]["name"], "get_current_time")
        self.assertEqual(msgs[1], {"role": "tool", "content": "Monday", "tool_name": "get_current_time"})

    def test_rescues_tool_call_written_as_text(self):
        tools = [{"type": "function", "function": {"name": "get_current_time"}}]
        reply = OllamaBrain("texty", host=self.host).chat([Message("user", "time?")], tools)
        self.assertEqual(reply.tool_calls[0].name, "get_current_time")
        self.assertTrue(reply.tool_calls_from_text)
        self.assertEqual(reply.content, "")
        self.assertEqual(reply.thinking, "hmm")

    def test_no_rescue_when_no_tools_offered(self):
        reply = OllamaBrain("texty", host=self.host).chat([Message("user", "time?")], [])
        self.assertEqual(reply.tool_calls, [])

    def test_think_tags_are_hidden_from_the_answer(self):
        reply = OllamaBrain("thinker", host=self.host).chat([Message("user", "capital?")], [])
        self.assertEqual(reply.content, "Paris.")
        self.assertIn("Let me see", reply.thinking)

    def test_streaming_shows_text_as_it_arrives(self):
        shown = []
        reply = OllamaBrain("streamer", host=self.host).chat([Message("user", "hi")], [], on_text=shown.append)
        self.assertEqual(shown, ["Hel", "lo ", "there."])
        self.assertEqual(reply.content, "Hello there.")
        self.assertEqual(reply.stats.output_tokens, 3)

    def test_streaming_hides_tool_requests_written_as_text(self):
        shown = []
        tools = [{"type": "function", "function": {"name": "get_current_time"}}]
        reply = OllamaBrain("jsonstreamer", host=self.host).chat(
            [Message("user", "time?")], tools, on_text=shown.append
        )
        self.assertEqual(shown, [])
        self.assertEqual(reply.tool_calls[0].name, "get_current_time")

    def test_missing_model_gives_pull_hint(self):
        with self.assertRaisesRegex(BrainError, "ollama pull missing"):
            OllamaBrain("missing", host=self.host).chat([Message("user", "hi")], [])

    def test_not_running_gives_clear_error(self):
        brain = OllamaBrain("x", host="http://127.0.0.1:9", timeout_seconds=2)
        self.assertFalse(brain.is_running())
        with self.assertRaisesRegex(BrainError, "Cannot reach Ollama"):
            brain.chat([Message("user", "hi")], [])


if __name__ == "__main__":
    unittest.main()
