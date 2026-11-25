"""Voice tests with fake microphone, ears and mouth: no sound card or models needed."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from achoose.voice.text import SentenceSplitter, clean_for_speech, heard_yes

HAS_NUMPY = importlib.util.find_spec("numpy") is not None


class TextTests(unittest.TestCase):
    def test_sentences_come_out_as_they_complete(self):
        s = SentenceSplitter()
        out = []
        for word in "Good evening, sir. Version 3.5 is ready. Your midterm covered inheritance. Anything else?".split():
            out += s.feed(word + " ")
        self.assertEqual(out[0], "Good evening, sir. Version 3.5 is ready.")  # short bits joined, 3.5 not cut
        out += s.flush()
        self.assertEqual(out[-1], "Anything else?")

    def test_clean_for_speech(self):
        self.assertEqual(clean_for_speech("**Done**, sir! 😊"), "Done , sir!")
        self.assertNotIn("~/", clean_for_speech("Saved to ~/Achoose/notes/todo.md"))

    def test_repeated_phrases_are_kept_once(self):
        from achoose.voice.text import drop_repeats

        looped = "Hey, I choose. " * 40 + "Hey, I"
        self.assertEqual(drop_repeats(looped.strip()), "Hey, I choose. Hey, I")
        self.assertEqual(drop_repeats("open open open the folder"), "open the folder")
        normal = "Open the Downloads folder, then open the Documents folder."
        self.assertEqual(drop_repeats(normal), normal)
        self.assertEqual(drop_repeats("no no"), "no no")

    def test_only_a_clear_yes_is_yes(self):
        for text in ["Yes.", "yeah go ahead", "Sure, do it", "okay", "Yes, sir.", "Of course.", "Absolutely", "Please do."]:
            self.assertTrue(heard_yes(text), text)
        for text in ["", "no", "yes, no wait", "hmm", "not now", "maybe"]:
            self.assertFalse(heard_yes(text), text)


class WakeWordNameTests(unittest.TestCase):
    def test_labels(self):
        from achoose.voice.engine import wake_word_label

        self.assertEqual(wake_word_label("hey_jarvis"), "Hey Jarvis")
        self.assertEqual(wake_word_label("hey_jarvis_v0.1"), "Hey Jarvis")
        self.assertEqual(wake_word_label("/home/j/Achoose/wake_words/hey_achoose.onnx"), "Hey Achoose")

    def test_trained_files_come_first_and_jarvis_stays_as_backup(self):
        from achoose.config import Config
        from achoose.voice.engine import find_wake_words

        with tempfile.TemporaryDirectory() as tmp:
            config = Config(data_dir=Path(tmp))
            self.assertEqual(find_wake_words(config), ["hey_jarvis"])
            (Path(tmp) / "wake_words").mkdir()
            (Path(tmp) / "wake_words" / "hey_achoose.onnx").write_bytes(b"")
            found = find_wake_words(config)
            self.assertEqual(found[-1], "hey_jarvis")
            self.assertTrue(found[0].endswith("hey_achoose.onnx"))
            config.wake_word = ["hey_jarvis", "~/x/alexa.onnx"]
            self.assertTrue(find_wake_words(config)[-1].endswith("x/alexa.onnx"))


@unittest.skipUnless(HAS_NUMPY, "numpy not installed")
class RecordingTests(unittest.TestCase):
    def blocks(self, pattern):
        import numpy as np

        from achoose.voice.engine import FRAME

        rng = np.random.default_rng(0)
        for loud in pattern:
            amp = 4000 if loud else 50
            yield (rng.standard_normal(FRAME) * amp).astype(np.int16)

    def test_stops_after_a_pause(self):
        from achoose.voice.engine import FRAME, record_until_silence

        # 4 quiet (noise check) + 3 quiet + 20 speech + 30 quiet + 50 more that must not be read
        pattern = [0] * 7 + [1] * 20 + [0] * 30 + [1] * 50
        audio = record_until_silence(self.blocks(pattern), silence_seconds=1.0)
        n_blocks = audio.size // FRAME
        self.assertGreaterEqual(n_blocks, 20)
        self.assertLess(n_blocks, 4 + 20 + 16)  # 0.3 s before + speech + ~1 s of silence, then stop

    def test_nobody_speaks(self):
        from achoose.voice.engine import record_until_silence

        audio = record_until_silence(self.blocks([0] * 200), wait_for_speech_seconds=2)
        self.assertEqual(audio.size, 0)

    def test_quick_answer_is_caught_with_known_noise(self):
        from achoose.voice.engine import FRAME, record_until_silence

        # "yes" starts at once, during what used to be the noise check
        pattern = [1] * 5 + [0] * 40
        old_way = record_until_silence(self.blocks(pattern), wait_for_speech_seconds=2)
        self.assertLess(old_way.size // FRAME, 5)  # the yes was lost
        heard = []
        audio = record_until_silence(self.blocks(pattern), noise_level=10.0, on_noise=heard.append)
        self.assertGreaterEqual(audio.size // FRAME, 5)  # the whole yes is kept
        self.assertEqual(heard, [])

    def test_noise_is_remembered_and_start_of_speech_kept(self):
        from achoose.voice.engine import FRAME, record_until_silence

        heard = []
        audio = record_until_silence(self.blocks([0] * 10 + [1] * 10 + [0] * 20), on_noise=heard.append)
        self.assertEqual(len(heard), 1)
        self.assertGreaterEqual(audio.size // FRAME, 10 + 4)  # 4 quiet blocks before speech kept


@unittest.skipUnless(HAS_NUMPY, "numpy not installed")
class VoiceLoopTests(unittest.TestCase):
    def setUp(self):
        from achoose.agent import Agent
        from achoose.brain import BrainReply, ToolCall
        from achoose.config import Config
        from achoose.tools import ToolRegistry, builtin_tools

        from .test_agent import ScriptedBrain

        self.tmp = tempfile.TemporaryDirectory()
        self.config = Config(data_dir=Path(self.tmp.name), read_folders=[Path(self.tmp.name)])
        self.brain = ScriptedBrain(
            [
                BrainReply("Good evening, sir. It is a fine day."),
                BrainReply("", tool_calls=[ToolCall("save_note", {"title": "Milk", "text": "buy milk"})]),
                BrainReply("Very good, sir. Noted."),
            ]
        )
        self.agent = Agent(self.brain, ToolRegistry(builtin_tools(self.config)), call_user="sir")

    def tearDown(self):
        self.tmp.cleanup()

    def run_loop(self, heard):
        import numpy as np

        from achoose.voice.loop import run_voice

        class Mic:
            def frames(self_):
                def gen():
                    while True:
                        yield np.zeros(1280, dtype=np.int16)

                return gen()

        class Ears:
            def transcribe(self_, audio, short_answer=False):
                return heard.pop(0)

        class Mouth:
            def __init__(self_):
                self_.said, self_.streamed = [], ""

            def say(self_, text):
                self_.said.append(text)

            def feed(self_, piece):
                self_.streamed += piece

            def finish(self_):
                pass

            def wait(self_):
                pass

            def beep(self_):
                pass

        class Wake:
            phrase, spoken, threshold = "“Hey Achoose”", "Hey Achoose", 0.5

            def wait(self_, frames, should_stop=lambda: False, on_near_miss=None):
                return True

        mouth = Mouth()
        import achoose.voice.loop as loop

        original = loop.record_until_silence
        loop.record_until_silence = lambda frames, **kw: np.ones(8000, dtype=np.int16)
        try:
            run_voice(self.agent, Mic(), Ears(), mouth, Wake(), lambda p: None, lambda: False, "sir")
        finally:
            loop.record_until_silence = original
        return mouth

    def test_conversation_confirmation_and_goodbye(self):
        mouth = self.run_loop(["How are you?", "Note that I need milk", "yes", "Goodbye"])
        self.assertIn("Good evening, sir", mouth.streamed)
        self.assertTrue(any(s.endswith("Shall I, sir?") for s in mouth.said))
        self.assertTrue(list((Path(self.tmp.name) / "notes").glob("*milk*.md")))  # saved after spoken yes
        self.assertEqual(mouth.said[-1], "Goodbye, sir.")
        self.assertIn("SPOKEN aloud", self.brain.seen[0][0].content)
        self.assertIn("'sir'", self.brain.seen[0][0].content)

    def test_spoken_no_declines(self):
        mouth = self.run_loop(["How are you?", "Note that I need milk", "no thanks", "goodbye"])
        self.assertFalse((Path(self.tmp.name) / "notes").exists())
        self.assertEqual(mouth.said[-1], "Goodbye, sir.")


if __name__ == "__main__":
    unittest.main()
