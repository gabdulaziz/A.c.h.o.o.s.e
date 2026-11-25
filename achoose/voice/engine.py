"""Ears, mouth and wake word. Everything runs on this computer.

  Mouth:     Piper (a British male voice by default)
  Ears:      faster-whisper (speech to text)
  Wake word: openWakeWord: ready-made "hey jarvis", plus any trained files
             in ~/Achoose/wake_words/ (like "hey achoose")

The heavy packages are imported only when voice mode starts, so text mode
keeps working without them.
"""

from __future__ import annotations

import queue
import re
import threading
from pathlib import Path
from typing import Callable

import numpy as np

from .text import SentenceSplitter, clean_for_speech

RATE = 16_000  # microphone sample rate that whisper and the wake word expect
FRAME = 1_280  # 80 ms of audio, the block size openWakeWord works with


class VoiceSetupError(RuntimeError):
    """Something voice mode needs is missing. The message says how to fix it."""


# ---- microphone ---------------------------------------------------------


class Microphone:
    def __init__(self, device: int | str | None = None) -> None:
        try:
            import sounddevice as sd
        except OSError:
            raise VoiceSetupError(
                "The sound library is missing. Run: sudo apt install -y libportaudio2"
            ) from None
        except ImportError:
            raise VoiceSetupError("Voice packages missing. See README, 'Voice setup'.") from None
        self.sd = sd
        self.device = device if device not in ("", None) else None

    def describe(self) -> str:
        info = self.sd.query_devices(self.device, "input")
        return f"#{info['index']} {info['name']}"

    def frames(self):
        """Yield 80 ms blocks of 16-bit audio until the caller stops."""
        q: queue.Queue = queue.Queue()

        def callback(indata, frames, time_info, status):
            q.put(indata[:, 0].copy())

        with self.sd.InputStream(
            samplerate=RATE,
            channels=1,
            dtype="int16",
            blocksize=FRAME,
            callback=callback,
            device=self.device,
        ):
            while True:
                yield q.get()


def record_until_silence(
    frames,
    silence_seconds: float = 1.0,
    max_seconds: float = 15.0,
    wait_for_speech_seconds: float = 5.0,
    noise_level: float | None = None,
    on_noise: Callable[[float], None] | None = None,
    skip_seconds: float = 0.0,
) -> np.ndarray:
    """Record one spoken request: starts when you speak, stops after a pause.

    Uses loudness only: anything clearly louder than the room's background
    noise counts as speech. The background is measured in the first few
    blocks, unless noise_level (measured earlier) is given: then speech that
    starts at once is caught too, as with a quick "yes" to "Shall I?".
    The last moment before speech is kept, so the first sound is not cut.
    skip_seconds: ignore the very start (the end of the beep can echo in).
    Returns an empty array if nobody spoke.
    """
    per_second = RATE / FRAME
    skip = int(skip_seconds * per_second)
    blocks: list[np.ndarray] = []
    before: list[np.ndarray] = []  # the last ~0.3 s before speech started
    noise: list[float] = []
    started = False
    quiet = 0
    waited = 0
    for block in frames:
        if skip > 0:
            skip -= 1
            continue
        level = float(np.sqrt(np.mean(block.astype(np.float32) ** 2)))
        if noise_level is None and len(noise) < 4:  # first ~0.3 s: learn the background level
            noise.append(level)
            before.append(block)
            if len(noise) == 4 and on_noise:
                on_noise(float(np.median(noise)))
            continue
        floor = noise_level if noise_level is not None else float(np.median(noise))
        threshold = max(floor * 3.0, 300.0)
        loud = level > threshold
        if not started:
            waited += 1
            if loud:
                started = True
                blocks.extend(before)
            elif waited > wait_for_speech_seconds * per_second:
                return np.zeros(0, dtype=np.int16)
            else:
                before = (before + [block])[-4:]
                continue
        blocks.append(block)
        quiet = 0 if loud else quiet + 1
        if quiet >= silence_seconds * per_second or len(blocks) >= max_seconds * per_second:
            break
    return np.concatenate(blocks) if blocks else np.zeros(0, dtype=np.int16)


# ---- ears ---------------------------------------------------------------


class Ears:
    def __init__(self, model: str = "base.en") -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise VoiceSetupError("faster-whisper is missing. See README, 'Voice setup'.") from None
        # int8 = small and fast on a laptop CPU. The model downloads on first use.
        self.model = WhisperModel(model, device="cpu", compute_type="int8")
        self.english_only = model.endswith(".en")

    def transcribe(self, audio: np.ndarray, short_answer: bool = False) -> str:
        """short_answer=True for a one-word reply like "yes": the speech filter
        is skipped, because it can throw away a word that short."""
        if audio.size < RATE // 4:  # under a quarter second: nothing said
            return ""
        samples = audio.astype(np.float32) / 32768.0
        segments, _ = self.model.transcribe(
            samples,
            language="en" if self.english_only else None,
            beam_size=1,  # fastest; good enough for short requests
            vad_filter=not short_answer,  # ignore silence and background noise
        )
        return " ".join(s.text.strip() for s in segments).strip()


# ---- mouth --------------------------------------------------------------


class Mouth:
    """Speaks text. Sentences are queued and spoken one after another in the
    background, so the brain can keep writing while Achoose talks."""

    def __init__(self, voice_path: Path, speed: float = 1.0) -> None:
        try:
            from piper import PiperVoice, SynthesisConfig
            import sounddevice as sd
        except OSError:
            raise VoiceSetupError(
                "The sound library is missing. Run: sudo apt install -y libportaudio2"
            ) from None
        except ImportError:
            raise VoiceSetupError("piper-tts is missing. See README, 'Voice setup'.") from None
        if not voice_path.exists():
            raise VoiceSetupError(
                f"Voice file not found: {voice_path}\n"
                f"Download it with: .venv/bin/python -m piper.download_voices "
                f"{voice_path.stem} --data-dir {voice_path.parent}"
            )
        self.sd = sd
        self.voice = PiperVoice.load(str(voice_path))
        self.config = SynthesisConfig(length_scale=1.0 / max(speed, 0.5))
        self.rate = self.voice.config.sample_rate
        self._queue: queue.Queue = queue.Queue()
        self._splitter = SentenceSplitter()
        threading.Thread(target=self._worker, daemon=True).start()

    # streaming use: feed pieces of an answer, then finish()
    def feed(self, piece: str) -> None:
        for sentence in self._splitter.feed(piece):
            self.say(sentence)

    def finish(self) -> None:
        for sentence in self._splitter.flush():
            self.say(sentence)
        self.wait()

    def say(self, text: str) -> None:
        text = clean_for_speech(text)
        if text:
            self._queue.put(text)

    def wait(self) -> None:
        self._queue.join()

    def beep(self, pitch: float = 880.0) -> None:
        t = np.linspace(0, 0.12, int(self.rate * 0.12), endpoint=False)
        tone = (np.sin(2 * np.pi * pitch * t) * 0.25 * 32767).astype(np.int16)
        self.sd.play(tone, self.rate, blocking=True)

    def _worker(self) -> None:
        while True:
            text = self._queue.get()
            try:
                audio = [c.audio_int16_array for c in self.voice.synthesize(text, syn_config=self.config)]
                if audio:
                    self.sd.play(np.concatenate(audio), self.rate, blocking=True)
            except Exception as err:  # never let a speech problem kill the chat
                print(f"  [voice problem: {err}]")
            finally:
                self._queue.task_done()


# ---- wake word ----------------------------------------------------------


def wake_word_label(name: str) -> str:
    """'hey_jarvis_v0.1' or '/x/hey_achoose.onnx' -> 'Hey Jarvis' / 'Hey Achoose'."""
    stem = Path(name).name
    stem = re.sub(r"\.(onnx|tflite)$", "", stem)
    stem = re.sub(r"_v\d+(\.\d+)*$", "", stem)
    return stem.replace("_", " ").strip().title()


def find_wake_words(config) -> list[str]:
    """Trained wake word files in <data_dir>/wake_words first, then the settings' wake_word(s)."""
    folder = Path(config.data_dir) / "wake_words"
    found = sorted(str(p) for p in folder.glob("*.onnx")) if folder.is_dir() else []
    extra = config.wake_word if isinstance(config.wake_word, list) else [config.wake_word]
    for name in extra:
        name = str(Path(name).expanduser()) if name.endswith(".onnx") else name
        if name and name not in found:
            found.append(name)
    return found


class WakeWord:
    def __init__(self, names: str | list[str] = "hey_jarvis", threshold: float = 0.5) -> None:
        """names: built-in wake words (like "hey_jarvis") and/or paths to trained .onnx files."""
        try:
            from openwakeword.model import Model
            from openwakeword.utils import download_models
        except ImportError:
            raise VoiceSetupError("openwakeword is missing. See README, 'Voice setup'.") from None
        names = [names] if isinstance(names, str) else list(names)
        files = [n for n in names if n.endswith(".onnx")]
        missing = [f for f in files if not Path(f).is_file()]
        if missing:
            raise VoiceSetupError(f"Wake word file not found: {missing[0]}")
        built_in = [n for n in names if not n.endswith(".onnx")]
        # A few MB, only the first time. Also fetches the shared sound-feature models.
        # (An empty list would download every model, so name at least one.)
        download_models(built_in or ["hey_jarvis"])
        self.model = Model(wakeword_models=names, inference_framework="onnx")
        self.labels = {key: wake_word_label(key) for key in self.model.models}
        self.names = names
        self.threshold = threshold
        self.heard = ""  # label of the wake word heard last

    @property
    def phrase(self) -> str:
        """For messages: “Hey Achoose” or “Hey Jarvis”."""
        return " or ".join(f"“{label}”" for label in self.labels.values())

    @property
    def spoken(self) -> str:
        """The main wake word, as said in the greeting."""
        return next(iter(self.labels.values()), "hey Jarvis")

    def wait(
        self,
        frames,
        should_stop: Callable[[], bool] = lambda: False,
        on_near_miss: Callable[[float], None] | None = None,
    ) -> bool:
        """Listen until the wake word is heard. Very light on the processor.

        on_near_miss is told about scores that came close (useful to tune the
        threshold or spot a quiet microphone)."""
        self.model.reset()
        best_miss = 0.0
        quiet_blocks = 0
        for block in frames:
            if should_stop():
                return False
            scores = self.model.predict(block)
            key = max(scores, key=scores.get)
            score = float(scores[key])
            if score >= self.threshold:
                self.heard = self.labels.get(key, key)
                return True
            if score >= 0.15:
                best_miss = max(best_miss, score)
                quiet_blocks = 0
            elif best_miss:
                quiet_blocks += 1
                if quiet_blocks > 10 and on_near_miss:  # the attempt is over: report it once
                    on_near_miss(best_miss)
                    best_miss, quiet_blocks = 0.0, 0
        return False
