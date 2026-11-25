"""Voice check: tests each part of voice mode on its own and says what works.

Run:  .venv/bin/python -m achoose --voice-check
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .engine import FRAME, RATE, Ears, Microphone, Mouth, VoiceSetupError, WakeWord, find_wake_words


def _step(title: str) -> None:
    print(f"\n=== {title} ===", flush=True)


def _record(mic: Microphone, seconds: float) -> np.ndarray:
    frames = mic.frames()
    blocks = []
    try:
        for block in frames:
            blocks.append(block)
            if len(blocks) * FRAME >= seconds * RATE:
                break
    finally:
        frames.close()
    return np.concatenate(blocks)


def _level(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio.astype(np.float32) ** 2))) if audio.size else 0.0


def run_check(config) -> int:
    problems = []

    _step("1. Sound devices")
    try:
        mic = Microphone(config.input_device)
        import sounddevice as sd

        print(sd.query_devices())
        default_in, default_out = sd.default.device
        print(f"\nMicrophone in use: {mic.describe()}")
        print(f"Default speaker: #{default_out}")
    except VoiceSetupError as err:
        print(f"FAILED: {err}")
        return 1

    _step("2. Speaker (you should hear a sentence)")
    mouth = None
    try:
        voice = Path(config.data_dir) / "voices" / f"{config.voice_name}.onnx"
        started = time.perf_counter()
        mouth = Mouth(voice, config.voice_speed)
        mouth.say("This is a test of the speaker. Can you hear me, sir?")
        mouth.wait()
        print(f"OK: spoke in {time.perf_counter() - started:.1f} s (did you hear it?)")
    except Exception as err:
        problems.append("speaker/voice")
        print(f"FAILED: {err}")

    _step("3. Microphone level (stay quiet 2 s, then talk 3 s)")
    try:
        print("Stay quiet...", flush=True)
        quiet = _level(_record(mic, 2))
        print("Now talk normally...", flush=True)
        talk = _level(_record(mic, 3))
        print(f"Background: {quiet:.0f}   Talking: {talk:.0f}")
        if talk < 300:
            problems.append("microphone too quiet")
            print("PROBLEM: your voice is very quiet. Turn up the input volume in Settings > Sound,")
            print("or pick another microphone with input_device in the settings file (numbers above).")
        elif talk < quiet * 3:
            problems.append("microphone: voice not louder than background")
            print("PROBLEM: your voice is hardly louder than the background noise.")
        else:
            print("OK")
    except Exception as err:
        problems.append("microphone")
        print(f"FAILED: {err}")

    _step("4. Understanding speech (say a short sentence)")
    try:
        started = time.perf_counter()
        print("Loading the speech model (the first time downloads about 145 MB)...", flush=True)
        ears = Ears(config.stt_model)
        print(f"Loaded in {time.perf_counter() - started:.1f} s. Speak now (4 s)...", flush=True)
        audio = _record(mic, 4)
        started = time.perf_counter()
        text = ears.transcribe(audio)
        print(f"Heard: {text!r}  ({time.perf_counter() - started:.1f} s to understand)")
        if not text:
            problems.append("speech not understood")
    except Exception as err:
        problems.append("speech-to-text")
        print(f"FAILED: {err}")

    _step("5. Wake word (say it a few times in the next 10 s)")
    try:
        wake = WakeWord(find_wake_words(config), config.wake_threshold)
        print(f"Say {wake.phrase} now...", flush=True)
        frames = mic.frames()
        best = {label: 0.0 for label in wake.labels.values()}
        try:
            for i, block in enumerate(frames):
                for key, score in wake.model.predict(block).items():
                    label = wake.labels.get(key, key)
                    best[label] = max(best[label], float(score))
                    if score >= 0.1:
                        mark = "  <- WAKE" if score >= wake.threshold else ""
                        print(f"  {label}: {score:.2f}{mark}", flush=True)
                if i * FRAME >= 10 * RATE:
                    break
        finally:
            frames.close()
        for label, score in best.items():
            print(f"Best score for “{label}”: {score:.2f} (wakes at {wake.threshold})")
        top = max(best.values(), default=0.0)
        if top < wake.threshold:
            problems.append("wake word not heard")
            if top >= 0.2:
                print(f"Tip: set wake_threshold = {max(0.2, round(top - 0.1, 1))} in the settings file.")
            else:
                print("Tip: check the microphone (step 3), or use --push-to-talk for now.")
    except Exception as err:
        problems.append("wake word")
        print(f"FAILED: {err}")

    _step("Result")
    if problems:
        print("Problems found: " + ", ".join(problems))
        return 1
    print("Everything works. Start voice mode with: .venv/bin/python -m achoose --voice")
    return 0


LANGUAGES = {"en": "English", "ru": "Russian", "tg": "Tajik", "auto": "any language (the model guesses)"}


def run_hear_test(config, language: str = "auto", model: str = "base", rounds: int = 3) -> int:
    """Try hearing another language with a multilingual speech model.

    Nothing in Achoose's settings changes: this only shows what the model
    heard and how long it took, so you can judge if it is good enough.
    """
    from .engine import record_until_silence

    language = language.lower()
    if language not in LANGUAGES:
        print(f"Unknown language '{language}'. Use one of: {', '.join(LANGUAGES)}")
        return 1
    if model.endswith(".en") and language not in ("en", "auto"):
        print("That model only knows English. Use base, small or medium.")
        return 1
    _step(f"Hearing test: {LANGUAGES[language]}, speech model '{model}'")
    try:
        mic = Microphone(config.input_device or None)
        started = time.perf_counter()
        print(f"Loading the speech model (the first time downloads it: base ~145 MB, small ~480 MB)...", flush=True)
        ears = Ears(model, language=None if language == "auto" else language)
        if language in ("en", "auto") and getattr(config, "stt_hints", False):
            from ..tools.computer import speech_hints

            ears.hints = speech_hints()  # same help as in voice mode, for a fair comparison
    except (VoiceSetupError, Exception) as err:
        print(f"FAILED: {err}")
        return 1
    print(f"Loaded in {time.perf_counter() - started:.1f} s.")
    results = []
    for n in range(1, rounds + 1):
        try:
            input(f"\n[{n}/{rounds}] Press Enter, then say one sentence (for example a command for Achoose)... ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        frames = mic.frames()
        try:
            audio = record_until_silence(frames, wait_for_speech_seconds=6.0)
        finally:
            frames.close()
        if audio.size == 0:
            print("  Heard nothing (was the microphone too quiet?)")
            continue
        started = time.perf_counter()
        text = ears.transcribe(audio)
        took = time.perf_counter() - started
        results.append(took)
        print(f"  Heard: {text!r}")
        print(f"  Language it thought: {ears.last_language or '?'}   ({took:.1f} s to understand)")
    if results:
        print(f"\nAverage {sum(results) / len(results):.1f} s per sentence.")
        print("Was the text right? Tell Claude what you said and what it heard.")
    return 0
