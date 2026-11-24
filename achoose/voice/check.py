"""Voice check: tests each part of voice mode on its own and says what works.

Run:  .venv/bin/python -m achoose --voice-check
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .engine import FRAME, RATE, Ears, Microphone, Mouth, VoiceSetupError, WakeWord


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

    _step("5. Wake word (say “Hey Jarvis” a few times in the next 10 s)")
    try:
        wake = WakeWord(config.wake_word, config.wake_threshold)
        frames = mic.frames()
        best = 0.0
        try:
            for i, block in enumerate(frames):
                score = max(wake.model.predict(block).values())
                best = max(best, float(score))
                if score >= 0.1:
                    print(f"  score {score:.2f}" + ("  <- WAKE" if score >= wake.threshold else ""), flush=True)
                if i * FRAME >= 10 * RATE:
                    break
        finally:
            frames.close()
        print(f"Best score: {best:.2f} (wakes at {wake.threshold})")
        if best < wake.threshold:
            problems.append("wake word not heard")
            if best >= 0.2:
                print(f"Tip: set wake_threshold = {max(0.2, round(best - 0.1, 1))} in the settings file.")
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
