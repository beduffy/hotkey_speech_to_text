#!/usr/bin/env python3
"""
Speech‑to‑text hot‑key utility for Ubuntu / Linux
================================================

Hold **Left Shift** → speak → release **Left Shift** → recognized words are typed into
whatever window currently has the keyboard focus.

This revision removes the hard dependency on **PortAudio** (which broke on
systems where the library is missing) by introducing a fallback recorder that
uses **FFmpeg**.  Behaviour matrix:

+----------------------+----------------+--------------------------+
| Sounddevice present? | FFmpeg present | Result                  |
+======================+================+==========================+
| ✔ yes                | (don't care)   | Use native microphone    |
+----------------------+----------------+--------------------------+
| ✘ no → PortAudio err | ✔ yes          | Use FFmpeg backend       |
+----------------------+----------------+--------------------------+
| ✘ no                 | ✘ no           | Graceful error & exit    |
+----------------------+----------------+--------------------------+

Quick start
-----------
1.  **Dependencies** (Python 3.9+):

        python3 -m pip install --user pynput numpy openai-whisper

    *Optional* – for the native backend:

        python3 -m pip install --user sounddevice

    FFmpeg must be installed and in $PATH for the fallback backend.

2.  Export your OpenAI API key so Whisper can be used::

        # export OPENAI_API_KEY="sk‑…"  (No longer required for local Whisper)

3.  Run the script (no sudo required)::

        python3 speech_to_text_hotkey.py

4.  Press and hold **Left Shift**, speak, and release. Text is injected at the caret.

Tests
-----
Run *pytest* to execute the lightweight, non‑audio unit tests::

    python3 -m pip install --user pytest
    pytest speech_to_text_hotkey.py::test_suite

"""
from __future__ import annotations

import os
import shutil
import subprocess as sp
import sys
import tempfile
import textwrap
import wave
from pathlib import Path
from typing import List, Protocol, runtime_checkable

import numpy as np
from pynput import keyboard

# <<< ADD WHISPER IMPORT >>>
try:
    import whisper
except ImportError:
    print(
        "[speech‑to‑text] whisper unavailable, transcription via local model will fail.\n" \
        "  Install it with: python3 -m pip install --user openai-whisper",
        file=sys.stderr,
    )
    whisper = None
# <<< END WHISPER IMPORT >>>

# TODO How can we run this in background?

# ---------------------------------------------------------------------------
# Optional sounddevice import ------------------------------------------------
# ---------------------------------------------------------------------------
try:
    import sounddevice as _sd  # noqa: N811 (alias to avoid collision below)

    _sounddevice_available = True
except Exception as _exc:  # ImportError *or* PortAudio OSError
    print(
        "[speech‑to‑text] sounddevice unavailable (" + str(_exc) + "), "
        "will attempt FFmpeg fallback…",
        file=sys.stderr,
    )
    _sd = None  # type: ignore[assignment]
    _sounddevice_available = False

# <<< LOAD WHISPER MODEL >>>
_whisper_model = None
if whisper:
    try:
        # Using "base.en" for a balance of speed and accuracy, like the example script
        print("[speech‑to‑text] Loading local Whisper model (base.en)...", file=sys.stderr)
        _whisper_model = whisper.load_model("base.en")
        print("[speech‑to‑text] Whisper model loaded.", file=sys.stderr)
    except Exception as exc:
        print(
            f"[speech‑to‑text] Failed to load Whisper model: {exc}\n" \
            "  Transcription will likely fail.",
            file=sys.stderr,
        )
# <<< END LOAD WHISPER MODEL >>>

# ---------------------------------------------------------------------------
# User‑tunable constants -----------------------------------------------------
# ---------------------------------------------------------------------------
HOTKEY = keyboard.Key.delete         # Hold this key to record
HOTKEY = keyboard.Key.space         # Hold this key to record
QUIT_KEY = keyboard.Key.esc      # Press this once to exit the program
SAMPLE_RATE = 48_000             # Use device's default sample rate (48 kHz)
CHANNELS = 1                     # Mono microphone
TMP_DIR = Path(tempfile.gettempdir())

# ---------------------------------------------------------------------------
# Recorder interface + back‑ends --------------------------------------------
# ---------------------------------------------------------------------------
@runtime_checkable
class Recorder(Protocol):
    """Common microphone recorder API."""

    def start(self) -> None:  # noqa: D401 – imperative mood
        """Begin recording."""

    def stop(self) -> np.ndarray:  # noqa: D401 – imperative mood
        """Stop recording and return mono float32 array (shape = [n, 1])."""


# ---------------- native PortAudio backend ----------------------------------
class _SounddeviceRecorder:
    def __init__(self, samplerate: int, channels: int):
        self.samplerate = samplerate
        self.channels = channels
        self._frames: List[np.ndarray] = []
        self._stream: "_sd.InputStream | None" = None  # type: ignore[name‑defined]

    # Callback collects data chunks
    def _callback(self, indata, _frames, _time, status):  # noqa: D401
        if status:
            print(status, file=sys.stderr, flush=True)
        self._frames.append(indata.copy())

    def start(self):  # noqa: D401 – imperative mood
        if self._stream is not None:
            return
        self._frames.clear()
        self._stream = _sd.InputStream(  # type: ignore[arg‑type]
            samplerate=self.samplerate,
            channels=self.channels,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> np.ndarray:  # noqa: D401
        if self._stream is None:
            return np.empty((0, self.channels), dtype="float32")
        self._stream.stop(); self._stream.close(); self._stream = None  # noqa: E702; one‑line tidy
        if not self._frames:
            return np.empty((0, self.channels), dtype="float32")
        return np.concatenate(self._frames, axis=0)


# ---------------- FFmpeg fallback backend -----------------------------------
class _FFmpegRecorder:
    """Microphone capture using *ffmpeg* piping to a temporary WAV."""

    def __init__(self, samplerate: int, channels: int):
        self.samplerate = samplerate
        self.channels = channels
        self._tmp_path: Path | None = None
        self._proc: sp.Popen[bytes] | None = None

    def start(self):  # noqa: D401
        if self._proc is not None:
            return
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("FFmpeg executable not found – please install it or provide PortAudio.")
        self._tmp_path = TMP_DIR / (next(tempfile._get_candidate_names()) + ".wav")  # type: ignore[attr‑defined]
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-f",
            "pulse" if shutil.which("pactl") else "alsa",  # prefer PulseAudio if available
            "-i",
            "default",
            "-ac",
            str(self.channels),
            "-ar",
            str(self.samplerate),
            "-y",
            str(self._tmp_path),
        ]
        self._proc = sp.Popen(cmd)

    def stop(self) -> np.ndarray:  # noqa: D401
        if self._proc is None:
            return np.empty((0, self.channels), dtype="float32")
        self._proc.terminate(); self._proc.wait(); self._proc = None  # noqa: E702
        assert self._tmp_path is not None
        data = _wav_to_float32(self._tmp_path)
        self._tmp_path.unlink(missing_ok=True)
        return data


# ---------------- Recorder factory -----------------------------------------

def get_recorder() -> Recorder:
    """Return the most appropriate recorder implementation for this system."""

    if _sounddevice_available:
        return _SounddeviceRecorder(SAMPLE_RATE, CHANNELS)  # type: ignore[arg‑type]
    return _FFmpegRecorder(SAMPLE_RATE, CHANNELS)


# ---------------------------------------------------------------------------
# Small helpers --------------------------------------------------------------
# ---------------------------------------------------------------------------

def _wav_to_float32(path: Path) -> np.ndarray:
    """Read *path* (PCM 16‑bit WAV) → float32 ±1.0, shape (n, 1)."""
    with wave.open(str(path), "rb") as wf:
        frames = wf.readframes(wf.getnframes())
        dtype = np.int16 if wf.getsampwidth() == 2 else np.uint8  # basic types
        ints = np.frombuffer(frames, dtype=dtype)
        if wf.getnchannels() > 1:
            ints = ints.reshape(-1, wf.getnchannels())[:, 0]  # take first channel
        floats = ints.astype(np.float32) / 32767.0
        return floats.reshape(-1, 1)


def _save_wav(signal: np.ndarray, samplerate: int) -> Path:
    """Write *signal* to a temporary WAV file and return its path."""
    path = TMP_DIR / (next(tempfile._get_candidate_names()) + ".wav")  # type: ignore[attr‑defined]
    pcm16 = np.clip(signal, -1.0, 1.0)
    pcm16 = (pcm16 * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(pcm16.tobytes())
    return path


# ---------------------------------------------------------------------------
# Whisper transcription ------------------------------------------------------
# ---------------------------------------------------------------------------

def transcribe(path: Path) -> str:
    """Return Whisper transcription of the WAV at *path* (blocking)."""
    if _whisper_model is None:
        raise RuntimeError(
            "Whisper model not loaded. Install 'openai-whisper' or check loading errors."
        )
    # Transcribe using the globally loaded local model
    result = _whisper_model.transcribe(str(path))
    return result["text"].strip()


# ---------------------------------------------------------------------------
# Virtual keyboard typing ----------------------------------------------------
# ---------------------------------------------------------------------------

def type_text(text: str) -> None:  # noqa: D401 – imperative mood
    keyboard.Controller().type(text)


# ---------------------------------------------------------------------------
# Main hot‑key loop ----------------------------------------------------------
# ---------------------------------------------------------------------------

def main() -> None:  # noqa: D401
    print(
        textwrap.dedent(
            f"""
            Press and hold {HOTKEY} to speak; release to insert text.
            Press {QUIT_KEY} to quit.  (Backend = {'sounddevice' if _sounddevice_available else 'ffmpeg'})
            """.strip()
        )
    )

    try:
        rec: Recorder = get_recorder()
    except RuntimeError as exc:
        print("Fatal: " + str(exc), file=sys.stderr)
        sys.exit(1)

    pressed = False

    def on_press(key):  # noqa: D401
        nonlocal pressed
        if key == HOTKEY and not pressed:
            pressed = True
            rec.start()
            print("🎙️  Recording… (release to transcribe)")

    def on_release(key):  # noqa: D401
        nonlocal pressed
        if key == HOTKEY and pressed:
            pressed = False
            print("⏹️  Stopped. Transcribing…")
            audio = rec.stop()
            if audio.size == 0:
                print("(No audio captured)")
                return
            wav_path = _save_wav(audio, SAMPLE_RATE)
            try:
                text = transcribe(wav_path)
            except Exception as exc:  # noqa: BLE001
                print("Transcription failed: " + str(exc), file=sys.stderr)
            else:
                print("→", text)
                type_text(text)
            finally:
                wav_path.unlink(missing_ok=True)

        elif key == QUIT_KEY:
            print("Exiting…")
            return False  # stops pynput listener

    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()


# ---------------------------------------------------------------------------
# Test suite (python -m pytest this_file.py::test_suite) ----------------------
# ---------------------------------------------------------------------------

def test__wav_to_float32_and_save_roundtrip(tmp_path):
    """_save_wav followed by _wav_to_float32 should round‑trip ≈ signal."""
    sig = np.sin(np.linspace(0, np.pi * 2, SAMPLE_RATE)).reshape(-1, 1).astype(np.float32)
    saved = _save_wav(sig, SAMPLE_RATE)
    try:
        recovered = _wav_to_float32(saved)
        assert recovered.shape == sig.shape
        assert np.allclose(recovered, sig, atol=1e-3)
    finally:
        saved.unlink(missing_ok=True)


def test_get_recorder_selects_backend(monkeypatch):
    """Ensure correct recorder class is picked depending on availability flags."""

    monkeypatch.setattr(sys.modules[__name__], "_sounddevice_available", True, raising=False)
    assert isinstance(get_recorder(), _SounddeviceRecorder)

    monkeypatch.setattr(sys.modules[__name__], "_sounddevice_available", False, raising=False)
    monkeypatch.setattr(shutil, "which", lambda x: "/usr/bin/ffmpeg" if x == "ffmpeg" else None)
    assert isinstance(get_recorder(), _FFmpegRecorder)


# When run directly ----------------------------------------------------------
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
