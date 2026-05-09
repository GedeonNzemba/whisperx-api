"""Pluggable TTS backend for /ws/s2s.

Phase 1 ships with **Chatterbox-Turbo** (Resemble AI, MIT license, 350M params,
English-only, paralinguistic tags, ~200ms latency on RTX 4090). Selected via
``TTS_BACKEND=chatterbox-turbo`` (default).

Future multilingual support will use **Qwen3-TTS 0.6B**
(``Qwen/Qwen3-TTS-12Hz-0.6B-Base``, Apache 2.0). Selected via
``TTS_BACKEND=qwen3-tts``. The Qwen3 backend is currently a stub that raises
``NotImplementedError`` so a deployment misconfig is loud rather than silent.

Why we *don't* ship Voxtral TTS: its CC-BY-NC 4.0 license forbids commercial
use; incompatible with our monetisation plan.

Common contract – every backend implements:

* ``sample_rate: int`` – output PCM sample rate
* ``warmup() -> None`` – load model + run a tiny dummy synth
* ``synthesize(text: str) -> np.ndarray`` – returns float32 mono in [-1, 1]
* ``language: str`` – ISO-639-1 of the produced audio (e.g. "en")

Reference voice clip: Chatterbox does zero-shot voice cloning from a 5-30s
mono WAV passed as ``audio_prompt_path``. Set ``TTS_REFERENCE_VOICE`` to a
file path; if unset, the model uses its built-in default voice.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Optional, Protocol

import numpy as np

logger = logging.getLogger("s2s.tts")

TTS_BACKEND = os.environ.get("TTS_BACKEND", "chatterbox-turbo").strip().lower() or "chatterbox-turbo"
TTS_REFERENCE_VOICE = os.environ.get("TTS_REFERENCE_VOICE", "").strip() or None


class TTSBackend(Protocol):
    """Structural type every backend must satisfy."""

    sample_rate: int
    language: str

    def warmup(self) -> None: ...

    def synthesize(self, text: str) -> np.ndarray: ...


# ---------------------------------------------------------------------------
# Chatterbox-Turbo (English, MIT)
# ---------------------------------------------------------------------------


class ChatterboxTurboBackend:
    """Resemble AI Chatterbox-Turbo. English only.

    The model itself lives in the ``chatterbox-tts`` PyPI package which
    downloads weights from ``ResembleAI/chatterbox-turbo`` on first load
    (cached under ``~/.cache/huggingface``).
    """

    sample_rate: int = 24_000  # populated for real once model loads
    language: str = "en"

    def __init__(
        self,
        device: Optional[str] = None,
        reference_voice: Optional[str] = None,
    ) -> None:
        self._device = device
        self._reference_voice = reference_voice or TTS_REFERENCE_VOICE
        if self._reference_voice and not Path(self._reference_voice).is_file():
            logger.warning(
                "TTS_REFERENCE_VOICE=%s does not exist on disk — falling "
                "back to model default voice.",
                self._reference_voice,
            )
            self._reference_voice = None
        self._model = None
        self._lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            import torch
            try:
                from chatterbox.tts_turbo import ChatterboxTurboTTS
            except ImportError as exc:
                raise RuntimeError(
                    "chatterbox-tts package not installed. "
                    "Run: pip install chatterbox-tts"
                ) from exc

            device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
            logger.info("Loading Chatterbox-Turbo on %s", device)
            model = ChatterboxTurboTTS.from_pretrained(device=device)
            self._model = model
            self._device = device
            try:
                self.sample_rate = int(model.sr)
            except Exception:  # noqa: BLE001
                pass
            logger.info("Chatterbox-Turbo ready on %s (sr=%d)", device, self.sample_rate)

    def warmup(self) -> None:
        try:
            self._ensure_loaded()
            _ = self.synthesize("Hello.")
            logger.info("Chatterbox-Turbo warm-up complete.")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Chatterbox-Turbo warm-up failed: %s", exc)

    def synthesize(self, text: str) -> np.ndarray:
        text = (text or "").strip()
        if not text:
            return np.zeros(0, dtype=np.float32)
        self._ensure_loaded()
        kwargs = {}
        if self._reference_voice:
            kwargs["audio_prompt_path"] = self._reference_voice
        # Returns a torch.Tensor shaped [1, T] (mono) at self.sample_rate.
        with self._tts_lock():
            wav = self._model.generate(text, **kwargs)
        # Squeeze + cast to numpy float32 for direct PCM streaming.
        try:
            arr = wav.detach().to("cpu").numpy()
        except Exception:  # noqa: BLE001
            arr = np.asarray(wav)
        if arr.ndim > 1:
            arr = arr.squeeze()
        return np.ascontiguousarray(arr, dtype=np.float32)

    def _tts_lock(self):
        # Chatterbox is not thread-safe for concurrent generate() calls on
        # the same module; serialise here. The TTS itself is fast enough
        # that this rarely blocks for long.
        return self._lock


# ---------------------------------------------------------------------------
# Qwen3-TTS 0.6B (multilingual, Apache 2.0) – future
# ---------------------------------------------------------------------------


class Qwen3TTSBackend:
    """Stub for Qwen3-TTS 0.6B multilingual TTS.

    Wired but not implemented in Phase 1. Keeps the env-var contract live so
    downstream config can target it once the integration lands.
    """

    sample_rate: int = 24_000
    language: str = "multi"

    def __init__(self, *args, **kwargs) -> None:  # noqa: D401, ANN002
        raise NotImplementedError(
            "TTS_BACKEND=qwen3-tts is not yet implemented. "
            "Use TTS_BACKEND=chatterbox-turbo for now."
        )

    def warmup(self) -> None:  # pragma: no cover
        raise NotImplementedError

    def synthesize(self, text: str) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Factory + module singleton
# ---------------------------------------------------------------------------


_singleton: Optional[TTSBackend] = None
_singleton_lock = threading.Lock()


def _build(name: str) -> TTSBackend:
    name = (name or "").strip().lower()
    if name in {"chatterbox-turbo", "chatterbox", ""}:
        return ChatterboxTurboBackend()
    if name == "qwen3-tts":
        return Qwen3TTSBackend()
    raise ValueError(
        f"Unknown TTS_BACKEND={name!r}. "
        f"Supported: chatterbox-turbo, qwen3-tts."
    )


def get_tts() -> TTSBackend:
    """Return the process-wide TTS singleton (constructs on first call)."""
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = _build(TTS_BACKEND)
    return _singleton


def is_available() -> bool:
    """Lightweight check (does NOT load weights). True when env-var maps to
    a non-stub backend AND the underlying package is import-able."""
    name = (TTS_BACKEND or "chatterbox-turbo").lower()
    if name in {"chatterbox-turbo", "chatterbox"}:
        try:
            import chatterbox.tts_turbo  # noqa: F401
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "chatterbox.tts_turbo import failed: %s: %s",
                type(exc).__name__, exc,
            )
            return False
    return False  # qwen3-tts stub not available


def availability_error() -> str:
    """Return the import error string if is_available() is False, else ''."""
    name = (TTS_BACKEND or "chatterbox-turbo").lower()
    if name in {"chatterbox-turbo", "chatterbox"}:
        try:
            import chatterbox.tts_turbo  # noqa: F401
            return ""
        except Exception as exc:  # noqa: BLE001
            return f"{type(exc).__name__}: {exc}"
    return "qwen3-tts backend not yet implemented"
