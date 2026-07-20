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

# ---------------------------------------------------------------------------
# Voice-design presets (OmniVoice ``instruct`` strings)
# ---------------------------------------------------------------------------
# Each preset id maps to a valid OmniVoice voice-design instruction. Valid
# vocabulary (docs/voice-design.md): gender=male|female; age=child|teenager|
# young adult|middle-aged|elderly; pitch=very low|low|moderate|high|very high
# pitch; style=whisper. (Accents are English-only and intentionally omitted so
# the presets stay neutral across the African/other target languages.)
VOICE_PRESETS: dict[str, str] = {
    "male_warm":      "male, middle-aged, moderate pitch",
    "male_deep":      "male, middle-aged, low pitch",
    "male_elderly":   "male, elderly, low pitch",
    "male_young":     "male, young adult, moderate pitch",
    "female_clear":   "female, young adult, moderate pitch",
    "female_bright":  "female, young adult, high pitch",
    "female_warm":    "female, middle-aged, moderate pitch",
    "female_elderly": "female, elderly, low pitch",
}
DEFAULT_VOICE = os.environ.get("TTS_DEFAULT_VOICE", "male_warm").strip() or "male_warm"


def voice_to_instruct(voice: Optional[str]) -> Optional[str]:
    """Map a preset id (or a raw instruct string) to an OmniVoice instruct.
    Unknown/blank -> the default preset. A value already containing a comma is
    treated as a raw instruct and passed through."""
    if voice and "," in voice:
        return voice  # caller supplied a raw instruct string
    return VOICE_PRESETS.get((voice or DEFAULT_VOICE), VOICE_PRESETS.get(DEFAULT_VOICE))


# ISO-639-1 -> ISO-639-3 for OmniVoice's ``language`` arg. Covers the S2S target
# set (major + African). Extend as needed; unknown codes fall through to None
# (OmniVoice then auto-detects from the text).
ISO1_TO_ISO3_TTS: dict[str, str] = {
    "en": "eng", "fr": "fra", "es": "spa", "de": "deu", "it": "ita", "pt": "por",
    "nl": "nld", "ru": "rus", "uk": "ukr", "pl": "pol", "ar": "arb", "he": "heb",
    "hi": "hin", "ja": "jpn", "ko": "kor", "zh": "cmn", "tr": "tur", "vi": "vie",
    # African languages (OmniVoice-supported)
    "sw": "swh", "yo": "yor", "ha": "hau", "ig": "ibo", "ln": "lin", "zu": "zul",
    "sn": "sna", "am": "amh", "wo": "wol", "tw": "twi",
}


def iso1_to_iso3_tts(code: Optional[str]) -> Optional[str]:
    if not code:
        return None
    code = code.strip().lower()
    if len(code) == 3:
        return code  # already ISO-639-3
    return ISO1_TO_ISO3_TTS.get(code)


class TTSBackend(Protocol):
    """Structural type every backend must satisfy."""

    sample_rate: int
    language: str        # ISO-639-1 of the fixed output lang, or "multi"
    multilingual: bool   # True if synthesize honours the `language` argument

    def warmup(self) -> None: ...

    def synthesize(self, text: str, *, language: Optional[str] = None,
                   voice: Optional[str] = None) -> np.ndarray: ...


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
    multilingual: bool = False

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

    def synthesize(self, text: str, *, language: Optional[str] = None,
                   voice: Optional[str] = None) -> np.ndarray:
        # English-only backend: `language` and `voice` are ignored.
        text = (text or "").strip()
        if not text:
            return np.zeros(0, dtype=np.float32)
        self._ensure_loaded()
        kwargs = {}
        # Re-check on every call: an operator may have provisioned the
        # reference clip after process start (e.g. mounted a volume).
        if self._reference_voice and Path(self._reference_voice).is_file():
            kwargs["audio_prompt_path"] = self._reference_voice
        elif self._reference_voice:
            logger.warning(
                "Reference voice %s missing at synth time — using default voice",
                self._reference_voice,
            )

        with self._tts_lock():
            try:
                wav = self._model.generate(text, **kwargs)
            except Exception as exc:  # noqa: BLE001
                # Most common failure: malformed/short reference clip. Retry
                # once without the prompt rather than failing the whole
                # /ws/s2s segment.
                if "audio_prompt_path" in kwargs:
                    logger.warning(
                        "Chatterbox generate() failed with reference clip "
                        "(%s: %s) — retrying with default voice",
                        type(exc).__name__, exc,
                    )
                    wav = self._model.generate(text)
                else:
                    raise
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
# OmniVoice (multilingual, Apache 2.0) — via isolated sidecar
# ---------------------------------------------------------------------------


class OmniVoiceBackend:
    """Multilingual TTS backed by the isolated OmniVoice sidecar.

    k2-fsa/OmniVoice — Apache 2.0, 646 languages including African (Lingala,
    Yoruba, Hausa, Igbo, Swahili, Zulu, Amharic, Twi, Wolof, Shona). Voice
    selection via voice-design presets (see ``VOICE_PRESETS``). The heavy model
    (transformers 5.x) runs in a separate venv; we only talk HTTP to it, so the
    main server's whisperx/ctranslate2 stack is untouched.
    """

    sample_rate: int = 24_000
    language: str = "multi"
    multilingual: bool = True

    def __init__(self) -> None:
        import omnivoice_client  # main-venv client; no transformers import

        self._client = omnivoice_client

    def warmup(self) -> None:
        # Raises on failure so the server's lifespan records the warm-up error
        # and /ws/s2s rejects connections until the sidecar is genuinely ready.
        if not self._client.ensure_sidecar_running():
            err = (self._client.health() or {}).get("error")
            raise RuntimeError(
                f"OmniVoice sidecar failed to become ready"
                + (f": {err}" if err else "")
            )
        h = self._client.health() or {}
        try:
            self.sample_rate = int(h.get("sample_rate", 24_000))
        except Exception:  # noqa: BLE001
            pass
        logger.info("OmniVoice sidecar ready (model=%s)", h.get("model"))

    def synthesize(self, text: str, *, language: Optional[str] = None,
                   voice: Optional[str] = None) -> np.ndarray:
        text = (text or "").strip()
        if not text:
            return np.zeros(0, dtype=np.float32)
        iso3 = iso1_to_iso3_tts(language)
        instruct = voice_to_instruct(voice)
        return self._client.synthesize(text, language=iso3, instruct=instruct)

    def supports_language(self, iso1: Optional[str]) -> bool:
        """OmniVoice covers 646 languages; we report support for anything we can
        map to an ISO-639-3 code (or any 3-letter code passed through)."""
        return iso1_to_iso3_tts(iso1) is not None


# ---------------------------------------------------------------------------
# Factory + module singleton
# ---------------------------------------------------------------------------


_singleton: Optional[TTSBackend] = None
_singleton_lock = threading.Lock()


def _build(name: str) -> TTSBackend:
    name = (name or "").strip().lower()
    if name in {"chatterbox-turbo", "chatterbox", ""}:
        return ChatterboxTurboBackend()
    if name in {"omnivoice", "omni"}:
        return OmniVoiceBackend()
    if name == "qwen3-tts":
        return Qwen3TTSBackend()
    raise ValueError(
        f"Unknown TTS_BACKEND={name!r}. "
        f"Supported: chatterbox-turbo, omnivoice, qwen3-tts."
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
    a usable backend.

    * chatterbox: the package must import in THIS venv.
    * omnivoice: the sidecar runs in a SEPARATE venv, so we don't import
      anything here — readiness is probed live via the sidecar /health and
      reported through availability_error().
    """
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
    if name in {"omnivoice", "omni"}:
        try:
            import omnivoice_client
            return omnivoice_client.is_available(timeout=2.0)
        except Exception as exc:  # noqa: BLE001
            logger.error("omnivoice_client probe failed: %s", exc)
            return False
    return False  # qwen3-tts stub not available


def availability_error() -> str:
    """Return the unavailability reason if is_available() is False, else ''."""
    name = (TTS_BACKEND or "chatterbox-turbo").lower()
    if name in {"chatterbox-turbo", "chatterbox"}:
        try:
            import chatterbox.tts_turbo  # noqa: F401
            return ""
        except Exception as exc:  # noqa: BLE001
            return f"{type(exc).__name__}: {exc}"
    if name in {"omnivoice", "omni"}:
        try:
            import omnivoice_client
            if omnivoice_client.is_available(timeout=2.0):
                return ""
            h = omnivoice_client.health()
            return (h or {}).get("error") or "OmniVoice sidecar not ready (still loading or failed to start)."
        except Exception as exc:  # noqa: BLE001
            return f"{type(exc).__name__}: {exc}"
    return "qwen3-tts backend not yet implemented"
