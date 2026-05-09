"""
WhisperX Large V3 API Server
============================

Production-ready FastAPI server exposing the full WhisperX pipeline:
- Whisper Large V3 transcription via faster-whisper
- Forced word-level alignment (wav2vec2)
- Silero VAD
- pyannote.audio speaker diarization (3.1)

Single-file server. Run with:
    python server.py
or:
    uvicorn server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import base64
import gc
import json
import logging
import os
import re
import shutil
import tempfile
import threading
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# torch 2.6 changed `torch.load` to default `weights_only=True`. Many trusted
# checkpoints (whisperx/pyannote/chatterbox) use omegaconf containers which
# aren't in the default allowlist. Whitelist them explicitly so loading works.
import torch.serialization  # noqa: E402
try:
    from omegaconf.listconfig import ListConfig
    from omegaconf.dictconfig import DictConfig
    from omegaconf import OmegaConf
    torch.serialization.add_safe_globals([ListConfig, DictConfig, OmegaConf])
except Exception:  # noqa: BLE001
    pass

# Belt-and-braces: also patch torch.load default to weights_only=False, since
# some checkpoints use globals beyond just omegaconf. We trust all sources
# (HuggingFace + bundled models).
_torch_load_orig = torch.load
def _torch_load_patched(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _torch_load_orig(*args, **kwargs)
torch.load = _torch_load_patched

import whisperx
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

try:  # Optional language detection for /align
    from langdetect import detect as _langdetect_detect  # type: ignore
    _LANGDETECT_OK = True
except Exception:  # noqa: BLE001
    _LANGDETECT_OK = False

# Optional: Apple MLX backend for fast transcription on Apple Silicon.
try:
    import platform as _platform
    if _platform.system() == "Darwin" and _platform.machine() == "arm64":
        from lightning_whisper_mlx import LightningWhisperMLX  # type: ignore
        _MLX_AVAILABLE = True
    else:
        _MLX_AVAILABLE = False
except Exception:  # noqa: BLE001
    _MLX_AVAILABLE = False

# Optional: HuggingFace Spaces ZeroGPU decorator (no-op outside Spaces).
try:
    import spaces  # type: ignore
    _SPACES_AVAILABLE = True
    def _gpu_decorator(duration: int = 120):
        return spaces.GPU(duration=duration)
except Exception:  # noqa: BLE001
    _SPACES_AVAILABLE = False
    def _gpu_decorator(duration: int = 120):
        def _wrap(fn):
            return fn
        return _wrap

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("whisperx-server")

PORT = int(os.environ.get("PORT", "8000"))
HOST = os.environ.get("HOST", "0.0.0.0")
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip() or None
MODEL_DIR = os.environ.get("MODEL_DIR", "./models")
MAX_AUDIO_DURATION = int(os.environ.get("MAX_AUDIO_DURATION", "86400"))  # seconds (default 24h)
MAX_FILE_SIZE = int(os.environ.get("MAX_FILE_SIZE", str(500 * 1024 * 1024)))  # bytes
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "large-v3")
COMPUTE_TYPE = os.environ.get("COMPUTE_TYPE", "float16")
BEAM_SIZE = int(os.environ.get("BEAM_SIZE", "5"))
DOWNLOAD_TTL_SECONDS = int(os.environ.get("DOWNLOAD_TTL_SECONDS", "3600"))
DIARIZATION_MODEL = os.environ.get("DIARIZATION_MODEL", "pyannote/speaker-diarization-3.2")
DIARIZATION_MODEL_FALLBACK = "pyannote/speaker-diarization-3.1"
DIARIZATION_DOMINANCE_THRESHOLD = float(os.environ.get("DIARIZATION_DOMINANCE_THRESHOLD", "0.80"))
# Tier-2 (VBx) trigger: if the Tier-1 backend produces a result where one
# speaker dominates more than this fraction of total speech AND the user
# expects ≥ 2 speakers, we re-cluster with the vendored VBx VB-HMM (see
# vbx_diarize.py). The HMM temporal prior fixes AHC's "long gap → merge"
# failure that the FoxNoseTech `diarize` library inherits from its
# clustering stage. Set VBX_ENABLED=0 to disable.
VBX_DOMINANCE_THRESHOLD = float(os.environ.get("VBX_DOMINANCE_THRESHOLD", "0.70"))
VBX_ENABLED = os.environ.get("VBX_ENABLED", "1").strip().lower() not in {"0", "false", "no"}
# Diarization backend selection:
#   "diarize"  → CPU-only FoxNoseTech/diarize (default; no HF token, ~4.8% DER)
#   "pyannote" → classic pyannote.audio path (requires HF_TOKEN)
#   "auto"     → diarize first; fall back to pyannote on dominance failure
# Backward compat: empty/unknown values are treated as "auto".
DIARIZATION_BACKEND = os.environ.get("DIARIZATION_BACKEND", "auto").strip().lower() or "auto"
if DIARIZATION_BACKEND not in {"diarize", "pyannote", "auto", "vibevoice"}:
    DIARIZATION_BACKEND = "auto"

# ---------------------------------------------------------------------------
# Speech-to-speech translation (Phase 1, instructions_1.txt)
# ---------------------------------------------------------------------------
# Set S2S_ENABLED=1 to enable the /ws/s2s endpoint and preload TTS + MT
# models alongside Whisper. Disabled by default to keep VRAM use minimal
# for deployments that don't need translation.
S2S_ENABLED = os.environ.get("S2S_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}
# Default target language for S2S sessions that don't specify one.
S2S_TARGET_LANG_DEFAULT = os.environ.get("S2S_TARGET_LANG_DEFAULT", "fr").strip().lower() or "fr"

# ASR backend selection: "auto" | "mlx" | "whisperx"
#   auto     → MLX on Apple Silicon, whisperx everywhere else
#   mlx      → force lightning-whisper-mlx (Apple Silicon only)
#   whisperx → force faster-whisper / CTranslate2 backend
ASR_BACKEND = os.environ.get("ASR_BACKEND", "auto").strip().lower()
# MLX-specific model name (defaults to a quantized large-v3 variant)
MLX_MODEL = os.environ.get("MLX_MODEL", "distil-large-v3")
MLX_QUANT = os.environ.get("MLX_QUANT", "").strip() or None  # e.g. "4bit"

DOWNLOAD_DIR = Path(tempfile.gettempdir()) / "whisperx_downloads"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    # Safer defaults for CPU fallback (mainly useful for development).
    COMPUTE_TYPE = os.environ.get("COMPUTE_TYPE_CPU", "int8")
    logger.warning("CUDA not available – running on CPU. Performance will be poor.")

# Resolve the effective ASR backend now that DEVICE is known.
def _resolve_backend(requested: str) -> str:
    if requested == "mlx":
        if not _MLX_AVAILABLE:
            logger.warning("ASR_BACKEND=mlx requested but lightning-whisper-mlx is not available; falling back to whisperx.")
            return "whisperx"
        return "mlx"
    if requested == "whisperx":
        return "whisperx"
    # auto
    if _MLX_AVAILABLE and DEVICE == "cpu":
        # On Apple Silicon (no CUDA) MLX is dramatically faster.
        return "mlx"
    return "whisperx"

EFFECTIVE_BACKEND = _resolve_backend(ASR_BACKEND)
logger.info("ASR backend: %s (requested=%s, mlx_available=%s, device=%s)",
            EFFECTIVE_BACKEND, ASR_BACKEND, _MLX_AVAILABLE, DEVICE)

ALLOWED_AUDIO_SUFFIXES = {
    ".wav", ".wave", ".mp3", ".mp4", ".m4a", ".flac",
    ".ogg", ".oga", ".opus", ".webm", ".aac", ".wma", ".aiff", ".aif",
}

START_TIME = time.time()

# ---------------------------------------------------------------------------
# MLX backend wrapper
# ---------------------------------------------------------------------------

class MLXWhisperBackend:
    """Wraps lightning-whisper-mlx so it exposes the same `transcribe(audio, **kw)`
    contract as the whisperx faster-whisper backend.

    Returns: {"language": str, "segments": [{"start", "end", "text"}, ...]}
    """

    def __init__(self, model_name: str = "distil-large-v3", quant: Optional[str] = None) -> None:
        if not _MLX_AVAILABLE:
            raise RuntimeError("lightning-whisper-mlx is not installed (Apple Silicon only).")
        logger.info("Loading MLX Whisper model '%s' (quant=%s)...", model_name, quant or "none")
        self._impl = LightningWhisperMLX(model=model_name, batch_size=12, quant=quant)
        self.model_name = model_name
        logger.info("MLX Whisper model loaded.")

    def transcribe(self, audio, **kwargs) -> Dict[str, Any]:
        """Transcribe a numpy audio array (16kHz, float32) or a file path."""
        language = kwargs.get("language")
        # MLX requires a file path; if we got a numpy array, write a temp wav.
        tmp_path: Optional[str] = None
        if isinstance(audio, str):
            audio_path = audio
        else:
            import numpy as _np
            import soundfile as _sf
            arr = _np.asarray(audio, dtype=_np.float32)
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav", prefix="mlx_")
            os.close(tmp_fd)
            _sf.write(tmp_path, arr, 16000, subtype="PCM_16")
            audio_path = tmp_path
        try:
            result = self._impl.transcribe(audio_path=audio_path, language=language)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
        # Normalize output to whisperx-compatible shape.
        text = (result.get("text") or "").strip()
        # `segments` is a list of dicts with keys: start, end, text
        raw_segments = result.get("segments") or result.get("chunks") or []
        segments: List[Dict[str, Any]] = []
        for seg in raw_segments:
            if isinstance(seg, dict):
                start = float(seg.get("start") or seg.get("timestamp", [0.0, 0.0])[0] or 0.0)
                end_val = seg.get("end")
                if end_val is None and "timestamp" in seg:
                    end_val = seg["timestamp"][1] if seg["timestamp"][1] is not None else start + 1.0
                segments.append({
                    "start": start,
                    "end": float(end_val or start + 1.0),
                    "text": (seg.get("text") or "").strip(),
                })
            elif isinstance(seg, (list, tuple)) and len(seg) >= 3:
                # Some versions return [start, end, text] tuples.
                segments.append({
                    "start": float(seg[0] or 0.0),
                    "end": float(seg[1] or 0.0),
                    "text": str(seg[2] or "").strip(),
                })
        if not segments and text:
            # Fallback: a single segment covering the whole audio.
            duration = float(len(audio)) / 16000.0 if hasattr(audio, "__len__") else 0.0
            segments = [{"start": 0.0, "end": duration, "text": text}]
        return {
            "language": result.get("language") or language or "en",
            "segments": segments,
        }


# ---------------------------------------------------------------------------
# Model registry (lazy-loaded singletons)
# ---------------------------------------------------------------------------

class ModelRegistry:
    """Holds lazily-loaded model singletons."""

    def __init__(self) -> None:
        self.whisper_model = None
        self.align_models: Dict[str, Tuple[Any, Dict[str, Any]]] = {}
        self.align_lock = threading.Lock()  # guards check-then-set in load_align
        self.diarize_pipeline = None
        self.diarization_error: Optional[str] = None
        self.diarization_model_loaded: Optional[str] = None
        # FoxNoseTech `diarize` library availability flag (lazy import).
        self._diarize_lib = None
        self._diarize_lib_error: Optional[str] = None
        self.tiny_model = None  # Used by /align for chunk discovery
        self.mms_model = None
        self.mms_processor = None
        self.mms_error: Optional[str] = None
        self.mms_loaded_adapters: set = set()
        # Silero VAD for streaming chunk gating; lazy-loaded, fail-open.
        self._silero_vad_fn = None
        self._silero_vad_failed = False
        self._silero_vad_lock = threading.Lock()

    def try_load_silero_vad(self):
        """Returns a callable `(audio_np, min_ms) -> [speech_segments]` or None if unavailable."""
        if self._silero_vad_fn is not None:
            return self._silero_vad_fn
        if self._silero_vad_failed:
            return None
        with self._silero_vad_lock:
            if self._silero_vad_fn is not None:
                return self._silero_vad_fn
            if self._silero_vad_failed:
                return None
            try:
                import torch
                model, utils = torch.hub.load(
                    "snakers4/silero-vad", "silero_vad", trust_repo=True, verbose=False,
                )
                get_speech_timestamps = utils[0]

                def _check(audio_np, min_ms: int = 300):
                    t = torch.from_numpy(audio_np)
                    return get_speech_timestamps(
                        t, model,
                        sampling_rate=16000,
                        min_speech_duration_ms=min_ms,
                    )

                self._silero_vad_fn = _check
                logger.info("Silero VAD loaded for streaming chunk gating.")
                return _check
            except Exception as exc:  # noqa: BLE001
                logger.warning("Silero VAD unavailable, will use energy-only gating: %s", exc)
                self._silero_vad_failed = True
                return None

    def load_whisper(self) -> Any:
        if self.whisper_model is None:
            if EFFECTIVE_BACKEND == "mlx":
                self.whisper_model = MLXWhisperBackend(model_name=MLX_MODEL, quant=MLX_QUANT)
            else:
                logger.info("Loading Whisper model %s on %s (%s)...", WHISPER_MODEL, DEVICE, COMPUTE_TYPE)
                self.whisper_model = whisperx.load_model(
                    WHISPER_MODEL,
                    device=DEVICE,
                    compute_type=COMPUTE_TYPE,
                    download_root=MODEL_DIR,
                    asr_options={"beam_size": BEAM_SIZE},
                )
                logger.info("Whisper model loaded.")
        return self.whisper_model

    def load_tiny(self) -> Any:
        """Lightweight Whisper used by /align purely to discover speech chunks
        and provide approximate word counts per chunk. Falls back to the main
        model only if 'tiny' cannot be loaded."""
        if self.tiny_model is None:
            try:
                if EFFECTIVE_BACKEND == "mlx":
                    logger.info("Loading tiny MLX Whisper model for /align chunking...")
                    self.tiny_model = MLXWhisperBackend(model_name="tiny", quant=None)
                else:
                    logger.info("Loading tiny Whisper model for /align chunking...")
                    self.tiny_model = whisperx.load_model(
                        "tiny",
                        device=DEVICE,
                        compute_type=COMPUTE_TYPE,
                        download_root=MODEL_DIR,
                        asr_options={"beam_size": 1},
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Tiny model load failed (%s); reusing main Whisper model.", exc)
                self.tiny_model = self.load_whisper()
        return self.tiny_model

    def load_align(self, language_code: str) -> Optional[Tuple[Any, Dict[str, Any]]]:
        with self.align_lock:
            if language_code in self.align_models:
                return self.align_models[language_code]
            try:
                logger.info("Loading alignment model for language=%s", language_code)
                model, metadata = whisperx.load_align_model(
                    language_code=language_code,
                    device=DEVICE,
                    model_dir=MODEL_DIR,
                )
                self.align_models[language_code] = (model, metadata)
                return self.align_models[language_code]
            except Exception as exc:  # noqa: BLE001
                logger.warning("Alignment model unavailable for %s: %s", language_code, exc)
                return None

    def load_mms(self) -> Optional[Tuple[Any, Any]]:
        """Load Meta MMS-1B-all (1,107 langs) for universal forced alignment.
        Returns (model, processor) or None on failure."""
        if self.mms_model is not None and self.mms_processor is not None:
            return (self.mms_model, self.mms_processor)
        if self.mms_error:
            return None
        try:
            logger.info("Loading Meta MMS-1B-all forced-alignment model…")
            from transformers import AutoProcessor, Wav2Vec2ForCTC
            model_id = os.environ.get("MMS_MODEL_ID", "facebook/mms-1b-all")
            processor = AutoProcessor.from_pretrained(model_id, cache_dir=MODEL_DIR)
            dtype = torch.float16 if DEVICE == "cuda" else torch.float32
            model = Wav2Vec2ForCTC.from_pretrained(
                model_id,
                cache_dir=MODEL_DIR,
                torch_dtype=dtype,
                target_lang="eng",
                ignore_mismatched_sizes=True,
            ).to(DEVICE)
            model.eval()
            self.mms_model = model
            self.mms_processor = processor
            logger.info("MMS loaded on %s (%s).", DEVICE, dtype)
            return (model, processor)
        except Exception as exc:  # noqa: BLE001
            self.mms_error = f"MMS load failed: {exc}"
            logger.warning(self.mms_error)
            return None

    def load_diarize(self):
        if self.diarize_pipeline is not None:
            return self.diarize_pipeline
        if not HF_TOKEN:
            self.diarization_error = (
                "HF_TOKEN environment variable is not set. "
                "Speaker diarization requires a HuggingFace access token "
                "with access to pyannote/speaker-diarization-3.x."
            )
            return None
        # Try the configured model first, then fall back to 3.1 if 3.2 (or any
        # newer revision) is not yet accepted on the user's HF account or has
        # download issues. Keeps the server usable on older setups.
        candidates: List[str] = [DIARIZATION_MODEL]
        if DIARIZATION_MODEL_FALLBACK and DIARIZATION_MODEL_FALLBACK != DIARIZATION_MODEL:
            candidates.append(DIARIZATION_MODEL_FALLBACK)
        last_err: Optional[str] = None
        for model_name in candidates:
            try:
                logger.info("Loading diarization pipeline %s", model_name)
                self.diarize_pipeline = whisperx.DiarizationPipeline(
                    model_name=model_name,
                    use_auth_token=HF_TOKEN,
                    device=DEVICE,
                )
                self.diarization_model_loaded = model_name
                return self.diarize_pipeline
            except Exception as exc:  # noqa: BLE001
                last_err = f"Failed to load diarization pipeline {model_name}: {exc}"
                logger.warning(last_err)
        self.diarization_error = last_err or "Failed to load any diarization pipeline."
        logger.error(self.diarization_error)
        return None

    def load_diarize_lib(self):
        """Lazy import the FoxNoseTech `diarize` package. Returns the module
        (which exposes the top-level ``diarize(...)`` function) or ``None`` if
        the package is not installed. The library does its own model caching
        so we only need to remember whether the import succeeded.
        """
        if self._diarize_lib is not None:
            return self._diarize_lib
        if self._diarize_lib_error is not None:
            return None
        try:
            import diarize as _diarize_lib  # noqa: WPS433 (lazy import is intentional)
            self._diarize_lib = _diarize_lib
            logger.info(
                "FoxNoseTech diarize library loaded (version %s)",
                getattr(_diarize_lib, "__version__", "unknown"),
            )
            return self._diarize_lib
        except Exception as exc:  # noqa: BLE001
            self._diarize_lib_error = (
                f"`diarize` package not available: {exc}. "
                "Install with `pip install diarize` or set "
                "DIARIZATION_BACKEND=pyannote."
            )
            logger.warning(self._diarize_lib_error)
            return None


registry = ModelRegistry()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _diarize_dataframe_to_records(diarize_segments: Any) -> List[Dict[str, Any]]:
    """Normalise the pyannote/whisperx diarization output into a plain list of
    ``{"start", "end", "speaker"}`` dicts. ``whisperx`` returns a pandas
    DataFrame, while raw pyannote returns an ``Annotation``; this hides the
    difference so the rest of the code stays simple and dependency-free."""
    records: List[Dict[str, Any]] = []
    if diarize_segments is None:
        return records
    # pandas DataFrame path (whisperx wrapper)
    if hasattr(diarize_segments, "iterrows"):
        for _, row in diarize_segments.iterrows():
            try:
                records.append(
                    {
                        "start": float(row["start"]),
                        "end": float(row["end"]),
                        "speaker": str(row["speaker"]),
                    }
                )
            except Exception:  # noqa: BLE001
                continue
        return records
    # pyannote Annotation path
    if hasattr(diarize_segments, "itertracks"):
        for turn, _, speaker in diarize_segments.itertracks(yield_label=True):
            records.append(
                {"start": float(turn.start), "end": float(turn.end), "speaker": str(speaker)}
            )
        return records
    # FoxNoseTech `diarize.DiarizeResult` path: has `.segments` of pydantic
    # ``Segment(start, end, speaker)`` objects. We accept either the result
    # object or the raw list.
    seg_list = getattr(diarize_segments, "segments", None)
    if seg_list is None and isinstance(diarize_segments, list):
        seg_list = diarize_segments
    if seg_list is not None:
        for s in seg_list:
            try:
                records.append(
                    {
                        "start": float(getattr(s, "start", s["start"] if isinstance(s, dict) else 0.0)),
                        "end": float(getattr(s, "end", s["end"] if isinstance(s, dict) else 0.0)),
                        "speaker": str(getattr(s, "speaker", s["speaker"] if isinstance(s, dict) else "")),
                    }
                )
            except Exception:  # noqa: BLE001
                continue
    return records


def _records_to_dataframe(records: List[Dict[str, Any]]):
    """Convert plain speaker records into the pandas DataFrame shape that
    ``whisperx.assign_word_speakers`` expects (columns: start, end, speaker).
    Returns ``None`` if pandas is not importable, in which case callers should
    fall back to the manual nearest-by-time word-labelling pass.
    """
    if not records:
        return None
    try:
        import pandas as _pd  # noqa: WPS433 (lazy import — only needed here)
    except Exception:  # noqa: BLE001
        return None
    return _pd.DataFrame(
        [{"start": r["start"], "end": r["end"], "speaker": r["speaker"]} for r in records]
    )


def _run_diarize_lib(
    audio_path: Path,
    min_speakers: Optional[int],
    max_speakers: Optional[int],
    diarize_module: Any,
) -> List[Dict[str, Any]]:
    """Run the FoxNoseTech `diarize` library and return normalised records.

    The library API:
        diarize(path, *, num_speakers=None, min_speakers=1, max_speakers=20)
    Returns a ``DiarizeResult`` with ``.segments`` (list of ``Segment``).

    If ``min_speakers == max_speakers`` we collapse to ``num_speakers`` for
    the strictest possible clustering; otherwise we pass through the bounds.
    """
    kwargs: Dict[str, Any] = {}
    if (
        min_speakers is not None
        and max_speakers is not None
        and min_speakers == max_speakers
        and min_speakers > 0
    ):
        kwargs["num_speakers"] = int(min_speakers)
    else:
        if min_speakers is not None and min_speakers > 0:
            kwargs["min_speakers"] = int(min_speakers)
        if max_speakers is not None and max_speakers > 0:
            kwargs["max_speakers"] = int(max_speakers)
    result = diarize_module.diarize(str(audio_path), **kwargs)
    return _diarize_dataframe_to_records(result)


def _dominant_speaker_ratio(records: List[Dict[str, Any]]) -> Tuple[Optional[str], float, float]:
    """Return (speaker_id, dominance_ratio, total_speech_seconds).

    Dominance ratio is ``max_speaker_speech / total_speech`` in [0, 1].
    Used both for re-split heuristics and for the `diarization_confidence`
    score surfaced to the UI.
    """
    totals: Dict[str, float] = {}
    total = 0.0
    for r in records:
        dur = max((r.get("end") or 0.0) - (r.get("start") or 0.0), 0.0)
        if dur <= 0:
            continue
        spk = r.get("speaker") or ""
        totals[spk] = totals.get(spk, 0.0) + dur
        total += dur
    if not totals or total <= 0:
        return None, 0.0, 0.0
    top_spk, top_dur = max(totals.items(), key=lambda kv: kv[1])
    return top_spk, top_dur / total, total


def _tune_clustering_threshold(pipeline: Any, threshold: Optional[float]) -> Optional[float]:
    """Try to set ``clustering.threshold`` on the underlying pyannote
    SpeakerDiarization pipeline. Returns the previous value if known, else
    None. Best-effort — silently no-ops if the wrapped pipeline doesn't
    expose the expected attribute path (e.g. older pyannote versions)."""
    try:
        inner = getattr(pipeline, "model", None) or pipeline
        # pyannote 3.x stores the configured params on the pipeline instance
        prev: Optional[float] = None
        clustering = getattr(inner, "clustering", None)
        if clustering is not None and hasattr(clustering, "threshold"):
            prev = float(getattr(clustering, "threshold"))
            if threshold is not None:
                setattr(clustering, "threshold", float(threshold))
        return prev
    except Exception:  # noqa: BLE001
        return None


def free_gpu_memory() -> None:
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


def gpu_memory_info() -> Dict[str, Any]:
    if DEVICE != "cuda":
        return {"device": "cpu"}
    info: Dict[str, Any] = {"device": "cuda", "devices": []}
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        info["devices"].append(
            {
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "total_mb": round(total / 1024 / 1024, 1),
                "used_mb": round((total - free) / 1024 / 1024, 1),
                "free_mb": round(free / 1024 / 1024, 1),
            }
        )
    return info


def select_batch_size(duration_seconds: float) -> int:
    if duration_seconds < 5 * 60:
        return 16
    if duration_seconds < 30 * 60:
        return 8
    return 4


def parse_output_formats(raw: Optional[str]) -> List[str]:
    if not raw:
        return ["json"]
    out = [p.strip().lower() for p in raw.split(",") if p.strip()]
    valid = {"txt", "srt", "vtt", "tsv", "json"}
    bad = [f for f in out if f not in valid]
    if bad:
        raise HTTPException(400, f"Invalid output_format(s): {bad}. Valid: {sorted(valid)}")
    if "json" not in out:
        out.append("json")  # Always include JSON envelope
    return out


def srt_timestamp(seconds: float) -> str:
    if seconds is None or seconds < 0:
        seconds = 0.0
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def vtt_timestamp(seconds: float) -> str:
    return srt_timestamp(seconds).replace(",", ".")


def render_txt(segments: List[Dict[str, Any]], diarized: bool) -> str:
    lines: List[str] = []
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        speaker = seg.get("speaker")
        if diarized and speaker:
            lines.append(f"[{speaker}]: {text}")
        else:
            lines.append(text)
    return "\n".join(lines) + ("\n" if lines else "")


def render_srt(segments: List[Dict[str, Any]], diarized: bool) -> str:
    out: List[str] = []
    idx = 1
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        start = seg.get("start", 0.0) or 0.0
        end = seg.get("end", start + 1.0) or (start + 1.0)
        speaker = seg.get("speaker")
        cue = f"{speaker}\n{text}" if diarized and speaker else text
        out.append(f"{idx}\n{srt_timestamp(start)} --> {srt_timestamp(end)}\n{cue}\n")
        idx += 1
    return "\n".join(out)


def render_vtt(segments: List[Dict[str, Any]], diarized: bool) -> str:
    out: List[str] = ["WEBVTT", ""]
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        start = seg.get("start", 0.0) or 0.0
        end = seg.get("end", start + 1.0) or (start + 1.0)
        speaker = seg.get("speaker")
        cue = f"{speaker}\n{text}" if diarized and speaker else text
        out.append(f"{vtt_timestamp(start)} --> {vtt_timestamp(end)}")
        out.append(cue)
        out.append("")
    return "\n".join(out)


def render_tsv(segments: List[Dict[str, Any]], diarized: bool) -> str:
    """Tab-separated: start_ms  end_ms  [speaker  ]text — whisper-native TSV format."""
    rows: List[str] = []
    header = "start\tend\tspeaker\ttext" if diarized else "start\tend\ttext"
    rows.append(header)
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        start_ms = int(round((seg.get("start") or 0.0) * 1000))
        end_ms = int(round((seg.get("end") or 0.0) * 1000))
        if diarized:
            speaker = seg.get("speaker") or ""
            rows.append(f"{start_ms}\t{end_ms}\t{speaker}\t{text}")
        else:
            rows.append(f"{start_ms}\t{end_ms}\t{text}")
    return "\n".join(rows) + "\n"


def apply_speaker_names(
    segments: List[Dict[str, Any]],
    word_segments: List[Dict[str, Any]],
    mapping: Dict[str, str],
) -> None:
    """Rename speaker labels in-place using the provided mapping dict.
    e.g. {"SPEAKER_00": "Pastor John", "SPEAKER_01": "Congregation"}
    Unknown speaker labels are left unchanged.
    """
    for seg in segments:
        spk = seg.get("speaker")
        if spk and spk in mapping:
            seg["speaker"] = mapping[spk]
        for w in seg.get("words", []) or []:
            wspk = w.get("speaker")
            if wspk and wspk in mapping:
                w["speaker"] = mapping[wspk]
    for w in word_segments:
        wspk = w.get("speaker")
        if wspk and wspk in mapping:
            w["speaker"] = mapping[wspk]


def parse_speaker_names(raw: Optional[str]) -> Optional[Dict[str, str]]:
    """Parse speaker_names JSON string. Returns None if not provided. Raises 400 on bad JSON."""
    if not raw:
        return None
    try:
        mapping = json.loads(raw)
        if not isinstance(mapping, dict):
            raise ValueError("speaker_names must be a JSON object")
        return {str(k): str(v) for k, v in mapping.items()}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"Invalid speaker_names JSON: {exc}") from exc


def cleanup_old_downloads() -> None:
    cutoff = time.time() - DOWNLOAD_TTL_SECONDS
    for child in DOWNLOAD_DIR.iterdir():
        try:
            if child.is_dir() and child.stat().st_mtime < cutoff:
                shutil.rmtree(child, ignore_errors=True)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(_: FastAPI):
    runpod_pod_id = os.environ.get("RUNPOD_POD_ID")
    if runpod_pod_id:
        logger.info("Running on RunPod — POD_ID=%s", runpod_pod_id)
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    try:
        registry.load_whisper()
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to preload Whisper model: %s", exc)
    # Preload diarization backends — diarize lib is preferred when available.
    if DIARIZATION_BACKEND in {"diarize", "auto"}:
        try:
            if registry.load_diarize_lib() is not None:
                logger.info("Diarization backend `diarize` is available.")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to probe diarize library: %s", exc)
    if HF_TOKEN and DIARIZATION_BACKEND in {"pyannote", "auto"}:
        try:
            registry.load_diarize()
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to preload pyannote diarization pipeline: %s", exc)
    elif not HF_TOKEN and registry._diarize_lib is None and DIARIZATION_BACKEND != "diarize":
        logger.warning(
            "HF_TOKEN not set and `diarize` package not installed – "
            "speaker diarization will be disabled. "
            "Run `pip install diarize` to enable the CPU backend."
        )

    # VibeVoice sidecar — isolated venv, GPU-only, optional. We always need
    # pyannote loaded as a fallback (handled above), so this is purely
    # additive.
    if DIARIZATION_BACKEND == "vibevoice":
        try:
            import vibevoice_client

            ok = vibevoice_client.ensure_sidecar_running()
            if ok:
                logger.info("VibeVoice sidecar is up — /session/diarize will use it")
            else:
                logger.warning(
                    "VibeVoice sidecar not available — /session/diarize will fall back "
                    "to pyannote (or `diarize` if pyannote is also unavailable)"
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("VibeVoice sidecar bootstrap failed: %s", exc)

    # Speech-to-speech (Phase 1) – preload NLLB + Chatterbox-Turbo so the
    # first /ws/s2s segment doesn't pay the ~5-30 s cold-start cost.
    if S2S_ENABLED:
        try:
            from s2s import translator as _s2s_translator
            from s2s import tts as _s2s_tts

            logger.info("S2S enabled — warming up MT (NLLB) and TTS (%s)", _s2s_tts.TTS_BACKEND)
            # Run warm-ups in background so server boot isn't blocked.
            def _s2s_warmup_async() -> None:
                try:
                    _s2s_translator.get_translator().warmup()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("S2S translator warm-up error: %s", exc)
                try:
                    _s2s_tts.get_tts().warmup()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("S2S TTS warm-up error: %s", exc)

            threading.Thread(target=_s2s_warmup_async, daemon=True, name="s2s-warmup").start()
        except Exception as exc:  # noqa: BLE001
            logger.warning("S2S preload failed: %s", exc)

    yield
    free_gpu_memory()


app = FastAPI(
    title="WhisperX Large V3 API",
    description="High-accuracy speech-to-text with word-level timestamps and speaker diarization.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the static web UI if the static/ directory exists.
_STATIC_DIR = Path(__file__).parent / "static"
if _STATIC_DIR.is_dir():
    app.mount("/ui", StaticFiles(directory=str(_STATIC_DIR), html=True), name="ui")


@app.get("/", include_in_schema=False)
def root():
    index = _STATIC_DIR / "index.html"
    if index.is_file():
        return HTMLResponse(index.read_text(encoding="utf-8"))
    return JSONResponse({
        "service": "WhisperX API",
        "endpoints": ["/health", "/transcribe", "/session/diarize", "/align", "/download/{file_id}/{name}"],
    })


@app.middleware("http")
async def access_log(request: Request, call_next):
    started = time.time()
    response = await call_next(request)
    elapsed = (time.time() - started) * 1000
    logger.info("%s %s -> %s (%.1fms)", request.method, request.url.path, response.status_code, elapsed)
    return response


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

def _vibevoice_health_snapshot() -> Optional[Dict[str, Any]]:
    """Return the VibeVoice sidecar /health payload, or None if not configured."""
    if DIARIZATION_BACKEND != "vibevoice":
        return None
    try:
        import vibevoice_client

        snap = vibevoice_client.health()
        return snap or {"ready": False, "error": "sidecar unreachable"}
    except Exception as exc:  # noqa: BLE001
        return {"ready": False, "error": str(exc)}


def _s2s_tts_import_error() -> str | None:
    """Return the chatterbox import error string, or None if import succeeds."""
    try:
        from s2s.tts import availability_error
        err = availability_error()
        return err or None
    except Exception as exc:  # noqa: BLE001
        return str(exc)


@app.get("/health")
@app.post("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "uptime_seconds": round(time.time() - START_TIME, 2),
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
        "asr_backend": EFFECTIVE_BACKEND,
        "asr_backend_requested": ASR_BACKEND,
        "mlx_available": _MLX_AVAILABLE,
        "mlx_model": MLX_MODEL if EFFECTIVE_BACKEND == "mlx" else None,
        "spaces_runtime": _SPACES_AVAILABLE,
        "whisper_model": MLX_MODEL if EFFECTIVE_BACKEND == "mlx" else WHISPER_MODEL,
        "whisper_loaded": registry.whisper_model is not None,
        "diarization_loaded": (
            registry.diarize_pipeline is not None or registry._diarize_lib is not None
        ),
        "diarization_backend": DIARIZATION_BACKEND,
        "diarize_lib_available": registry._diarize_lib is not None,
        "pyannote_loaded": registry.diarize_pipeline is not None,
        "vibevoice_sidecar": _vibevoice_health_snapshot(),
        "s2s_enabled": S2S_ENABLED,
        "s2s_tts_backend": (os.environ.get("TTS_BACKEND", "chatterbox-turbo") if S2S_ENABLED else None),
        "s2s_tts_error": _s2s_tts_import_error() if S2S_ENABLED else None,
        "diarization_error": registry.diarization_error,
        "hf_token_present": bool(HF_TOKEN),
        "gpu": gpu_memory_info(),
        "loaded_align_models": list(registry.align_models.keys()),
        "mms_loaded": registry.mms_model is not None,
        "mms_error": registry.mms_error,
    }


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(..., description="Audio file (wav, mp3, m4a, flac, ogg, opus, webm, ...)"),
    language: Optional[str] = Form(None, description="ISO 639-1 language code; auto-detect if omitted."),
    diarize: Optional[bool] = Form(None, description="Run speaker diarization. Defaults to True if HF_TOKEN is set."),
    min_speakers: Optional[int] = Form(None),
    max_speakers: Optional[int] = Form(None),
    vad: bool = Form(True, description="Use Silero VAD before transcription."),
    output_format: str = Form("json", description="Comma-separated list: json,txt,srt,vtt,tsv"),
    initial_prompt: Optional[str] = Form(None),
    task: str = Form("transcribe", description="'transcribe' or 'translate'"),
    speaker_names: Optional[str] = Form(None, description='JSON object mapping speaker IDs to names. e.g. {"SPEAKER_00":"Alice","SPEAKER_01":"Bob"}'),
    include_speakers: bool = Form(True, description="Include speaker labels in output. Set False to hide SPEAKER_xx in TXT/SRT/VTT/TSV even when diarization is on."),
) -> JSONResponse:
    if task not in ("transcribe", "translate"):
        raise HTTPException(400, "task must be 'transcribe' or 'translate'.")

    suffix = Path(file.filename or "").suffix.lower()
    if suffix and suffix not in ALLOWED_AUDIO_SUFFIXES:
        raise HTTPException(415, f"Unsupported file extension '{suffix}'. Allowed: {sorted(ALLOWED_AUDIO_SUFFIXES)}")

    formats = parse_output_formats(output_format)
    speaker_map = parse_speaker_names(speaker_names)

    # Stream upload to disk with size enforcement.
    work_dir = Path(tempfile.mkdtemp(prefix="whisperx_"))
    audio_path = work_dir / f"input{suffix or '.wav'}"
    total = 0
    try:
        with open(audio_path, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_FILE_SIZE:
                    raise HTTPException(413, f"File exceeds max size of {MAX_FILE_SIZE} bytes.")
                f.write(chunk)
        if total == 0:
            raise HTTPException(400, "Empty upload.")

        result = _run_pipeline(
            audio_path=audio_path,
            language=language,
            diarize_requested=diarize,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            vad=vad,
            formats=formats,
            initial_prompt=initial_prompt,
            task=task,
            speaker_map=speaker_map,
            include_speakers=include_speakers,
        )
        return JSONResponse(result)
    except HTTPException:
        raise
    except torch.cuda.OutOfMemoryError as exc:
        free_gpu_memory()
        logger.error("CUDA OOM: %s", exc)
        raise HTTPException(507, "GPU out of memory. Try a shorter file or lower batch size.") from exc
    except Exception as exc:  # noqa: BLE001
        logger.error("Transcription failed: %s\n%s", exc, traceback.format_exc())
        raise HTTPException(500, f"Transcription failed: {exc}") from exc
    finally:
        try:
            shutil.rmtree(work_dir, ignore_errors=True)
        except OSError:
            pass
        free_gpu_memory()


@app.get("/download/{file_id}/{name}")
def download(file_id: str, name: str):
    cleanup_old_downloads()
    safe_name = Path(name).name
    target = DOWNLOAD_DIR / file_id / safe_name
    if not target.is_file():
        raise HTTPException(404, "File not found or expired.")
    return FileResponse(str(target), filename=safe_name)


# ---------------------------------------------------------------------------
# /session/diarize — dedicated diarized transcription endpoint
# ---------------------------------------------------------------------------
#
# This is the "source of truth" entry point for downstream features
# (speech-to-speech, summarizer, note-taker). Behaves like /transcribe but
# forces diarize=True and returns a clear 503 when HF_TOKEN is missing
# instead of silently falling back. /transcribe is preserved for backward
# compatibility.

@app.post("/session/diarize")
async def session_diarize(
    file: UploadFile = File(..., description="Audio file (wav, mp3, m4a, flac, ogg, opus, webm, ...)"),
    language: Optional[str] = Form(None, description="ISO 639-1 language code; auto-detect if omitted."),
    min_speakers: Optional[int] = Form(None),
    max_speakers: Optional[int] = Form(None),
    vad: bool = Form(True, description="Use Silero VAD before transcription."),
    output_format: str = Form("json,txt,srt,vtt", description="Comma-separated list: json,txt,srt,vtt,tsv"),
    initial_prompt: Optional[str] = Form(None),
    task: str = Form("transcribe", description="'transcribe' or 'translate'"),
    speaker_names: Optional[str] = Form(None, description='JSON object mapping speaker IDs to names. e.g. {"SPEAKER_00":"Alice","SPEAKER_01":"Bob"}'),
    include_speakers: bool = Form(True),
) -> JSONResponse:
    """Diarized transcription — speaker-labelled segments + word timings.

    Same form fields as /transcribe but always runs speaker diarization.
    The backend is selected by the ``DIARIZATION_BACKEND`` env var:

    * ``diarize`` (default) — CPU-only FoxNoseTech `diarize` library.
      No HuggingFace token required.
    * ``pyannote`` — classic pyannote.audio (requires ``HF_TOKEN``).
    * ``auto`` — try `diarize` first, fall back to pyannote when the result
      collapses to one speaker but the user expected ≥ 2.

    Returns ``503`` only when no backend is available (i.e. neither the
    `diarize` package is installed nor ``HF_TOKEN`` is set).

    Tips:
    * For mixed-language audio, leave `language` blank so Whisper can
      auto-detect; forcing a single language on multi-lingual recordings
      causes large blocks to be transcribed in the wrong language and
      mis-aligned, which can starve diarization of usable word timings.
    * Use `min_speakers` / `max_speakers` (or set them equal for an exact
      count) when you know the room size — both backends use these as hard
      hints and produce noticeably cleaner turns.
    """
    diar_lib_available = registry.load_diarize_lib() is not None
    if not diar_lib_available and not HF_TOKEN:
        raise HTTPException(
            503,
            "Speaker diarization is unavailable: neither the `diarize` "
            "package is installed nor HF_TOKEN is set. "
            "Install with `pip install diarize` (recommended, CPU-only, no "
            "account needed) or set HF_TOKEN to a Hugging Face access token "
            "that has accepted the user agreement on "
            "https://huggingface.co/pyannote/speaker-diarization-3.1 "
            "and restart the server.",
        )
    if task not in ("transcribe", "translate"):
        raise HTTPException(400, "task must be 'transcribe' or 'translate'.")

    suffix = Path(file.filename or "").suffix.lower()
    if suffix and suffix not in ALLOWED_AUDIO_SUFFIXES:
        raise HTTPException(415, f"Unsupported file extension '{suffix}'. Allowed: {sorted(ALLOWED_AUDIO_SUFFIXES)}")

    formats = parse_output_formats(output_format)
    speaker_map = parse_speaker_names(speaker_names)

    work_dir = Path(tempfile.mkdtemp(prefix="whisperx_session_"))
    audio_path = work_dir / f"input{suffix or '.wav'}"
    total = 0
    try:
        with open(audio_path, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_FILE_SIZE:
                    raise HTTPException(413, f"File exceeds max size of {MAX_FILE_SIZE} bytes.")
                f.write(chunk)
        if total == 0:
            raise HTTPException(400, "Empty upload.")

        result = _run_pipeline(
            audio_path=audio_path,
            language=language,
            diarize_requested=True,  # forced
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            vad=vad,
            formats=formats,
            initial_prompt=initial_prompt,
            task=task,
            speaker_map=speaker_map,
            include_speakers=include_speakers,
        )
        return JSONResponse(result)
    except HTTPException:
        raise
    except torch.cuda.OutOfMemoryError as exc:
        free_gpu_memory()
        logger.error("CUDA OOM (session/diarize): %s", exc)
        raise HTTPException(507, "GPU out of memory. Try a shorter file.") from exc
    except Exception as exc:  # noqa: BLE001
        logger.error("Session diarization failed: %s\n%s", exc, traceback.format_exc())
        raise HTTPException(500, f"Session diarization failed: {exc}") from exc
    finally:
        try:
            shutil.rmtree(work_dir, ignore_errors=True)
        except OSError:
            pass
        free_gpu_memory()


# ---------------------------------------------------------------------------
# Forced alignment endpoint (/align)
# ---------------------------------------------------------------------------

# Match speaker prefixes at the start of a turn. Examples:
#   "Speaker 1: hello"   "SPEAKER_00: hello"   "[A]: hello"
#   "John: hello"        "Narrator - hello"    "A> hello"
_SPEAKER_PREFIX_RE = re.compile(
    r"""
    ^\s*
    (?:\[\s*)?                                # optional [
    (?P<speaker>
        (?:SPEAKER[_\- ]?\d{1,3})
      | (?:Speaker\s+[A-Za-z0-9]+)
      | (?:[A-Z][A-Za-z0-9_\- ]{0,24})
    )
    \s*\]?\s*[:>\-—–]\s+                      # delimiter : > - — –
    """,
    re.VERBOSE,
)


def parse_transcript(text: str) -> Tuple[List[Dict[str, str]], bool]:
    """Split a transcript into speaker turns.

    Returns (turns, has_speakers) where each turn is {"speaker": str|None, "text": str}.
    """
    text = (text or "").strip()
    if not text:
        return [], False

    # Try line-by-line speaker prefixes first.
    lines = [ln for ln in (l.strip() for l in text.splitlines()) if ln]
    turns: List[Dict[str, str]] = []
    has_speakers = False
    current: Optional[Dict[str, str]] = None

    for line in lines:
        m = _SPEAKER_PREFIX_RE.match(line)
        if m:
            has_speakers = True
            if current and current["text"].strip():
                turns.append(current)
            current = {"speaker": m.group("speaker").strip(), "text": line[m.end():].strip()}
        else:
            if current is None:
                current = {"speaker": None, "text": line}
            else:
                current["text"] = (current["text"] + " " + line).strip()
    if current and current["text"].strip():
        turns.append(current)

    if not has_speakers:
        # Single-line input with inline "Speaker X:" markers — split on those.
        single = " ".join(lines)
        parts = []
        last = 0
        for m in _SPEAKER_PREFIX_RE.finditer(single):
            if m.start() > 0:
                parts.append((m.start(), m))
        if parts:
            has_speakers = True
            turns = []
            cursor = 0
            current_speaker: Optional[str] = None
            for start, m in parts:
                preceding = single[cursor:start].strip()
                if preceding:
                    turns.append({"speaker": current_speaker, "text": preceding})
                current_speaker = m.group("speaker").strip()
                cursor = m.end()
            tail = single[cursor:].strip()
            if tail:
                turns.append({"speaker": current_speaker, "text": tail})
            # Drop empty leading turn with no speaker.
            turns = [t for t in turns if t["text"]]

    if not turns:
        turns = [{"speaker": None, "text": text}]

    return turns, has_speakers


def detect_language_from_text(text: str) -> str:
    """Best-effort language detection. Returns ISO 639-1 code."""
    if _LANGDETECT_OK:
        try:
            code = _langdetect_detect(text)  # may return e.g. "en", "fr"
            return code.split("-")[0].lower()
        except Exception:  # noqa: BLE001
            pass
    return "en"


# Languages with native wav2vec2 forced-alignment models in WhisperX.
# For any language not in this set, we skip wav2vec2 entirely and use
# chunk-based uniform word distribution (much more accurate than falling
# back to English alignment, which produces bogus anchors).
WAV2VEC2_SUPPORTED_LANGS = {
    "en", "fr", "de", "es", "it", "ja", "zh", "nl", "uk", "pt", "ar", "cs",
    "ru", "pl", "hu", "fi", "fa", "el", "tr", "da", "he", "vi", "ko", "ur",
    "te", "hi", "ca", "ml", "no", "nn", "sk", "sl", "hr", "ro", "eu", "gl",
    "ka", "lv", "tl",
}


# ─────────────────────────────────────────────────────────────────────────────
# Meta MMS — Universal Forced Alignment (1,107 languages)
# ─────────────────────────────────────────────────────────────────────────────
# MMS uses ISO 639-3 codes. Whisper uses 639-1. Map common ones; for codes
# already in 639-3 form, MMS will accept them directly via the adapter loader.
# Reference: https://huggingface.co/facebook/mms-1b-all
ISO1_TO_ISO3 = {
    "en": "eng", "fr": "fra", "de": "deu", "es": "spa", "it": "ita",
    "pt": "por", "ru": "rus", "ja": "jpn", "zh": "cmn", "ko": "kor",
    "ar": "ara", "hi": "hin", "nl": "nld", "pl": "pol", "tr": "tur",
    "sw": "swa", "uk": "ukr", "cs": "ces", "hu": "hun", "fi": "fin",
    "el": "ell", "he": "heb", "vi": "vie", "ur": "urd", "te": "tel",
    "ca": "cat", "ml": "mal", "no": "nor", "sk": "slk", "sl": "slv",
    "hr": "hrv", "ro": "ron", "eu": "eus", "gl": "glg", "ka": "kat",
    "lv": "lav", "tl": "tgl", "fa": "fas", "da": "dan", "id": "ind",
    "ms": "msa", "th": "tha", "bn": "ben", "ta": "tam",
    "kn": "kan", "mr": "mar", "gu": "guj", "pa": "pan", "ne": "nep",
    "si": "sin", "my": "mya", "km": "khm", "lo": "lao", "am": "amh",
    "ti": "tir", "so": "som", "sn": "sna", "yo": "yor", "ig": "ibo",
    "ha": "hau", "zu": "zul", "xh": "xho", "af": "afr", "ln": "lin",
    "lg": "lug", "ny": "nya", "rw": "kin", "rn": "run", "sg": "sag",
    "wo": "wol", "ff": "ful", "bm": "bam", "ee": "ewe", "tw": "twi",
    "lt": "lit", "et": "est", "is": "isl", "mk": "mkd", "bg": "bul",
    "sr": "srp", "be": "bel", "hy": "hye", "az": "aze", "uz": "uzb",
    "kk": "kaz", "ky": "kir", "tg": "tgk", "tk": "tuk", "mn": "mon",
    "ps": "pus", "sd": "snd", "ku": "kmr", "yue": "yue", "br": "bre",
    "cy": "cym", "ga": "gle", "gd": "gla", "fo": "fao", "lb": "ltz",
    "mt": "mlt", "sq": "sqi", "as": "asm", "or": "ori", "tt": "tat",
    "ba": "bak", "yi": "yid", "la": "lat", "su": "sun", "jw": "jav",
    "haw": "haw", "ht": "hat", "mg": "mlg", "mi": "mri",
}

# Languages whose script MMS expects in romanized form (most non-Latin scripts).
# MMS-1B-all was trained on a mix of native and romanized text. The safest
# approach is to romanize anything outside Latin/Cyrillic/Greek.
NEEDS_ROMANIZATION_SCRIPTS = {
    "ara", "heb", "hin", "ben", "tam", "tel", "kan", "mal", "mar", "guj",
    "pan", "nep", "sin", "mya", "khm", "lao", "tha", "amh", "tir", "kat",
    "hye", "yi",  "yid", "ori", "asm", "snd", "urd", "pus", "uig",
    "jpn", "kor", "cmn", "yue", "tib", "bod",
}


def _whisper_to_mms_code(lang: Optional[str]) -> Optional[str]:
    if not lang:
        return None
    lang = lang.lower().strip()
    if lang in ISO1_TO_ISO3:
        return ISO1_TO_ISO3[lang]
    if len(lang) == 3:
        return lang
    return None


_uroman_instance = None


def _romanize(text: str) -> str:
    """Romanize text using uroman. Returns lowercase ASCII-friendly text."""
    global _uroman_instance
    if _uroman_instance is None:
        try:
            import uroman as _ur
            _uroman_instance = _ur.Uroman()
        except Exception as exc:  # noqa: BLE001
            logger.warning("uroman unavailable: %s — falling back to original text", exc)
            _uroman_instance = False
    if _uroman_instance is False:
        return text
    try:
        return _uroman_instance.romanize_string(text)
    except Exception as exc:  # noqa: BLE001
        logger.warning("uroman failed (%s); using original text", exc)
        return text


def _mms_normalize(text: str) -> str:
    """Lowercase + strip everything that isn't a letter or whitespace.
    MMS tokenizers use a small char vocab; punctuation isn't in the dict.
    Curly apostrophes (U+2019) and similar Unicode quotes are normalised to
    ASCII so that words like ``l'église`` stay as a single token instead of
    being split into ``l`` and ``église`` (which would corrupt word counts
    and timestamps downstream)."""
    text = text.lower()
    # Normalise smart quotes / apostrophes → ASCII apostrophe.
    text = (text
            .replace("\u2019", "'")  # right single quote
            .replace("\u2018", "'")  # left single quote
            .replace("\u02bc", "'")  # modifier letter apostrophe
            .replace("\u02b9", "'")  # modifier letter prime
            .replace("\u201c", '"').replace("\u201d", '"'))
    text = re.sub(r"[^\w\s']", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _load_audio_16k_mono(audio_path: Path) -> "torch.Tensor":
    """Load audio as 16kHz mono float32 tensor of shape (samples,)."""
    import torchaudio
    waveform, sr = torchaudio.load(str(audio_path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    return waveform.squeeze(0)


def align_with_mms(
    audio_input,        # Path OR numpy float32 array at 16 kHz mono
    transcript: str,
    language_iso3: str,
    model,
    processor,
) -> List[Dict[str, Any]]:
    """Run Meta MMS forced alignment (torchaudio tutorial approach).

    Returns [{word, start, end, score}, ...] in seconds referenced to
    the start of audio_input.  Raises on hard failure so caller can
    fall back gracefully.

    Critical design choices vs the previous implementation:
      1. Tokenise *word by word*, never including separator tokens in the
         CTC target sequence.  The previous approach included '|' tokens
         which caused sp_idx to drift, systematically pushing every word
         timestamp ~7 s later on a 3-minute audio.
      2. Use torchaudio.functional.merge_tokens to collapse consecutive
         same-token frames — more robust than manual span building.
      3. Process the full audio in a single model forward pass so that
         per-chunk normalisation differences cannot bias the frame-rate
         calculation.
    """
    import torchaudio
    import torch.nn.functional as F

    if isinstance(audio_input, (str, Path)):
        waveform = _load_audio_16k_mono(audio_input)
    else:
        waveform = torch.from_numpy(audio_input.astype("float32"))
    audio_duration = waveform.shape[-1] / 16000.0

    # Switch tokenizer + model adapter to target language.
    try:
        processor.tokenizer.set_target_lang(language_iso3)
        model.load_adapter(language_iso3)
    except Exception as exc:
        raise RuntimeError(
            f"MMS does not support language code '{language_iso3}': {exc}"
        ) from exc

    # Romanize if the script is non-Latin.
    text = transcript
    if language_iso3 in NEEDS_ROMANIZATION_SCRIPTS:
        text = _romanize(text)
    norm_text = _mms_normalize(text)
    if not norm_text:
        raise RuntimeError("Transcript is empty after normalization for MMS alignment.")

    raw_words = transcript.split()
    norm_words = norm_text.split()
    if len(norm_words) != len(raw_words):
        raw_words = norm_words

    # ------------------------------------------------------------------
    # Build CTC target token IDs — word by word, NO separator tokens.
    #
    # Including separator tokens ('|' or ' ') in the target sequence is
    # the root cause of the timestamp drift: torchaudio.forced_align
    # assigns many CTC frames to each separator, but our earlier grouping
    # loop skipped separators without advancing sp_idx, causing every
    # subsequent character span to map to the WRONG (earlier) position in
    # token_spans — producing a cumulative ~7 s shift on long audio.
    #
    # By tokenising character-by-character per word and never inserting
    # a separator ID we guarantee len(merged) == sum(word_token_lengths)
    # and the unflatten step maps spans to words perfectly.
    # ------------------------------------------------------------------
    blank_id: int = processor.tokenizer.pad_token_id or 0
    vocab: Dict[str, int] = processor.tokenizer.get_vocab()
    # Collect IDs that are word/segment separators (must be excluded).
    sep_ids: set = set()
    for sep_char in ("|", " ", "<pad>", "<unk>"):
        if sep_char in vocab:
            sep_ids.add(vocab[sep_char])
    sep_ids.add(blank_id)

    word_token_lengths: List[int] = []
    all_token_ids: List[int] = []
    for word in norm_words:
        # Encode the word; fall back to character-by-character if needed.
        encoded = processor.tokenizer.encode(word, add_special_tokens=False)
        # Strip blank / separator IDs that sneak in.
        clean = [t for t in encoded if t not in sep_ids]
        if not clean:
            # Last resort: encode each character independently.
            for ch in word:
                tid = vocab.get(ch, vocab.get(ch.lower(), vocab.get(ch.upper())))
                if tid is not None and tid not in sep_ids:
                    clean.append(tid)
        # Every word must contribute at least 1 token or forced_align will
        # produce fewer spans than words and the unflatten will be off.
        if not clean:
            unk_id = processor.tokenizer.unk_token_id
            clean = [unk_id] if unk_id is not None and unk_id not in sep_ids else [1]
        word_token_lengths.append(len(clean))
        all_token_ids.extend(clean)

    if not all_token_ids:
        raise RuntimeError("Tokenization produced no tokens for MMS alignment.")

    targets = torch.tensor(all_token_ids, dtype=torch.int32).unsqueeze(0)  # (1, N)

    # ------------------------------------------------------------------
    # Run model — full audio in one pass to keep normalisation consistent.
    # For very long audio (>5 min) we chunk to stay within VRAM, but we
    # still use the *per-utterance* audio_duration for the frame-rate
    # calculation rather than a per-chunk duration.
    # ------------------------------------------------------------------
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    CHUNK_SECONDS = 300  # process up to 5 min in one shot on T4 (16 GB)
    chunk_samples = CHUNK_SECONDS * 16000
    total_samples = waveform.shape[-1]

    all_log_probs: List[torch.Tensor] = []
    with torch.inference_mode():
        for offset in range(0, total_samples, chunk_samples):
            end = min(offset + chunk_samples, total_samples)
            raw_chunk = waveform[offset:end].cpu().numpy()
            inputs = processor(raw_chunk, sampling_rate=16000, return_tensors="pt")
            chunk_tensor = inputs.input_values.to(device=device, dtype=dtype)
            attn_mask = inputs.get("attention_mask")
            if attn_mask is not None:
                logits = model(chunk_tensor, attention_mask=attn_mask.to(device)).logits
            else:
                logits = model(chunk_tensor).logits  # (1, T, V)
            log_probs = F.log_softmax(logits.float(), dim=-1)
            all_log_probs.append(log_probs.cpu())
            del logits, log_probs, chunk_tensor, inputs

    log_probs = torch.cat(all_log_probs, dim=1)  # (1, T_total, V)
    n_frames = log_probs.shape[1]
    sec_per_frame = audio_duration / max(n_frames, 1)
    logger.info(
        "MMS: audio=%.2fs  frames=%d  sec/frame=%.4f  tokens=%d",
        audio_duration, n_frames, sec_per_frame, len(all_token_ids),
    )

    # ------------------------------------------------------------------
    # Forced CTC alignment → per-token spans via merge_tokens.
    # ------------------------------------------------------------------
    try:
        alignments, scores = torchaudio.functional.forced_align(
            log_probs, targets.cpu(), blank=blank_id,
        )
    except Exception as exc:
        raise RuntimeError(f"torchaudio.forced_align failed: {exc}") from exc

    # merge_tokens collapses consecutive frames with the same label.
    # Result: one TokenSpan per non-blank target token, in order.
    merged = torchaudio.functional.merge_tokens(alignments[0], scores[0])
    logger.info(
        "MMS: forced_align merged %d token spans for %d target tokens.",
        len(merged), len(all_token_ids),
    )

    # ------------------------------------------------------------------
    # Unflatten merged spans into per-word groups — direct index arithmetic,
    # no separator scanning needed because there are no separators in targets.
    # ------------------------------------------------------------------
    timed_words: List[Dict[str, Any]] = []
    span_idx = 0
    for i, (raw_word, wlen) in enumerate(zip(raw_words, word_token_lengths)):
        if span_idx >= len(merged):
            break
        word_spans = merged[span_idx: span_idx + wlen]
        span_idx += wlen
        if not word_spans:
            continue
        fs = min(s.start for s in word_spans)
        fe = max(s.end for s in word_spans)
        avg_score = float(sum(s.score for s in word_spans) / len(word_spans))
        timed_words.append({
            "word": raw_word,
            "start": float(fs * sec_per_frame),
            "end": float((fe + 1) * sec_per_frame),
            "score": avg_score,
        })

    # Fill any remaining words with uniform interpolation to the audio end.
    if len(timed_words) < len(raw_words):
        last_end = timed_words[-1]["end"] if timed_words else 0.0
        remaining = audio_duration - last_end
        tail_words = raw_words[len(timed_words):]
        per = remaining / max(len(tail_words), 1)
        for j, w in enumerate(tail_words):
            timed_words.append({
                "word": w,
                "start": float(last_end + per * j),
                "end": float(last_end + per * (j + 1)),
                "score": 0.0,
                "interpolated": True,
            })

    return timed_words


def _distribute_words_in_chunks(
    chunks: List[Dict[str, Any]],
    turns: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Distribute each chunk's `_gt_words` uniformly across [chunk.start, chunk.end].

    Used when wav2vec2 alignment is unavailable or unreliable. Produces
    one timed-word entry per ground-truth word, each tagged with its
    speaker label from the originating turn.
    """
    timed: List[Dict[str, Any]] = []
    for chunk in chunks:
        words = chunk.get("_gt_words") or []
        if not words:
            continue
        start = float(chunk["start"])
        end = float(chunk["end"])
        if end <= start:
            end = start + max(0.05 * len(words), 0.5)
        span = end - start
        n = len(words)
        for i, gt in enumerate(words):
            ws = start + span * i / n
            we = start + span * (i + 1) / n
            timed.append({
                "word": gt["word"],
                "start": ws,
                "end": we,
                "score": None,
                "speaker": turns[gt["turn"]]["speaker"],
                "interpolated": True,
            })
    return timed


def _strip_speakers(
    segments: List[Dict[str, Any]],
    word_segments: List[Dict[str, Any]],
) -> None:
    """Remove all speaker labels from output (in-place)."""
    for seg in segments:
        seg.pop("speaker", None)
        for w in seg.get("words", []) or []:
            w.pop("speaker", None)
    for w in word_segments:
        w.pop("speaker", None)


# ---------------------------------------------------------------------------
# Output reconciliation — preserve original transcript punctuation & casing.
# ---------------------------------------------------------------------------
# The wav2vec2 / MMS aligners normalise text aggressively (lowercase, strip
# punctuation, split contractions on apostrophes).  Their output `word_segments`
# therefore never matches the original transcript word-for-word.  We reconcile
# them here so every emitted word, segment and SRT line preserves the EXACT
# original wording — including ``Donc,`` ``d'abord,`` ``l'église.`` — while
# still using the precise timestamps the aligner produced.
# Works for any language (no French-specific logic).
# ---------------------------------------------------------------------------

_PUNCT_STRIP_RE = re.compile(r"[^\w]", flags=re.UNICODE)
# Apostrophe variants treated as identical for matching purposes.
_APOSTROPHES = "'\u2019\u2018\u02bc\u02b9"
# Sentence-final punctuation (multilingual).
_SENTENCE_END_CHARS = ".!?\u2026\u3002\uff01\uff1f"  # also CJK . ! ?


def _norm_for_match(s: str) -> str:
    """Normalise a word for fuzzy matching: lowercase, strip non-letters,
    collapse all apostrophe variants. Keeps Unicode letters intact."""
    if not s:
        return ""
    out = s.lower()
    for ap in _APOSTROPHES:
        out = out.replace(ap, "")
    return _PUNCT_STRIP_RE.sub("", out)


def _reconcile_to_gt(
    aligner_words: List[Dict[str, Any]],
    gt_words: List[Dict[str, Any]],
    turns: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], int]:
    """Walk ``aligner_words`` and ``gt_words`` in parallel, emitting one
    output dict per GT word with the **original** ``word`` text and timing
    derived from the aligner output.

    Greedy character-prefix matching means one GT word can absorb several
    aligner tokens (the apostrophe-split case: GT ``l'église`` → aligner
    [``l``, ``église``]) and one aligner token can cover several GT words
    in degenerate cases (rare).  Unmatched GT words are emitted with
    ``interpolated=True`` for later linear-interpolation.

    Returns ``(reconciled_word_segments, interpolated_count)``.
    """
    out: List[Dict[str, Any]] = []
    interpolated = 0
    j = 0  # index into aligner_words
    n_align = len(aligner_words)

    for gt in gt_words:
        target = _norm_for_match(gt["word"])
        speaker = turns[gt["turn"]]["speaker"] if 0 <= gt["turn"] < len(turns) else None

        if not target:
            # Punctuation-only GT token (rare); attach zero-length placeholder.
            out.append({
                "word": gt["word"],
                "start": None, "end": None, "score": None,
                "interpolated": True, "speaker": speaker,
            })
            interpolated += 1
            continue

        # Greedily consume aligner tokens whose concatenated normalised form
        # equals (or is a prefix of) the target.
        acc = ""
        consumed: List[Dict[str, Any]] = []
        k = j
        while k < n_align and len(acc) < len(target):
            cand = _norm_for_match(aligner_words[k].get("word", ""))
            if not cand:
                k += 1
                continue
            if target.startswith(acc + cand):
                acc += cand
                consumed.append(aligner_words[k])
                k += 1
            else:
                break

        if acc == target and consumed:
            start = next((w.get("start") for w in consumed if w.get("start") is not None), None)
            end = next((w.get("end") for w in reversed(consumed) if w.get("end") is not None), start)
            score_vals = [w.get("score") for w in consumed if isinstance(w.get("score"), (int, float))]
            out.append({
                "word": gt["word"],  # ← preserves original casing & punctuation
                "start": start,
                "end": end,
                "score": (sum(score_vals) / len(score_vals)) if score_vals else None,
                "speaker": speaker,
            })
            j = k
        else:
            out.append({
                "word": gt["word"],
                "start": None, "end": None, "score": None,
                "interpolated": True, "speaker": speaker,
            })
            interpolated += 1

    return out, interpolated


def _word_ends_sentence(w: str) -> bool:
    if not w:
        return False
    # Strip trailing closing punctuation/quotes so ``)``, ``"``, ``»`` don't hide ``.``
    stripped = w.rstrip(")\u201d\u00bb\"'\u2019")
    return bool(stripped) and stripped[-1] in _SENTENCE_END_CHARS


def _segment_by_sentences(
    turns: List[Dict[str, Any]],
    gt_words: List[Dict[str, Any]],
    word_segments: List[Dict[str, Any]],
    audio_duration: float,
) -> List[Dict[str, Any]]:
    """Build subtitle segments split at **sentence boundaries** in the
    original transcript text.  ``word_segments`` MUST be 1:1 with
    ``gt_words`` (use ``_reconcile_to_gt`` first).

    A new segment starts whenever the previous word ended a sentence
    (``.``, ``!``, ``?``, ``…``, plus CJK equivalents) OR the speaker turn
    changes.  This guarantees the SRT output reads exactly like the
    original transcript broken into sentences — preserving punctuation,
    capitalisation, and apostrophes.
    """
    segments: List[Dict[str, Any]] = []
    if not gt_words or len(word_segments) != len(gt_words):
        return segments

    n = len(gt_words)
    i = 0
    while i < n:
        turn_idx = gt_words[i]["turn"]
        speaker = turns[turn_idx]["speaker"] if 0 <= turn_idx < len(turns) else None
        sent_words_text: List[str] = []
        sent_words_seg: List[Dict[str, Any]] = []
        start_i = i
        while i < n and gt_words[i]["turn"] == turn_idx:
            w_text = gt_words[i]["word"]
            sent_words_text.append(w_text)
            sent_words_seg.append(word_segments[i])
            ends = _word_ends_sentence(w_text)
            i += 1
            if ends:
                break

        if not sent_words_seg:
            continue

        seg_start = next(
            (w.get("start") for w in sent_words_seg if w.get("start") is not None),
            0.0,
        )
        seg_end = next(
            (w.get("end") for w in reversed(sent_words_seg) if w.get("end") is not None),
            seg_start,
        )
        # Last-word fallback — if the very last sentence has no end timestamp.
        if seg_end is None or seg_end <= seg_start:
            seg_end = audio_duration

        segments.append({
            "start": float(seg_start),
            "end": float(seg_end),
            "text": " ".join(sent_words_text).strip(),
            "speaker": speaker,
            "words": sent_words_seg,
        })

    return segments


def _split_long_segments(
    segments: List[Dict[str, Any]],
    max_duration: float = 7.0,
    max_chars: int = 110,
) -> List[Dict[str, Any]]:
    """Split segments that are too long for readable subtitle display.

    Splits at sentence/clause boundaries first, then by time/char limits,
    using word-level timestamps. Safe to call even when words are uniformly
    interpolated — the proportional timestamps still give sensible splits.
    """
    result: List[Dict[str, Any]] = []
    for seg in segments:
        seg_start = float(seg.get("start") or 0.0)
        seg_end = float(seg.get("end") or seg_start)
        duration = seg_end - seg_start
        text = (seg.get("text") or "").strip()
        words = seg.get("words") or []

        # Nothing to split
        if (duration <= max_duration and len(text) <= max_chars) or not words:
            result.append(seg)
            continue

        # Group words into subtitle-sized chunks
        groups: List[List[Dict[str, Any]]] = []
        current: List[Dict[str, Any]] = []
        cur_start: Optional[float] = None
        cur_chars = 0
        cur_dur = 0.0

        for w in words:
            wtext = (w.get("word") or "").strip()
            ws = w.get("start")
            we = w.get("end")
            if ws is None:
                ws = cur_start or seg_start
            if we is None:
                we = float(ws)

            if cur_start is None:
                cur_start = float(ws)

            cur_chars += len(wtext) + 1
            cur_dur = float(we) - cur_start
            current.append(w)

            is_sent_end = bool(wtext) and wtext[-1] in ".!?\u2026"
            is_clause = bool(wtext) and wtext[-1] in ",;:"
            at_hard_limit = cur_dur >= max_duration or cur_chars >= max_chars
            at_soft_limit = cur_dur >= max_duration * 0.65 and (is_clause or is_sent_end)

            if is_sent_end or (at_hard_limit and len(current) > 1) or at_soft_limit:
                groups.append(current)
                current = []
                cur_start = None
                cur_chars = 0
                cur_dur = 0.0

        if current:
            groups.append(current)

        if len(groups) <= 1:
            result.append(seg)
            continue

        speaker = seg.get("speaker")
        for grp in groups:
            g_start = next(
                (float(w["start"]) for w in grp if w.get("start") is not None), seg_start
            )
            g_end = next(
                (float(w["end"]) for w in reversed(grp) if w.get("end") is not None),
                g_start + 1.0,
            )
            g_text = " ".join((w.get("word") or "").strip() for w in grp).strip()
            if not g_text:
                continue
            result.append({
                "start": g_start,
                "end": g_end,
                "text": g_text,
                "speaker": speaker,
                "words": grp,
            })

    return result


@app.post("/align")
async def align_endpoint(
    audio: UploadFile = File(..., description="Audio file (wav, mp3, m4a, flac, ogg, opus, webm, ...)"),
    transcript: str = Form(..., description="Exact, ground-truth transcript text. May include speaker labels."),
    language: Optional[str] = Form(None, description="ISO 639-1 code. Auto-detected from transcript if omitted."),
    vad: bool = Form(True, description="Run Silero VAD before alignment to skip silence."),
    diarize: bool = Form(False, description="Also run pyannote diarization and assign speakers acoustically."),
    min_speakers: Optional[int] = Form(None),
    max_speakers: Optional[int] = Form(None),
    output_format: str = Form("json", description="Comma-separated list: json,txt,srt,vtt,tsv"),
    speaker_names: Optional[str] = Form(None, description='JSON object mapping speaker IDs to names. e.g. {"SPEAKER_00":"Alice","SPEAKER_01":"Bob"}'),
    include_speakers: bool = Form(True, description="Include speaker labels in output. Set False to hide SPEAKER_xx in TXT/SRT/VTT/TSV even when diarization is on."),
) -> JSONResponse:
    """Aligns the user-provided ground-truth transcript to the audio using the
    WhisperX wav2vec2 alignment pipeline. Does NOT run ASR.
    """
    # Validate input transcript.
    cleaned_transcript = (transcript or "").strip()
    if not cleaned_transcript or len(cleaned_transcript) < 2:
        raise HTTPException(400, "Transcript is empty or too short.")

    formats = parse_output_formats(output_format)
    speaker_map = parse_speaker_names(speaker_names)

    # Validate audio extension.
    suffix = Path(audio.filename or "").suffix.lower()
    if suffix and suffix not in ALLOWED_AUDIO_SUFFIXES:
        raise HTTPException(415, f"Unsupported file extension '{suffix}'. Allowed: {sorted(ALLOWED_AUDIO_SUFFIXES)}")

    # Stream upload to disk with size enforcement.
    work_dir = Path(tempfile.mkdtemp(prefix="whisperx_align_"))
    audio_path = work_dir / f"input{suffix or '.wav'}"
    total = 0
    try:
        with open(audio_path, "wb") as f:
            while True:
                chunk = await audio.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_FILE_SIZE:
                    raise HTTPException(413, f"File exceeds max size of {MAX_FILE_SIZE} bytes.")
                f.write(chunk)
        if total == 0:
            raise HTTPException(400, "Empty audio upload.")

        return JSONResponse(
            _run_align_pipeline(
                audio_path=audio_path,
                transcript_text=cleaned_transcript,
                language=language,
                vad=vad,
                diarize=diarize,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                formats=formats,
                speaker_map=speaker_map,
                include_speakers=include_speakers,
            )
        )
    except HTTPException:
        raise
    except torch.cuda.OutOfMemoryError as exc:
        free_gpu_memory()
        logger.error("CUDA OOM during alignment: %s", exc)
        raise HTTPException(507, "GPU out of memory. Try a shorter file.") from exc
    except Exception as exc:  # noqa: BLE001
        logger.error("Alignment failed: %s\n%s", exc, traceback.format_exc())
        raise HTTPException(500, f"Alignment failed: {exc}") from exc
    finally:
        try:
            shutil.rmtree(work_dir, ignore_errors=True)
        except OSError:
            pass
        free_gpu_memory()


def _run_align_pipeline(
    *,
    audio_path: Path,
    transcript_text: str,
    language: Optional[str],
    vad: bool,
    diarize: bool,
    min_speakers: Optional[int],
    max_speakers: Optional[int],
    formats: List[str],
    speaker_map: Optional[Dict[str, str]] = None,
    include_speakers: bool = True,
) -> Dict[str, Any]:
    pipeline_started = time.time()

    # 1) Parse transcript into speaker turns.
    turns, has_speakers = parse_transcript(transcript_text)
    if not turns:
        raise HTTPException(400, "Transcript could not be parsed into any text turns.")

    plain_text = " ".join(t["text"] for t in turns).strip()
    if not plain_text:
        raise HTTPException(400, "Transcript contains no alignable text.")

    # 2) Decide language.
    detected_language = (language or "").strip().lower() or detect_language_from_text(plain_text)
    if not detected_language:
        detected_language = "en"

    # 3) Load audio.
    audio = whisperx.load_audio(str(audio_path))
    duration = float(len(audio)) / 16000.0
    if duration > MAX_AUDIO_DURATION:
        raise HTTPException(413, f"Audio length {duration:.1f}s exceeds max {MAX_AUDIO_DURATION}s.")

    # 4) Optional VAD: trim leading/trailing silence to improve alignment.
    audio_for_align = audio
    vad_offset = 0.0
    vad_end = duration
    vad_warning: Optional[str] = None
    if vad:
        try:
            vad_offset, vad_end = _vad_speech_window(audio_path)
            if vad_end > vad_offset and (vad_end - vad_offset) > 0.5:
                start_sample = max(int(vad_offset * 16000), 0)
                end_sample = min(int(vad_end * 16000), len(audio))
                audio_for_align = audio[start_sample:end_sample]
            else:
                vad_offset, vad_end = 0.0, duration
        except Exception as exc:  # noqa: BLE001
            vad_warning = f"VAD failed, using full audio: {exc}"
            logger.warning(vad_warning)
            vad_offset, vad_end = 0.0, duration

    align_audio_duration = float(len(audio_for_align)) / 16000.0

    # ------------------------------------------------------------------
    # NEW STRATEGY (per fix-alignment-issue.txt):
    # 1. Use a small Whisper model to transcribe the audio. The output is
    #    naturally chunked by VAD into short segments (each ≤30s) with a
    #    word count we can use to slice the ground-truth transcript.
    # 2. Replace each ASR chunk's text with the corresponding slice of
    #    ground-truth words (preserving the exact user text).
    # 3. Align each chunk individually with whisperx.align — short chunks
    #    are exactly what the wav2vec2 forced-aligner is built for.
    # 4. Concatenate the aligned word lists. Interpolate any words that
    #    the aligner dropped, so every ground-truth word ends up timed.
    # ------------------------------------------------------------------

    # 5) Tokenise the ground-truth transcript while remembering each word's
    # original turn index (so we can re-attach speaker labels later).
    gt_words: List[Dict[str, Any]] = []
    for turn_idx, turn in enumerate(turns):
        for w in turn["text"].split():
            gt_words.append({"word": w, "turn": turn_idx})
    if not gt_words:
        raise HTTPException(400, "Transcript contains no alignable words.")

    # 6) Lightweight ASR pass to discover speech chunks + per-chunk word counts.
    tiny = registry.load_tiny()
    if EFFECTIVE_BACKEND == "mlx":
        asr_kwargs: Dict[str, Any] = {}
    else:
        asr_kwargs = {"batch_size": 8, "task": "transcribe"}
    if language:
        asr_kwargs["language"] = language
    try:
        asr_result = tiny.transcribe(audio_for_align, **asr_kwargs)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"Helper ASR pass failed: {exc}") from exc

    # If the user did not specify a language, prefer ASR detection over text-based.
    asr_language = (asr_result.get("language") or "").lower() or None
    if not language and asr_language:
        detected_language = asr_language

    asr_segments = asr_result.get("segments", []) or []
    if not asr_segments:
        # Fallback: one big chunk covering the whole audio.
        asr_segments = [{"start": 0.0, "end": align_audio_duration, "text": ""}]

    # 7) Distribute the ground-truth words across the ASR chunks proportional
    # to the ASR-emitted word counts. We work in *aligned* word counts so
    # every ground-truth word gets placed somewhere.
    def _wc(seg: Dict[str, Any]) -> int:
        return max(len((seg.get("text") or "").split()), 1)

    asr_word_counts = [_wc(s) for s in asr_segments]
    total_asr_words = sum(asr_word_counts)
    total_gt_words = len(gt_words)

    # Compute how many GT words to assign to each chunk (proportional, sum = N).
    assigned: List[int] = []
    running = 0.0
    for i, c in enumerate(asr_word_counts):
        if i == len(asr_word_counts) - 1:
            assigned.append(total_gt_words - sum(assigned))
        else:
            share = c / total_asr_words
            n = int(round(share * total_gt_words))
            n = max(0, min(n, total_gt_words - sum(assigned) - (len(asr_word_counts) - i - 1)))
            assigned.append(n)

    # 8) Build chunked alignment input segments using the ASR timestamps + GT text.
    # IMPORTANT: expand each chunk's start time backwards to fill any gap left by
    # the previous chunk. Whisper/VAD sometimes reports a segment starting at e.g.
    # 0:20 when speech actually began at 0:12; if we pass [0:20, 0:37] to wav2vec2
    # it never sees the 0:07–0:20 window, causing a blank gap in the subtitles.
    # By setting effective_start = previous_chunk_end we give wav2vec2 the full
    # continuous audio and it will find the true speech onset.
    chunks: List[Dict[str, Any]] = []
    cursor_word = 0
    prev_chunk_end: float = 0.0
    for seg, n in zip(asr_segments, assigned):
        if n <= 0:
            continue
        words_slice = gt_words[cursor_word: cursor_word + n]
        cursor_word += n
        seg_start = float(seg.get("start") or 0.0)
        seg_end = float(seg.get("end") or 0.0)
        # Expand start backwards to fill any inter-segment gap.
        effective_start = min(prev_chunk_end, seg_start) if chunks else seg_start
        prev_chunk_end = seg_end
        # Use the dominant turn (and therefore speaker) of this slice.
        turn_indices = [w["turn"] for w in words_slice]
        dom_turn = max(set(turn_indices), key=turn_indices.count)
        chunks.append({
            "start": effective_start,
            "end": seg_end,
            "text": " ".join(w["word"] for w in words_slice),
            "speaker": turns[dom_turn]["speaker"],
            "_gt_words": words_slice,
        })

    # Any leftover GT words (rounding) → append to the last chunk.
    if cursor_word < total_gt_words and chunks:
        leftover = gt_words[cursor_word:]
        chunks[-1]["text"] += " " + " ".join(w["word"] for w in leftover)
        chunks[-1]["_gt_words"].extend(leftover)

    logger.info(
        "Align: %d turns / %d GT words → %d ASR chunks (audio %.1fs).",
        len(turns), total_gt_words, len(chunks), align_audio_duration,
    )

    # 9) Decide alignment strategy.
    # Strategy A (preferred): wav2vec2 forced alignment when a native model
    #   exists for the detected language. High accuracy.
    # Strategy B (fallback): chunk-based uniform word distribution. Used when:
    #   - no native wav2vec2 model exists for the language, OR
    #   - wav2vec2 alignment drops more than 40% of words (too unreliable).
    # This avoids the previous catastrophic failure mode where falling back
    # to English alignment on (e.g.) Lingala produced nonsense anchor points
    # that left huge gaps in the timestamps.
    # 9) Decide alignment strategy.
    # Strategy A (preferred): WhisperX wav2vec2 forced alignment when a native
    #   model exists for the detected language. High accuracy.
    # Strategy B: Meta MMS (Massively Multilingual Speech) forced alignment.
    #   Covers 1,107 languages including Lingala, Kikongo, Xhosa, Hausa, etc.
    #   Used when no native wav2vec2 model exists OR wav2vec2 was unreliable.
    # Strategy C (last resort): chunk-based uniform word distribution.
    align_warning: Optional[str] = None
    used_strategy = "wav2vec2"
    aligned_segments: List[Dict[str, Any]] = []
    word_segments: List[Dict[str, Any]] = []
    interpolated_count = 0

    align_loaded = None
    if detected_language in WAV2VEC2_SUPPORTED_LANGS:
        align_loaded = registry.load_align(detected_language)
        if align_loaded is None:
            align_warning = (
                f"Failed to load wav2vec2 model for '{detected_language}'."
            )

    if align_loaded is not None:
        # Strategy A: full-audio single-pass wav2vec2 forced alignment.
        # CRITICAL: do NOT chunk by ASR segment — CTC alignment must see the
        # full audio so it can place words at their true positions, including
        # inside gaps that Whisper/VAD falsely detected as silence.
        align_model, align_metadata = align_loaded
        full_gt_text = " ".join(w["word"] for w in gt_words)
        full_seg = {"start": 0.0, "end": float(align_audio_duration), "text": full_gt_text}
        try:
            res = whisperx.align(
                [full_seg], align_model, align_metadata, audio_for_align, DEVICE,
                return_char_alignments=False,
            )
            word_segments = res.get("word_segments", []) or []
            aligned_segments = res.get("segments", []) or []
        except Exception as exc:  # noqa: BLE001
            logger.warning("Full-audio wav2vec2 alignment failed: %s", exc)
            word_segments = []
            aligned_segments = []

        # Quality check: if too few words got real timestamps, try MMS instead.
        timed_word_ratio = (len(word_segments) / total_gt_words) if total_gt_words else 0.0
        if timed_word_ratio < 0.6:
            logger.info(
                "wav2vec2 only timed %d / %d words (%.0f%%); attempting MMS fallback.",
                len(word_segments), total_gt_words, timed_word_ratio * 100,
            )
            align_warning = (
                f"wav2vec2 aligned only {len(word_segments)}/{total_gt_words} words "
                f"({timed_word_ratio*100:.0f}%); falling back to Meta MMS."
            )
            aligned_segments = []
            word_segments = []
            align_loaded = None  # signal to try MMS below

    if align_loaded is None:
        # Strategy B: Meta MMS — universal forced alignment for 1,107 languages.
        mms_lang = _whisper_to_mms_code(detected_language)
        mms_loaded = registry.load_mms() if mms_lang else None
        mms_words: Optional[List[Dict[str, Any]]] = None
        if mms_loaded is not None and mms_lang:
            mms_model, mms_processor = mms_loaded
            try:
                # Build full transcript text in turn order to align as one pass.
                full_text = " ".join(w["word"] for w in gt_words)
                mms_words = align_with_mms(
                    audio_for_align, full_text, mms_lang, mms_model, mms_processor,
                )
                # Attach speaker labels from gt_words (1:1 if MMS produced same count).
                if len(mms_words) == len(gt_words):
                    for mw, gt in zip(mms_words, gt_words):
                        mw["speaker"] = turns[gt["turn"]]["speaker"]
                else:
                    # Best-effort speaker assignment by relative index.
                    for i, mw in enumerate(mms_words):
                        gt_idx = min(i, len(gt_words) - 1)
                        mw["speaker"] = turns[gt_words[gt_idx]["turn"]]["speaker"]
                used_strategy = "mms"
                align_warning = (
                    f"Aligned with Meta MMS ({mms_lang}) — "
                    f"{len(mms_words)} words timed."
                )
                word_segments = mms_words
            except Exception as exc:  # noqa: BLE001
                logger.warning("MMS alignment failed: %s", exc)
                align_warning = (
                    (align_warning + " | " if align_warning else "")
                    + f"MMS alignment failed: {exc}. Using uniform fallback."
                )

        if not word_segments:
            # Strategy C: uniform word distribution within ASR chunks.
            word_segments = _distribute_words_in_chunks(chunks, turns)
            interpolated_count = len(word_segments)
            used_strategy = "uniform" if used_strategy == "wav2vec2" else used_strategy
            if used_strategy != "mms":
                used_strategy = "uniform"

    # 10) Reconcile aligner output back to GT transcript words.
    # ------------------------------------------------------------------
    # The wav2vec2 / MMS aligners normalise text aggressively (lowercase,
    # punctuation stripped, contractions split on apostrophes).  Their
    # `word_segments` therefore never matches the original transcript
    # word-for-word.  We map every aligner token back onto the original
    # GT word(s) so the OUTPUT preserves exact transcript wording —
    # including ``Donc,``, ``d'abord,``, ``l'église.`` — while still
    # using the aligner's precise timestamps.  This makes SRT/VTT/TXT
    # read identically to the source transcript.  Works for ANY language.
    if used_strategy in ("wav2vec2", "mms"):
        word_segments, recon_interpolated = _reconcile_to_gt(
            word_segments, gt_words, turns,
        )
        interpolated_count += recon_interpolated

        # Linearly interpolate any words that didn't match the aligner.
        n = len(word_segments)
        first_known = next(
            (k for k in range(n) if word_segments[k].get("start") is not None), None
        )
        last_known = next(
            (k for k in range(n - 1, -1, -1) if word_segments[k].get("end") is not None),
            None,
        )
        if first_known is None:
            # Total miss — uniform spread across the audio.
            step = align_audio_duration / max(n, 1)
            for k in range(n):
                word_segments[k]["start"] = k * step
                word_segments[k]["end"] = (k + 1) * step
                word_segments[k]["interpolated"] = True
        else:
            # Pad before / after known region.
            for k in range(first_known):
                word_segments[k]["start"] = 0.0
                word_segments[k]["end"] = (
                    word_segments[first_known]["start"] * (k + 1) / (first_known + 1)
                )
                word_segments[k]["interpolated"] = True
            for k in range(last_known + 1, n):
                base = word_segments[last_known]["end"]
                remaining = align_audio_duration - base
                step = remaining / max(n - last_known, 1)
                word_segments[k]["start"] = base + step * (k - last_known - 1)
                word_segments[k]["end"] = base + step * (k - last_known)
                word_segments[k]["interpolated"] = True
            # Fill interior gaps.
            i = first_known
            while i < last_known:
                if word_segments[i + 1].get("start") is None:
                    j = i + 1
                    while j <= last_known and word_segments[j].get("start") is None:
                        j += 1
                    span = word_segments[j]["start"] - word_segments[i]["end"]
                    gap = j - i
                    for k in range(1, gap):
                        word_segments[i + k]["start"] = word_segments[i]["end"] + span * (k - 1) / gap
                        word_segments[i + k]["end"] = word_segments[i]["end"] + span * k / gap
                        word_segments[i + k]["interpolated"] = True
                    i = j
                else:
                    i += 1

    # 12) Build segments split at sentence boundaries in the ORIGINAL
    # transcript text — this is what makes the SRT read naturally.
    # Each segment ends at the next ``.``, ``!``, ``?``, ``…`` (or speaker
    # turn change), so subtitle lines correspond to real sentences with
    # all original punctuation, casing and apostrophes intact.
    if used_strategy in ("wav2vec2", "mms") and len(word_segments) == len(gt_words):
        aligned_segments = _segment_by_sentences(
            turns, gt_words, word_segments, align_audio_duration,
        )
    elif (has_speakers or len(turns) > 1) and len(word_segments) == len(gt_words):
        # Uniform fallback path: keep per-turn rebuilding.
        gt_turn_idx_per_word = [w["turn"] for w in gt_words]
        buckets: Dict[int, List[Dict[str, Any]]] = {}
        for tidx, w in zip(gt_turn_idx_per_word, word_segments):
            buckets.setdefault(tidx, []).append(w)
        rebuilt_segments: List[Dict[str, Any]] = []
        for tidx, ws in buckets.items():
            if not ws:
                continue
            start = next((w["start"] for w in ws if w.get("start") is not None), 0.0)
            end = next((w["end"] for w in reversed(ws) if w.get("end") is not None), start)
            rebuilt_segments.append({
                "start": float(start),
                "end": float(end),
                "text": turns[tidx]["text"],
                "speaker": turns[tidx]["speaker"],
                "words": ws,
            })
        aligned_segments = rebuilt_segments
    else:
        # Last-resort: keep the aligner's segments or build one big segment.
        if not aligned_segments and word_segments:
            aligned_segments = [{
                "start": word_segments[0].get("start") or 0.0,
                "end": word_segments[-1].get("end") or align_audio_duration,
                "text": " ".join(w.get("word", "") for w in word_segments),
                "speaker": None,
                "words": word_segments,
            }]

    # Split any sentence that is still too long for a readable subtitle line
    # (handles run-on sentences and the uniform fallback case).
    aligned_segments = _split_long_segments(aligned_segments)

    # 13) Shift timestamps back by VAD offset so they reference the original audio.
    if vad_offset > 0:
        for seg in aligned_segments:
            if seg.get("start") is not None:
                seg["start"] = float(seg["start"]) + vad_offset
            if seg.get("end") is not None:
                seg["end"] = float(seg["end"]) + vad_offset
            for w in seg.get("words", []) or []:
                if w.get("start") is not None:
                    w["start"] = float(w["start"]) + vad_offset
                if w.get("end") is not None:
                    w["end"] = float(w["end"]) + vad_offset
        for w in word_segments:
            if w.get("start") is not None:
                w["start"] = float(w["start"]) + vad_offset
            if w.get("end") is not None:
                w["end"] = float(w["end"]) + vad_offset

    # 14) Optional acoustic diarization (pyannote) on top of transcript-driven speakers.
    diarized_acoustic = False
    diarization_warning: Optional[str] = None
    if diarize:
        pipeline = registry.load_diarize()
        if pipeline is None:
            raise HTTPException(
                400,
                registry.diarization_error
                or "Diarization requested but pipeline is unavailable. Set HF_TOKEN.",
            )
        try:
            diar_kwargs: Dict[str, Any] = {}
            if min_speakers is not None:
                diar_kwargs["min_speakers"] = min_speakers
            if max_speakers is not None:
                diar_kwargs["max_speakers"] = max_speakers
            diarize_segments = pipeline(str(audio_path), **diar_kwargs)
            merged = whisperx.assign_word_speakers(
                diarize_segments,
                {"segments": aligned_segments, "word_segments": word_segments},
            )
            aligned_segments = merged.get("segments", aligned_segments) or aligned_segments
            word_segments = merged.get("word_segments", word_segments) or word_segments
            diarized_acoustic = True
        except Exception as exc:  # noqa: BLE001
            diarization_warning = f"Acoustic diarization failed: {exc}"
            logger.warning(diarization_warning)

    diarized = bool(has_speakers or diarized_acoustic)

    # Reconstruct any missing segment-level speakers from words.
    if diarized:
        for seg in aligned_segments:
            if seg.get("speaker"):
                continue
            counts: Dict[str, float] = {}
            for w in seg.get("words", []) or []:
                spk = w.get("speaker")
                if not spk:
                    continue
                dur = max((w.get("end") or 0.0) - (w.get("start") or 0.0), 0.01)
                counts[spk] = counts.get(spk, 0.0) + dur
            if counts:
                seg["speaker"] = max(counts.items(), key=lambda kv: kv[1])[0]

    processing_time = time.time() - pipeline_started

    # Apply speaker name remapping if provided (before rendering)
    if speaker_map and diarized:
        apply_speaker_names(aligned_segments, word_segments, speaker_map)

    # If user wants no speaker labels in output, strip them now (after diarization
    # has done its job of helping the alignment quality).
    show_speakers = bool(diarized and include_speakers)
    if not include_speakers:
        _strip_speakers(aligned_segments, word_segments)

    # 10) Render output formats.
    rendered: Dict[str, str] = {}
    if "txt" in formats:
        rendered["txt"] = render_txt(aligned_segments, show_speakers)
    if "srt" in formats:
        rendered["srt"] = render_srt(aligned_segments, show_speakers)
    if "vtt" in formats:
        rendered["vtt"] = render_vtt(aligned_segments, show_speakers)
    if "tsv" in formats:
        rendered["tsv"] = render_tsv(aligned_segments, show_speakers)

    files: Dict[str, str] = {}
    download_links: Dict[str, str] = {}
    if rendered:
        file_id = uuid.uuid4().hex
        out_dir = DOWNLOAD_DIR / file_id
        out_dir.mkdir(parents=True, exist_ok=True)
        for ext, content in rendered.items():
            name = f"transcript.{ext}"
            (out_dir / name).write_text(content, encoding="utf-8")
            files[ext] = content
            files[f"{ext}_base64"] = base64.b64encode(content.encode("utf-8")).decode("ascii")
            download_links[ext] = f"/download/{file_id}/{name}"

    cleanup_old_downloads()

    return {
        "mode": "align",
        "language": detected_language,
        "language_detection": "user" if language else ("langdetect" if _LANGDETECT_OK else "default-en"),
        "duration_seconds": round(duration, 3),
        "processing_time_seconds": round(processing_time, 3),
        "realtime_factor": round(duration / processing_time, 2) if processing_time > 0 else None,
        "diarized": diarized,
        "diarized_acoustic": diarized_acoustic,
        "diarization_warning": diarization_warning,
        "alignment_warning": align_warning,
        "alignment_strategy": used_strategy,
        "vad_warning": vad_warning,
        "vad_window": {"start": round(vad_offset, 3), "end": round(vad_end, 3)} if vad else None,
        "turn_count": len(turns),
        "ground_truth_word_count": total_gt_words,
        "interpolated_word_count": interpolated_count,
        "warning": (
            f"{interpolated_count} of {total_gt_words} words could not be force-aligned "
            "and were interpolated between neighbouring timestamps."
        ) if interpolated_count else None,
        "word_level": True,
        "segments": aligned_segments,
        "word_segments": word_segments,
        "files": files,
        "download_links": download_links,
        "download_expires_at": (datetime.utcnow() + timedelta(seconds=DOWNLOAD_TTL_SECONDS)).isoformat() + "Z",
    }


def _vad_speech_window(audio_path: Path) -> Tuple[float, float]:
    """Use Silero VAD via WhisperX to find speech start/end seconds."""
    try:
        from whisperx.vads import Silero  # type: ignore
        vad_model = Silero(device=DEVICE)
        result = vad_model({"waveform": None, "sample_rate": 16000, "audio_path": str(audio_path)})
        # `result` is a pyannote-style timeline list of segments with start/end.
        segs = []
        if hasattr(result, "itersegments"):
            for s in result.itersegments():
                segs.append((float(s.start), float(s.end)))
        elif isinstance(result, list):
            for s in result:
                segs.append((float(s["start"]), float(s["end"])))
        if segs:
            return segs[0][0], segs[-1][1]
    except Exception:  # noqa: BLE001
        pass
    return 0.0, 0.0


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _run_pipeline(
    *,
    audio_path: Path,
    language: Optional[str],
    diarize_requested: Optional[bool],
    min_speakers: Optional[int],
    max_speakers: Optional[int],
    vad: bool,
    formats: List[str],
    initial_prompt: Optional[str],
    task: str,
    speaker_map: Optional[Dict[str, str]] = None,
    include_speakers: bool = True,
) -> Dict[str, Any]:
    pipeline_started = time.time()

    # 1) Load audio
    audio = whisperx.load_audio(str(audio_path))
    duration = float(len(audio)) / 16000.0
    if duration > MAX_AUDIO_DURATION:
        raise HTTPException(413, f"Audio length {duration:.1f}s exceeds max {MAX_AUDIO_DURATION}s.")
    batch_size = select_batch_size(duration)

    # 2) Transcribe (backend-agnostic)
    whisper_model = registry.load_whisper()
    transcribe_kwargs: Dict[str, Any] = {"task": task}
    if language:
        transcribe_kwargs["language"] = language

    if EFFECTIVE_BACKEND == "mlx":
        # MLX backend: only supports `language`. Strip out whisperx-specific kwargs.
        mlx_kwargs = {k: v for k, v in transcribe_kwargs.items() if k in {"language"}}
        t0 = time.time()
        transcription = whisper_model.transcribe(audio, **mlx_kwargs)
    else:
        transcribe_kwargs["batch_size"] = batch_size
        if not vad:
            transcribe_kwargs["chunk_size"] = 30
        if initial_prompt:
            transcribe_kwargs["initial_prompt"] = initial_prompt
        t0 = time.time()
        transcription = whisper_model.transcribe(
            audio,
            **{k: v for k, v in transcribe_kwargs.items() if v is not None},
        )
    transcribe_time = time.time() - t0
    detected_language = transcription.get("language") or language or "en"

    # Language-mismatch warning. If the user forced a language and Whisper
    # detected a different one, the audio likely contains another language
    # (or is mixed-language). We log a warning rather than re-transcribe
    # automatically (re-transcribe would double the GPU cost). Callers facing
    # mixed-language audio should omit `language` to enable auto-detection.
    language_warning: Optional[str] = None
    if language and detected_language and language.lower() != detected_language.lower():
        language_warning = (
            f"Requested language '{language}' but Whisper detected "
            f"'{detected_language}'. For mixed-language audio, omit the "
            f"language parameter to enable per-segment auto-detection."
        )
        logger.warning(language_warning)

    segments = transcription.get("segments", []) or []

    # 3) Forced alignment for word-level timestamps
    align_warning: Optional[str] = None
    align_loaded = registry.load_align(detected_language)
    if align_loaded is None and detected_language != "en":
        align_warning = (
            f"No alignment model available for language '{detected_language}'. "
            "Falling back to English alignment; word timestamps may be inaccurate."
        )
        align_loaded = registry.load_align("en")

    aligned_segments = segments
    word_segments: List[Dict[str, Any]] = []
    if align_loaded is not None and segments:
        align_model, align_metadata = align_loaded
        try:
            aligned = whisperx.align(
                segments,
                align_model,
                align_metadata,
                audio,
                DEVICE,
                return_char_alignments=False,
            )
            aligned_segments = aligned.get("segments", segments) or segments
            word_segments = aligned.get("word_segments", []) or []
        except Exception as exc:  # noqa: BLE001
            align_warning = f"Alignment failed: {exc}"
            logger.warning(align_warning)

    # 4) Diarization
    # Backend selection (env-driven, see DIARIZATION_BACKEND):
    #   "diarize"  → CPU diarize lib only (no fallback)
    #   "pyannote" → pyannote.audio only (legacy path; needs HF_TOKEN)
    #   "auto"     → diarize first, fall back to pyannote on dominance failure
    diarize_lib = registry.load_diarize_lib() if DIARIZATION_BACKEND in {"diarize", "auto", "vibevoice"} else None
    diarize_default = (diarize_lib is not None) or (HF_TOKEN is not None)
    do_diarize = diarize_default if diarize_requested is None else bool(diarize_requested)
    diarized = False
    diarization_warning: Optional[str] = None
    diarization_records: List[Dict[str, Any]] = []
    diarization_confidence: Optional[float] = None
    diarization_resplit: bool = False
    diarization_backend_used: Optional[str] = None
    diarize_segments: Any = None  # raw object kept around for assign_word_speakers
    if do_diarize:
        # ── Phase A0: VibeVoice sidecar (optional, isolated venv) ──────────
        # When DIARIZATION_BACKEND=vibevoice, try the sidecar first. On any
        # failure (sidecar down, no GPU, model OOM, transcription error) we
        # fall through to the existing diarize/pyannote path so the request
        # never hard-fails just because the sidecar is unhappy.
        if DIARIZATION_BACKEND == "vibevoice":
            try:
                import vibevoice_client

                if vibevoice_client.is_available(timeout=2.0):
                    vv_payload = vibevoice_client.transcribe(audio_path)
                    vv_segments = vv_payload.get("segments") or []
                    if vv_segments:
                        # Replace WhisperX segments with VibeVoice's joint
                        # ASR+diar output. Word-level timestamps are not
                        # provided by VibeVoice; we leave words empty and
                        # let downstream renderers handle that gracefully.
                        aligned_segments = [
                            {
                                "start": float(s["start"]),
                                "end": float(s["end"]),
                                "text": str(s.get("text", "")).strip(),
                                "speaker": str(s["speaker"]),
                                "words": [],
                            }
                            for s in vv_segments
                            if s.get("text")
                        ]
                        word_segments = []
                        diarization_records = vibevoice_client.to_diarization_records(
                            vv_segments
                        )
                        diarize_segments = _records_to_dataframe(diarization_records)
                        diarization_backend_used = "vibevoice"
                        diarized = True
                        logger.info(
                            "VibeVoice produced %d segments across %d speakers (%.1fs latency)",
                            len(aligned_segments),
                            len({s["speaker"] for s in aligned_segments}),
                            float(vv_payload.get("latency_seconds", 0.0)),
                        )
                else:
                    logger.warning(
                        "VibeVoice sidecar unreachable — falling back to pyannote/diarize"
                    )
            except Exception as exc:  # noqa: BLE001
                diarization_warning = f"vibevoice backend failed: {exc}"
                logger.warning(diarization_warning)
                diarized = False

        # ── Smart default for short audio (instructions_16 Fix 2) ─────────
        # When the caller leaves both hints on "auto" and the audio is short
        # (< 10 min), pyannote's clustering is prone to collapsing two real
        # speakers into one. Seed min_speakers=2 so the very first diarization
        # call already biases towards splitting. Long audio is left alone —
        # pyannote behaves well on longer material and a hard floor of 2 would
        # incorrectly split true monologues.
        if (
            min_speakers is None
            and max_speakers is None
            and duration < 600.0
        ):
            logger.info(
                "No speaker hints provided, defaulting min_speakers=2 for short "
                "audio (%.1fs < 600s).", duration,
            )
            min_speakers = 2

        # ── Phase A: primary backend ──────────────────────────────────────
        primary_backend = "pyannote" if DIARIZATION_BACKEND == "pyannote" else "diarize"
        if primary_backend == "diarize" and diarize_lib is None:
            # Asked for diarize (or auto) but the library isn't installed —
            # silently degrade to pyannote so existing setups keep working.
            primary_backend = "pyannote"

        if not diarized and primary_backend == "diarize":
            try:
                diarization_records = _run_diarize_lib(
                    audio_path, min_speakers, max_speakers, diarize_lib
                )
                diarize_segments = _records_to_dataframe(diarization_records)
                diarization_backend_used = "diarize"
                diarized = True
                logger.info(
                    "Diarization (diarize lib) produced %d turns across %d speakers",
                    len(diarization_records),
                    len({r["speaker"] for r in diarization_records}),
                )
            except Exception as exc:  # noqa: BLE001
                diarization_warning = f"diarize backend failed: {exc}"
                logger.warning(diarization_warning)
                diarized = False

        if not diarized and primary_backend != "diarize":
            # Primary was already pyannote; will be handled by Phase B below.
            pass

        # ── Phase B: pyannote (used as primary or as auto-fallback) ────────
        need_pyannote = False
        if not diarized and DIARIZATION_BACKEND in {"pyannote", "auto", "vibevoice"}:
            need_pyannote = True
        if (
            DIARIZATION_BACKEND == "auto"
            and diarized
            and diarization_records
            and HF_TOKEN
        ):
            # If the diarize library produced a heavily-collapsed result and
            # the user expects ≥ 2 speakers, try pyannote as a second opinion.
            _, dom_check, _ = _dominant_speaker_ratio(diarization_records)
            wants_multi = (max_speakers is None or max_speakers >= 2) and (
                min_speakers is None or min_speakers != 1
            )
            if (
                dom_check >= DIARIZATION_DOMINANCE_THRESHOLD
                and wants_multi
                and len({r["speaker"] for r in diarization_records}) < 2
            ):
                logger.info(
                    "diarize backend collapsed to single speaker (dom=%.2f); "
                    "trying pyannote as second opinion.",
                    dom_check,
                )
                need_pyannote = True

        if need_pyannote:
            pipeline = registry.load_diarize()
            if pipeline is None:
                if not diarized and diarize_requested is True:
                    raise HTTPException(
                        400,
                        registry.diarization_error
                        or "Diarization requested but pipeline is unavailable. "
                        "Install `diarize` or set HF_TOKEN.",
                    )
                if not diarized:
                    diarization_warning = (
                        registry.diarization_error
                        or "Diarization unavailable: install `diarize` or set HF_TOKEN."
                    )
            else:
                try:
                    def _call_pyannote(extra: Optional[Dict[str, Any]] = None) -> Any:
                        diar_kwargs: Dict[str, Any] = {}
                        if min_speakers is not None:
                            diar_kwargs["min_speakers"] = min_speakers
                        if max_speakers is not None:
                            diar_kwargs["max_speakers"] = max_speakers
                        if extra:
                            diar_kwargs.update(extra)
                        return pipeline(str(audio_path), **diar_kwargs)

                    py_segments = _call_pyannote()
                    py_records = _diarize_dataframe_to_records(py_segments)

                    # Dominance re-split (instructions_16 Fix 3).
                    # If a single speaker dominates above threshold AND the
                    # effective min_speakers is >= 2, force pyannote to produce
                    # exactly that many clusters via num_speakers (which is a
                    # harder constraint than min_speakers). This works around
                    # WhisperX issue #516 where assign_word_speakers() can
                    # cascade the first segment's label across the whole
                    # transcript when pyannote's clustering is too permissive.
                    _, dom_ratio, _ = _dominant_speaker_ratio(py_records)
                    effective_min = min_speakers if min_speakers else 2
                    allow_resplit = (max_speakers is None or max_speakers >= 2)
                    if (
                        dom_ratio >= DIARIZATION_DOMINANCE_THRESHOLD
                        and allow_resplit
                        and effective_min >= 2
                    ):
                        logger.info(
                            "Detected single-speaker dominance (%.0f%%), "
                            "re-running pyannote with exact num_speakers=%d.",
                            dom_ratio * 100.0, effective_min,
                        )
                        prev_threshold = _tune_clustering_threshold(pipeline, 0.5)
                        try:
                            py_segments_2 = _call_pyannote(
                                {"num_speakers": effective_min}
                            )
                            records_2 = _diarize_dataframe_to_records(py_segments_2)
                            distinct_2 = len({r["speaker"] for r in records_2})
                            # Always adopt the forced re-run — even if it also
                            # collapses we surface a warning so the caller
                            # knows the result is unreliable.
                            py_segments = py_segments_2
                            py_records = records_2
                            diarization_resplit = True
                            if distinct_2 < effective_min:
                                msg = (
                                    f"Speaker separation may be unreliable: "
                                    f"forced num_speakers={effective_min} but "
                                    f"pyannote produced {distinct_2} distinct "
                                    f"cluster(s)."
                                )
                                logger.warning(msg)
                                if not diarization_warning:
                                    diarization_warning = msg
                        finally:
                            if prev_threshold is not None:
                                _tune_clustering_threshold(pipeline, prev_threshold)

                    # Adopt pyannote result only if it actually beats the
                    # diarize result (more speakers found).
                    diarize_distinct = len({r["speaker"] for r in diarization_records})
                    py_distinct = len({r["speaker"] for r in py_records})
                    if (not diarized) or py_distinct > diarize_distinct:
                        diarization_records = py_records
                        diarize_segments = py_segments
                        diarization_backend_used = "pyannote"
                        diarized = True
                        logger.info(
                            "Adopted pyannote result (%d speakers vs %d from diarize)",
                            py_distinct, diarize_distinct,
                        )

                    # Final per-speaker stats (instructions_16 Fix G).
                    try:
                        per_spk: Dict[str, int] = {}
                        for r in diarization_records:
                            per_spk[r["speaker"]] = per_spk.get(r["speaker"], 0) + 1
                        if per_spk:
                            logger.info(
                                "Diarization final stats (%s): %s",
                                diarization_backend_used,
                                ", ".join(
                                    f"{spk}={n}" for spk, n in sorted(per_spk.items())
                                ),
                            )
                    except Exception:  # noqa: BLE001
                        pass
                except Exception as exc:  # noqa: BLE001
                    msg = f"pyannote backend failed: {exc}"
                    logger.warning(msg)
                    if not diarized:
                        diarization_warning = msg

        # ── Phase B2: VBx Tier-2 referee (vendored BUTSpeechFIT/VBx) ────────
        # Trigger: any backend produced a result whose dominance is above the
        # VBx threshold (one speaker > 70% of speech) AND the user expects
        # ≥ 2 speakers. The VB-HMM temporal prior re-clusters per-window
        # x-vectors and reliably splits the "speaker returns after long gap"
        # case that AHC clustering merges. See vbx_diarize.py module docstring.
        vbx_applied = False
        if (
            VBX_ENABLED
            and diarized
            and diarization_records
            and diarize_lib is not None
        ):
            _, dom_t1, _ = _dominant_speaker_ratio(diarization_records)
            wants_multi_vbx = (max_speakers is None or max_speakers >= 2) and (
                min_speakers is None or min_speakers >= 2 or min_speakers == 0
            )
            if dom_t1 >= VBX_DOMINANCE_THRESHOLD and wants_multi_vbx:
                logger.info(
                    "VBx trigger: Tier-1 dominance %.2f >= %.2f — running VB-HMM resegmentation",
                    dom_t1, VBX_DOMINANCE_THRESHOLD,
                )
                try:
                    from vbx_diarize import vbx_resegment  # local import

                    refined = vbx_resegment(
                        str(audio_path),
                        diarization_records,
                        diarize_lib,
                        min_speakers=min_speakers,
                        max_speakers=max_speakers,
                    )
                    if refined:
                        _, dom_vbx, _ = _dominant_speaker_ratio(refined)
                        n_vbx = len({r["speaker"] for r in refined})
                        n_t1 = len({r["speaker"] for r in diarization_records})
                        # Adopt VBx result if it improved dominance (closer to
                        # balanced) and produced at least as many distinct
                        # speakers. Otherwise keep Tier-1 (safer).
                        if dom_vbx < dom_t1 and n_vbx >= max(n_t1, 2):
                            logger.info(
                                "VBx adopted: dom %.2f → %.2f, speakers %d → %d",
                                dom_t1, dom_vbx, n_t1, n_vbx,
                            )
                            diarization_records = refined
                            diarize_segments = _records_to_dataframe(refined)
                            diarization_backend_used = (
                                f"{diarization_backend_used}+vbx"
                                if diarization_backend_used else "vbx"
                            )
                            vbx_applied = True
                        else:
                            logger.info(
                                "VBx discarded: dom %.2f vs Tier-1 %.2f, speakers %d vs %d",
                                dom_vbx, dom_t1, n_vbx, n_t1,
                            )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("VBx resegmentation failed: %s", exc)

        # ── Phase C: shared post-processing (confidence + word assignment) ──
        if diarized and diarization_records:
            _, dom_after, _ = _dominant_speaker_ratio(diarization_records)
            diarization_confidence = round(
                max(0.0, 1.0 - max(0.0, dom_after - 0.5) * 2), 3
            )
            try:
                # whisperx.assign_word_speakers accepts either the pyannote
                # Annotation OR a pandas DataFrame with [start, end, speaker].
                # _records_to_dataframe handles the diarize-lib path.
                diar_for_assign = (
                    diarize_segments
                    if diarize_segments is not None
                    else _records_to_dataframe(diarization_records)
                )
                if diar_for_assign is not None:
                    merged = whisperx.assign_word_speakers(
                        diar_for_assign,
                        {"segments": aligned_segments, "word_segments": word_segments},
                    )
                    aligned_segments = merged.get("segments", aligned_segments) or aligned_segments
                    word_segments = merged.get("word_segments", word_segments) or word_segments
            except Exception as exc:  # noqa: BLE001
                # The nearest-by-time pass below will recover even if the
                # whisperx merger fails — log and continue.
                logger.warning("assign_word_speakers failed: %s", exc)

            # Per deepseek brief: explicit max-overlap mapper as a safety net
            # that GUARANTEES WhisperX segment start/end are never modified
            # (only `speaker` is set/overwritten). This is a no-op for segments
            # that whisperx.assign_word_speakers already labelled correctly,
            # but it heals any segment that the word-level merger left blank
            # (e.g. silence-bounded segments with zero word coverage).
            try:
                from vbx_diarize import map_diar_to_whisperx_segments
                map_diar_to_whisperx_segments(aligned_segments, diarization_records)
            except Exception as exc:  # noqa: BLE001
                logger.warning("segment-level diar mapping failed: %s", exc)

    # Reconstruct segment-level speakers from majority of word-level speakers
    if diarized:
        # First pass: derive speaker from word-level majority where missing.
        for seg in aligned_segments:
            if seg.get("speaker"):
                continue
            counts: Dict[str, float] = {}
            for w in seg.get("words", []) or []:
                spk = w.get("speaker")
                if not spk:
                    continue
                dur = max((w.get("end") or 0.0) - (w.get("start") or 0.0), 0.01)
                counts[spk] = counts.get(spk, 0.0) + dur
            if counts:
                seg["speaker"] = max(counts.items(), key=lambda kv: kv[1])[0]

        # Second pass: chronologically-closest fallback. For any segment still
        # without a usable speaker, find the diarization turn whose time range
        # is nearest to the segment's [start, end] (overlap distance = 0,
        # otherwise the gap to the closest edge). This is far better than
        # inheriting the previous segment's speaker, which causes long runs
        # to drift to whichever speaker happened to come first. The original
        # segment's start/end are NEVER overwritten — preventing the
        # "end = audio_duration" bug.
        def _nearest_diar_speaker(s_start: float, s_end: float) -> Optional[str]:
            if not diarization_records:
                return None
            best: Optional[Tuple[float, str]] = None
            for r in diarization_records:
                rs, re_, spk = r["start"], r["end"], r["speaker"]
                if not spk:
                    continue
                if rs <= s_end and re_ >= s_start:
                    dist = 0.0  # overlap
                elif re_ < s_start:
                    dist = s_start - re_
                else:
                    dist = rs - s_end
                if best is None or dist < best[0]:
                    best = (dist, spk)
            return best[1] if best else None

        for seg in aligned_segments:
            spk = seg.get("speaker")
            if isinstance(spk, str):
                spk_norm = spk.strip()
                if spk_norm and spk_norm.upper() != "UNKNOWN":
                    seg["speaker"] = spk_norm
                    continue
            s_start = float(seg.get("start") or 0.0)
            s_end = float(seg.get("end") or s_start)
            nearest = _nearest_diar_speaker(s_start, s_end)
            seg["speaker"] = nearest or "SPEAKER_00"

        # Same nearest-in-time normalisation for word-level labels.
        for w in word_segments:
            spk = w.get("speaker")
            if isinstance(spk, str):
                spk_norm = spk.strip()
                if spk_norm and spk_norm.upper() != "UNKNOWN":
                    w["speaker"] = spk_norm
                    continue
            w_start = float(w.get("start") or 0.0)
            w_end = float(w.get("end") or w_start)
            nearest = _nearest_diar_speaker(w_start, w_end)
            w["speaker"] = nearest or "SPEAKER_00"

    processing_time = time.time() - pipeline_started

    # Apply speaker name remapping if provided (before rendering)
    if speaker_map and diarized:
        apply_speaker_names(aligned_segments, word_segments, speaker_map)

    show_speakers = bool(diarized and include_speakers)
    if not include_speakers:
        _strip_speakers(aligned_segments, word_segments)

    # Ensure no segment is too long for subtitle display.
    aligned_segments = _split_long_segments(aligned_segments)

    # 5) Render outputs
    payload: Dict[str, Any] = {
        "language": detected_language,
        "duration_seconds": round(duration, 3),
        "processing_time_seconds": round(processing_time, 3),
        "transcribe_time_seconds": round(transcribe_time, 3),
        "realtime_factor": round(duration / processing_time, 2) if processing_time > 0 else None,
        "diarized": diarized,
        "diarization_warning": diarization_warning,
        "diarization_confidence": diarization_confidence,
        "diarization_resplit": diarization_resplit,
        "diarization_backend": diarization_backend_used,
        "diarization_vbx_applied": vbx_applied if diarized else False,
        "diarization_model": (
            registry.diarization_model_loaded
            if diarized and diarization_backend_used and "pyannote" in diarization_backend_used
            else (
                f"diarize {getattr(diarize_lib, '__version__', '')}".strip()
                + (" + VBx" if vbx_applied else "")
                if diarized and diarization_backend_used and "diarize" in diarization_backend_used
                else (
                    "VBx (BUTSpeechFIT, vendored)"
                    if diarized and diarization_backend_used == "vbx"
                    else None
                )
            )
        ),
        "alignment_warning": align_warning,
        "language_warning": language_warning,
        "segments": aligned_segments,
        "word_segments": word_segments,
    }

    files: Dict[str, str] = {}
    download_links: Dict[str, str] = {}
    rendered: Dict[str, str] = {}
    if "txt" in formats:
        rendered["txt"] = render_txt(aligned_segments, show_speakers)
    if "srt" in formats:
        rendered["srt"] = render_srt(aligned_segments, show_speakers)
    if "vtt" in formats:
        rendered["vtt"] = render_vtt(aligned_segments, show_speakers)
    if "tsv" in formats:
        rendered["tsv"] = render_tsv(aligned_segments, show_speakers)

    if rendered:
        file_id = uuid.uuid4().hex
        out_dir = DOWNLOAD_DIR / file_id
        out_dir.mkdir(parents=True, exist_ok=True)
        for ext, content in rendered.items():
            name = f"transcript.{ext}"
            (out_dir / name).write_text(content, encoding="utf-8")
            files[ext] = content
            files[f"{ext}_base64"] = base64.b64encode(content.encode("utf-8")).decode("ascii")
            download_links[ext] = f"/download/{file_id}/{name}"

    payload["files"] = files
    payload["download_links"] = download_links
    payload["download_expires_at"] = (datetime.utcnow() + timedelta(seconds=DOWNLOAD_TTL_SECONDS)).isoformat() + "Z"

    cleanup_old_downloads()
    return payload


# ---------------------------------------------------------------------------
# Batch processing — job store + /batch + /jobs endpoints
# ---------------------------------------------------------------------------

JOB_TTL_SECONDS = int(os.environ.get("JOB_TTL_SECONDS", "3600"))  # 1 hour


class JobStore:
    """In-memory async-safe job store with TTL cleanup."""

    def __init__(self) -> None:
        self._jobs: Dict[str, Any] = {}
        self._lock = asyncio.Lock()

    async def create(self, job_id: str, filenames: List[str]) -> Dict[str, Any]:
        job: Dict[str, Any] = {
            "job_id": job_id,
            "status": "pending",
            "created_at": datetime.utcnow().isoformat() + "Z",
            "total": len(filenames),
            "done": 0,
            "failed": 0,
            "results": [
                {"filename": fn, "status": "pending", "result": None, "error": None}
                for fn in filenames
            ],
        }
        async with self._lock:
            self._jobs[job_id] = job
        return job

    async def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        async with self._lock:
            return self._jobs.get(job_id)

    async def update_item(self, job_id: str, index: int, status: str, result: Any, error: Optional[str]) -> None:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job["results"][index]["status"] = status
            job["results"][index]["result"] = result
            job["results"][index]["error"] = error
            if status == "done":
                job["done"] += 1
            elif status == "failed":
                job["failed"] += 1
            total_finished = job["done"] + job["failed"]
            if total_finished >= job["total"]:
                job["status"] = "done" if job["failed"] == 0 else "partial"
            else:
                job["status"] = "processing"

    async def set_processing(self, job_id: str) -> None:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job["status"] = "processing"

    async def list_recent(self) -> List[Dict[str, Any]]:
        cutoff = time.time() - JOB_TTL_SECONDS
        async with self._lock:
            return [
                {k: v for k, v in j.items() if k != "results"}
                for j in self._jobs.values()
                if _iso_to_ts(j["created_at"]) > cutoff
            ]

    async def cleanup(self) -> None:
        cutoff = time.time() - JOB_TTL_SECONDS
        async with self._lock:
            expired = [jid for jid, j in self._jobs.items() if _iso_to_ts(j["created_at"]) <= cutoff]
            for jid in expired:
                del self._jobs[jid]


def _iso_to_ts(iso: str) -> float:
    try:
        return datetime.fromisoformat(iso.rstrip("Z")).timestamp()
    except Exception:  # noqa: BLE001
        return 0.0


job_store = JobStore()


@app.post("/batch")
async def batch_transcribe(
    files: List[UploadFile] = File(..., description="One or more audio files to transcribe."),
    language: Optional[str] = Form(None),
    diarize: Optional[bool] = Form(None),
    min_speakers: Optional[int] = Form(None),
    max_speakers: Optional[int] = Form(None),
    vad: bool = Form(True),
    output_format: str = Form("json", description="Comma-separated list: json,txt,srt,vtt,tsv"),
    initial_prompt: Optional[str] = Form(None),
    task: str = Form("transcribe"),
    speaker_names: Optional[str] = Form(None),
    include_speakers: bool = Form(True, description="Include speaker labels in output."),
) -> JSONResponse:
    """Process multiple audio files sequentially. Blocks until all files are done.
    Works on ZeroGPU (HF Spaces) because GPU access is held for the duration of the call.
    """
    if not files:
        raise HTTPException(400, "No files provided.")
    if task not in ("transcribe", "translate"):
        raise HTTPException(400, "task must be 'transcribe' or 'translate'.")

    formats = parse_output_formats(output_format)
    speaker_map = parse_speaker_names(speaker_names)

    file_paths: List[Path] = []
    filenames: List[str] = []
    work_dirs: List[Path] = []

    # Save all uploads to disk first
    for upload in files:
        suffix = Path(upload.filename or "").suffix.lower()
        if suffix and suffix not in ALLOWED_AUDIO_SUFFIXES:
            raise HTTPException(415, f"Unsupported file extension '{suffix}' in '{upload.filename}'.")
        work_dir = Path(tempfile.mkdtemp(prefix="whisperx_batch_"))
        audio_path = work_dir / f"input{suffix or '.wav'}"
        total = 0
        with open(audio_path, "wb") as f:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_FILE_SIZE:
                    shutil.rmtree(work_dir, ignore_errors=True)
                    raise HTTPException(413, f"File '{upload.filename}' exceeds max size.")
                f.write(chunk)
        if total == 0:
            shutil.rmtree(work_dir, ignore_errors=True)
            raise HTTPException(400, f"File '{upload.filename}' is empty.")
        file_paths.append(audio_path)
        filenames.append(upload.filename or f"file_{len(filenames)}")
        work_dirs.append(work_dir)

    # Process each file synchronously (required for ZeroGPU GPU access)
    results = []
    done_count = 0
    failed_count = 0
    for audio_path, filename, work_dir in zip(file_paths, filenames, work_dirs):
        try:
            result = _run_pipeline(
                audio_path=audio_path,
                language=language,
                diarize_requested=diarize,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                vad=vad,
                formats=formats,
                initial_prompt=initial_prompt,
                task=task,
                speaker_map=speaker_map,
                include_speakers=include_speakers,
            )
            results.append({"filename": filename, "status": "done", "result": result, "error": None})
            done_count += 1
        except Exception as exc:  # noqa: BLE001
            logger.error("Batch file %s failed: %s", filename, exc)
            results.append({"filename": filename, "status": "failed", "result": None, "error": str(exc)})
            failed_count += 1
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)
            free_gpu_memory()

    overall = "done" if failed_count == 0 else "partial"
    return JSONResponse({
        "status": overall,
        "total": len(files),
        "done": done_count,
        "failed": failed_count,
        "results": results,
    })


@app.post("/batch-align")
async def batch_align(
    files: List[UploadFile] = File(..., description="Audio files — one per row, in order."),
    transcripts: List[str] = Form(..., description="Transcripts — one per file, in order."),
    languages: Optional[str] = Form(None, description="JSON array of language codes, one per file, e.g. [\"fr\",\"en\",null]. null = auto-detect."),
    diarize: Optional[bool] = Form(None),
    min_speakers: Optional[int] = Form(None),
    max_speakers: Optional[int] = Form(None),
    vad: bool = Form(True),
    output_format: str = Form("json", description="Comma-separated list: json,txt,srt,vtt,tsv"),
    speaker_names: Optional[str] = Form(None),
    include_speakers: bool = Form(True, description="Include speaker labels in output."),
) -> JSONResponse:
    """Align multiple audio+transcript pairs in one request.
    files and transcripts must be provided in the same order.
    Works on ZeroGPU (HF Spaces) — GPU is held for the full duration.
    """
    try:
        return await _batch_align_impl(
            files=files,
            transcripts=transcripts,
            languages=languages,
            diarize=diarize,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            vad=vad,
            output_format=output_format,
            speaker_names=speaker_names,
            include_speakers=include_speakers,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("batch_align unhandled error: %s\n%s", exc, traceback.format_exc())
        raise HTTPException(500, f"Batch align error: {exc}") from exc


async def _batch_align_impl(
    files: List[UploadFile],
    transcripts: List[str],
    languages: Optional[str],
    diarize: Optional[bool],
    min_speakers: Optional[int],
    max_speakers: Optional[int],
    vad: bool,
    output_format: str,
    speaker_names: Optional[str],
    include_speakers: bool = True,
) -> JSONResponse:
    if not files:
        raise HTTPException(400, "No files provided.")
    if len(transcripts) != len(files):
        raise HTTPException(400, f"Got {len(files)} file(s) but {len(transcripts)} transcript(s). They must match.")

    # Parse per-file languages: JSON array or None (use same for all)
    per_file_languages: List[Optional[str]] = [None] * len(files)
    if languages:
        try:
            lang_list = json.loads(languages)
            if not isinstance(lang_list, list):
                raise ValueError("languages must be a JSON array")
            if len(lang_list) != len(files):
                raise HTTPException(400, f"languages array has {len(lang_list)} entries but {len(files)} files.")
            per_file_languages = [str(l).strip() if l else None for l in lang_list]
        except json.JSONDecodeError as exc:
            raise HTTPException(400, f"Invalid JSON in languages: {exc}") from exc

    formats = parse_output_formats(output_format)
    speaker_map = parse_speaker_names(speaker_names)

    # Save all uploads
    file_paths: List[Path] = []
    filenames: List[str] = []
    work_dirs: List[Path] = []
    for upload in files:
        suffix = Path(upload.filename or "").suffix.lower()
        if suffix and suffix not in ALLOWED_AUDIO_SUFFIXES:
            raise HTTPException(415, f"Unsupported file extension '{suffix}' in '{upload.filename}'.")
        work_dir = Path(tempfile.mkdtemp(prefix="whisperx_balign_"))
        audio_path = work_dir / f"input{suffix or '.wav'}"
        total_bytes = 0
        with open(audio_path, "wb") as f:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > MAX_FILE_SIZE:
                    shutil.rmtree(work_dir, ignore_errors=True)
                    raise HTTPException(413, f"File '{upload.filename}' exceeds max size.")
                f.write(chunk)
        if total_bytes == 0:
            shutil.rmtree(work_dir, ignore_errors=True)
            raise HTTPException(400, f"File '{upload.filename}' is empty.")
        file_paths.append(audio_path)
        filenames.append(upload.filename or f"file_{len(filenames)}")
        work_dirs.append(work_dir)

    # Process each pair sequentially
    results = []
    done_count = 0
    failed_count = 0
    for audio_path, filename, work_dir, transcript, lang in zip(
        file_paths, filenames, work_dirs, transcripts, per_file_languages
    ):
        try:
            result = _run_align_pipeline(
                audio_path=audio_path,
                transcript_text=transcript,
                language=lang,
                diarize=bool(diarize),
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                vad=vad,
                formats=formats,
                speaker_map=speaker_map,
                include_speakers=include_speakers,
            )
            results.append({"filename": filename, "status": "done", "result": result, "error": None})
            done_count += 1
        except Exception as exc:  # noqa: BLE001
            logger.error("Batch-align file %s failed: %s", filename, exc)
            results.append({"filename": filename, "status": "failed", "result": None, "error": str(exc)})
            failed_count += 1
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)
            free_gpu_memory()

    overall = "done" if failed_count == 0 else "partial"
    return JSONResponse({
        "status": overall,
        "total": len(files),
        "done": done_count,
        "failed": failed_count,
        "results": results,
    })


@app.get("/jobs/{job_id}")
async def get_job(job_id: str) -> JSONResponse:
    """Get the status and results of a batch job."""
    job = await job_store.get(job_id)
    if job is None:
        raise HTTPException(404, f"Job '{job_id}' not found or expired.")
    return JSONResponse(job)


@app.get("/jobs")
async def list_jobs() -> JSONResponse:
    """List recent batch jobs (summary, no per-file results)."""
    jobs = await job_store.list_recent()
    return JSONResponse({"jobs": jobs, "count": len(jobs)})


# ---------------------------------------------------------------------------
# Real-time streaming STT (WebSocket /ws/stt)
# ---------------------------------------------------------------------------
#
# Wire protocol (browser <-> server):
#
#   Client → Server (text JSON):
#     {"type":"start", "language": "en"|null}
#     {"type":"end"}
#
#   Client → Server (binary):
#     Raw PCM, mono, 16 kHz, little-endian Float32 frames (any length).
#
#   Server → Client (text JSON):
#     {"type":"ready",   "session_id": "..."}
#     {"type":"final",   "segment_id": N, "text": "...", "start": s, "end": s,
#                        "words": [{"word":"...","start":s,"end":s}, ...]}
#     {"type":"done",    "srt_url": "...", "txt_url": "...", "duration_s": s,
#                        "segments": N}
#     {"type":"error",   "error": "..."}
#
# Design:
#   * Audio is buffered server-side as a single Float32 numpy array at 16 kHz.
#   * Every ~1s, a periodic task tries to commit the next chunk:
#       - chunks are CHUNK_DURATION_S long with OVERLAP_S leading overlap into
#         the previously-committed audio (helps Whisper avoid splitting words).
#       - the committed chunk is transcribed (WhisperX / faster-whisper) then
#         force-aligned (wav2vec2). Words whose end-time falls inside the
#         already-committed region are discarded as duplicates.
#   * On {"type":"end"} the remaining tail is force-committed and a session
#     transcript is written to DOWNLOAD_DIR/<session_id>/{transcript.srt,
#     transcript.txt}, served by the existing /download endpoint.
#
# Notes:
#   * No diarization is performed inline (kept for batch /transcribe to keep
#     latency low). Speaker labels could be appended later via a separate pass.
#   * The handler is GPU-bound; it serializes work behind asyncio.to_thread to
#     keep the event loop responsive. Concurrent streaming sessions on a single
#     RTX 4090 are technically possible but may compete for VRAM.

DEFAULT_STREAM_INITIAL_PROMPT = (
    "This transcript includes numbers, years, and dollar amounts."
)

# Common WhisperX/faster-whisper hallucination phrases that appear on
# silence, music, or fast/noisy speech. Normalized form (lowercase, no punct).
_WHISPER_HALLUCINATIONS = frozenset({
    "thank you", "thanks", "thanks for watching", "thank you for watching",
    "thank you so much", "thank you very much", "thanks a lot",
    "thank you for your attention", "thank you bye", "thank you thank you",
    "bye", "bye bye", "okay bye", "okay", "ok", "you", "yeah",
    "im", "im im", "im im im", "hmm", "uh", "um",
    "subscribe", "please subscribe", "subtitles by the amaraorg community",
    "music", "applause",
})

# Tokens that, when standing alone or repeated, are almost always hallucinations
# from the chunk-tail force-commit on near-silence audio.
_HALLUCINATION_TOKENS = frozenset({
    "thank", "thanks", "im", "hmm", "uh", "um", "you", "okay", "ok", "bye",
})

# Known proper-name corrections (case-insensitive substring replacement on
# whole words). Built-in defaults; extended at startup from names.json if
# present (so users can edit corrections without touching server code).
_NAME_CORRECTIONS: Dict[str, str] = {
    "Sam Maltman": "Sam Altman",
    "Sal Maltman": "Sam Altman",
    "Sal Altman": "Sam Altman",
    "Amadeo Enthropic": "Dario Amodei",
    "Amadea Anthropic": "Dario Amodei",
    "Amadeo Anthropic": "Dario Amodei",
    "Amadea Enthropic": "Dario Amodei",
    "Amadei and Fropec": "Dario Amodei from Anthropic",
    "Amadei and Fropek": "Dario Amodei from Anthropic",
    "Amadei Fropec": "Dario Amodei",
    "Enthropic": "Anthropic",
    "Fropec": "Anthropic",
    "Fropek": "Anthropic",
}


def _load_names_json() -> None:
    """Merge optional names.json (alongside server.py) into _NAME_CORRECTIONS.
    File format: {"Wrong Name": "Right Name", ...}. Silently ignored if
    missing or malformed — user-editable without touching code.
    """
    try:
        path = Path(__file__).parent / "names.json"
        if not path.is_file():
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            added = 0
            for k, v in data.items():
                if isinstance(k, str) and isinstance(v, str) and k.strip() and v.strip():
                    _NAME_CORRECTIONS[k] = v
                    added += 1
            logger.info("Loaded %d name corrections from names.json", added)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load names.json: %s", exc)


_load_names_json()


def _apply_name_corrections(text: str) -> str:
    out = text
    for wrong, right in _NAME_CORRECTIONS.items():
        out = re.sub(r"\b" + re.escape(wrong) + r"\b", right, out, flags=re.IGNORECASE)
    return out


class StreamingSession:
    SAMPLE_RATE = 16000
    CHUNK_DURATION_S = 5.0     # how much new audio to commit per pass
    OVERLAP_S = 0.5            # overlap into prior audio for context
    MIN_TAIL_S = 0.3           # below this, finalize() will skip processing
    MAX_BUFFER_S = 90.0        # safety cap to bound memory use

    def __init__(
        self,
        websocket: WebSocket,
        language: Optional[str] = None,
        initial_prompt: Optional[str] = None,
    ) -> None:
        self.ws = websocket
        self.language = (language or "").strip().lower() or None
        # Optional Whisper conditioning prompt — biases the decoder toward
        # numbers, years, dollar amounts (instructions_10). MLX backend
        # ignores this; whisperx/faster-whisper consumes it.
        ip = (initial_prompt or "").strip()
        self.initial_prompt: Optional[str] = ip if ip else DEFAULT_STREAM_INITIAL_PROMPT
        self.session_id = uuid.uuid4().hex
        self.buffer = np.zeros(0, dtype=np.float32)
        self.committed_samples = 0  # samples of buffer[] already finalized
        self.absolute_offset_s = 0.0  # wall-time second corresponding to buffer[0]
        self.segments: List[Dict[str, Any]] = []
        self.next_segment_id = 0
        self.lock = asyncio.Lock()
        self.started_at = time.time()
        # Hard-stop flag: when set (by /ws/stt on `end` or disconnect),
        # finalize() will NOT process any remaining tail audio. This prevents
        # the post-stop hallucinated tail (e.g. "I'm... Thank you.").
        self.stop_requested = False
        logger.info("Streaming session %s started (language=%s)", self.session_id, self.language)

    # ---- audio ingestion -------------------------------------------------

    def add_audio(self, pcm: np.ndarray) -> None:
        if pcm.size == 0:
            return
        self.buffer = np.concatenate([self.buffer, pcm.astype(np.float32, copy=False)])

    # ---- chunking driver -------------------------------------------------

    async def maybe_process(self) -> None:
        # Brief lock to check how much unprocessed audio exists.
        async with self.lock:
            unprocessed = len(self.buffer) - self.committed_samples
        if unprocessed < int(self.CHUNK_DURATION_S * self.SAMPLE_RATE):
            return
        await self._commit_one_chunk(force=False)

    async def finalize(self) -> Dict[str, Any]:
        # Hard-stop: if the user clicked Stop (or disconnected), do NOT process
        # any remaining tail audio. Whisper hallucinates filler tokens
        # ("Thank you.", "I'm I'm") on the trailing silence after speech ends,
        # and the user expects Stop to mean Stop. Just flush what we have.
        if not self.stop_requested:
            # Force-commit any remaining audio in chunks of CHUNK_DURATION_S.
            # Each pass either advances committed_samples (success or skipped silence)
            # or force-advances on error to avoid infinite loops.
            while True:
                async with self.lock:
                    tail = len(self.buffer) - self.committed_samples
                    prev_committed = self.committed_samples
                if tail < int(self.MIN_TAIL_S * self.SAMPLE_RATE):
                    break
                await self._commit_one_chunk(force=True)
                async with self.lock:
                    if self.committed_samples == prev_committed:
                        self.committed_samples = min(
                            prev_committed + int(self.CHUNK_DURATION_S * self.SAMPLE_RATE),
                            len(self.buffer),
                        )
        async with self.lock:
            outputs = self._write_outputs()
            outputs["duration_s"] = round(
                self.absolute_offset_s + (len(self.buffer) / self.SAMPLE_RATE), 3
            )
            outputs["segments"] = len(self.segments)
        logger.info(
            "Streaming session %s finalized (%d segments, %.1fs)",
            self.session_id, len(self.segments), outputs["duration_s"],
        )
        return outputs

    # ---- speech / silence gating -----------------------------------------

    @staticmethod
    def _norm_word(w: str) -> str:
        """Lowercase, strip non-alphanumerics for robust word comparison."""
        return "".join(c for c in (w or "").lower() if c.isalnum())

    def _chunk_has_speech(self, audio: np.ndarray) -> bool:
        """Return True if the chunk contains real speech.

        Two-stage gate: cheap RMS energy filter (rejects dead silence) followed
        by Silero VAD (rejects background noise / breath / hum). Fail-open: if
        VAD is unavailable or errors, fall back to the energy decision.
        """
        if audio.size < int(0.3 * self.SAMPLE_RATE):
            return False
        try:
            rms = float(np.sqrt(np.mean(np.square(audio.astype(np.float64)))))
        except Exception:
            rms = 0.0
        # ~ -50 dBFS — anything below this is room tone / dead silence.
        if rms < 0.003:
            return False
        vad_fn = registry.try_load_silero_vad()
        if vad_fn is None:
            return True
        try:
            ts = vad_fn(audio, min_ms=300)
            return bool(ts) and len(ts) > 0
        except Exception as exc:  # noqa: BLE001
            logger.debug("Silero VAD call failed (falling back to energy): %s", exc)
            return True

    # Words for which immediate repetition is intentional and should be kept.
    _ALLOWED_REPEATS = frozenset({
        "very", "really", "no", "yes", "yeah", "ha", "haha",
        "ok", "okay", "bye", "hi", "so", "well",
    })

    @staticmethod
    def _edit_distance_le_1(a: str, b: str) -> bool:
        """Fast check: True iff Levenshtein(a, b) <= 1. Both strings expected
        to be already normalized (lowercase, alnum only)."""
        if a == b:
            return True
        la, lb = len(a), len(b)
        if abs(la - lb) > 1:
            return False
        if la == lb:
            diffs = sum(1 for x, y in zip(a, b) if x != y)
            return diffs <= 1
        # One insertion/deletion. Walk the longer with a one-char skip budget.
        if la > lb:
            a, b = b, a
            la, lb = lb, la
        i = j = 0
        skipped = False
        while i < la and j < lb:
            if a[i] == b[j]:
                i += 1
                j += 1
            elif skipped:
                return False
            else:
                skipped = True
                j += 1
        return True

    def _filter_repeat_bigrams(self, words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Drop the first of two adjacent words that are identical or
        near-identical (edit distance ≤ 1) when their time-gap is ≤ 1.5s and
        the word is not in the allowed-repeats list. Catches "AIC, AI",
        "fall in furthest", "the operator. the operating" boundary remnants.
        """
        if len(words) < 2:
            return words
        out: List[Dict[str, Any]] = []
        for w in words:
            if out:
                prev = out[-1]
                a = self._norm_word(prev.get("word", ""))
                b = self._norm_word(w.get("word", ""))
                if a and b and len(a) >= 2 and len(b) >= 2 and a not in self._ALLOWED_REPEATS:
                    try:
                        gap = float(w.get("start", 0.0)) - float(prev.get("end", 0.0))
                    except Exception:
                        gap = 999.0
                    if gap <= 1.5 and self._edit_distance_le_1(a, b):
                        # Drop the shorter / earlier one (prev) — keep the
                        # later word, which is more likely to be the
                        # speaker's actual word.
                        out.pop()
            out.append(w)
        return out

    def _dedup_against_prior(self, new_words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove the leading words of `new_words` that duplicate the trailing
        words of the prior committed transcript.

        Whisper re-transcribes the overlap region in successive chunks. Pure
        timestamp-based filtering is unreliable because Whisper's word timings
        drift by 50–200ms across chunks. We do three passes:

        1. Find the longest k such that the new chunk's first k normalized
           words appear as a contiguous run *anywhere* in the prior transcript's
           trailing 24 words. Drop those k words from new. This is more lenient
           than strict suffix-prefix matching and catches cases like prior
           ending with a stray "a" or punctuation token.
        2. As a fallback, drop new[0] if it equals prior[-1] AND the time gap
           is < 0.3s (handles single-word boundary repeats).
        """
        if not new_words or not self.segments:
            return new_words
        # Collect up to 24 trailing words from prior segments.
        prior_words: List[Dict[str, Any]] = []
        for seg in reversed(self.segments):
            prior_words = list(seg.get("words", [])) + prior_words
            if len(prior_words) >= 24:
                break
        if not prior_words:
            return new_words
        prior_words = prior_words[-24:]
        prior_norm = [self._norm_word(w.get("word", "")) for w in prior_words]
        new_norm = [self._norm_word(w.get("word", "")) for w in new_words]

        # Pass 1: find longest k (>=2) such that new_norm[:k] appears as a
        # contiguous sublist within prior_norm. Prefer larger k.
        max_k = min(len(new_norm), len(prior_norm), 14)
        for k in range(max_k, 1, -1):
            needle = new_norm[:k]
            if not all(needle):
                continue
            for i in range(len(prior_norm) - k + 1):
                if prior_norm[i:i + k] == needle:
                    return new_words[k:]

        # Pass 2: single-word boundary dedup with tight time gap (≤ 0.3s).
        if new_norm and prior_norm and new_norm[0] and new_norm[0] == prior_norm[-1]:
            try:
                gap = float(new_words[0].get("start", 0.0)) - float(prior_words[-1].get("end", 0.0))
            except Exception:
                gap = 999.0
            if gap <= 0.3:
                return new_words[1:]
        return new_words

    def _dedup_within_chunk(self, words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove adjacent repeated 2-3 word phrases produced by Whisper
        hallucination loops within a single chunk (e.g. 'see it as innovation.
        See it as innovation.')."""
        if len(words) < 4:
            return words
        out: List[Dict[str, Any]] = []
        for w in words:
            out.append(w)
            for n in (3, 2):
                if len(out) >= 2 * n:
                    a = [self._norm_word(out[-2 * n + i].get("word", "")) for i in range(n)]
                    b = [self._norm_word(out[-n + i].get("word", "")) for i in range(n)]
                    if a == b and all(a):
                        del out[-n:]
                        break
        return out

    @staticmethod
    def _is_hallucination_only(words: List[Dict[str, Any]]) -> bool:
        """True if the chunk's content is entirely a known Whisper hallucination
        phrase or short repeated filler. Used to drop the chunk completely."""
        if not words:
            return True
        norm_tokens = [
            "".join(c for c in (w.get("word", "") or "").lower() if c.isalnum())
            for w in words
        ]
        norm_tokens = [t for t in norm_tokens if t]
        if not norm_tokens:
            return True
        norm_text = " ".join(norm_tokens)
        if norm_text in _WHISPER_HALLUCINATIONS:
            return True
        # Short content (≤ 4 tokens) made entirely of hallucination tokens.
        if len(norm_tokens) <= 4 and all(t in _HALLUCINATION_TOKENS for t in norm_tokens):
            return True
        return False

    async def _commit_one_chunk(self, *, force: bool) -> None:
        # ── Phase 1: snapshot state under lock (fast) ──────────────────────
        async with self.lock:
            chunk_target = self.committed_samples + int(self.CHUNK_DURATION_S * self.SAMPLE_RATE)
            chunk_end = min(chunk_target, len(self.buffer))
            if not force and chunk_end < chunk_target:
                return
            if chunk_end <= self.committed_samples:
                return

            chunk_start = max(0, self.committed_samples - int(self.OVERLAP_S * self.SAMPLE_RATE))
            chunk_audio = self.buffer[chunk_start:chunk_end].copy()
            offset_s = self.absolute_offset_s + (chunk_start / self.SAMPLE_RATE)
            non_overlap_start_s = self.absolute_offset_s + (self.committed_samples / self.SAMPLE_RATE)
            committed_before = self.committed_samples
            # Tentative commit — prevents another concurrent tick from re-processing
            # this region.  Rolled back if inference fails.
            self.committed_samples = chunk_end
        # ── Lock released ── audio ingest is unblocked during GPU inference ─

        # VAD-gate: silent / non-speech chunks would make Whisper hallucinate
        # filler tokens ("Thank you.", "I'm I'm", ".") so skip them entirely.
        # The committed_samples advance above is kept (no rollback) — the
        # silence is simply dropped from the transcript.
        if not self._chunk_has_speech(chunk_audio):
            logger.debug(
                "Streaming session %s: chunk [%.2f-%.2f] skipped (no speech)",
                self.session_id, offset_s, offset_s + chunk_audio.size / self.SAMPLE_RATE,
            )
            async with self.lock:
                self._trim_buffer_if_large()
            return

        try:
            result = await asyncio.to_thread(
                self._transcribe_and_align, chunk_audio, offset_s, non_overlap_start_s,
            )
        except asyncio.CancelledError:
            # The periodic task was cancelled (session ending). Roll back so that
            # finalize() can re-process this audio chunk cleanly.
            async with self.lock:
                self.committed_samples = committed_before
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("Streaming chunk transcribe/align failed")
            # Rollback so the chunk is retried on the next tick.
            async with self.lock:
                self.committed_samples = committed_before
            try:
                await self.ws.send_json({"type": "error", "error": f"chunk processing failed: {exc}"})
            except Exception:
                pass
            return

        # ── Phase 2: append result under lock (fast) ────────────────────────
        seg = None
        async with self.lock:
            if result and result.get("words"):
                # 1) Drop the entire chunk if it's a known Whisper hallucination
                #    phrase (e.g. "Thank you.", "I'm I'm", "Bye.").
                if self._is_hallucination_only(result["words"]):
                    logger.debug(
                        "Streaming session %s: chunk dropped (hallucination-only): %r",
                        self.session_id, result.get("text"),
                    )
                else:
                    # 2) Cross-chunk dedup against trailing words of prior segments.
                    deduped = self._dedup_against_prior(result["words"])
                    # 3) Intra-chunk dedup of repeated 2-3 word phrases.
                    deduped = self._dedup_within_chunk(deduped)
                    # 4) Bigram near-identical filter (instructions_10).
                    deduped = self._filter_repeat_bigrams(deduped)
                    # 5) Cross-chunk single-word near-dupe at the boundary
                    #    (e.g. prior ends "AIC", new starts "AI" → drop prior's "AIC").
                    if deduped and self.segments:
                        prev_seg = self.segments[-1]
                        prev_words = prev_seg.get("words") or []
                        if prev_words:
                            a = self._norm_word(prev_words[-1].get("word", ""))
                            b = self._norm_word(deduped[0].get("word", ""))
                            if (
                                a and b and len(a) >= 2 and len(b) >= 2
                                and a not in self._ALLOWED_REPEATS
                                and self._edit_distance_le_1(a, b)
                            ):
                                try:
                                    gap = float(deduped[0].get("start", 0.0)) - float(prev_words[-1].get("end", 0.0))
                                except Exception:
                                    gap = 999.0
                                if gap <= 1.5:
                                    prev_words.pop()
                                    if prev_words:
                                        prev_seg["text"] = " ".join(
                                            w["word"] for w in prev_words if w.get("word")
                                        ).strip()
                                        prev_seg["start"] = prev_words[0]["start"]
                                        prev_seg["end"] = prev_words[-1]["end"]
                                    else:
                                        # Prior segment is now empty; remove it.
                                        self.segments.pop()
                    if deduped:
                        text = " ".join(w["word"] for w in deduped if w.get("word")).strip()
                        if text:
                            seg_id = self.next_segment_id
                            self.next_segment_id += 1
                            seg = {
                                "segment_id": seg_id,
                                "text": text,
                                "start": deduped[0]["start"],
                                "end": deduped[-1]["end"],
                                "words": deduped,
                            }
                            self.segments.append(seg)
            self._trim_buffer_if_large()

        if seg is not None:
            try:
                await self.ws.send_json({"type": "final", **seg})
            except Exception:
                pass

    def _trim_buffer_if_large(self) -> None:
        if len(self.buffer) <= int(self.MAX_BUFFER_S * self.SAMPLE_RATE):
            return
        keep_from = max(0, self.committed_samples - int(2.0 * self.SAMPLE_RATE))
        if keep_from <= 0:
            return
        self.buffer = self.buffer[keep_from:]
        self.absolute_offset_s += keep_from / self.SAMPLE_RATE
        self.committed_samples -= keep_from

    # ---- inference (blocking; runs in a thread) --------------------------

    def _transcribe_and_align(
        self, audio: np.ndarray, offset_s: float, non_overlap_start_s: float,
    ) -> Optional[Dict[str, Any]]:
        whisper = registry.load_whisper()

        if EFFECTIVE_BACKEND == "mlx":
            mlx_kwargs: Dict[str, Any] = {}
            if self.language:
                mlx_kwargs["language"] = self.language
            transcription = whisper.transcribe(audio, **mlx_kwargs)
        else:
            kwargs: Dict[str, Any] = {"batch_size": 4, "chunk_size": 30}
            if self.language:
                kwargs["language"] = self.language
            if self.initial_prompt:
                kwargs["initial_prompt"] = self.initial_prompt
            transcription = whisper.transcribe(audio, **kwargs)

        segments = transcription.get("segments") or []
        if not segments:
            return None
        detected_language = transcription.get("language") or self.language or "en"

        # Forced alignment for word-level timestamps
        words: List[Dict[str, Any]] = []
        align_loaded = registry.load_align(detected_language)
        if align_loaded is None and detected_language != "en":
            align_loaded = registry.load_align("en")
        if align_loaded is not None:
            try:
                align_model, align_metadata = align_loaded
                aligned = whisperx.align(
                    segments, align_model, align_metadata, audio, DEVICE,
                    return_char_alignments=False,
                )
                for w in aligned.get("word_segments", []) or []:
                    if w.get("start") is None or w.get("end") is None:
                        continue
                    abs_start = float(w["start"]) + offset_s
                    abs_end = float(w["end"]) + offset_s
                    # Discard words from the overlap region; they were already
                    # emitted by the previous chunk.
                    if abs_end <= non_overlap_start_s + 1e-3:
                        continue
                    words.append({
                        "word": str(w.get("word", "")).strip(),
                        "start": round(abs_start, 3),
                        "end": round(abs_end, 3),
                    })
            except Exception as exc:  # noqa: BLE001
                logger.warning("Streaming alignment failed: %s", exc)

        if not words:
            # Fallback: synthesize approximate per-word timings from segment text
            for s in segments:
                s_start = float(s.get("start", 0.0) or 0.0) + offset_s
                s_end = float(s.get("end", 0.0) or 0.0) + offset_s
                if s_end <= non_overlap_start_s + 1e-3:
                    continue
                tokens = (s.get("text", "") or "").split()
                if not tokens:
                    continue
                step = max((s_end - s_start) / max(len(tokens), 1), 0.05)
                for i, tok in enumerate(tokens):
                    words.append({
                        "word": tok,
                        "start": round(s_start + i * step, 3),
                        "end": round(s_start + (i + 1) * step, 3),
                    })

        if not words:
            return None
        text = " ".join(w["word"] for w in words if w["word"]).strip()
        if not text:
            return None
        return {
            "text": text,
            "start": words[0]["start"],
            "end": words[-1]["end"],
            "words": words,
        }

    # ---- output writers --------------------------------------------------

    def _write_outputs(self) -> Dict[str, str]:
        out_dir = DOWNLOAD_DIR / self.session_id
        out_dir.mkdir(parents=True, exist_ok=True)

        srt_lines: List[str] = []
        cue_idx = 1
        cur_words: List[Dict[str, Any]] = []
        cur_chars = 0
        MAX_CUE_CHARS = 80

        def flush_cue() -> None:
            nonlocal cue_idx, cur_words, cur_chars
            if not cur_words:
                return
            cue_text = " ".join(w["word"] for w in cur_words if w["word"]).strip()
            if cue_text:
                srt_lines.append(str(cue_idx))
                srt_lines.append(
                    f"{srt_timestamp(cur_words[0]['start'])} --> {srt_timestamp(cur_words[-1]['end'])}"
                )
                srt_lines.append(cue_text)
                srt_lines.append("")
                cue_idx += 1
            cur_words = []
            cur_chars = 0

        for seg in self.segments:
            for w in seg["words"]:
                cur_words.append(w)
                cur_chars += len(w["word"]) + 1
                if cur_chars >= MAX_CUE_CHARS:
                    flush_cue()
            # Always break cue at segment boundary for readability.
            flush_cue()
        flush_cue()

        srt_path = out_dir / "transcript.srt"
        srt_path.write_text("\n".join(srt_lines), encoding="utf-8")

        txt_path = out_dir / "transcript.txt"
        txt_body = " ".join(seg["text"] for seg in self.segments).strip()
        txt_body = _apply_name_corrections(txt_body) + "\n"
        txt_path.write_text(txt_body, encoding="utf-8")

        return {
            "srt_url": f"/download/{self.session_id}/transcript.srt",
            "txt_url": f"/download/{self.session_id}/transcript.txt",
        }


@app.websocket("/ws/stt")
async def websocket_stt(ws: WebSocket) -> None:
    """Real-time streaming STT over WebSocket.

    See StreamingSession for the wire protocol.
    """
    await ws.accept()
    session: Optional[StreamingSession] = None
    process_task: Optional[asyncio.Task] = None

    async def periodic_processor() -> None:
        try:
            while True:
                await asyncio.sleep(1.0)
                if session is None:
                    continue
                try:
                    await session.maybe_process()
                except Exception:  # noqa: BLE001
                    logger.exception("Streaming periodic processing error")
        except asyncio.CancelledError:
            pass

    try:
        while True:
            msg = await ws.receive()
            mtype = msg.get("type")
            if mtype == "websocket.disconnect":
                break

            text_payload = msg.get("text")
            bytes_payload = msg.get("bytes")

            if text_payload is not None:
                try:
                    data = json.loads(text_payload)
                except Exception:
                    await ws.send_json({"type": "error", "error": "invalid JSON control message"})
                    continue
                ctype = data.get("type")
                if ctype == "start":
                    if session is not None:
                        await ws.send_json({"type": "error", "error": "session already started"})
                        continue
                    session = StreamingSession(
                        ws,
                        language=data.get("language"),
                        initial_prompt=data.get("initial_prompt"),
                    )
                    process_task = asyncio.create_task(periodic_processor())
                    await ws.send_json({
                        "type": "ready",
                        "session_id": session.session_id,
                        "sample_rate": StreamingSession.SAMPLE_RATE,
                        "language": session.language,
                    })
                elif ctype == "end":
                    if session is None:
                        await ws.send_json({"type": "error", "error": "no active session"})
                        break
                    # Mark hard stop FIRST so the periodic processor and finalize
                    # both know the user has clicked Stop. Then cancel the
                    # processor; finalize() will skip remaining tail audio
                    # entirely (preventing post-stop hallucinated tails).
                    session.stop_requested = True
                    if process_task is not None:
                        process_task.cancel()
                        try:
                            await process_task
                        except (asyncio.CancelledError, Exception):
                            pass
                        process_task = None
                    outputs = await session.finalize()
                    await ws.send_json({"type": "done", **outputs})
                    break
                else:
                    await ws.send_json({"type": "error", "error": f"unknown control type: {ctype}"})

            elif bytes_payload is not None:
                if session is None:
                    # silently drop pre-start audio
                    continue
                if len(bytes_payload) % 4 != 0:
                    await ws.send_json({
                        "type": "error",
                        "error": f"PCM frame size {len(bytes_payload)} not aligned to float32 (must be multiple of 4)",
                    })
                    continue
                try:
                    pcm = np.frombuffer(bytes_payload, dtype=np.float32)
                except Exception as exc:  # noqa: BLE001
                    await ws.send_json({"type": "error", "error": f"invalid PCM frame: {exc}"})
                    continue
                async with session.lock:
                    session.add_audio(pcm)

    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001
        logger.exception("WebSocket /ws/stt unexpected error")
        try:
            await ws.send_json({"type": "error", "error": str(exc)})
        except Exception:
            pass
    finally:
        if session is not None:
            session.stop_requested = True
        if process_task is not None:
            process_task.cancel()
            try:
                await process_task
            except Exception:
                pass
        try:
            await ws.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Real-time speech-to-speech translation (WebSocket /ws/s2s)
# ---------------------------------------------------------------------------
# instructions_1.txt — Phase 1: STT (existing) → MT (NLLB) → TTS (Chatterbox)
# Reuses the StreamingSession infrastructure for STT, then hooks each
# committed segment through the translator + TTS in a background task so
# audio synthesis never blocks STT chunking.
#
# Wire protocol (delta vs /ws/stt):
#   Client → Server (start):
#     {"type": "start", "language": "en", "target_language": "fr",
#      "initial_prompt": "..."}
#   Server → Client:
#     • All existing /ws/stt messages (ready / final / done / error).
#     • Plus per-segment translation events:
#       {"type": "translation", "segment_id": N, "source_text": "...",
#        "translated_text": "...", "source_lang": "en", "target_lang": "fr"}
#       {"type": "audio", "segment_id": N, "sample_rate": 24000,
#        "duration_s": 1.23}
#       <binary frame: float32 PCM, mono, sample_rate as above>


async def _s2s_translate_and_speak(
    ws: WebSocket,
    seg: Dict[str, Any],
    *,
    source_lang: str,
    target_lang: str,
) -> None:
    """Translate a finalised STT segment and stream synthesised audio back.

    Runs as a background asyncio task so STT chunking is never blocked by
    MT or TTS latency. CPU/GPU work is dispatched to a thread pool.
    """
    from s2s import translator as s2s_translator
    from s2s import tts as s2s_tts

    try:
        text = (seg.get("text") or "").strip()
        if not text:
            return
        seg_id = int(seg.get("segment_id", 0))

        loop = asyncio.get_running_loop()
        translator = s2s_translator.get_translator()
        tts = s2s_tts.get_tts()

        # 1) Translate (off-loop; ~50-200 ms on RTX 4090).
        translated = await loop.run_in_executor(
            None,
            lambda: translator.translate(
                text, source_lang=source_lang, target_lang=target_lang
            ),
        )
        if not translated:
            return

        await ws.send_json({
            "type": "translation",
            "segment_id": seg_id,
            "source_text": text,
            "translated_text": translated,
            "source_lang": source_lang,
            "target_lang": target_lang,
        })

        # 2) Synthesise (off-loop; <300 ms TTFA on Chatterbox-Turbo).
        # Phase 1 ships English-only TTS, so for non-English targets we
        # *announce* the translation but do not synthesise audio. The
        # client can fall back to its OS TTS until Qwen3-TTS lands.
        if tts.language != "en" and target_lang.lower() != "en":
            audio_lang_supported = True
        elif tts.language == "en" and target_lang.lower() != "en":
            audio_lang_supported = False
        else:
            audio_lang_supported = True

        if not audio_lang_supported:
            await ws.send_json({
                "type": "audio_skipped",
                "segment_id": seg_id,
                "reason": (
                    f"Active TTS backend ({s2s_tts.TTS_BACKEND}) supports "
                    f"language {tts.language!r}; cannot synthesise "
                    f"target_language={target_lang!r}. Set "
                    "TTS_BACKEND=qwen3-tts when multilingual support lands."
                ),
            })
            return

        audio = await loop.run_in_executor(None, lambda: tts.synthesize(translated))
        if audio is None or audio.size == 0:
            return

        sr = int(getattr(tts, "sample_rate", 24_000))
        duration_s = float(audio.shape[-1]) / float(sr)
        await ws.send_json({
            "type": "audio",
            "segment_id": seg_id,
            "sample_rate": sr,
            "duration_s": round(duration_s, 4),
        })
        # Send raw float32 PCM (matches /ws/stt input format on the
        # other direction). Client should treat this as a 24 kHz mono
        # float32 array for direct playback via WebAudio.
        await ws.send_bytes(np.ascontiguousarray(audio, dtype=np.float32).tobytes())

    except Exception as exc:  # noqa: BLE001
        logger.exception("s2s translate/speak failed for segment %s", seg.get("segment_id"))
        try:
            await ws.send_json({
                "type": "error",
                "stage": "s2s",
                "segment_id": seg.get("segment_id"),
                "error": str(exc),
            })
        except Exception:
            pass


@app.websocket("/ws/s2s")
async def websocket_s2s(ws: WebSocket) -> None:
    """Real-time speech-to-speech translation.

    Cascaded pipeline: STT (WhisperX, existing) → MT (NLLB-200) → TTS
    (Chatterbox-Turbo). See module-level comment above for the wire
    protocol (a superset of /ws/stt).

    Set ``S2S_ENABLED=1`` on the server to enable. If disabled, this
    endpoint immediately rejects connections with a clear error message.
    """
    await ws.accept()

    if not S2S_ENABLED:
        await ws.send_json({
            "type": "error",
            "error": "S2S endpoint disabled. Set S2S_ENABLED=1 on the server.",
        })
        await ws.close()
        return

    # Verify the TTS backend is available BEFORE accepting audio so the
    # client gets an immediate, actionable error instead of a partial
    # session with broken audio.
    try:
        from s2s import tts as s2s_tts
        if not s2s_tts.is_available():
            await ws.send_json({
                "type": "error",
                "error": (
                    f"TTS backend {s2s_tts.TTS_BACKEND!r} is not available. "
                    f"Reason: {s2s_tts.availability_error()}"
                ),
            })
            await ws.close()
            return
    except Exception as exc:  # noqa: BLE001
        await ws.send_json({"type": "error", "error": f"S2S init failed: {exc}"})
        await ws.close()
        return

    session: Optional[StreamingSession] = None
    process_task: Optional[asyncio.Task] = None
    pending_s2s_tasks: List[asyncio.Task] = []
    target_lang: str = S2S_TARGET_LANG_DEFAULT
    last_emitted_seg_id: int = -1

    async def periodic_processor() -> None:
        nonlocal last_emitted_seg_id
        try:
            while True:
                await asyncio.sleep(1.0)
                if session is None:
                    continue
                try:
                    await session.maybe_process()
                except Exception:  # noqa: BLE001
                    logger.exception("S2S periodic processing error")
                # Watch for newly-committed segments and dispatch S2S work.
                try:
                    new_segs = [
                        s for s in session.segments
                        if int(s.get("segment_id", -1)) > last_emitted_seg_id
                    ]
                    for seg in new_segs:
                        last_emitted_seg_id = max(
                            last_emitted_seg_id, int(seg.get("segment_id", -1))
                        )
                        src = (session.language or "en").lower()
                        task = asyncio.create_task(
                            _s2s_translate_and_speak(
                                ws, seg, source_lang=src, target_lang=target_lang
                            )
                        )
                        pending_s2s_tasks.append(task)
                except Exception:  # noqa: BLE001
                    logger.exception("S2S segment dispatch error")
        except asyncio.CancelledError:
            pass

    try:
        while True:
            msg = await ws.receive()
            mtype = msg.get("type")
            if mtype == "websocket.disconnect":
                break

            text_payload = msg.get("text")
            bytes_payload = msg.get("bytes")

            if text_payload is not None:
                try:
                    data = json.loads(text_payload)
                except Exception:
                    await ws.send_json({"type": "error", "error": "invalid JSON control message"})
                    continue
                ctype = data.get("type")
                if ctype == "start":
                    if session is not None:
                        await ws.send_json({"type": "error", "error": "session already started"})
                        continue
                    target_lang = (
                        (data.get("target_language") or S2S_TARGET_LANG_DEFAULT)
                        .strip()
                        .lower()
                    )
                    # Validate target_lang up-front so the user gets a clear
                    # error rather than silent fall-through to default.
                    try:
                        from s2s.translator import iso1_to_flores
                        iso1_to_flores(target_lang)
                    except Exception as exc:  # noqa: BLE001
                        await ws.send_json({
                            "type": "error",
                            "error": f"invalid target_language: {exc}",
                        })
                        continue

                    session = StreamingSession(
                        ws,
                        language=data.get("language"),
                        initial_prompt=data.get("initial_prompt"),
                    )
                    process_task = asyncio.create_task(periodic_processor())
                    await ws.send_json({
                        "type": "ready",
                        "session_id": session.session_id,
                        "sample_rate": StreamingSession.SAMPLE_RATE,
                        "language": session.language,
                        "target_language": target_lang,
                        "tts_backend": s2s_tts.TTS_BACKEND,
                    })
                elif ctype == "end":
                    if session is None:
                        await ws.send_json({"type": "error", "error": "no active session"})
                        break
                    session.stop_requested = True
                    if process_task is not None:
                        process_task.cancel()
                        try:
                            await process_task
                        except (asyncio.CancelledError, Exception):
                            pass
                        process_task = None
                    outputs = await session.finalize()
                    # Dispatch any newly-committed final segments before
                    # closing so the client receives their translation+audio.
                    try:
                        new_segs = [
                            s for s in session.segments
                            if int(s.get("segment_id", -1)) > last_emitted_seg_id
                        ]
                        for seg in new_segs:
                            last_emitted_seg_id = max(
                                last_emitted_seg_id, int(seg.get("segment_id", -1))
                            )
                            src = (session.language or "en").lower()
                            pending_s2s_tasks.append(asyncio.create_task(
                                _s2s_translate_and_speak(
                                    ws, seg, source_lang=src, target_lang=target_lang
                                )
                            ))
                    except Exception:  # noqa: BLE001
                        logger.exception("S2S final-segment dispatch error")
                    # Wait for any in-flight S2S tasks so the client gets
                    # all audio before the {type: done} sentinel.
                    if pending_s2s_tasks:
                        await asyncio.gather(*pending_s2s_tasks, return_exceptions=True)
                    await ws.send_json({"type": "done", **outputs})
                    break
                else:
                    await ws.send_json({"type": "error", "error": f"unknown control type: {ctype}"})

            elif bytes_payload is not None:
                if session is None:
                    continue
                if len(bytes_payload) % 4 != 0:
                    await ws.send_json({
                        "type": "error",
                        "error": f"PCM frame size {len(bytes_payload)} not aligned to float32 (must be multiple of 4)",
                    })
                    continue
                try:
                    pcm = np.frombuffer(bytes_payload, dtype=np.float32)
                except Exception as exc:  # noqa: BLE001
                    await ws.send_json({"type": "error", "error": f"invalid PCM frame: {exc}"})
                    continue
                async with session.lock:
                    session.add_audio(pcm)

    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001
        logger.exception("WebSocket /ws/s2s unexpected error")
        try:
            await ws.send_json({"type": "error", "error": str(exc)})
        except Exception:
            pass
    finally:
        if session is not None:
            session.stop_requested = True
        if process_task is not None:
            process_task.cancel()
            try:
                await process_task
            except Exception:
                pass
        for t in pending_s2s_tasks:
            if not t.done():
                t.cancel()
        try:
            await ws.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host=HOST, port=PORT, log_level="info")
