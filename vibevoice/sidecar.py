"""VibeVoice-ASR sidecar — runs in its OWN venv (transformers 5.x).

Exposes a tiny FastAPI on 127.0.0.1:9001 with two endpoints:

    GET  /health      -> readiness probe (model loaded, device, dtype)
    POST /transcribe  -> multipart `file` upload OR JSON {"path": "..."}
                         returns:
                            {
                              "ok": true,
                              "duration": <seconds>,
                              "segments": [
                                  {"start": float, "end": float,
                                   "speaker": "SPEAKER_0", "text": "..."}
                              ],
                              "model_id": "microsoft/VibeVoice-ASR-HF",
                              "device": "cuda:0", "dtype": "torch.bfloat16"
                            }

Designed to be launched as a subprocess by the main server when
``DIARIZATION_BACKEND=vibevoice``. Loads the model ONCE on startup and
keeps it warm.

Required env vars (with sensible defaults):
    VIBEVOICE_HOST        default 127.0.0.1
    VIBEVOICE_PORT        default 9001
    VIBEVOICE_MODEL_ID    default microsoft/VibeVoice-ASR-HF
    VIBEVOICE_DTYPE       default bfloat16  (one of: bfloat16, float16, float32)
    VIBEVOICE_MAX_NEW_TOKENS  default 16000
    VIBEVOICE_CHUNK_MINUTES   default 55  (long-form chunking threshold)
    VIBEVOICE_OVERLAP_SECONDS default 300

Long-form audio (> CHUNK_MINUTES * 60s) is split into overlapping chunks
and the parsed segments are stitched back together with timestamps offset
into the global timeline.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] vibevoice-sidecar: %(message)s",
)
logger = logging.getLogger("vibevoice-sidecar")


HOST = os.environ.get("VIBEVOICE_HOST", "127.0.0.1")
PORT = int(os.environ.get("VIBEVOICE_PORT", "9001"))
MODEL_ID = os.environ.get("VIBEVOICE_MODEL_ID", "microsoft/VibeVoice-ASR-HF")
DTYPE_NAME = os.environ.get("VIBEVOICE_DTYPE", "bfloat16").lower()
MAX_NEW_TOKENS = int(os.environ.get("VIBEVOICE_MAX_NEW_TOKENS", "16000"))
CHUNK_MINUTES = float(os.environ.get("VIBEVOICE_CHUNK_MINUTES", "55"))
OVERLAP_SECONDS = float(os.environ.get("VIBEVOICE_OVERLAP_SECONDS", "300"))


# Globals populated on startup
_model = None
_processor = None
_device: Optional[str] = None
_dtype: Optional[str] = None
_load_error: Optional[str] = None


def _resolve_dtype(name: str):
    import torch

    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }.get(name, torch.bfloat16)


def _load_model() -> None:
    """Load VibeVoice once. Sets module-level globals on success."""
    global _model, _processor, _device, _dtype, _load_error

    try:
        import torch
        from transformers import AutoProcessor, VibeVoiceAsrForConditionalGeneration
    except Exception as exc:
        _load_error = (
            f"VibeVoice-ASR requires transformers>=5.3.0 in this venv. Import error: {exc}"
        )
        logger.error(_load_error)
        return

    if not torch.cuda.is_available():
        _load_error = (
            "VibeVoice-ASR requires CUDA. No GPU detected — sidecar will refuse "
            "transcription requests so the main server falls back to pyannote."
        )
        logger.error(_load_error)
        return

    dtype = _resolve_dtype(DTYPE_NAME)
    logger.info("Loading %s with dtype=%s, device_map=auto…", MODEL_ID, DTYPE_NAME)
    t0 = time.time()
    try:
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        model = VibeVoiceAsrForConditionalGeneration.from_pretrained(
            MODEL_ID,
            dtype=dtype,
            device_map="auto",
        )
    except Exception as exc:
        _load_error = f"Model load failed: {exc}"
        logger.exception("VibeVoice load failed")
        return

    _model = model
    _processor = processor
    try:
        _device = str(model.device)
    except Exception:
        _device = "cuda"
    _dtype = str(getattr(model, "dtype", dtype))
    logger.info("VibeVoice ready on %s (%s) in %.1fs", _device, _dtype, time.time() - t0)


def _audio_duration_seconds(path: str) -> float:
    try:
        import soundfile as sf

        info = sf.info(path)
        return float(info.frames) / float(info.samplerate)
    except Exception:
        # Fallback via ffprobe
        import subprocess

        try:
            out = subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=nw=1:nk=1",
                    path,
                ],
                text=True,
            ).strip()
            return float(out)
        except Exception:
            return 0.0


def _to_mono16k_wav(src_path: str) -> str:
    """Decode any input to 16 kHz mono WAV via ffmpeg. Returns path to a
    NamedTemporaryFile-managed file (caller deletes)."""
    import subprocess

    fd, dst = tempfile.mkstemp(suffix=".wav", prefix="vv_")
    os.close(fd)
    cmd = [
        "ffmpeg",
        "-y",
        "-nostdin",
        "-loglevel",
        "error",
        "-i",
        src_path,
        "-ac",
        "1",
        "-ar",
        "16000",
        "-acodec",
        "pcm_s16le",
        dst,
    ]
    subprocess.run(cmd, check=True)
    return dst


def _slice_wav(src_wav: str, start_s: float, end_s: float) -> str:
    """Cut a fragment from src_wav into a new temp WAV (16k mono)."""
    import subprocess

    fd, dst = tempfile.mkstemp(suffix=".wav", prefix="vv_chunk_")
    os.close(fd)
    duration = max(0.01, end_s - start_s)
    cmd = [
        "ffmpeg",
        "-y",
        "-nostdin",
        "-loglevel",
        "error",
        "-ss",
        f"{start_s:.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        src_wav,
        "-ac",
        "1",
        "-ar",
        "16000",
        "-acodec",
        "pcm_s16le",
        dst,
    ]
    subprocess.run(cmd, check=True)
    return dst


def _normalise_segments(parsed: Any, time_offset: float = 0.0) -> List[Dict[str, Any]]:
    """Normalise VibeVoice's parsed output into our schema."""
    out: List[Dict[str, Any]] = []
    if not isinstance(parsed, list):
        return out
    for row in parsed:
        if not isinstance(row, dict):
            continue
        try:
            start = float(row.get("Start", row.get("start", 0.0)) or 0.0) + time_offset
            end = float(row.get("End", row.get("end", start)) or start) + time_offset
        except (TypeError, ValueError):
            continue
        spk_raw = row.get("Speaker", row.get("speaker", 0))
        try:
            spk_idx = int(spk_raw)
            speaker = f"SPEAKER_{spk_idx:02d}"
        except (TypeError, ValueError):
            speaker = str(spk_raw)
        text = str(row.get("Content", row.get("content", row.get("text", "")))).strip()
        if not text:
            continue
        out.append({"start": start, "end": end, "speaker": speaker, "text": text})
    return out


def _stitch_chunks(chunks: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Merge per-chunk segment lists into a single timeline.

    Speaker IDs across chunks are NOT guaranteed to be the same person — we
    re-label conservatively by appending a chunk suffix; downstream callers
    can re-cluster if needed. This avoids silently merging two different
    speakers across chunk boundaries.
    """
    if not chunks:
        return []
    if len(chunks) == 1:
        return sorted(chunks[0], key=lambda s: s["start"])

    merged: List[Dict[str, Any]] = []
    for ci, segs in enumerate(chunks):
        for s in segs:
            new = dict(s)
            new["speaker"] = f"{s['speaker']}_C{ci}"
            merged.append(new)
    merged.sort(key=lambda s: s["start"])
    return merged


def _chunk_plan(duration: float) -> List[tuple[float, float]]:
    """Return list of (start_s, end_s) windows. Single window if short."""
    chunk_s = CHUNK_MINUTES * 60.0
    if duration <= chunk_s:
        return [(0.0, duration)]
    plan = []
    start = 0.0
    while start < duration:
        end = min(start + chunk_s, duration)
        plan.append((start, end))
        if end >= duration:
            break
        start = end - OVERLAP_SECONDS
    return plan


def _transcribe_one(wav_path: str) -> List[Dict[str, Any]]:
    """Run a single VibeVoice forward pass on a (≤chunk) wav file."""
    inputs = _processor.apply_transcription_request(audio=wav_path).to(
        _model.device, _model.dtype
    )
    output_ids = _model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
    parsed_list = _processor.decode(generated_ids, return_format="parsed")
    parsed = parsed_list[0] if parsed_list else []
    return _normalise_segments(parsed, time_offset=0.0)


def _transcribe_file(audio_path: str) -> Dict[str, Any]:
    if _model is None or _processor is None:
        raise RuntimeError(_load_error or "VibeVoice model not loaded")

    wav_path = _to_mono16k_wav(audio_path)
    cleanup = [wav_path]
    try:
        duration = _audio_duration_seconds(wav_path)
        plan = _chunk_plan(duration)
        all_chunks: List[List[Dict[str, Any]]] = []

        if len(plan) == 1:
            segs = _transcribe_one(wav_path)
            all_chunks.append(segs)
        else:
            logger.info("Long audio %.1fs → %d chunks", duration, len(plan))
            for start_s, end_s in plan:
                chunk_path = _slice_wav(wav_path, start_s, end_s)
                cleanup.append(chunk_path)
                segs = _transcribe_one(chunk_path)
                # Offset timestamps to the global timeline
                segs = [
                    {**s, "start": s["start"] + start_s, "end": s["end"] + start_s}
                    for s in segs
                ]
                all_chunks.append(segs)

        merged = _stitch_chunks(all_chunks)
        return {
            "ok": True,
            "duration": duration,
            "segments": merged,
            "model_id": MODEL_ID,
            "device": _device,
            "dtype": _dtype,
        }
    finally:
        for p in cleanup:
            try:
                os.unlink(p)
            except OSError:
                pass


# ── FastAPI app ──────────────────────────────────────────────────────────────
def create_app():
    from fastapi import FastAPI, File, HTTPException, UploadFile
    from fastapi.responses import JSONResponse

    app = FastAPI(title="VibeVoice ASR sidecar")

    @app.on_event("startup")
    def _startup():
        _load_model()

    @app.get("/health")
    def health():
        ready = _model is not None and _processor is not None
        return {
            "ready": ready,
            "model_id": MODEL_ID,
            "device": _device,
            "dtype": _dtype,
            "error": _load_error,
        }

    @app.post("/transcribe")
    async def transcribe(file: UploadFile = File(...)):
        if _model is None:
            raise HTTPException(
                status_code=503,
                detail=_load_error or "VibeVoice not ready",
            )
        # Persist upload to a temp file
        suffix = Path(file.filename or "audio").suffix or ".bin"
        fd, tmp = tempfile.mkstemp(suffix=suffix, prefix="vv_in_")
        os.close(fd)
        try:
            with open(tmp, "wb") as fh:
                while True:
                    chunk = await file.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
            t0 = time.time()
            result = _transcribe_file(tmp)
            result["latency_seconds"] = round(time.time() - t0, 3)
            return JSONResponse(result)
        except Exception as exc:
            logger.exception("transcribe failed")
            raise HTTPException(status_code=500, detail=str(exc))
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    return app


def main():
    import uvicorn

    app = create_app()
    logger.info("Starting VibeVoice sidecar on %s:%d", HOST, PORT)
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
