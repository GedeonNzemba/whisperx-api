"""OmniVoice-TTS sidecar — runs in its OWN venv (transformers 5.x + omnivoice).

Exposes a tiny FastAPI on 127.0.0.1:9002:

    GET  /health      -> {ready, model_id/path, device, dtype, sample_rate, error}
    POST /synthesize  -> JSON body:
                            {
                              "text": "...",                 # required
                              "language": "fra",             # ISO-639-3 (optional; auto if omitted)
                              "instruct": "male, low pitch", # voice-design (optional)
                              "ref_audio": "/path/ref.wav",  # voice-cloning (optional)
                              "ref_text": "..."              # reference transcript (optional)
                            }
                         returns raw little-endian float32 PCM (mono) bytes with
                         headers  X-Sample-Rate: 24000  and  X-Duration-Seconds.

Loads OmniVoice ONCE on startup and keeps it warm. Designed to be launched as a
subprocess by :mod:`omnivoice_client` from the main server.

Why a sidecar: OmniVoice needs transformers>=5.3 which we do not want to impose on
whisperx / faster-whisper / ctranslate2 in the main venv. Full isolation = zero
risk to the working STT pipeline.

Env vars (sensible defaults):
    OMNIVOICE_HOST          default 127.0.0.1
    OMNIVOICE_PORT          default 9002
    OMNIVOICE_MODEL_PATH    default /models/omnivoice_local  (falls back to the
                            HF id k2-fsa/OmniVoice if the local dir is absent)
    OMNIVOICE_DEVICE        default cuda:0
    OMNIVOICE_DTYPE         default float16  (bfloat16 | float16 | float32)
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] omnivoice-sidecar: %(message)s",
)
logger = logging.getLogger("omnivoice-sidecar")

HOST = os.environ.get("OMNIVOICE_HOST", "127.0.0.1")
PORT = int(os.environ.get("OMNIVOICE_PORT", "9002"))
MODEL_PATH = os.environ.get("OMNIVOICE_MODEL_PATH", "/models/omnivoice_local")
MODEL_HF_ID = os.environ.get("OMNIVOICE_MODEL_ID", "k2-fsa/OmniVoice")
DEVICE = os.environ.get("OMNIVOICE_DEVICE", "cuda:0")
DTYPE_NAME = os.environ.get("OMNIVOICE_DTYPE", "float16").lower()
SAMPLE_RATE = 24_000

# Globals populated on startup
_model = None
_loaded_from: Optional[str] = None
_dtype: Optional[str] = None
_load_error: Optional[str] = None


def _resolve_dtype(name: str):
    import torch

    return {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }.get(name, torch.float16)


def _load_model() -> None:
    """Load OmniVoice once. Prefers the baked/cached local dir, falls back to the
    HF id (which downloads on first use)."""
    global _model, _loaded_from, _dtype, _load_error
    try:
        import torch
        from omnivoice import OmniVoice
    except Exception as exc:  # noqa: BLE001
        _load_error = (
            f"OmniVoice import failed (needs `pip install omnivoice` in this venv): {exc}"
        )
        logger.error(_load_error)
        return

    if not torch.cuda.is_available() and DEVICE.startswith("cuda"):
        _load_error = "CUDA not available — OmniVoice sidecar requires a GPU."
        logger.error(_load_error)
        return

    src = MODEL_PATH if Path(MODEL_PATH).is_dir() else MODEL_HF_ID
    dtype = _resolve_dtype(DTYPE_NAME)
    logger.info("Loading OmniVoice from %s on %s [%s]…", src, DEVICE, DTYPE_NAME)
    t0 = time.time()
    try:
        model = OmniVoice.from_pretrained(src, device_map=DEVICE, dtype=dtype)
    except Exception as exc:  # noqa: BLE001
        _load_error = f"OmniVoice load failed: {exc}"
        logger.exception("OmniVoice load failed")
        return
    _model = model
    _loaded_from = src
    _dtype = str(dtype)
    logger.info("OmniVoice ready (loaded from %s) in %.1fs", src, time.time() - t0)


def _synthesize(
    text: str,
    *,
    language: Optional[str] = None,
    instruct: Optional[str] = None,
    ref_audio: Optional[str] = None,
    ref_text: Optional[str] = None,
):
    """Run OmniVoice.generate and return a contiguous float32 mono numpy array."""
    import numpy as np

    if _model is None:
        raise RuntimeError(_load_error or "OmniVoice model not loaded")
    text = (text or "").strip()
    if not text:
        return np.zeros(0, dtype=np.float32)

    kwargs: Dict[str, Any] = {"text": text}
    if language:
        kwargs["language"] = language
    # Voice cloning takes precedence over voice design if a reference clip is given.
    if ref_audio and Path(ref_audio).is_file():
        kwargs["ref_audio"] = ref_audio
        if ref_text:
            kwargs["ref_text"] = ref_text
    elif instruct:
        kwargs["instruct"] = instruct

    out = _model.generate(**kwargs)
    wav = out[0] if isinstance(out, (list, tuple)) else out
    arr = np.asarray(wav, dtype=np.float32).squeeze()
    return np.ascontiguousarray(arr, dtype=np.float32)


# ── FastAPI app ──────────────────────────────────────────────────────────────
def create_app():
    from fastapi import FastAPI, HTTPException, Request, Response
    from fastapi.responses import JSONResponse

    app = FastAPI(title="OmniVoice TTS sidecar")

    @app.on_event("startup")
    def _startup():
        _load_model()
        # Warm-up so the first real request doesn't pay the JIT/cudnn-tune cost.
        if _model is not None:
            try:
                _ = _synthesize("Hello.", language="eng")
                logger.info("OmniVoice warm-up complete.")
            except Exception as exc:  # noqa: BLE001
                logger.warning("OmniVoice warm-up failed (non-fatal): %s", exc)

    @app.get("/health")
    def health():
        return {
            "ready": _model is not None,
            "model": _loaded_from or MODEL_PATH,
            "device": DEVICE,
            "dtype": _dtype,
            "sample_rate": SAMPLE_RATE,
            "error": _load_error,
        }

    @app.post("/synthesize")
    async def synthesize(request: Request):
        if _model is None:
            raise HTTPException(status_code=503, detail=_load_error or "OmniVoice not ready")
        try:
            body = await request.json()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"invalid JSON: {exc}")
        text = (body.get("text") or "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="`text` is required")
        try:
            t0 = time.time()
            arr = _synthesize(
                text,
                language=body.get("language") or None,
                instruct=body.get("instruct") or None,
                ref_audio=body.get("ref_audio") or None,
                ref_text=body.get("ref_text") or None,
            )
            latency = time.time() - t0
        except Exception as exc:  # noqa: BLE001
            logger.warning("synthesize failed: %s", exc)
            # 422 for caller-correctable errors (e.g. bad instruct vocabulary),
            # 500 otherwise. OmniVoice raises a ValueError-like message for
            # "Unsupported instruct items".
            msg = str(exc)
            code = 422 if "instruct" in msg.lower() or "language" in msg.lower() else 500
            raise HTTPException(status_code=code, detail=msg[:400])
        data = arr.tobytes()
        return Response(
            content=data,
            media_type="application/octet-stream",
            headers={
                "X-Sample-Rate": str(SAMPLE_RATE),
                "X-Duration-Seconds": f"{len(arr) / SAMPLE_RATE:.4f}",
                "X-Latency-Seconds": f"{latency:.4f}",
            },
        )

    return app


def main():
    import uvicorn

    app = create_app()
    logger.info("Starting OmniVoice sidecar on %s:%d", HOST, PORT)
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
