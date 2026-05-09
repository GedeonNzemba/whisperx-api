"""HuggingFace Spaces entrypoint.

Re-exports the FastAPI app from server.py and applies the @spaces.GPU
decorator to the heavy endpoints so they run on ZeroGPU (H200) hardware.

On HF Spaces:
    SDK: docker
    app_port: 7860

Locally this file is unused; run `python server.py` directly.
"""

from __future__ import annotations

import os

# Force whisperx backend on Spaces (CUDA, no MLX).
os.environ.setdefault("ASR_BACKEND", "whisperx")

import server  # noqa: E402  (must be imported AFTER setting env vars)
from server import app  # noqa: F401,E402  (re-export for `uvicorn app:app`)

try:
    import spaces  # type: ignore

    # Wrap the heavy endpoints. ZeroGPU only allocates a GPU during the call.
    # Duration is the *max* seconds the GPU is reserved for one request.
    # 600s / 900s covers ~60 min audio at float16 on H200.
    server.transcribe = spaces.GPU(duration=600)(server.transcribe)  # type: ignore[attr-defined]
    server.align_endpoint = spaces.GPU(duration=900)(server.align_endpoint)  # type: ignore[attr-defined]
    server.batch_transcribe = spaces.GPU(duration=3600)(server.batch_transcribe)  # type: ignore[attr-defined]
    server.batch_align = spaces.GPU(duration=3600)(server.batch_align)  # type: ignore[attr-defined]
    print("[app.py] ZeroGPU decorators applied to /transcribe, /align, /batch, and /batch-align.")
except ImportError:
    print("[app.py] `spaces` SDK not available — running without ZeroGPU.")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "7860")))
