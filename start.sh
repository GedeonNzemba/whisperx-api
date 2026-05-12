#!/usr/bin/env bash
# Entrypoint for the RunPod pod.
# - Pulls latest code from GitHub if GIT_REPO is set (fast code updates).
# - If DIARIZATION_BACKEND=vibevoice, ensures the isolated VibeVoice venv is
#   installed under /models/vibevoice-venv (Network Volume — persisted across
#   pod restarts) before starting the server.
# - All other backends: starts server directly.

set -euo pipefail

# ── DNS fix: RunPod's internal resolver sometimes fails on pod start ───────────
# Write public resolvers so pip/HF downloads work during startup.
if [ -w /etc/resolv.conf ]; then
    printf 'nameserver 8.8.8.8\nnameserver 1.1.1.1\n' > /etc/resolv.conf
fi

# ── cuDNN: belt-and-suspenders LD_LIBRARY_PATH ────────────────────────────────
# ldconfig baked into the image is the primary fix; this covers edge cases
# where the linker cache is stale or a sub-process resets the environment.
# Include both cuDNN 8 (for CTranslate2) and cuDNN 9 (for PyTorch).
for _dir in /usr/local/lib/cudnn8 /usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib; do
    if [ -d "$_dir" ]; then
        export LD_LIBRARY_PATH="${_dir}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
done

# Auto-pull latest code from GitHub on every pod start (set GIT_REPO env var).
# Uses curl (always available) since git is not installed in the runtime image.
if [ -n "${GIT_REPO:-}" ]; then
    echo "[start.sh] Pulling latest code from $GIT_REPO"
    _BRANCH="${GIT_BRANCH:-main}"
    _RAW_BASE="https://raw.githubusercontent.com/$(echo "$GIT_REPO" | sed 's|https://github.com/||' | sed 's|\.git$||')"
    for _file in server.py static/index.html s2s/translator.py s2s/tts.py start.sh; do
        curl -fsSL "${_RAW_BASE}/${_BRANCH}/${_file}" -o "/app/${_file}" 2>/dev/null && \
            echo "[start.sh] Updated ${_file}" || \
            echo "[start.sh] Skipped ${_file} (not found or no network)"
    done
fi

VENV_DIR="${VIBEVOICE_VENV_DIR:-/models/vibevoice-venv}"
REQUIREMENTS="/app/vibevoice/requirements-vibevoice.txt"

if [ "${DIARIZATION_BACKEND:-auto}" = "vibevoice" ]; then
    echo "[start.sh] DIARIZATION_BACKEND=vibevoice — checking isolated venv at $VENV_DIR"

    if [ ! -f "$VENV_DIR/bin/python" ]; then
        echo "[start.sh] Venv not found — installing (this takes ~5 min on first boot)..."
        python3 -m venv "$VENV_DIR"
        "$VENV_DIR/bin/pip" install --upgrade pip --quiet
        "$VENV_DIR/bin/pip" install -r "$REQUIREMENTS" --quiet
        echo "[start.sh] VibeVoice venv installed at $VENV_DIR"
    else
        echo "[start.sh] VibeVoice venv found at $VENV_DIR"
    fi

    # Tell the main server where the isolated venv lives.
    export VIBEVOICE_VENV="$VENV_DIR"
fi

# ── Pre-cache NLLB tokenizer so S2S warmup succeeds on first start ────────────
# Writes to HF_HOME (/models/hf) which is on the Network Volume — persisted.
# Skips silently if already cached or if network is unavailable.
if python -c "
import os, sys
cache = os.path.join(os.environ.get('HF_HOME', '/models/hf'), 'hub')
# Check if sentencepiece model already cached
import glob
cached = glob.glob(cache + '/**/sentencepiece.bpe.model', recursive=True)
sys.exit(0 if cached else 1)
" 2>/dev/null; then
    echo "[start.sh] NLLB tokenizer already cached — skipping download"
else
    echo "[start.sh] Pre-caching NLLB tokenizer..."
    python -c "
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained(
    'facebook/nllb-200-distilled-600M',
    cache_dir='${HF_HOME:-/models/hf}',
)
print('[start.sh] NLLB tokenizer cached.')
" 2>&1 || echo "[start.sh] NLLB tokenizer pre-cache failed (no network?) — will retry at first translation"
fi

echo "[start.sh] Starting uvicorn on ${HOST:-0.0.0.0}:${PORT:-8000}"
# WS ping/timeout extended so /ws/s2s survives the GPU-bound chunks
# of MT + TTS that briefly stall pong replies. Default uvicorn is
# 20/20 which yields code=1011 keepalive-ping-timeout under load.
exec uvicorn server:app \
    --host "${HOST:-0.0.0.0}" \
    --port "${PORT:-8000}" \
    --ws-ping-interval "${WS_PING_INTERVAL:-60}" \
    --ws-ping-timeout "${WS_PING_TIMEOUT:-300}" \
    --timeout-keep-alive "${TIMEOUT_KEEP_ALIVE:-300}"
