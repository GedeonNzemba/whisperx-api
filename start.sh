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

# ── cuDNN 9: belt-and-suspenders LD_LIBRARY_PATH ──────────────────────────────
# ldconfig baked into the image is the primary fix; this covers edge cases where
# the linker cache is stale or a sub-process resets the environment. Since
# CTranslate2 4.5.0, BOTH PyTorch 2.6 and CTranslate2 use the SAME cuDNN 9 that
# torch bundles — no separate cuDNN 8 needed (that caused the old
# libcudnn_ops_infer.so.8 crash). See Dockerfile for the full rationale.
for _dir in /usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib; do
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
    for _file in server.py static/index.html s2s/translator.py s2s/tts.py \
                 streaming_asr.py omnivoice_client.py omnivoice_tts/sidecar.py start.sh; do
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

# ── Ensure MADLAD-400 MT model (Apache 2.0 — the commercial-safe default) ─────
# CT2 int8 conversion (~3 GB) on the Network Volume. curl with resume/retries
# (same rationale as the OmniVoice fetch: HF's downloader stalls on this pod).
# Runs only when MT_BACKEND is madlad (the default).
if [ "${MT_BACKEND:-madlad}" = "madlad" ]; then
    MADLAD_DIR="${MADLAD_MODEL_DIR:-/models/madlad400-3b-mt-ct2-int8}"
    if [ ! -s "$MADLAD_DIR/model.bin" ]; then
        echo "[start.sh] Fetching MADLAD-400 CT2 model to $MADLAD_DIR ..."
        mkdir -p "$MADLAD_DIR"
        _MDBASE="https://huggingface.co/Nextcloud-AI/madlad400-3b-mt-ct2-int8/resolve/main"
        for _f in config.json generation_config.json added_tokens.json special_tokens_map.json tokenizer_config.json spiece.model shared_vocabulary.json model.bin; do
            if [ ! -s "$MADLAD_DIR/$_f" ]; then
                curl -fL --retry 12 --retry-delay 3 --retry-all-errors -C - \
                    -o "$MADLAD_DIR/$_f" "${_MDBASE}/${_f}" 2>/dev/null \
                    && echo "[start.sh] MADLAD got $_f" \
                    || echo "[start.sh] MADLAD fetch FAILED for $_f (translator will error until present)"
            fi
        done
    else
        echo "[start.sh] MADLAD model already present at $MADLAD_DIR"
    fi
fi

# ── Pre-cache NLLB tokenizer so S2S warmup succeeds on first start ────────────
# Only relevant for the NON-COMMERCIAL dev fallback (MT_BACKEND=nllb).
# Writes to HF_HOME (/models/hf) which is on the Network Volume — persisted.
# Skips silently if already cached or if network is unavailable.
if [ "${MT_BACKEND:-madlad}" != "nllb" ]; then
    echo "[start.sh] MT_BACKEND is not nllb — skipping NLLB tokenizer pre-cache"
elif python -c "
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

# ── Ensure OmniVoice TTS model is present (only when using that backend) ──────
# The sidecar venv is baked into the image; the model lives on the Network
# Volume (persisted). HF's own downloader hangs on the 805MB audio_tokenizer
# file, so we fetch with curl (robust resume + retries). No-op if already
# complete. Runs only when TTS_BACKEND=omnivoice to avoid needless work.
if [ "${TTS_BACKEND:-}" = "omnivoice" ] || [ "${TTS_BACKEND:-}" = "omni" ]; then
    OMNI_DIR="${OMNIVOICE_MODEL_PATH:-/models/omnivoice_local}"
    if [ ! -s "$OMNI_DIR/model.safetensors" ] || [ ! -s "$OMNI_DIR/audio_tokenizer/model.safetensors" ]; then
        echo "[start.sh] Fetching OmniVoice model to $OMNI_DIR (curl, robust) ..."
        mkdir -p "$OMNI_DIR/audio_tokenizer"
        _OVBASE="https://huggingface.co/k2-fsa/OmniVoice/resolve/main"
        for _f in config.json tokenizer.json tokenizer_config.json chat_template.jinja model.safetensors audio_tokenizer/config.json audio_tokenizer/model.safetensors; do
            if [ ! -s "$OMNI_DIR/$_f" ]; then
                curl -fL --retry 12 --retry-delay 3 --retry-all-errors -C - \
                    -o "$OMNI_DIR/$_f" "${_OVBASE}/${_f}" 2>/dev/null \
                    && echo "[start.sh] OmniVoice got $_f" \
                    || echo "[start.sh] OmniVoice fetch FAILED for $_f (will retry at sidecar load)"
            fi
        done
    else
        echo "[start.sh] OmniVoice model already present at $OMNI_DIR"
    fi
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
