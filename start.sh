#!/usr/bin/env bash
# Entrypoint for the RunPod pod.
# - Pulls latest code from GitHub if GIT_REPO is set (fast code updates).
# - If DIARIZATION_BACKEND=vibevoice, ensures the isolated VibeVoice venv is
#   installed under /models/vibevoice-venv (Network Volume — persisted across
#   pod restarts) before starting the server.
# - All other backends: starts server directly.

set -euo pipefail

# Auto-pull latest code from GitHub on every pod start (set GIT_REPO env var).
if [ -n "${GIT_REPO:-}" ]; then
    echo "[start.sh] Pulling latest code from $GIT_REPO"
    if [ -d /app/.git ]; then
        git -C /app pull origin "${GIT_BRANCH:-main}" 2>&1 || echo "[start.sh] git pull failed — using baked code"
    else
        git clone --depth=1 --branch "${GIT_BRANCH:-main}" "$GIT_REPO" /tmp/repo 2>&1 && \
        cp -r /tmp/repo/. /app/ && rm -rf /tmp/repo && \
        echo "[start.sh] Cloned fresh code from GitHub"
    fi
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

echo "[start.sh] Starting uvicorn on ${HOST:-0.0.0.0}:${PORT:-8000}"
exec uvicorn server:app --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}"
