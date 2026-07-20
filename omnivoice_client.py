"""OmniVoice sidecar client + auto-launcher (lives in the MAIN venv).

No `omnivoice` / transformers-5.x import here — keeps the main venv clean. The
heavy multilingual TTS model runs in an isolated venv (see omnivoice_tts/sidecar.py);
this module launches it and talks to it over localhost HTTP.

Responsibilities:
1. ``ensure_sidecar_running()`` — spawn the sidecar subprocess using the isolated
   venv's interpreter (``OMNIVOICE_VENV``). Idempotent.
2. ``is_available(timeout)`` / ``health()`` — readiness probe.
3. ``synthesize(text, language, instruct, ref_audio, ref_text)`` — returns a
   float32 mono numpy array at 24 kHz. Raises on failure so the caller can react.
4. ``shutdown()`` — terminate the spawned subprocess on server exit.
"""
from __future__ import annotations

import atexit
import logging
import os
import shlex
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import requests

logger = logging.getLogger("whisperx-server.omnivoice")

SIDE_HOST = os.environ.get("OMNIVOICE_HOST", "127.0.0.1")
SIDE_PORT = int(os.environ.get("OMNIVOICE_PORT", "9002"))
SIDE_BASE = f"http://{SIDE_HOST}:{SIDE_PORT}"
SIDE_TIMEOUT = float(os.environ.get("OMNIVOICE_TIMEOUT", "120"))
SIDE_VENV = os.environ.get("OMNIVOICE_VENV", "/models/omnivoice-venv").strip()
SIDE_AUTOSTART = os.environ.get("OMNIVOICE_AUTOSTART", "1").strip() not in {"0", "false", "no"}
SIDE_STARTUP_TIMEOUT = float(os.environ.get("OMNIVOICE_STARTUP_TIMEOUT", "300"))
SAMPLE_RATE = 24_000

_proc: Optional[subprocess.Popen] = None
_proc_lock = threading.Lock()
_log_thread: Optional[threading.Thread] = None


def _sidecar_python() -> Optional[str]:
    if not SIDE_VENV:
        return None
    py = Path(SIDE_VENV) / "bin" / "python"
    if not py.exists():
        logger.warning(
            "OMNIVOICE_VENV=%s but %s does not exist — cannot auto-start OmniVoice sidecar",
            SIDE_VENV, py,
        )
        return None
    return str(py)


def is_available(timeout: float = 2.0) -> bool:
    try:
        r = requests.get(f"{SIDE_BASE}/health", timeout=timeout)
        return r.status_code == 200 and bool(r.json().get("ready"))
    except Exception:  # noqa: BLE001
        return False


def health() -> Optional[Dict[str, Any]]:
    try:
        r = requests.get(f"{SIDE_BASE}/health", timeout=2.0)
        if r.status_code == 200:
            return r.json()
    except Exception:  # noqa: BLE001
        pass
    return None


def _stream_logs(proc: subprocess.Popen) -> None:
    assert proc.stdout is not None
    for raw in iter(proc.stdout.readline, b""):
        try:
            line = raw.decode("utf-8", errors="replace").rstrip()
        except Exception:  # noqa: BLE001
            continue
        if line:
            logger.info("[omnivoice-sidecar] %s", line)
    logger.info("[omnivoice-sidecar] log stream closed (exit=%s)", proc.poll())


def ensure_sidecar_running() -> bool:
    """Start the sidecar subprocess if not already running. Returns True if it is
    reachable after this call."""
    global _proc, _log_thread

    if is_available(timeout=1.5):
        return True
    if not SIDE_AUTOSTART:
        logger.info("OMNIVOICE_AUTOSTART=0 — not launching sidecar; expecting external manager")
        return False
    py = _sidecar_python()
    if py is None:
        return False

    with _proc_lock:
        if _proc is None or _proc.poll() is not None:
            cwd = Path(__file__).resolve().parent
            cmd = [py, "-u", "-m", "omnivoice_tts.sidecar"]
            env = os.environ.copy()
            logger.info("Launching OmniVoice sidecar: %s (cwd=%s)", shlex.join(cmd), cwd)
            try:
                _proc = subprocess.Popen(
                    cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    env=env, start_new_session=True,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to spawn OmniVoice sidecar: %s", exc)
                _proc = None
                return False
            _log_thread = threading.Thread(
                target=_stream_logs, args=(_proc,), name="omnivoice-sidecar-logs", daemon=True,
            )
            _log_thread.start()
            atexit.register(shutdown)

    deadline = time.time() + SIDE_STARTUP_TIMEOUT
    while time.time() < deadline:
        if _proc is not None and _proc.poll() is not None:
            logger.error("OmniVoice sidecar exited prematurely (rc=%s)", _proc.returncode)
            return False
        if is_available(timeout=2.0):
            logger.info("OmniVoice sidecar is ready at %s", SIDE_BASE)
            return True
        time.sleep(2.0)
    logger.error("OmniVoice sidecar failed to become ready within %.0fs", SIDE_STARTUP_TIMEOUT)
    return False


def shutdown() -> None:
    global _proc
    with _proc_lock:
        if _proc is None:
            return
        if _proc.poll() is None:
            try:
                _proc.terminate()
                try:
                    _proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    _proc.kill()
            except Exception:  # noqa: BLE001
                pass
        _proc = None


def synthesize(
    text: str,
    *,
    language: Optional[str] = None,
    instruct: Optional[str] = None,
    ref_audio: Optional[str] = None,
    ref_text: Optional[str] = None,
) -> np.ndarray:
    """Synthesize speech via the sidecar. Returns float32 mono @ 24 kHz.

    ``language`` is an ISO-639-3 code (e.g. ``"fra"``, ``"lin"``, ``"yor"``).
    Raises ``RuntimeError`` on failure (caller falls back / reports to the client).
    """
    text = (text or "").strip()
    if not text:
        return np.zeros(0, dtype=np.float32)
    payload: Dict[str, Any] = {"text": text}
    if language:
        payload["language"] = language
    if instruct:
        payload["instruct"] = instruct
    if ref_audio:
        payload["ref_audio"] = ref_audio
    if ref_text:
        payload["ref_text"] = ref_text
    try:
        r = requests.post(f"{SIDE_BASE}/synthesize", json=payload, timeout=SIDE_TIMEOUT)
    except requests.RequestException as exc:
        raise RuntimeError(f"OmniVoice sidecar unreachable: {exc}") from exc
    if r.status_code != 200:
        raise RuntimeError(f"OmniVoice sidecar HTTP {r.status_code}: {r.text[:300]}")
    arr = np.frombuffer(r.content, dtype=np.float32)
    return np.ascontiguousarray(arr, dtype=np.float32)
