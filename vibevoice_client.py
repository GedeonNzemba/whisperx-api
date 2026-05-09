"""VibeVoice sidecar client + auto-launcher (lives in the MAIN venv).

Responsibilities (no transformers 5.x import here — keeps main venv clean):

1. ``ensure_sidecar_running()`` — if ``DIARIZATION_BACKEND=vibevoice`` and
   ``VIBEVOICE_VENV`` is set, spawn the sidecar as a subprocess using the
   isolated venv's interpreter. Idempotent.

2. ``is_available(timeout)`` — quick health probe.

3. ``transcribe(audio_path)`` — POST the audio to the sidecar, return the
   parsed segment list; raises on failure so the caller can fall back to
   pyannote.

4. ``shutdown()`` — terminate the spawned subprocess on server exit.
"""
from __future__ import annotations

import atexit
import logging
import os
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger("whisperx-server.vibevoice")


SIDE_HOST = os.environ.get("VIBEVOICE_HOST", "127.0.0.1")
SIDE_PORT = int(os.environ.get("VIBEVOICE_PORT", "9001"))
SIDE_BASE = f"http://{SIDE_HOST}:{SIDE_PORT}"
SIDE_TIMEOUT = float(os.environ.get("VIBEVOICE_TIMEOUT", "1800"))
SIDE_VENV = os.environ.get("VIBEVOICE_VENV", "").strip()
SIDE_AUTOSTART = os.environ.get("VIBEVOICE_AUTOSTART", "1").strip() not in {"0", "false", "no"}
SIDE_STARTUP_TIMEOUT = float(os.environ.get("VIBEVOICE_STARTUP_TIMEOUT", "300"))


_proc: Optional[subprocess.Popen] = None
_proc_lock = threading.Lock()
_log_thread: Optional[threading.Thread] = None


def _sidecar_python() -> Optional[str]:
    if not SIDE_VENV:
        return None
    py = Path(SIDE_VENV) / "bin" / "python"
    if not py.exists():
        logger.warning(
            "VIBEVOICE_VENV=%s but %s does not exist — cannot auto-start sidecar",
            SIDE_VENV, py,
        )
        return None
    return str(py)


def is_available(timeout: float = 2.0) -> bool:
    """Return True iff the sidecar /health reports ready=True."""
    try:
        r = requests.get(f"{SIDE_BASE}/health", timeout=timeout)
        if r.status_code != 200:
            return False
        data = r.json()
        return bool(data.get("ready"))
    except Exception:
        return False


def health() -> Optional[Dict[str, Any]]:
    try:
        r = requests.get(f"{SIDE_BASE}/health", timeout=2.0)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def _stream_logs(proc: subprocess.Popen) -> None:
    assert proc.stdout is not None
    for raw in iter(proc.stdout.readline, b""):
        try:
            line = raw.decode("utf-8", errors="replace").rstrip()
        except Exception:
            continue
        if line:
            logger.info("[sidecar] %s", line)
    logger.info("[sidecar] log stream closed (exit=%s)", proc.poll())


def ensure_sidecar_running() -> bool:
    """Start the sidecar subprocess if it isn't already running.

    Returns True if (after this call) the sidecar appears reachable. False
    if not configured, not autostarted, or failed to start.
    """
    global _proc, _log_thread

    # Already up?
    if is_available(timeout=1.5):
        return True

    if not SIDE_AUTOSTART:
        logger.info("VIBEVOICE_AUTOSTART=0 — not launching sidecar; expecting external manager")
        return False

    py = _sidecar_python()
    if py is None:
        logger.info(
            "Cannot auto-launch VibeVoice sidecar — set VIBEVOICE_VENV to a venv that has "
            "transformers>=5.3.0 installed (see vibevoice/README.md)."
        )
        return False

    with _proc_lock:
        if _proc is not None and _proc.poll() is None:
            # Already launched, just waiting for ready
            pass
        else:
            sidecar_module = "vibevoice.sidecar"
            cwd = Path(__file__).resolve().parent
            cmd = [py, "-u", "-m", sidecar_module]
            env = os.environ.copy()
            # Make sure the sidecar's CWD includes the package dir
            logger.info("Launching VibeVoice sidecar: %s (cwd=%s)", shlex.join(cmd), cwd)
            try:
                _proc = subprocess.Popen(
                    cmd,
                    cwd=str(cwd),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    env=env,
                    start_new_session=True,
                )
            except Exception as exc:
                logger.error("Failed to spawn VibeVoice sidecar: %s", exc)
                _proc = None
                return False

            _log_thread = threading.Thread(
                target=_stream_logs, args=(_proc,), name="vibevoice-sidecar-logs", daemon=True
            )
            _log_thread.start()
            atexit.register(shutdown)

    # Wait for ready
    deadline = time.time() + SIDE_STARTUP_TIMEOUT
    while time.time() < deadline:
        if _proc is not None and _proc.poll() is not None:
            logger.error("VibeVoice sidecar exited prematurely (rc=%s)", _proc.returncode)
            return False
        if is_available(timeout=2.0):
            logger.info("VibeVoice sidecar is ready at %s", SIDE_BASE)
            return True
        time.sleep(2.0)

    logger.error(
        "VibeVoice sidecar failed to become ready within %.0fs", SIDE_STARTUP_TIMEOUT
    )
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
            except Exception:
                pass
        _proc = None


def transcribe(audio_path: str) -> Dict[str, Any]:
    """Send `audio_path` to the sidecar; return parsed payload.

    Raises ``RuntimeError`` if the sidecar is unreachable or returned an
    error — callers are expected to catch and fall back to pyannote.
    """
    if not Path(audio_path).exists():
        raise RuntimeError(f"audio file not found: {audio_path}")

    url = f"{SIDE_BASE}/transcribe"
    with open(audio_path, "rb") as fh:
        files = {"file": (Path(audio_path).name, fh, "application/octet-stream")}
        try:
            r = requests.post(url, files=files, timeout=SIDE_TIMEOUT)
        except requests.RequestException as exc:
            raise RuntimeError(f"VibeVoice sidecar unreachable: {exc}") from exc

    if r.status_code != 200:
        raise RuntimeError(
            f"VibeVoice sidecar HTTP {r.status_code}: {r.text[:300]}"
        )
    try:
        data = r.json()
    except ValueError as exc:
        raise RuntimeError(f"sidecar returned non-JSON: {exc}") from exc
    if not data.get("ok"):
        raise RuntimeError(f"sidecar error: {data}")
    return data


def to_diarization_records(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Translate VibeVoice segments into our diarization-record schema:
    [{start, end, speaker}] sorted by start. Used as a safety-net mapper for
    word-level assignment.
    """
    out: List[Dict[str, Any]] = []
    for s in segments or []:
        try:
            out.append({
                "start": float(s["start"]),
                "end": float(s["end"]),
                "speaker": str(s["speaker"]),
            })
        except (KeyError, TypeError, ValueError):
            continue
    out.sort(key=lambda r: r["start"])
    return out
