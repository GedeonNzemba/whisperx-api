# VibeVoice-ASR sidecar

Optional, **isolated** end-to-end speaker-diarized ASR backend for the
`/session/diarize` endpoint. Runs in its own Python venv with its own
`transformers>=5.3.0` install so the main server's WhisperX stack
(transformers 4.x) is **not** disturbed.

## When it is used

Only when `DIARIZATION_BACKEND=vibevoice` is set on the main server.
Other endpoints (`/session/audio`, `/transcribe`, etc.) are not affected.

## Hardware

* **CUDA GPU required**, ~8–12 GB VRAM (bf16). On CPU the sidecar refuses
  requests so the main server auto-falls-back to pyannote 3.1.

## Install (RunPod, fresh shell)

```bash
cd /app                            # or wherever the repo lives
python3 -m venv vibevoice/.venv
vibevoice/.venv/bin/pip install --upgrade pip
vibevoice/.venv/bin/pip install -r vibevoice/requirements-vibevoice.txt
```

This venv is ~6 GB (torch + transformers + accelerate). The model itself
(~10 GB) is cached under `~/.cache/huggingface/` on first run.

## Run

The main server launches the sidecar automatically when
`DIARIZATION_BACKEND=vibevoice` and `VIBEVOICE_VENV` is set:

```bash
export DIARIZATION_BACKEND=vibevoice
export VIBEVOICE_VENV=/app/vibevoice/.venv
python server.py
```

Or run it stand-alone for debugging:

```bash
vibevoice/.venv/bin/python -m vibevoice.sidecar
# health probe
curl http://127.0.0.1:9001/health
```

## Configuration env vars

| Var | Default | Description |
|---|---|---|
| `VIBEVOICE_HOST` | `127.0.0.1` | Sidecar bind host |
| `VIBEVOICE_PORT` | `9001` | Sidecar bind port |
| `VIBEVOICE_MODEL_ID` | `microsoft/VibeVoice-ASR-HF` | HF model id |
| `VIBEVOICE_DTYPE` | `bfloat16` | `bfloat16`, `float16`, `float32` |
| `VIBEVOICE_MAX_NEW_TOKENS` | `16000` | Generation budget per chunk |
| `VIBEVOICE_CHUNK_MINUTES` | `55` | Long-form chunk size |
| `VIBEVOICE_OVERLAP_SECONDS` | `300` | Long-form chunk overlap |
| `VIBEVOICE_VENV` | _(unset)_ | If set, server auto-launches sidecar with `${VIBEVOICE_VENV}/bin/python` |
| `VIBEVOICE_AUTOSTART` | `1` | Set to `0` to skip auto-launch (manage sidecar yourself) |
| `VIBEVOICE_TIMEOUT` | `1800` | Per-request HTTP timeout (seconds) |

## Long-form audio

Audio longer than `VIBEVOICE_CHUNK_MINUTES * 60s` is split into overlapping
chunks. Speaker IDs across chunks are suffixed (`SPEAKER_00_C0`,
`SPEAKER_00_C1`) because VibeVoice has no cross-chunk speaker memory —
fusing them would risk silent mis-attribution.

## Schema returned by the sidecar

```json
{
  "ok": true,
  "duration": 412.3,
  "segments": [
    {"start": 0.0, "end": 15.43, "speaker": "SPEAKER_00", "text": "..."},
    ...
  ],
  "model_id": "microsoft/VibeVoice-ASR-HF",
  "device": "cuda:0",
  "dtype": "torch.bfloat16",
  "latency_seconds": 4.21
}
```

The main server maps this into the standard `/session/diarize` response.
