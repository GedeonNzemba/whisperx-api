---
title: WhisperX Transcription API
emoji: 🎙️
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Self-hosted WhisperX large-v3 with diarization & alignment.
hardware: cpu-basic
---

# WhisperX Transcription API (HF Spaces)

This Space runs the `server.py` FastAPI app on **ZeroGPU** (H200/A10G).

## Endpoints
- `GET  /health` — status, GPU info, model load state
- `POST /transcribe` — full pipeline (Whisper → wav2vec2 alignment → pyannote diarization)
- `POST /align` — forced alignment of a known transcript to audio
- `POST /transcript` — extract plain text from a previous align/transcribe JSON
- `GET  /` — minimal web UI (`static/index.html`)

## Configuration (Space → Settings → Variables and secrets)

| Var | Value | Notes |
| --- | --- | --- |
| `HF_TOKEN` | your HF token | Must have access to `pyannote/speaker-diarization-3.1` |
| `WHISPER_MODEL` | `large-v3` | Most accurate; already set in Dockerfile |
| `ASR_BACKEND` | `whisperx` | MLX is Apple-only; already set in Dockerfile |
| `COMPUTE_TYPE` | `float16` | Full precision on GPU; already set in Dockerfile |
| `BEAM_SIZE` | `10` | Wider beam = higher accuracy; already set in Dockerfile |

> All five vars above are pre-set in `Dockerfile.hf`. Only `HF_TOKEN` must
> be added as a **secret** in Space settings (never hard-code it).

## Curl (replace `<space>`)

```bash
curl -s https://<space>.hf.space/health
curl -X POST https://<space>.hf.space/transcribe \
     -F file=@audio.m4a -F language=fr -F diarize=true
```

## Hardware

This Space is configured for **ZeroGPU**: GPU is borrowed only during a
request thanks to `@spaces.GPU(duration=...)` decorators in `app.py`.
Free for HF Pro members; community access via the *Request a GPU* button
on the Space page.

## Local development

See the main `README.md` and `DOCUMENTATION.md` in this repo.
