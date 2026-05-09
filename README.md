# WhisperX Large V3 API Server

A production-ready, drop-in **WhisperX** REST API exposing the full pipeline:
**Whisper Large V3** transcription (via `faster-whisper`) → **wav2vec2 forced alignment**
for word-level timestamps → **Silero VAD** → **pyannote.audio 3.1 speaker diarization**.

Single endpoint (`POST /transcribe`), one Docker image, GPU-accelerated.

---

## Features

- 🚀 **Whisper Large V3** with `faster-whisper` backend (~70× real-time on a modern GPU).
- 🎯 **Word-level timestamps** (±50 ms) via wav2vec2 forced alignment, auto-selected per language.
- 🌍 **99+ languages** with auto-detection; graceful fallback to English alignment for unsupported langs.
- 🗣️ **Speaker diarization** via `pyannote/speaker-diarization-3.1` (requires `HF_TOKEN`).
- ⚡ **Silero VAD** to strip silence and reduce hallucinations.
- 🧠 **Dynamic batch sizing** to keep VRAM under ~8 GB at `beam_size=5`.
- 📁 **Multiple outputs**: `json`, `txt`, `srt`, `vtt` — inline, base64, and download links.
- 🔌 **CORS enabled** so any frontend can call it.
- 🐳 **Single-command Docker run**.
- 🎤 **Real-time streaming STT** via WebSocket `/ws/stt` — stream microphone PCM (16 kHz mono Float32) and receive word-level transcripts every ~5 s. Try it from the **Realtime STT** tab in the web UI. See `DOCUMENTATION.md` § *5.x WebSocket /ws/stt* for the wire protocol.

---

## Quick start

### Option A — Docker (recommended)

```bash
# 1. Build
docker build -t whisperx-server .

# 2. Run (GPU required; mount ./models to persist downloads across restarts)
docker run --gpus all --rm -p 8000:8000 \
    -e HF_TOKEN=hf_xxx_your_token \
    -v "$PWD/models:/app/models" \
    --name whisperx whisperx-server
```

First start downloads Whisper Large V3 (~3 GB), the alignment model, the VAD model
and the diarization pipeline into `/app/models`. Subsequent starts are fast.

### Option B — Local Python

Requirements: Python 3.10+, NVIDIA GPU with CUDA 12.1, `ffmpeg`.

```bash
python -m venv .venv && source .venv/bin/activate
pip install --extra-index-url https://download.pytorch.org/whl/cu121 \
    torch==2.3.1+cu121 torchaudio==2.3.1+cu121
pip install -r requirements.txt

cp .env.example .env   # then fill in HF_TOKEN
export $(grep -v '^#' .env | xargs)

python server.py
```

---

## Environment variables

| Variable               | Default             | Purpose                                                          |
| ---------------------- | ------------------- | ---------------------------------------------------------------- |
| `HF_TOKEN`             | _(unset)_           | HuggingFace token with access to `pyannote/speaker-diarization-3.1`. Required for diarization. |
| `PORT`                 | `8000`              | HTTP port.                                                       |
| `HOST`                 | `0.0.0.0`           | Bind address.                                                    |
| `MODEL_DIR`            | `./models`          | Where models are cached.                                         |
| `MAX_AUDIO_DURATION`   | `7200` (2h)         | Hard cap on audio length (seconds).                              |
| `MAX_FILE_SIZE`        | `524288000` (500MB) | Max upload size (bytes).                                         |
| `WHISPER_MODEL`        | `large-v3`          | Override whisper model name.                                     |
| `COMPUTE_TYPE`         | `float16`           | `float16`, `int8_float16`, or `int8`.                            |
| `BEAM_SIZE`            | `5`                 | Whisper beam width.                                              |
| `DOWNLOAD_TTL_SECONDS` | `3600`              | How long generated subtitle files stay downloadable.             |

> **HF_TOKEN setup**: visit <https://huggingface.co/pyannote/speaker-diarization-3.1>,
> accept the user agreement, then create a token at
> <https://huggingface.co/settings/tokens>.

---

## API reference

### `GET /health`

Returns service status, GPU memory, model load state, uptime.

```bash
curl http://localhost:8000/health
```

### `POST /transcribe`

Multipart form. Parameters:

| Field           | Type                | Default              | Description |
| --------------- | ------------------- | -------------------- | ----------- |
| `file`          | file (required)     | —                    | Audio: wav, mp3, flac, ogg, m4a, opus, webm, … |
| `language`      | string              | _(auto-detect)_      | ISO 639-1 code (`en`, `fr`, `de`, `ja`, …). |
| `diarize`       | bool                | `true` if `HF_TOKEN` set, else `false` | Run speaker diarization. |
| `min_speakers`  | int                 | _(unset)_            | Hint for diarization. |
| `max_speakers`  | int                 | _(unset)_            | Hint for diarization. |
| `vad`           | bool                | `true`               | Silero VAD pre-filter. |
| `output_format` | string              | `json`               | Comma-separated subset of `json,txt,srt,vtt`. JSON is always included. |
| `initial_prompt`| string              | _(none)_             | Whisper biasing prompt. |
| `task`          | `transcribe`/`translate` | `transcribe`    | `translate` translates to English. |

#### Examples

```bash
# Simplest call — JSON only, auto language, diarize if HF_TOKEN is set.
curl -X POST http://localhost:8000/transcribe \
     -F file=@meeting.mp3

# Force English, all formats, no diarization.
curl -X POST http://localhost:8000/transcribe \
     -F file=@interview.wav \
     -F language=en \
     -F diarize=false \
     -F output_format=json,txt,srt,vtt

# Diarize with speaker hints.
curl -X POST http://localhost:8000/transcribe \
     -F file=@panel.m4a \
     -F diarize=true \
     -F min_speakers=2 -F max_speakers=4

# Translate any language to English.
curl -X POST http://localhost:8000/transcribe \
     -F file=@spanish.mp3 \
     -F task=translate
```

#### Response shape (truncated)

```jsonc
{
  "language": "en",
  "duration_seconds": 312.4,
  "processing_time_seconds": 11.7,
  "realtime_factor": 26.7,
  "diarized": true,
  "diarization_warning": null,
  "alignment_warning": null,
  "segments": [
    {
      "start": 0.12, "end": 4.55,
      "text": " Hello and welcome to the show.",
      "speaker": "SPEAKER_00",
      "words": [
        {"word": "Hello", "start": 0.12, "end": 0.41, "speaker": "SPEAKER_00", "score": 0.98}
      ]
    }
  ],
  "word_segments": [ /* every aligned word, flat */ ],
  "files": {
    "txt":        "[SPEAKER_00]: Hello and welcome to the show.\n…",
    "txt_base64": "W1NQRUFLRVJfMDBdOiBI…",
    "srt":        "1\n00:00:00,120 --> 00:00:04,550\nSPEAKER_00\nHello and welcome to the show.\n…",
    "srt_base64": "…",
    "vtt":        "WEBVTT\n\n00:00:00.120 --> 00:00:04.550\nSPEAKER_00\nHello and welcome to the show.\n…",
    "vtt_base64": "…"
  },
  "download_links": {
    "txt": "/download/4f6b…/transcript.txt",
    "srt": "/download/4f6b…/transcript.srt",
    "vtt": "/download/4f6b…/transcript.vtt"
  },
  "download_expires_at": "2026-05-01T23:25:13Z"
}
```

### `GET /download/{file_id}/{name}`

Serves the generated subtitle/text files referenced in the JSON `download_links`.
Files expire after `DOWNLOAD_TTL_SECONDS` (default 1 hour).

---

### `POST /session/diarize`  — Dedicated diarized transcription

Same form fields as `/transcribe` but **always** runs pyannote speaker
diarization. Returns a JSON payload with speaker-labelled segments and word
timestamps — the canonical "source of truth" input for downstream features
(real-time speech-to-speech, summarizer, note-taker). Returns **HTTP 503**
with a clear message if `HF_TOKEN` is not set.

```bash
curl -X POST http://localhost:8000/session/diarize \
  -F "file=@meeting.m4a" \
  -F "language=en" \
  -F "min_speakers=2" \
  -F "max_speakers=4" \
  -F "output_format=json,txt,srt,vtt" \
  -F 'speaker_names={"SPEAKER_00":"Alice","SPEAKER_01":"Bob"}'
```

Response shape is identical to `/transcribe` (`segments[]`, `word_segments[]`,
`download_links{}`, etc.) — every segment is guaranteed to carry a `speaker`
field.

#### Architecture: real-time vs. post-session diarized

The server intentionally exposes **two complementary STT entry points**:

| Endpoint | Use case | Diarization | Latency |
|---|---|---|---|
| `WS /ws/stt` | Live captions, live agents — stream microphone PCM, get word-level chunks every ~5 s | ❌ (kept off for low latency) | ~2–6 s |
| `POST /session/diarize` | Post-session ground-truth transcript for downstream services | ✅ (forced on) | seconds-to-minutes per file |

The realtime path is optimised for responsiveness; the diarized path is
optimised for accuracy and produces the canonical record of the session.

---

### `POST /align`  — Forced alignment from a known transcript

Aligns a **ground-truth transcript** to the audio using the wav2vec2 forced-alignment
pipeline. **No ASR is performed** — Whisper is not invoked. Use this when you already
have a 100 % accurate transcript and only need the timestamps.

Multipart form parameters:

| Field           | Type        | Default        | Description |
| --------------- | ----------- | -------------- | ----------- |
| `audio`         | file        | _(required)_   | Audio file. |
| `transcript`    | string      | _(required)_   | Ground-truth text. May include speaker labels: `Speaker 1: …`, `SPEAKER_00: …`, `[A]: …`. |
| `language`      | string      | _(auto)_       | ISO 639-1 code. Auto-detected from the transcript text via `langdetect` if omitted. |
| `vad`           | bool        | `true`         | Use Silero VAD to trim leading/trailing silence before alignment. |
| `diarize`       | bool        | `false`        | Additionally run pyannote acoustic diarization (needs `HF_TOKEN`). |
| `min_speakers`  | int         | _(unset)_      | Diarization hint. |
| `max_speakers`  | int         | _(unset)_      | Diarization hint. |
| `output_format` | string      | `json`         | Same as `/transcribe`: `json,txt,srt,vtt`. |

Speaker labels in the transcript are preserved in the output even when `diarize=false`.

#### Examples

```bash
# Plain text transcript
curl -X POST http://localhost:8000/align \
     -F audio=@speech.mp3 \
     -F 'transcript=Hello world. This is a test of forced alignment.' \
     -F output_format=json,srt

# Speaker-annotated transcript
curl -X POST http://localhost:8000/align \
     -F audio=@interview.wav \
     -F 'transcript=Speaker 1: Welcome to the show.
Speaker 2: Thanks for having me.
Speaker 1: Let us get started.' \
     -F language=en \
     -F output_format=json,srt,vtt

# With acoustic diarization on top of transcript-driven speakers
curl -X POST http://localhost:8000/align \
     -F audio=@panel.m4a \
     -F transcript="$(cat transcript.txt)" \
     -F diarize=true
```

Response (truncated):

```jsonc
{
  "mode": "align",
  "language": "en",
  "duration_seconds": 12.4,
  "processing_time_seconds": 1.8,
  "diarized": true,
  "diarized_acoustic": false,
  "word_level": true,
  "segments": [
    {
      "start": 0.21, "end": 2.4, "text": "Welcome to the show.",
      "speaker": "Speaker 1",
      "words": [
        {"word": "Welcome", "start": 0.21, "end": 0.74, "speaker": "Speaker 1", "score": 0.97}
      ]
    }
  ],
  "word_segments": [ /* flat list of every aligned word */ ],
  "files":           { "txt": "...", "srt": "...", "vtt": "..." },
  "download_links":  { "txt": "/download/.../transcript.txt", ... }
}
```

---

## Web UI

Open <http://localhost:8000/> in a browser to use the built-in **WhisperX Studio** UI.
It has two tabs:

1. **Transcribe** — upload audio, get a Whisper transcript with optional diarization.
2. **Align from Transcript** — upload audio + paste a known transcript, get word-level timestamps.

Both tabs render results in a JSON / TXT / SRT / VTT / Word-table viewer and provide
download links for the generated files.

---

## Performance characteristics

Measured on RTX 4090, CUDA 12.1, `large-v3` + alignment + VAD + diarization:

| Audio length | Batch size | Wall time   | RTF      | Peak VRAM |
| ------------ | ---------- | ----------- | -------- | --------- |
| 1 min        | 16         | ~2 s        | ~30×     | ~5 GB     |
| 10 min       | 8          | ~25 s       | ~24×     | ~6 GB     |
| 1 h          | 4          | ~80–120 s   | ~30–45×  | ~7.5 GB   |

Throughput up to **~70× real-time** without diarization on the same hardware.
The server picks `batch_size` automatically based on audio length:

- `< 5 min` → `16`
- `< 30 min` → `8`
- otherwise → `4`

---

## Error handling

All errors return JSON:

```json
{ "detail": "GPU out of memory. Try a shorter file or lower batch size." }
```

Common cases:

| Status | Meaning |
| ------ | ------- |
| `400`  | Bad parameters; e.g. `diarize=true` but `HF_TOKEN` missing. |
| `413`  | File or audio duration exceeds limit. |
| `415`  | Unsupported file extension. |
| `500`  | Pipeline failure (full traceback in server logs). |
| `507`  | CUDA out of memory. |

---

## Test client

A minimal Python client is included:

```bash
python client_test.py path/to/audio.mp3 --formats json,srt,vtt,txt --out ./output
```

---

## File layout

```
.
├── server.py            # FastAPI app — all endpoints + pipeline
├── requirements.txt     # Pinned Python dependencies
├── Dockerfile           # CUDA 12.1 multi-stage image
├── .env.example         # Documented env variables
├── client_test.py       # Sample client
└── README.md
```

---

## License & credits

This project glues together open-source components governed by their own licenses:
[WhisperX](https://github.com/m-bain/whisperX) (BSD-2),
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) (MIT),
[pyannote.audio](https://github.com/pyannote/pyannote-audio) (MIT, models gated on HF),
[OpenAI Whisper](https://github.com/openai/whisper) (MIT).
