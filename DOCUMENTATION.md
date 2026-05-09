# WhisperX API Server — Complete Documentation

> **Version:** 2.2.0  
> **Engine:** [WhisperX](https://github.com/m-bain/whisperX) wrapping OpenAI Whisper Large V3 + Meta MMS-1b-all  
> **Framework:** FastAPI (Python 3.10+)

---

## Table of Contents

1. [Overview](#1-overview)
2. [Technology Stack](#2-technology-stack)
3. [Setup & Running](#3-setup--running)
4. [Configuration (Environment Variables)](#4-configuration-environment-variables)
5. [API Endpoints](#5-api-endpoints)
   - [GET /health](#51-get-health)
   - [POST /transcribe](#52-post-transcribe)
   - [POST /session/diarize](#53-post-sessiondiarize)
   - [POST /align](#54-post-align)
   - [POST /batch](#55-post-batch)
   - [POST /batch-align](#56-post-batch-align)
   - [GET /download/{file_id}/{name}](#57-get-downloadfile_idname)
   - [GET / (Web UI)](#58-get--web-ui)
   - [WebSocket /ws/stt (Real-time STT)](#59-websocket-wsstt--real-time-streaming-stt)
6. [Output Formats](#6-output-formats)
7. [Speaker Labels](#7-speaker-labels)
8. [Speaker Name Remapping](#8-speaker-name-remapping)
9. [How Forced Alignment Works](#9-how-forced-alignment-works)
10. [Universal Alignment — Meta MMS Backend](#10-universal-alignment--meta-mms-backend)
11. [File Downloads](#11-file-downloads)
12. [Error Reference](#12-error-reference)
13. [Supported Audio Formats](#13-supported-audio-formats)
14. [Supported Languages](#14-supported-languages)
15. [Project Structure](#15-project-structure)
16. [Docker](#16-docker)
17. [MLX Backend (Apple Silicon)](#17-mlx-backend-apple-silicon)
18. [HuggingFace Spaces Deployment (ZeroGPU)](#18-huggingface-spaces-deployment-zerogpu)
19. [Always-on Monitoring](#19-always-on-monitoring)
20. [Session Diarization — Deep Reference](#20-session-diarization--deep-reference)

---

## 1. Overview

This is a self-hosted REST API that provides:

| Capability | How |
|---|---|
| **Automatic transcription** | WhisperX Large V3 (ASR) |
| **Word-level timestamps** | wav2vec2 forced alignment (automatic, post-ASR) |
| **Forced alignment from known transcript** | Provide your own text → get timestamps for every word |
| **Universal alignment (any language)** | Meta MMS-1b-all — 1,107 languages, zero-drift CTC alignment |
| **Speaker diarization** | pyannote.audio 3.2 (who spoke when, with intelligent fallback) |
| **Session Diarization** | Dedicated endpoint (`POST /session/diarize`) with accuracy heuristics, confidence score, and coloured Web UI |
| **Speaker name remapping** | Replace auto labels (SPEAKER_00) with real names |
| **Multiple output formats** | JSON, TXT, SRT, VTT, TSV |
| **Batch transcription** | Submit multiple files in one request (`POST /batch`) |
| **Batch alignment** | Submit multiple audio+transcript pairs in one request (`POST /batch-align`) |
| **Web UI** | Browser-based studio: Transcribe, Align, Batch Transcribe, Batch Align, Session Diarization, Subtitle Player |

WhisperX is **not** the same as plain Whisper. It adds:
- Faster inference via `faster-whisper`
- Automatic word-level timestamps via a separate wav2vec2 pass
- Speaker diarization via pyannote.audio

---

## 2. Technology Stack

| Component | Library | Purpose |
|---|---|---|
| ASR engine | `whisperx` + `faster-whisper` | Speech recognition (Whisper Large V3) |
| Word alignment (39 languages) | `wav2vec2` (via whisperx) | Millisecond-accurate word timestamps for supported languages |
| Word alignment (1,107 languages) | `facebook/mms-1b-all` (Meta MMS) | Universal CTC forced alignment — any language |
| VAD | Silero VAD (via whisperx) | Remove silence before processing |
| Speaker diarization | `pyannote.audio` 3.2 (falls back to 3.1) | Identify who is speaking |
| Language detection | `langdetect` | Auto-detect transcript language for `/align` |
| Script romanization | `uroman` | Convert non-Latin scripts before MMS alignment |
| Web framework | `FastAPI` + `uvicorn` | REST API server |
| Audio loading | `ffmpeg` (via whisperx) | Decode any audio/video format |

---

## 3. Setup & Running

### Prerequisites

- Python 3.10
- ffmpeg installed (`brew install ffmpeg` on macOS)
- A virtual environment with dependencies installed

### Install

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run

```bash
source .venv/bin/activate
PORT=8001 python server.py
```

Or with uvicorn directly:
```bash
uvicorn server:app --host 0.0.0.0 --port 8001
```

### Verify

```bash
curl http://localhost:8001/health
```

---

## 4. Configuration (Environment Variables)

| Variable | Default | Description |
|---|---|---|
| `PORT` | `8000` | Port to listen on |
| `HOST` | `0.0.0.0` | Bind address |
| `WHISPER_MODEL` | `large-v3` | Whisper model size: `tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3` |
| `COMPUTE_TYPE` | `float16` (GPU) / `int8` (CPU) | Quantization: `float16`, `int8`, `int8_float16` |
| `BEAM_SIZE` | `5` | Beam search width (higher = more accurate, slower) |
| `DEVICE` | auto (`cuda` or `cpu`) | Inferred automatically from available hardware |
| `HF_TOKEN` | *(unset)* | HuggingFace access token. **Optional** — only required if you want pyannote (default `diarize` backend works without it). |
| `MODEL_DIR` | `./models` | Local cache directory for downloaded models |
| `MAX_AUDIO_DURATION` | `86400` | Maximum audio length in seconds (24 hours) |
| `MAX_FILE_SIZE` | `524288000` | Maximum upload size in bytes (500 MB) |
| `DOWNLOAD_TTL_SECONDS` | `3600` | How long generated files are kept before auto-deletion (1 hour) |
| `JOB_TTL_SECONDS` | `3600` | How long batch job records are kept in memory (1 hour) |
| `DIARIZATION_BACKEND` | `auto` | Which diarization stack to use: `diarize` (FoxNoseTech, CPU, no token, default in `auto`), `pyannote` (HF, needs token), `vibevoice` (end-to-end speaker-diarized ASR via isolated sidecar — requires CUDA + a separate venv, see `vibevoice/README.md`), or `auto` (try `diarize` first; if it collapses to 1 speaker but the user requested ≥2 and `HF_TOKEN` is set, automatically retry with pyannote as a second opinion). |
| `VIBEVOICE_VENV` | _(unset)_ | Path to the isolated venv containing `transformers>=5.3.0` (see `vibevoice/requirements-vibevoice.txt`). When set together with `DIARIZATION_BACKEND=vibevoice`, the server auto-launches `vibevoice/sidecar.py` as a subprocess on `127.0.0.1:9001`. |
| `VIBEVOICE_AUTOSTART` | `1` | Set to `0` to skip the subprocess launch (manage the sidecar yourself). |
| `VIBEVOICE_HOST` / `VIBEVOICE_PORT` | `127.0.0.1` / `9001` | Sidecar bind address. |
| `VIBEVOICE_DTYPE` | `bfloat16` | Sidecar model dtype: `bfloat16`, `float16`, `float32`. |
| `VIBEVOICE_CHUNK_MINUTES` | `55` | Long-form chunk size (audio longer than this is split). |
| `VIBEVOICE_OVERLAP_SECONDS` | `300` | Long-form chunk overlap. |
| `VIBEVOICE_TIMEOUT` | `1800` | Per-request HTTP timeout from server → sidecar (seconds). |
| `VBX_ENABLED` | `1` | Enable Tier-2 VB-HMM resegmentation (vendored BUTSpeechFIT/VBx). When Tier-1 dominance ≥ `VBX_DOMINANCE_THRESHOLD` and the caller wants ≥2 speakers, VBx is run as a "second-pass referee" using the same WeSpeaker embeddings. The result is adopted only if it lowers dominance and finds at least as many speakers as Tier-1. Set `0` to disable. |
| `VBX_DOMINANCE_THRESHOLD` | `0.70` | Tier-1 single-speaker dominance ratio above which VBx is triggered. |
| `DIARIZATION_MODEL` | `pyannote/speaker-diarization-3.2` | Primary pyannote model ID (only used when pyannote backend runs). Server tries this first, falls back to `DIARIZATION_MODEL_FALLBACK` if download fails (e.g. EULA not accepted). |
| `DIARIZATION_MODEL_FALLBACK` | `pyannote/speaker-diarization-3.1` | Fallback pyannote model if primary is unavailable. Set to empty string to disable fallback. |
| `DIARIZATION_DOMINANCE_THRESHOLD` | `0.80` | If a single speaker covers more than this fraction of total speech, diarization is automatically re-run with `min_speakers=2` and a stricter clustering threshold. Set to `1.0` to disable. |

> **Note:** On CPU, `COMPUTE_TYPE` is automatically overridden to `int8`.  
> Set via env before starting: `export HF_TOKEN=hf_xxx` or `HF_TOKEN=hf_xxx python server.py`

---

## 5. API Endpoints

### 5.1 GET /health

Returns server status and model load state.

**Request:**
```bash
curl http://localhost:8001/health
```

**Response:**
```json
{
  "status": "ok",
  "uptime_seconds": 123.4,
  "device": "cpu",
  "compute_type": "int8",
  "whisper_model": "large-v3",
  "whisper_loaded": true,
  "diarization_loaded": false,
  "diarization_error": "HF_TOKEN not set",
  "hf_token_present": false,
  "gpu": { "device": "cpu" },
  "loaded_align_models": ["en"],
  "mms_loaded": true,
  "mms_error": null
}
```

---

### 5.2 POST /transcribe

Transcribes an audio file using WhisperX Large V3. Returns word-level timestamps. Optionally diarizes speakers.

**Form fields:**

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `file` | file | ✅ | — | Audio/video file |
| `language` | string | ❌ | auto-detect | ISO 639-1 code e.g. `en`, `fr`, `de`. See [Supported Languages](#13-supported-languages) |
| `diarize` | bool | ❌ | `true` if HF_TOKEN set | Run speaker diarization |
| `min_speakers` | int | ❌ | — | Hint for diarization |
| `max_speakers` | int | ❌ | — | Hint for diarization |
| `vad` | bool | ❌ | `true` | Strip silence with Silero VAD |
| `output_format` | string | ❌ | `json` | Comma-separated: `json,txt,srt,vtt,tsv` |
| `initial_prompt` | string | ❌ | — | Optional prompt to guide transcription style |
| `task` | string | ❌ | `transcribe` | `transcribe` or `translate` (to English) |
| `speaker_names` | string | ❌ | — | JSON map to rename speakers: `{"SPEAKER_00":"Alice","SPEAKER_01":"Bob"}` |

**Example:**
```bash
curl -X POST http://localhost:8001/transcribe \
     -F file=@audio.m4a \
     -F language=en \
     -F output_format=json,srt,vtt,txt \
     -o result.json
```

**Response fields:**

| Field | Description |
|---|---|
| `mode` | `"transcribe"` |
| `language` | Detected or specified language code |
| `duration_seconds` | Audio length |
| `processing_time_seconds` | How long it took |
| `realtime_factor` | `duration / processing_time` (higher = faster than real-time) |
| `diarized` | Whether speakers were assigned |
| `segments` | List of transcript segments with `start`, `end`, `text`, `speaker` (if diarized), `words` |
| `word_segments` | Flat list of all words with individual `start`, `end`, `score` |
| `files` | Dict of format → file content as string (and `_base64` variants) |
| `download_links` | Dict of format → `/download/{id}/transcript.{ext}` URLs |
| `download_expires_at` | ISO timestamp when download links expire |

**Extract SRT from response:**
```bash
python3 -c "import json; open('transcript.srt','w').write(json.load(open('result.json'))['files']['srt'])"
```

---

### 5.3 POST /session/diarize

The **dedicated speaker diarization endpoint**. Identical to `POST /transcribe` with `diarize=true`, but:
- Diarization is **always forced** — it cannot be turned off.
- Returns `503` immediately if `HF_TOKEN` is not configured (rather than silently falling back to no-speaker output).
- Includes additional response fields: `diarization_confidence`, `diarization_resplit`, `diarization_model`.
- This is the intended "source of truth" endpoint for any downstream feature that needs to know who said what (summarizer, note-taker, speech-to-speech pipeline, etc.).

**Prerequisites:**
- `HF_TOKEN` env var must be set to a Hugging Face token that has accepted the user agreement on https://hf.co/pyannote/speaker-diarization-3.1 (minimum) or https://hf.co/pyannote/speaker-diarization-3.2 (recommended).
- Audio must be uploaded as a file (not a URL).

**Form fields:**

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `file` | file | ✅ | — | Audio/video file (any ffmpeg-decodable format) |
| `language` | string | ❌ | auto-detect | ISO 639-1 code e.g. `en`, `fr`. **Leave blank for mixed-language audio.** |
| `min_speakers` | int | ❌ | — | Minimum number of speakers. Strongly recommended for known speaker counts. |
| `max_speakers` | int | ❌ | — | Maximum number of speakers. Set equal to `min_speakers` for exact count. |
| `vad` | bool | ❌ | `true` | Strip silence with Silero VAD before transcription |
| `output_format` | string | ❌ | `json,txt,srt,vtt` | Comma-separated: `json,txt,srt,vtt,tsv` |
| `initial_prompt` | string | ❌ | — | Optional Whisper prompt to guide transcription style or vocabulary |
| `task` | string | ❌ | `transcribe` | `transcribe` or `translate` (translate to English) |
| `speaker_names` | string | ❌ | — | JSON map to rename speakers: `{"SPEAKER_00":"Alice","SPEAKER_01":"Bob"}` |
| `include_speakers` | bool | ❌ | `true` | Set `false` to strip `[SPEAKER_xx]:` labels from TXT/SRT/VTT output |

**Example — 2-person podcast (French):**
```bash
curl -X POST http://localhost:8000/session/diarize \
     -F file=@podcast.m4a \
     -F language=fr \
     -F min_speakers=2 \
     -F max_speakers=2 \
     -F output_format=json,srt,txt \
     -o diarized.json
```

**Example — unknown number of speakers, auto language:**
```bash
curl -X POST http://localhost:8000/session/diarize \
     -F file=@meeting.mp3 \
     -F output_format=json,txt \
     -o diarized.json
```

**Example — with speaker name remapping:**
```bash
curl -X POST http://localhost:8000/session/diarize \
     -F file=@interview.wav \
     -F min_speakers=2 \
     -F max_speakers=2 \
     -F 'speaker_names={"SPEAKER_00":"Host","SPEAKER_01":"Guest"}' \
     -F output_format=json,srt \
     -o diarized.json
```

**Response fields:**

| Field | Type | Description |
|---|---|---|
| `language` | string | Detected or specified language code |
| `duration_seconds` | float | Audio length in seconds |
| `processing_time_seconds` | float | Total wall-clock time for the full pipeline |
| `transcribe_time_seconds` | float | Time spent on ASR transcription alone |
| `realtime_factor` | float | `duration / processing_time` — higher means faster than real-time |
| `diarized` | bool | Always `true` for this endpoint |
| `diarization_warning` | string\|null | Set if diarization encountered a non-fatal issue |
| `diarization_confidence` | float\|null | Score from 0.0–1.0 indicating balance of speaker time. Below 0.5 means one speaker dominated — consider setting `min_speakers`. |
| `diarization_resplit` | bool | `true` if the dominance heuristic triggered an automatic re-run with `min_speakers=2` |
| `diarization_model` | string\|null | Which pyannote model was actually used (e.g. `"pyannote/speaker-diarization-3.1"`) |
| `alignment_warning` | string\|null | Set if the alignment model had to fall back |
| `language_warning` | string\|null | Set if the requested `language` differed from Whisper's detected language |
| `segments` | array | Transcript segments — see [Segment Schema](#segment-schema) |
| `word_segments` | array | Flat list of all words with individual `start`, `end`, `score`, `speaker` |
| `files` | object | Format → file content as string (and `_base64` variants) |
| `download_links` | object | Format → `/download/{id}/transcript.{ext}` URLs |
| `download_expires_at` | string | ISO timestamp when download links expire |

**Segment schema:**

Each element of `segments` has:

```json
{
  "start": 0.0,
  "end": 4.25,
  "text": "There have been few brands that have fallen further faster.",
  "speaker": "SPEAKER_00",
  "words": [
    {"word": "There",   "start": 0.00, "end": 0.24, "score": 0.98, "speaker": "SPEAKER_00"},
    {"word": "have",    "start": 0.25, "end": 0.45, "score": 0.99, "speaker": "SPEAKER_00"},
    ...
  ]
}
```

- `speaker` is always set (never `null`, never `"UNKNOWN"`) — see [Speaker Fallback Algorithm](#speaker-fallback-algorithm).
- `start`/`end` come from the ASR segment, never from audio duration or diarization turn boundaries.
- `words` contains per-word `start`, `end`, `score` (alignment confidence), and `speaker`.

**Extract SRT from response:**
```bash
python3 -c "import json; open('diarized.srt','w').write(json.load(open('diarized.json'))['files']['srt'])"
```

**Error codes specific to this endpoint:**

| Code | Cause |
|---|---|
| `400` | Empty file, invalid `task` value, `max_speakers < min_speakers` |
| `413` | File exceeds `MAX_FILE_SIZE` |
| `415` | Unsupported audio file extension |
| `503` | `HF_TOKEN` not set — diarization unavailable |
| `507` | GPU out of memory |
| `500` | Internal pipeline failure |

---

### 5.4 POST /align

**Forced alignment** — you provide the exact, correct transcript text and the audio. The server returns word-level timestamps for every single word in your transcript.

Use this when:
- You already have a 100% accurate transcript
- You need precise timestamps for subtitles/captions
- You want to avoid ASR errors

**How it works internally (three-tier alignment):**
1. Parse transcript → detect speaker turns
2. Detect language (from text or your `language` param)
3. Optional VAD to trim silence and extract speech region
4. **Strategy A — wav2vec2** (for the ~39 languages with a dedicated model):
   - Pass the full audio + full GT transcript as a single segment to `whisperx.align()`
   - One continuous scan — no chunk boundaries, no timestamp gaps
5. **Strategy B — Meta MMS** (fallback for all other languages):
   - Load the `facebook/mms-1b-all` language adapter for the detected language
   - Tokenize word-by-word (no separator tokens) → CTC forced alignment via `torchaudio.functional.forced_align`
   - Collapse token spans with `merge_tokens` → unflatten directly to words
   - Romanize non-Latin scripts with `uroman` if needed
6. **Strategy C — uniform interpolation** (last resort if both A and B fail):
   - Distribute words evenly across the audio duration
7. Inject speaker labels from the parsed turn structure
8. Rebuild output segments per speaker turn; add VAD offset to all timestamps

The three-tier chain means **every language on earth is supported** — wav2vec2 for the 39 fastest, MMS for the remaining 1,068, uniform as a non-failing safety net.

**Form fields:**

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `audio` | file | ✅ | — | Audio file |
| `transcript` | string | ✅ | — | Your exact ground-truth transcript text |
| `language` | string | ❌ | auto-detect | ISO 639-1 code. See [Supported Languages](#13-supported-languages) |
| `vad` | bool | ❌ | `true` | Strip silence before aligning |
| `diarize` | bool | ❌ | `false` | Also run pyannote acoustic diarization |
| `min_speakers` | int | ❌ | — | Hint for diarization |
| `max_speakers` | int | ❌ | — | Hint for diarization |
| `output_format` | string | ❌ | `json` | Comma-separated: `json,txt,srt,vtt,tsv` |
| `speaker_names` | string | ❌ | — | JSON map to rename speakers: `{"SPEAKER_00":"Alice"}` |

**Example — transcript from a file (recommended):**
```bash
curl -X POST http://localhost:8001/align \
     -F audio=@audio.m4a \
     -F "transcript=<transcript.txt" \
     -F output_format=json,srt,vtt,txt \
     -o aligned.json
```

> ⚠️ Always use `-F "transcript=<yourfile.txt"` (curl file-read syntax) instead of  
> `-F transcript="$(cat yourfile.txt)"` — shell expansion truncates text containing  
> special characters like `;`, `"`, curly quotes (`""`), or `!`.

**Example — inline short transcript:**
```bash
curl -X POST http://localhost:8001/align \
     -F audio=@audio.m4a \
     -F 'transcript=Hello world. This is a test.' \
     -F output_format=json,srt \
     -o aligned.json
```

**Example — with speaker labels:**
```bash
# transcript.txt:
# Speaker 1: Hello, welcome to the show.
# Speaker 2: Thanks for having me.

curl -X POST http://localhost:8001/align \
     -F audio=@audio.m4a \
     -F "transcript=<transcript.txt" \
     -F output_format=json,srt \
     -o aligned.json
```

**Response fields (in addition to standard fields):**

| Field | Description |
|---|---|
| `mode` | `"align"` |
| `ground_truth_word_count` | Total words in your transcript |
| `interpolated_word_count` | Words that couldn't be force-aligned (given estimated timestamps) |
| `warning` | Human-readable message if any words were interpolated |
| `turn_count` | Number of speaker turns parsed |
| `language_detection` | `"user"`, `"langdetect"`, or `"default-en"` |
| `vad_window` | `{start, end}` of speech region used (if `vad=true`) |
| `alignment_strategy` | `"wav2vec2"`, `"mms"`, or `"uniform"` — which backend was used |
| `alignment_warning` | Set if alignment model had to fall back to a different strategy |
| `segments` | Segments per speaker turn with word-level timestamps |
| `word_segments` | Flat list of all words (one per GT word, 1:1 with transcript) |

**Extract SRT after align:**
```bash
python3 -c "import json; open('transcript.srt','w').write(json.load(open('aligned.json'))['files']['srt'])"
```

**Check alignment quality:**
```bash
python3 -c "
import json
d = json.load(open('aligned.json'))
ws = d['word_segments']
print('GT words:', d['ground_truth_word_count'])
print('Interpolated:', d['interpolated_word_count'])
print('First word:', ws[0]['word'], '@', ws[0]['start'])
print('Last word:', ws[-1]['word'], '@', ws[-1]['end'])
"
```

---

### 5.5 POST /batch

Transcribe multiple audio files in a single request. Files are processed **sequentially** (one after another). The request blocks until all files are done, then returns all results at once.

Works on ZeroGPU (HF Spaces) because the GPU is held for the full duration.

**Form fields:**

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `files` | file[] | ✅ | — | One or more audio/video files (repeat `-F files=@...` for each) |
| `language` | string | ❌ | auto-detect | ISO 639-1 code applied to all files |
| `diarize` | bool | ❌ | auto | Speaker diarization for all files |
| `min_speakers` | int | ❌ | — | Hint for diarization |
| `max_speakers` | int | ❌ | — | Hint for diarization |
| `vad` | bool | ❌ | `true` | Silero VAD for all files |
| `output_format` | string | ❌ | `json` | Comma-separated: `json,txt,srt,vtt,tsv` |
| `initial_prompt` | string | ❌ | — | Prompt applied to all files |
| `task` | string | ❌ | `transcribe` | `transcribe` or `translate` |
| `speaker_names` | string | ❌ | — | JSON speaker name map applied to all files |

**Example:**
```bash
curl -X POST https://tourousou-whisperx-api.hf.space/batch \
  -F files=@interview1.mp3 \
  -F files=@interview2.mp3 \
  -F files=@interview3.mp3 \
  -F language=fr \
  -F output_format=srt,txt,json \
  -o batch_results.json
```

**Response:**
```json
{
  "status": "done",
  "total": 3,
  "done": 3,
  "failed": 0,
  "results": [
    {
      "filename": "interview1.mp3",
      "status": "done",
      "result": { "language": "fr", "segments": [...], "download_links": {...} },
      "error": null
    },
    ...
  ]
}
```

`status` is `"done"` if all files succeeded, `"partial"` if some failed.

---

### 5.6 POST /batch-align

Align multiple audio+transcript pairs in a single request. Each pair has its own audio file, transcript, and optional language code. Processes sequentially, returns all results when done.

**Form fields:**

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `files` | file[] | ✅ | — | Audio files, one per pair (in order) |
| `transcripts` | string[] | ✅ | — | Transcripts, one per file (in order, repeat `-F transcripts=...`) |
| `languages` | string | ❌ | — | JSON array of language codes, one per file: `["fr","en",null]`. `null` = auto-detect |
| `diarize` | bool | ❌ | `false` | Speaker diarization for all files |
| `min_speakers` | int | ❌ | — | Hint for diarization |
| `max_speakers` | int | ❌ | — | Hint for diarization |
| `vad` | bool | ❌ | `true` | Silero VAD |
| `output_format` | string | ❌ | `json` | Comma-separated: `json,txt,srt,vtt,tsv` |
| `speaker_names` | string | ❌ | — | JSON speaker name map applied to all files |

**Example:**
```bash
curl -X POST https://tourousou-whisperx-api.hf.space/batch-align \
  -F files=@interview_fr.mp3 \
  -F files=@meeting_en.mp3 \
  -F 'transcripts=<transcript_fr.txt' \
  -F 'transcripts=<transcript_en.txt' \
  -F 'languages=["fr","en"]' \
  -F output_format=srt,txt,json \
  -o batch_align_results.json
```

**Response:** Same structure as `/batch`.

---

### 5.7 GET /download/{file_id}/{name}

Download a generated transcript file by its ID and filename.

- Links are returned in the `download_links` field of `/transcribe` and `/align` responses
- Files expire after `DOWNLOAD_TTL_SECONDS` (default: 1 hour)
- Returns `404` if file has expired or never existed

**Example:**
```bash
curl http://localhost:8001/download/abc123/transcript.srt -o my.srt
```

---

### 5.8 GET / (Web UI)

Opens the **WhisperX Studio** browser UI at `http://localhost:8001`.

- **Transcribe tab** — upload audio, set options, get transcript + download links
- **Align from Transcript tab** — upload audio + paste/upload transcript, get timed output
- **Batch Transcribe tab** — drag-and-drop multiple files, process all at once
- **Batch Align tab** — add rows of audio+transcript pairs with per-file language, process all at once
- **👥 Session Diarization tab** — speaker-labelled transcription with colour-coded turns, confidence display, and downloads (see [§20](#20-session-diarization--deep-reference))
- **Production Server tab** — live health status + monitoring info
- **Realtime STT tab** — record from your microphone and get word-level transcripts streamed back live (uses `/ws/stt`, see §5.9 below)

Also available at `http://localhost:8001/ui`.

---

## 5.9 WebSocket /ws/stt — Real-time streaming STT

A low-latency streaming endpoint that accepts microphone PCM and returns word-level
transcripts every ~5 seconds. Backed by the same WhisperX large-v3 + wav2vec2 stack as
the batch `/transcribe` endpoint. **No diarization is performed inline** (use
`/transcribe` for that).

### Wire protocol

**Audio frames (client → server, binary):**
- Raw PCM, **mono, 16 kHz, little-endian Float32** (any frame length).

**Control messages (client → server, text JSON):**
```json
{"type": "start", "language": "en"}    // language is optional; null = auto-detect
{"type": "end"}                         // flush remaining audio + emit done
```

**Server → client (text JSON):**
```json
{"type": "ready",   "session_id": "...", "sample_rate": 16000, "language": "en"}
{"type": "final",   "segment_id": 0, "text": "Hello world",
                    "start": 0.12, "end": 0.95,
                    "words": [{"word":"Hello","start":0.12,"end":0.42},
                              {"word":"world","start":0.55,"end":0.95}]}
{"type": "done",    "srt_url": "/download/<sid>/transcript.srt",
                    "txt_url": "/download/<sid>/transcript.txt",
                    "duration_s": 12.4, "segments": 3}
{"type": "error",   "error": "..."}
```

### Behavior

- Audio is committed in **5-second chunks** with **0.5 s leading overlap**. Words from
  the overlap region are deduplicated against the previously-emitted segment.
- The full session transcript is written to `DOWNLOAD_DIR/<session_id>/transcript.{srt,txt}`
  on `end`, and served via the existing `/download/{file_id}/{name}` endpoint
  (subject to `DOWNLOAD_TTL_SECONDS`).
- Buffer is capped at 90 s of audio; older audio is discarded once committed.

### Browser example

```js
const ws = new WebSocket('wss://YOUR_HOST/ws/stt');
ws.binaryType = 'arraybuffer';
ws.onopen = () => ws.send(JSON.stringify({type: 'start', language: 'en'}));
ws.onmessage = ev => console.log(JSON.parse(ev.data));
// later, send Float32Array audio buffers:
//   ws.send(float32Buffer.buffer);
// when done:
//   ws.send(JSON.stringify({type: 'end'}));
```

The bundled web UI (`Realtime STT` tab) implements the full client.

---

## 6. Output Formats

All endpoints accept `output_format` as a comma-separated list. `json` is always included.

| Format | Description |
|---|---|
| `json` | Full response with all metadata, segments, word timestamps, and inline file content |
| `txt` | Plain text transcript. Prefixed with `[Speaker]:` if diarized |
| `srt` | SubRip subtitle format. Speaker label on its own line above text if diarized |
| `vtt` | WebVTT subtitle format. Same structure as SRT |
| `tsv` | Tab-separated values (whisper-native). Columns: `start` (ms), `end` (ms), `speaker`, `text` |

---

## 7. Speaker Labels

### In `/transcribe` and `/batch`

Speaker labels are added automatically when `diarize=true` and `HF_TOKEN` is set. Labels are machine-generated: `SPEAKER_00`, `SPEAKER_01`, etc.

### In `/align` and `/batch-align`

The server parses speaker prefixes from your transcript automatically. Supported formats:

| Format | Example |
|---|---|
| `Speaker N:` | `Speaker 1: Hello there.` |
| `SPEAKER_00:` | `SPEAKER_00: Hello there.` |
| `[A]:` | `[A]: Hello there.` |
| `Name:` | `John: Hello there.` |
| `Name -` | `Narrator - Hello there.` |
| No prefix | Plain text (single speaker, no label) |

You can also run acoustic diarization on top with `diarize=true` to override/augment the transcript-based labels.

---

## 8. Speaker Name Remapping

All endpoints accept an optional `speaker_names` field — a JSON string mapping auto-generated speaker IDs to real names. Applied **after** processing, **before** rendering, so all output formats (SRT, VTT, TSV, TXT) use the real names.

```bash
curl -X POST http://localhost:8001/transcribe \
  -F file=@audio.mp3 \
  -F diarize=true \
  -F 'speaker_names={"SPEAKER_00":"Alice","SPEAKER_01":"Bob"}'
```

In the resulting SRT/TXT/TSV, `SPEAKER_00` is replaced by `Alice` everywhere.

> **Note:** Only applied when diarization ran (`diarized: true` in response). Ignored for non-diarized output.

---

## 9. How Forced Alignment Works

### Three-tier alignment chain

```
Audio + Transcript
       │
       ▼
[1] Detect language
       │
       ├──► wav2vec2 supported? ──YES──► [A] whisperx.align()
       │         (39 languages)               full audio, single pass
       │                                           │
       │                                           ▼
       │                              aligned ≥60% of words?
       │                                     │         │
       │                                    YES        NO
       │                                     │          │
       │                         ◄───────────┘          │
       │                         done                    │
       │                                                 ▼
       └──► NOT supported ──────────────────► [B] MMS alignment
                                                facebook/mms-1b-all
                                                1,107 languages
                                                          │
                                                          ▼
                                               both A fail? ──► [C] Uniform
                                                                  interpolation
```

### Strategy A — wav2vec2 (39 languages)

WhisperX's built-in alignment model. One forward pass over the **full audio** (no chunking) with the full GT transcript as a single segment. This eliminates the old chunking bug where words were locked into segment time windows and gaps appeared wherever Whisper had silence.

### Strategy B — Meta MMS (1,107 languages)

`facebook/mms-1b-all` is a wav2vec2-CTC model fine-tuned across 1,107 languages. Each language has a small loadable adapter (~10 MB). The alignment procedure:

1. Load the language adapter: `model.load_adapter(iso3)`
2. Tokenize the transcript **word by word** — separator tokens (`|`, space) are **never** inserted into the CTC target sequence. This is the critical correctness invariant: including separators caused a systematic ~7-second timestamp drift on 3-minute audio in earlier implementations.
3. Run `torchaudio.functional.forced_align(log_probs, targets)` — Viterbi forced alignment.
4. Collapse consecutive frames with `torchaudio.functional.merge_tokens`.
5. Unflatten merged spans directly to words by token length — zero drift possible.
6. Non-Latin scripts (Arabic, Hindi, Chinese, etc.) are romanized with `uroman` before tokenization, then the original words are used in the output.

### Strategy C — uniform interpolation (safety net)

If both wav2vec2 and MMS fail (e.g. model load error), words are distributed evenly across the audio duration. Timestamps will be approximate but the output shape is always valid. `interpolated_word_count` will equal the full word count and `alignment_strategy` will be `"uniform"`.

### Quality gate between wav2vec2 and MMS

If wav2vec2 aligns fewer than 60% of GT words, the pipeline automatically retries with MMS. This handles edge cases where a language is technically supported by wav2vec2 but the audio quality or accent causes the model to drop most words.

### Transcript reconciliation & sentence segmentation

After alignment (regardless of strategy), the pipeline runs two post-processing steps:

1. **Reconciliation (`_reconcile_to_gt`)** — The aligners normalise text internally (lowercase, punctuation stripped, contractions split on apostrophes). This step maps every aligner token back to the **original** GT word using greedy character-prefix matching, so the output preserves exact wording — including `Donc,`, `d'abord,`, `l'église.`, Arabic diacritics, etc. — while keeping the aligner's precise timestamps.

2. **Sentence segmentation (`_segment_by_sentences`)** — Segments are split at sentence-ending punctuation (`. ! ? …` and CJK equivalents `。！？`) in the original transcript. Speaker turn changes also start a new segment. This makes every SRT/VTT cue correspond to a real sentence, matching the flow of the source text.

`_split_long_segments` runs afterwards as a safety net for run-on sentences (max ~7 s / 110 chars), and can now detect clause boundaries (`,` `;` `:`) because original punctuation is preserved.

**Both steps are language-agnostic** — they operate on Unicode text and work identically for French, Lingala, Arabic, Chinese, or any other language.

---

## 10. Universal Alignment — Meta MMS Backend

### What is MMS?

Meta's [Massively Multilingual Speech](https://ai.meta.com/blog/multilingual-model-speech-recognition/) project trained wav2vec2-CTC models on 1,107 languages — covering virtually every language that has ever had recorded audio. The `facebook/mms-1b-all` model is a single 1B-parameter model with per-language adapters. It runs in fp16 and fits within 8 GB VRAM.

### Languages covered

MMS supports all languages Whisper supports, plus hundreds more including:
- All major African languages (Lingala, Kinyarwanda, Wolof, Igbo, Xhosa, Zulu, Twi, Fula, Bambara, ...)
- All South/Southeast Asian languages
- Indigenous languages of the Americas
- Minority European languages

### Non-Latin script support

Languages that use non-Latin scripts (Arabic, Devanagari, Cyrillic, CJK, etc.) are romanized using `uroman` before tokenization. The original (non-romanized) words always appear in the output — romanization is an internal alignment step only.

### VRAM usage

| Model | VRAM (fp16) |
|---|---|
| `facebook/mms-1b-all` | ~4 GB |
| Language adapter | ~10 MB each (cached after first use) |
| wav2vec2 align models | ~200 MB each |

Total typical peak (MMS + Whisper large-v3): ~10–12 GB. Fits on a T4 (16 GB).

### `alignment_strategy` field

Every `/align` and `/batch-align` response includes `alignment_strategy`:

| Value | Meaning |
|---|---|
| `"wav2vec2"` | Aligned with whisperx native model (fastest) |
| `"mms"` | Aligned with Meta MMS-1b-all adapter |
| `"uniform"` | Could not align — timestamps are evenly distributed estimates |

---

## 11. File Downloads

When you request `output_format=srt` (or any non-JSON format), the response includes:

```json
{
  "files": {
    "srt": "1\n00:00:00,251 --> ...\n...",
    "srt_base64": "MQowMD...",
    "txt": "Hello world...",
    "txt_base64": "SGVsb..."
  },
  "download_links": {
    "srt": "/download/abc123/transcript.srt",
    "txt": "/download/abc123/transcript.txt"
  },
  "download_expires_at": "2026-05-02T02:30:00Z"
}
```

Files live in the system temp directory and are cleaned up after `DOWNLOAD_TTL_SECONDS`.

---

## 12. Error Reference

| HTTP Status | Meaning |
|---|---|
| `400` | Bad request — empty file, invalid params, unparseable transcript, mismatched files/transcripts count |
| `413` | File too large or audio too long |
| `415` | Unsupported audio file extension |
| `404` | Download file not found or expired |
| `500` | Internal server error (model failure, ffmpeg error, etc.) |
| `507` | GPU out of memory — try a shorter file or lower batch size |
| `503` | Diarization unavailable — `HF_TOKEN` not set (only from `/session/diarize`) |

All errors return `{"detail": "...message..."}`.

---

## 13. Supported Audio Formats

Any format decodable by **ffmpeg**:

`.wav` `.wave` `.mp3` `.mp4` `.m4a` `.flac` `.ogg` `.oga` `.opus` `.webm` `.aac` `.wma` `.aiff` `.aif`

---

## 14. Supported Languages

Pass the **Code** as the `language` parameter. If omitted, language is auto-detected.

| Language | Code | Language | Code | Language | Code |
|---|---|---|---|---|---|
| Afrikaans | `af` | Greek | `el` | Portuguese | `pt` |
| Albanian | `sq` | Gujarati | `gu` | Punjabi | `pa` |
| Amharic | `am` | Haitian Creole | `ht` | Romanian | `ro` |
| Arabic | `ar` | Hausa | `ha` | Russian | `ru` |
| Armenian | `hy` | Hawaiian | `haw` | Sanskrit | `sa` |
| Assamese | `as` | Hebrew | `he` | Serbian | `sr` |
| Azerbaijani | `az` | Hindi | `hi` | Shona | `sn` |
| Bashkir | `ba` | Hungarian | `hu` | Sindhi | `sd` |
| Basque | `eu` | Icelandic | `is` | Sinhala | `si` |
| Belarusian | `be` | Indonesian | `id` | Slovak | `sk` |
| Bengali | `bn` | Italian | `it` | Slovenian | `sl` |
| Bosnian | `bs` | Japanese | `ja` | Somali | `so` |
| Breton | `br` | Javanese | `jw` | Spanish | `es` |
| Bulgarian | `bg` | Kannada | `kn` | Sundanese | `su` |
| Cantonese | `yue` | Kazakh | `kk` | Swahili | `sw` |
| Catalan | `ca` | Khmer | `km` | Swedish | `sv` |
| Chinese | `zh` | Korean | `ko` | Tagalog | `tl` |
| Croatian | `hr` | Lao | `lo` | Tajik | `tg` |
| Czech | `cs` | Latin | `la` | Tamil | `ta` |
| Danish | `da` | Latvian | `lv` | Tatar | `tt` |
| Dutch | `nl` | Lingala | `ln` | Telugu | `te` |
| English | `en` | Lithuanian | `lt` | Thai | `th` |
| Estonian | `et` | Luxembourgish | `lb` | Tibetan | `bo` |
| Faroese | `fo` | Macedonian | `mk` | Turkish | `tr` |
| Finnish | `fi` | Malagasy | `mg` | Turkmen | `tk` |
| French | `fr` | Malay | `ms` | Ukrainian | `uk` |
| Galician | `gl` | Malayalam | `ml` | Urdu | `ur` |
| Georgian | `ka` | Maltese | `mt` | Uzbek | `uz` |
| German | `de` | Maori | `mi` | Vietnamese | `vi` |
| Greek | `el` | Marathi | `mr` | Welsh | `cy` |
| Gujarati | `gu` | Mongolian | `mn` | Yiddish | `yi` |
| | | Myanmar | `my` | Yoruba | `yo` |
| | | Nepali | `ne` | | |
| | | Norwegian | `no` | | |
| | | Nynorsk | `nn` | | |
| | | Occitan | `oc` | | |
| | | Pashto | `ps` | | |
| | | Persian | `fa` | | |
| | | Polish | `pl` | | |

> **Note:** Languages like Kikongo, Xhosa, Zulu, Igbo, Kinyarwanda, Chichewa, and Twi are **not** supported by Whisper large-v3 — they were not included in the model's training data. African languages that *are* supported: Afrikaans (`af`), Amharic (`am`), Hausa (`ha`), Lingala (`ln`), Shona (`sn`), Somali (`so`), Swahili (`sw`), Yoruba (`yo`).
>
> **For `/align` and `/batch-align`:** the Meta MMS backend supports alignment for **1,107 languages** — including Kikongo, Xhosa, Zulu, Igbo, Kinyarwanda, Chichewa, Twi, Wolof, Fula, and hundreds more. You can pass any Whisper-supported language code for transcription, and alignment will automatically use MMS if wav2vec2 does not support that language.

---

## 15. Project Structure

```
speech to text/
├── server.py              # Single-file FastAPI server (all logic)
├── requirements.txt       # Python dependencies
├── Dockerfile             # CUDA 12.1 multi-stage Docker image
├── .env.example           # Example environment variables
├── client_test.py         # Sample Python client
├── colab_server.ipynb     # Google Colab notebook (GPU testing via ngrok)
├── static/
│   └── index.html         # WhisperX Studio web UI
├── models/                # Downloaded model cache (auto-created)
├── README.md              # Short quickstart README
└── DOCUMENTATION.md       # This file
```

### server.py internal structure

| Section | Line range | Description |
|---|---|---|
| Configuration | ~50–85 | Env vars, constants, allowed formats |
| `ModelRegistry` | ~217+ | Lazy-load singletons: Whisper, tiny, align, diarize, MMS |
| MMS helpers | ~758–890 | `ISO1_TO_ISO3` map, `_whisper_to_mms_code()`, `_romanize()`, `_mms_normalize()` |
| `align_with_mms()` | ~892+ | Full MMS forced-alignment implementation |
| Helpers | ~175–290 | `render_txt/srt/vtt`, timestamp formatters, cleanup |
| FastAPI app + CORS | ~295–335 | App init, middleware, static mount |
| `GET/POST /health` | ~583+ | Status endpoint (includes `mms_loaded`, `mms_error`) |
| Speaker prefix parser | ~460–535 | `parse_transcript()`, `_SPEAKER_PREFIX_RE` |
| Language detection | ~538–546 | `detect_language_from_text()` |
| `POST /align` | ~549–620 | Upload handler → `_run_align_pipeline` |
| `_run_align_pipeline` | ~1278+ | Three-tier alignment pipeline (wav2vec2 → MMS → uniform) |
| `_run_pipeline` | ~1700+ | Transcription pipeline |

---

## 16. Docker

Build and run with GPU support:

```bash
docker build -t whisperx-api .
docker run --gpus all -p 8001:8000 \
  -e HF_TOKEN=hf_xxx \
  -v $(pwd)/models:/app/models \
  whisperx-api
```

Without GPU (CPU only, slow):
```bash
docker run -p 8001:8000 \
  -e COMPUTE_TYPE=int8 \
  whisperx-api
```

The Dockerfile uses CUDA 12.1 + Python 3.10. Models are downloaded at first request and cached in `/app/models` (mount a volume to persist between container restarts).

## 17. MLX Backend (Apple Silicon)

On Apple Silicon Macs (M1/M2/M3) the server can use Apple's **MLX** framework
via [`lightning-whisper-mlx`](https://github.com/mustafaaljadery/lightning-whisper-mlx)
for **4-10× faster** local transcription than CPU faster-whisper.

### Selection
The backend is chosen by the `ASR_BACKEND` env var:

| Value | Behaviour |
| --- | --- |
| `auto` (default) | MLX on Apple Silicon (no CUDA), else WhisperX |
| `mlx` | Force MLX (errors if not installed) |
| `whisperx` | Force the original faster-whisper backend |

### Config

| Var | Default | Notes |
| --- | --- | --- |
| `MLX_MODEL` | `distil-large-v3` | Any model from `lightning-whisper-mlx`: `tiny`, `small`, `large-v3`, `distil-large-v3`, … |
| `MLX_QUANT` | unset | `"4bit"` or `"8bit"` for quantized inference |

MLX models are downloaded to `~/.cache/lightning_whisper_mlx/mlx_models/`
(separate cache from the WhisperX `MODEL_DIR`).

### Pipeline impact
- MLX returns **segment-level** timestamps only. Word-level timestamps still
  come from the existing wav2vec2 alignment pass — unchanged.
- Diarization (pyannote) is unchanged.
- `/align` uses a tiny model just for chunk discovery; if MLX is the active
  backend, a tiny MLX model is loaded for that pass too.

### Verifying

```bash
curl -s http://localhost:8001/health | grep -o '"asr_backend":"[a-z]*"'
# → "asr_backend":"mlx"
```

### Memory note
Holding both an MLX Whisper model **and** the wav2vec2 alignment model in
RAM simultaneously can exceed 16 GB on a base M1. If the worker is killed
during alignment, set `ASR_BACKEND=whisperx` or use a smaller `MLX_MODEL`
(e.g. `small` or `distil-large-v3` with `MLX_QUANT=4bit`).

---

## 18. HuggingFace Spaces Deployment (ZeroGPU)

Three files in this repo enable HF Spaces deployment:

| File | Purpose |
| --- | --- |
| `app.py` | Spaces entrypoint — re-exports `server:app`, applies `@spaces.GPU` decorators |
| `requirements-hf.txt` | Linux/CUDA deps (no MLX) + the `spaces` SDK |
| `Dockerfile.hf` | CUDA 12.1 + ffmpeg base image, exposes port 7860 |
| `Spaces-README.md` | Has the YAML frontmatter the Space needs |

### Deploy steps (HF Pro account)

1. Create a new Space on https://huggingface.co/new-space
   - SDK: **Docker**
   - Hardware: **ZeroGPU**
2. Clone it locally: `git clone https://huggingface.co/spaces/<user>/<name>`
3. Copy these files into the cloned dir:
   ```bash
   cp server.py app.py static/ requirements-hf.txt <space>/
   cp Dockerfile.hf <space>/Dockerfile
   cp Spaces-README.md <space>/README.md
   ```
4. In the Space's **Settings → Variables and secrets**, set:
   - `HF_TOKEN` = your token (with access to `pyannote/speaker-diarization-3.1`)
   - `ASR_BACKEND` = `whisperx`
5. Push:
   ```bash
   cd <space> && git add . && git commit -m "Initial deploy" && git push
   ```

ZeroGPU borrows a GPU only during requests — `@spaces.GPU(duration=N)` in
`app.py` controls the max seconds reserved per call.

### Free community ZeroGPU
If you are NOT on HF Pro, request community access via the *"Request a GPU"*
button on your Space page; approval is usually within a day.

---

## 19. Always-on Monitoring

The `/health` endpoint is cheap (no model inference) and ideal for cron pings.

### Free monitors

| Service | Free tier | Notes |
| --- | --- | --- |
| [cron-job.org](https://cron-job.org) | unlimited | 1-min intervals, simple |
| [UptimeRobot](https://uptimerobot.com) | 50 monitors | 5-min intervals |
| [Better Stack](https://betteruptime.com) | 10 monitors | Nicer dashboard, status page |

Point any of them at `https://<your-space>.hf.space/health` (or your
self-hosted URL). HF Spaces sleep after inactivity; periodic pings keep
them warm.

### Self-hosted crontab

```cron
*/5 * * * * curl -fsS http://localhost:8001/health > /dev/null \
            || echo "WhisperX down" | mail -s alert you@example.com
```

The new **"Production Server"** tab in the web UI polls `/health` every
10 s and shows backend / device / uptime live.

---

## 20. Session Diarization — Deep Reference

This section is a complete technical reference for the `POST /session/diarize` pipeline.
For the API quick-reference see [§5.3](#53-post-sessiondiarize).

---

### 20.1 What is Session Diarization?

**Diarization** answers the question *"who spoke when?"* by segmenting audio into
speaker-homogeneous time intervals (turns). Combined with WhisperX transcription and forced
alignment it produces a time-coded transcript where each segment is labelled `SPEAKER_00`,
`SPEAKER_01`, etc.

#### Why a separate endpoint?

| Feature | `/transcribe` | `/session/diarize` |
|---|---|---|
| Transcription | ✅ | ✅ |
| Forced alignment (word timestamps) | ✅ | ✅ |
| Speaker labels | ❌ | ✅ |
| HuggingFace token required | ❌ | ✅ |
| Returns per-speaker colour in UI | ❌ | ✅ |
| Dominance re-split heuristic | ❌ | ✅ |
| Nearest-by-time fallback | ❌ | ✅ |

---

### 20.2 Full Pipeline Diagram

```
                         POST /session/diarize
                                  │
              ┌───────────────────┼───────────────────┐
              │ Form fields       │                   │
              │  audio_file ──────┤                   │
              │  language         │                   │
              │  min_speakers     │                   │
              │  max_speakers     │                   │
              │  initial_prompt   │                   │
              └───────────────────┘                   │
                                  │
                          ┌───────▼────────┐
                          │   ffmpeg       │
                          │ decode → WAV   │
                          │  16 kHz mono   │
                          └───────┬────────┘
                                  │
               ┌──────────────────▼──────────────────┐
               │          ASR pass                   │
               │  WhisperX (faster-whisper / MLX)    │
               │  → List[Segment] with rough times   │
               └──────────────────┬──────────────────┘
                                  │
               ┌──────────────────▼──────────────────┐
               │        Forced Alignment              │
               │  wav2vec2 model per language         │
               │  → List[Segment] with word-level     │
               │    start/end timestamps              │
               └──────────────────┬──────────────────┘
                                  │
               ┌──────────────────▼──────────────────┐
               │  pyannote Speaker Diarization        │
               │  (3.2 with fallback to 3.1)          │
               │  → List[{start, end, speaker}]       │
               │        (raw diar records)            │
               └──────────────────┬──────────────────┘
                                  │
               ┌──────────────────▼──────────────────┐
               │  Dominance Check                    │
               │  Is one speaker > 80% of audio?     │
               └──────┬───────────────────┬──────────┘
                 YES  │                   │  NO
                      ▼                   │
         ┌────────────────────┐           │
         │  Re-split pass     │           │
         │  min_speakers=2    │           │
         │  cluster thresh=0.5│           │
         │  → new diar records│           │
         │  (only if ≥2 spkrs │           │
         │   found)           │           │
         └────────┬───────────┘           │
                  └───────────────────────┘
                                  │
                  ┌───────────────▼────────────────┐
                  │  Speaker-to-Segment Assignment  │
                  │  For each ASR segment:           │
                  │   1. Find best overlapping diar  │
                  │      record (nearest-by-time)   │
                  │   2. Label segment with speaker  │
                  │   3. Apply same label to every   │
                  │      word inside the segment     │
                  └───────────────┬────────────────┘
                                  │
                  ┌───────────────▼────────────────┐
                  │  Confidence Score Computation   │
                  │  formula: max(0, 1 - max(0,     │
                  │    dom_ratio - 0.5) * 2)        │
                  └───────────────┬────────────────┘
                                  │
                  ┌───────────────▼────────────────┐
                  │  Response assembly              │
                  │  SRT + TXT written to disk      │
                  │  JSON returned to client        │
                  └────────────────────────────────┘
```

---

### 20.3 Model Versions and EULA

The server tries models in priority order:

| Priority | Model | Notes |
|---|---|---|
| 1 | `pyannote/speaker-diarization-3.2` | Best accuracy. Requires HF EULA acceptance at https://hf.co/pyannote/speaker-diarization-3.2 |
| 2 | `pyannote/speaker-diarization-3.1` | Stable fallback. Most users already have EULA accepted. |

**How to enable 3.2:**
1. Visit https://hf.co/pyannote/speaker-diarization-3.2 and click *Accept*
2. Restart the server — it will load 3.2 automatically
3. The response field `diarization_model` confirms which version was loaded

The env var `DIARIZATION_MODEL` overrides the primary model slug.
The env var `DIARIZATION_MODEL_FALLBACK` overrides the fallback slug.

---

### 20.4 Dominant-Speaker Re-Split Heuristic

#### Problem
pyannote's clustering sometimes collapses two speakers into one cluster when speaker turns are
short or their voice characteristics are similar. This produces output where one speaker has
>80% of the audio labelled as theirs.

#### Algorithm

```
dom_ratio = seconds_of_top_speaker / total_diarized_seconds

if dom_ratio >= DIARIZATION_DOMINANCE_THRESHOLD   # default 0.80
   AND max_speakers != 1
   AND min_speakers < 2:

    save current clustering.threshold
    set clustering.threshold = 0.5      # force splits at smaller gap
    re-run diarization with min_speakers=2

    if result has >= 2 distinct speakers:
        adopt new diar records
        set diarization_resplit = True
    else:
        revert to original diar records

    restore clustering.threshold
```

#### Example

| Scenario | dom_ratio | Re-split triggered? | Result |
|---|---|---|---|
| 2 guests, roughly equal talk | 0.52 | No | Use original |
| Host-heavy podcast (80/20) | 0.81 | Yes | Re-split forced |
| Monologue / lecture | 1.0 (user set max=1) | No (max=1 guard) | Use original |
| Interview where host talks 78% | 0.78 | No (below 0.80) | Use original |

**Tip:** If you know the audio has exactly 2 speakers, set `min_speakers=2` — this bypasses
the re-split guard and forces pyannote to always find 2 clusters from the start.

---

### 20.5 Nearest-by-Time Speaker Assignment

#### Problem
Forced alignment and diarization are independent models — they may not agree on exact
segment boundaries. A naive "inherit last seen speaker" strategy causes long wrong-speaker
runs when the diarizer disagrees with the ASR slicer.

#### Algorithm

For each ASR segment `[s_start, s_end]`, the server iterates all diarization records and
computes a **distance score** to each record `[d_start, d_end]`:

```python
overlap = max(0, min(s_end, d_end) - max(s_start, d_start))

if overlap > 0:
    distance = 0                          # perfect: record covers segment
else:
    distance = min(
        abs(s_start - d_end),             # segment starts after record ends
        abs(s_end   - d_start)            # segment ends before record starts
    )
```

The record with the **smallest distance** (ties broken by overlap size) wins.
If no diarization records exist at all, the segment is labelled `SPEAKER_00`.

#### Why this matters

Consider a 10-second gap where pyannote found no speech (e.g., music bridge).
The ASR model may still emit a low-confidence segment. Old code would inherit
whatever speaker was active before the gap — potentially wrong for minutes.
The nearest-by-time approach picks the closest speaker boundary (ahead or behind)
which is almost always correct.

---

### 20.6 Confidence Score

The **diarization confidence** is a single float `[0.0, 1.0]` computed from the
dominant-speaker ratio *after* any re-split:

```
confidence = max(0.0,  1.0 - max(0.0, dom_ratio - 0.5) * 2.0)
```

| dom_ratio | confidence | Interpretation |
|---|---|---|
| 0.50 | 1.00 | Perfectly balanced — two equal speakers |
| 0.60 | 0.80 | Slightly uneven, still good |
| 0.70 | 0.60 | One speaker dominating a bit |
| 0.80 | 0.40 | Quite uneven — check labels carefully |
| 0.90 | 0.20 | ⚠️ Low confidence warning shown in UI |
| 1.00 | 0.00 | Single speaker or collapsed diarization |

The UI shows a **⚠️ Low confidence** badge when confidence < 0.50 and diarization
was used (i.e., the response contains speaker labels).

---

### 20.7 Response Fields Reference

Complete field list for the `/session/diarize` JSON response:

| Field | Type | Description |
|---|---|---|
| `segments` | `Array` | One entry per ASR segment (see §20.7.1) |
| `language` | `string` | Detected or requested language code |
| `language_probability` | `float` | Language detection confidence `[0–1]` |
| `language_warning` | `string\|null` | Non-null if audio seems mixed-language |
| `duration` | `float` | Audio duration in seconds |
| `processing_time` | `float` | Wall-clock time for the full pipeline (s) |
| `speakers` | `Array[string]` | Unique speaker labels found, e.g. `["SPEAKER_00","SPEAKER_01"]` |
| `diarization_confidence` | `float` | See §20.6 |
| `diarization_resplit` | `bool` | `true` if dominance re-split was triggered |
| `diarization_model` | `string` | Model slug that was actually loaded, e.g. `"pyannote/speaker-diarization-3.1"` |
| `txt_file` | `string` | Relative download path for `.txt` |
| `srt_file` | `string` | Relative download path for `.srt` |
| `model_used` | `string` | Whisper model slug |
| `device` | `string` | `"cuda"` / `"mps"` / `"cpu"` |

#### 20.7.1 Segment schema

| Field | Type | Description |
|---|---|---|
| `start` | `float` | Segment start time (seconds, from ASR — never modified) |
| `end` | `float` | Segment end time (seconds, from ASR — never modified) |
| `text` | `string` | Transcribed text |
| `speaker` | `string` | `SPEAKER_XX` label assigned by nearest-time algorithm |
| `words` | `Array` | Word-level entries (see below) |

Word entry:

| Field | Type | Description |
|---|---|---|
| `word` | `string` | The word token |
| `start` | `float` | Word start (seconds) |
| `end` | `float` | Word end (seconds) |
| `score` | `float` | Alignment confidence `[0–1]` |
| `speaker` | `string` | Same as parent segment speaker |

---

### 20.8 Web UI — Speaker Display

The **👥 Session Diarization** tab renders results with:

- **Colour-coded turns** — each unique `SPEAKER_XX` gets a distinct CSS colour (up to 8; wraps after that)
- **Speaker badge** — shown at the start of each new speaker turn
- **Confidence line** — `Confidence: 87%` (or `⚠️ Low confidence: 31%`)
- **Re-split notice** — `↻ Dominant-speaker re-split applied` shown when `diarization_resplit=true`
- **Language warning** — shown when `language_warning` is non-null
- **Model badge** — `pyannote 3.1` or `pyannote 3.2`
- **Download buttons** — `.txt` (speaker-labelled) and `.srt` (timed with speaker labels)

Speaker colours are assigned in order of first appearance:

```
#4A90D9  #E67E22  #27AE60  #9B59B6
#E74C3C  #1ABC9C  #F39C12  #2980B9
```

---

### 20.9 Output Files

#### TXT format (speaker-labelled)

```
[SPEAKER_00]
When I look at some of the quotes from the CVs...

[SPEAKER_01]
That's a really interesting point. Let me add...

[SPEAKER_00]
Exactly. And what we found was...
```

Consecutive segments from the same speaker are merged into one paragraph.

#### SRT format (timed with speaker tags)

```
1
00:00:01,240 --> 00:00:07,830
[SPEAKER_00] When I look at some of the quotes from the CVs...

2
00:00:08,150 --> 00:00:15,460
[SPEAKER_01] That's a really interesting point.
```

---

### 20.10 Best Practices

#### Hints that improve accuracy

| Situation | Recommended settings |
|---|---|
| You know the speaker count | Set both `min_speakers` and `max_speakers` to that exact number |
| Podcast with 2 hosts | `min_speakers=2`, `max_speakers=2` |
| Panel discussion (3–5 people) | `min_speakers=3`, `max_speakers=5` |
| Long monologue | `max_speakers=1` (disables re-split, faster) |
| Technical domain (medical, legal) | Set `initial_prompt` with vocabulary |
| Mixed-language audio | Leave `language` blank for auto-detection; check `language_warning` |

#### When confidence is low

1. Check `diarization_resplit` — if `false` and confidence is low, try setting `min_speakers` explicitly
2. Accept the pyannote 3.2 EULA (better accuracy than 3.1 on 2-speaker audio)
3. Ensure audio quality — background music, heavy reverb, and overlapping speech all hurt diarization
4. For interviews: pre-process audio to separate channels if possible (one mic per speaker)

#### GPU memory

Diarization loads a separate neural model (pyannote) alongside Whisper. On a 24 GB GPU
this is fine. On smaller cards (e.g., RTX 3060 12 GB), run `/session/diarize` and `/transcribe`
sequentially, not concurrently.

---

### 20.11 Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `503 Diarization service unavailable` | `HF_TOKEN` not set | Add `HF_TOKEN` to `.env` and restart |
| All segments labelled `SPEAKER_00` | pyannote collapsed to 1 cluster | Set `min_speakers=2`; accept pyannote 3.2 EULA |
| End timestamp equals audio duration | Old server.py bug (fixed in v2.2.0) | Pull latest code |
| Confidence 0% / re-split always triggers | Very uneven audio (e.g., interviewer barely speaks) | Normal; set `min_speakers=1` if truly single speaker |
| Wrong speaker on long silence | VAD and diarizer boundary mismatch | Expected; nearest-by-time should handle it |
| pyannote 3.2 not loading | EULA not accepted on HF account | Visit https://hf.co/pyannote/speaker-diarization-3.2 |
| `RuntimeError: CUDA out of memory` during diarize | GPU VRAM exhausted | Use a shorter clip or reduce `WHISPER_BATCH_SIZE` |
| `language_warning` non-null | Audio contains multiple languages | Leave `language` blank; accuracy may be lower |
