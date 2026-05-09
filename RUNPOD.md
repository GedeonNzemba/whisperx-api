# RunPod Deployment Guide — WhisperX Large V3 API

Deploy the WhisperX API as an always-on HTTP REST service on a RunPod GPU pod.
No real-time/streaming features are included in this deployment — only the standard
`/transcribe`, `/align`, and `/health` endpoints.

---

## Cost Summary

| GPU             | Tier            | Per Hour | ~8 hr/day | 24/7      |
|-----------------|-----------------|----------|-----------|-----------|
| RTX 4090 (24 GB)| Secure Cloud    | $0.59    | ~$144/mo  | ~$430/mo  |
| RTX 4090 (24 GB)| Community Cloud | $0.34    | ~$83/mo   | ~$248/mo  |
| Network Volume (50 GB) | —        | —        | —         | $3.50/mo  |

> **Tip:** Stop the pod when not in use — you only pay for the Network Volume
> ($3.50/mo) while the pod is stopped.

---

## Step 1 — Set Up Secrets

RunPod Secrets are encrypted environment variables injected into your pod at
runtime. Reference them in your pod template with the syntax
`{{RUNPOD_SECRET_<NAME>}}`.

1. Go to **RunPod Console → Settings → Secrets**.
2. Create the following secrets:

| Secret Name      | Value                                  |
|------------------|----------------------------------------|
| `hf_token`       | Your HuggingFace token (for pyannote)  |
| `gemini_api_key` | Your Google Gemini API key (optional)  |

3. In your pod template's environment variables, reference them as:
   ```
   HF_TOKEN = {{RUNPOD_SECRET_hf_token}}
   GEMINI_API_KEY = {{RUNPOD_SECRET_gemini_api_key}}
   ```

---

## Step 2 — Create a Network Volume

A Network Volume persists your downloaded models across pod restarts, saving
~10–15 minutes of re-download time each start.

1. Go to **RunPod Console → Storage → Network Volumes → + New Volume**.
2. Configure:
   - **Name:** `whisperx-models`
   - **Size:** 50 GB
   - **Datacenter:** Choose the **same datacenter** as where you will deploy your pod
     (e.g. `US-TX-3`). Cross-datacenter volumes cannot be attached.
3. Cost: **$3.50/month** regardless of whether the pod is running.

> The server will download models to `/models` on first request and cache them there
> for all subsequent requests.

---

## Step 3 — Build and Push Docker Image

```bash
# Clone / enter your project directory
cd "speech to text"

# Build the image (takes ~10–15 min on first build)
docker build -f Dockerfile -t whisperx-api:latest .

# Tag for Docker Hub
docker tag whisperx-api:latest YOUR_DOCKERHUB_USERNAME/whisperx-api:latest
docker push YOUR_DOCKERHUB_USERNAME/whisperx-api:latest

# --- OR tag for GitHub Container Registry (GHCR) ---
docker tag whisperx-api:latest ghcr.io/YOUR_GITHUB_USERNAME/whisperx-api:latest
docker push ghcr.io/YOUR_GITHUB_USERNAME/whisperx-api:latest
```

> Make sure you are logged in: `docker login` (Docker Hub) or
> `echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin` (GHCR).

---

## Step 4 — Create a Pod Template

1. Go to **RunPod Console → Templates → + New Template**.
2. Fill in:

| Field                  | Value                                             |
|------------------------|---------------------------------------------------|
| Template Name          | `whisperx-api`                                    |
| Container Image        | `YOUR_DOCKERHUB_USERNAME/whisperx-api:latest`     |
| Container Disk         | **20 GB**                                         |
| Volume Disk            | **50 GB** (from the Network Volume you created)   |
| Volume Mount Path      | `/models`                                         |
| Expose HTTP Ports      | `8000`                                            |

3. Environment Variables:

| Variable       | Value                            |
|----------------|----------------------------------|
| `MODEL_DIR`    | `/models`                        |
| `PORT`         | `8000`                           |
| `HF_TOKEN`     | `{{RUNPOD_SECRET_hf_token}}`     |
| `COMPUTE_TYPE` | `float16`                        |
| `WHISPER_MODEL`| `large-v3`                       |
| `BEAM_SIZE`    | `5`                              |

---

## Step 5 — Deploy the Pod

1. Go to **RunPod Console → Pods → + Deploy**.
2. Select:
   - **Cloud:** Secure Cloud (for reliability) or Community Cloud (cheaper).
   - **GPU:** RTX 4090 (24 GB VRAM) — recommended for full pipeline (Whisper + diarization).
   - **Template:** `whisperx-api` (the one you just created).
3. Click **Deploy**.
4. The pod will take ~2–3 minutes to start. Wait for status **Running**.
5. Click the pod → **Connect → HTTP Service [8000]** — this gives you your public URL,
   e.g. `https://XXXXXXXX-8000.proxy.runpod.net`.

> **Estimated cost:** ~$0.59/hr (Secure Cloud) or ~$0.34/hr (Community Cloud).

---

## Step 6 — Verify the Deployment

Replace `<POD_URL>` with your pod's public URL.

```bash
# 1. Health check — should return HTTP 200 with "whisper_loaded": true
curl https://<POD_URL>/health | python3 -m json.tool

# 2. Transcribe a test audio file
curl -X POST https://<POD_URL>/transcribe \
  -F "audio=@/path/to/test.wav" \
  -F "language=en" \
  | python3 -m json.tool

# 3. Forced alignment
curl -X POST https://<POD_URL>/align \
  -F "audio=@/path/to/test.wav" \
  -F "transcript=Hello world this is a test" \
  -F "language=en" \
  | python3 -m json.tool
```

Expected `/health` response:
```json
{
  "status": "ok",
  "whisper_loaded": true,
  "diarization_loaded": true,
  "device": "cuda",
  "compute_type": "float16",
  "model": "large-v3"
}
```

Or run the automated validation script:
```bash
bash test_deployment.sh https://<POD_URL>
```

---

## Step 7 — Keep Costs Low

### Stop the pod when not in use
- Go to **RunPod Console → Pods** → click **Stop** on your pod.
- Stopped pods pay **only for storage** (~$3.50/mo for the 50 GB volume).
- Restart from the same console when you need it — models are already cached.

### Use Community Cloud for testing
- Community Cloud GPUs cost ~$0.34/hr vs $0.59/hr for Secure Cloud.
- Slightly higher chance of interruption; fine for development.

### Auto-stop cron script

Save as `auto_stop_pod.sh` and run it on your laptop or a cheap cron server:

```bash
#!/usr/bin/env bash
# Auto-stop the RunPod pod after IDLE_MINUTES of no requests.
# Requires: RUNPOD_API_KEY and POD_ID environment variables.
# Usage: RUNPOD_API_KEY=xxx POD_ID=yyy IDLE_MINUTES=30 bash auto_stop_pod.sh

set -euo pipefail

API_KEY="${RUNPOD_API_KEY:?Set RUNPOD_API_KEY}"
POD_ID="${POD_ID:?Set POD_ID}"
IDLE_MINUTES="${IDLE_MINUTES:-30}"
POD_URL="${POD_URL:?Set POD_URL (e.g. https://XXXXXX-8000.proxy.runpod.net)}"

LAST_REQUEST_FILE="/tmp/whisperx_last_request"
[ -f "$LAST_REQUEST_FILE" ] || echo "0" > "$LAST_REQUEST_FILE"

# Check if the pod is responsive
if curl -sf --max-time 5 "${POD_URL}/health" > /dev/null 2>&1; then
    echo "$(date +%s)" > "$LAST_REQUEST_FILE"
fi

LAST=$(cat "$LAST_REQUEST_FILE")
NOW=$(date +%s)
ELAPSED=$(( (NOW - LAST) / 60 ))

if [ "$ELAPSED" -ge "$IDLE_MINUTES" ]; then
    echo "Pod idle for ${ELAPSED}m — stopping pod ${POD_ID}..."
    curl -s -X POST "https://api.runpod.io/graphql?api_key=${API_KEY}" \
      -H "Content-Type: application/json" \
      -d "{\"query\": \"mutation { podStop(input: {podId: \\\"${POD_ID}\\\"}) { id desiredStatus } }\"}"
    echo "Pod stop requested."
else
    echo "Pod active (last seen ${ELAPSED}m ago)."
fi
```

Add to crontab to run every 5 minutes:
```
*/5 * * * * POD_URL=https://XXXX-8000.proxy.runpod.net POD_ID=xxx RUNPOD_API_KEY=yyy bash /path/to/auto_stop_pod.sh >> /tmp/auto_stop.log 2>&1
```

---

## Step 8 — Future-Proofing for Real-Time Streaming

When you are ready to add FastRTC real-time STT:

1. The pod already exposes port `8000/tcp` which supports WebSocket upgrades — no
   firewall changes needed.
2. Add `fastrtc` to `requirements.txt` and implement the `/ws/stt` WebSocket endpoint
   in `server.py` (mount a FastRTC `Stream` object on the FastAPI app).
3. No template changes are required — RunPod's HTTP proxy transparently forwards
   WebSocket connections on port 8000.
4. For sub-500ms latency, use the Voxtral Realtime engine (set `RUNPOD_POD_ID` env var
   to trigger Voxtral engine selection in the server).
