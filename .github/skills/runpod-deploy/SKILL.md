---
name: runpod-deploy
description: How to deploy changes to the RunPod pod. Use this whenever the user wants to push code or dependency changes to RunPod. Code-only changes use SSH (seconds). Dependency changes use Docker (30+ min, last resort).
---

# RunPod Deployment

## Rule: Docker is the LAST resort

Rebuilding Docker takes 30+ minutes to build + push. Only do it when `requirements.txt` or `Dockerfile` change.

---

## ✅ Default: SSH + rsync (code-only changes — ~10 seconds)

Use this for any `.py` file change (server.py, s2s/*.py, vbx_diarize.py, etc.).

### Setup (one-time)
1. RunPod pod → **Connect** → **SSH over exposed TCP** → note host + port
2. Add your public key in RunPod account settings (`~/.ssh/id_rsa.pub`)

### Deploy code change
```bash
# 1. Sync changed files to the pod
rsync -avz -e "ssh -p <PORT>" server.py s2s/ vbx_diarize.py vibevoice_client.py vibevoice/ \
  root@<RUNPOD_SSH_HOST>:/app/

# 2. Restart uvicorn on the pod
ssh -p <PORT> root@<RUNPOD_SSH_HOST> \
  "pkill -f uvicorn; cd /app && nohup uvicorn server:app --host 0.0.0.0 --port 8000 > /app/server.log 2>&1 &"

# 3. Verify
curl https://<POD_ID>-8000.proxy.runpod.net/health | python3 -m json.tool
```

---

## 🐳 Docker rebuild (only for requirements.txt / Dockerfile changes)

```bash
docker buildx build --platform linux/amd64 -f Dockerfile -t nzemba48/whisperx-api:latest --push .
```

Then restart the RunPod pod (Update Pod → restart or terminate + redeploy).

### Docker DNS requirements (daemon.json must have these)
```json
{
  "ipv6": false,
  "dns": ["8.8.8.8", "1.1.1.1"]
}
```

---

## Current RunPod setup
- **Image:** `nzemba48/whisperx-api:latest`
- **GPU:** RTX 4090
- **Proxy URL:** `https://kt209obk22m548-8000.proxy.runpod.net`
- **Base image:** `nvidia/cuda:12.4.0-runtime-ubuntu22.04`
- **torch:** `2.6.0+cu124`

### Required env vars on pod
```
S2S_ENABLED=1
TTS_BACKEND=chatterbox-turbo
S2S_TARGET_LANG_DEFAULT=en
HF_TOKEN=<token>
```
