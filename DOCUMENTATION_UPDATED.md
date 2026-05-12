# WhisperX API — RunPod Deployment & Debugging Log

Version: 2.2.0 (working build)

This document captures the full operational context, recent debugging timeline, fixes applied, and recommended permanent changes for the WhisperX FastAPI server running on RunPod (RTX 3090). It is intended as both user-facing documentation and an operational runbook for future deployments.

---

## 1. Purpose & Overview

Goal: deliver a real-time speech-to-speech (S2S) pipeline:
- Browser mic → WhisperX (STT) → NLLB-200 (MT) → Chatterbox-Turbo (TTS) → Audio back in browser
- UX: /ws/s2s WebSocket for streaming and low-latency round trip


## 2. High-level architecture

- FastAPI server (server.py) running under uvicorn
- GPU runtime: nvidia CUDA 12.4 base image (RunPod)
- Models cached on Network Volume mounted at /models
- Client: static/index.html (Web UI) opens WebSocket to /ws/s2s and streams Float32 PCM


## 3. Recent failure mode (symptoms)

- WebSocket closes unexpectedly (code 1006) or sometimes 1001 when speech triggers actual inference.
- Logs show "Could not load library libcudnn_ops_infer.so.8" immediately when first non-silent chunks arrive — process hard-crashes (native abort), container restarts, WebSocket closes.
- After restart, ephemeral fixes (pip install, LD_LIBRARY_PATH) were lost → problem reappears.


## 4. Root causes found

1. cuDNN missing from container runtime:
   - WhisperX / CTranslate2 require cuDNN 8 (libcudnn_ops_infer.so.8) for GPU kernels.
   - PyTorch 2.6 (used by some components) wants cuDNN 9 (libcudnn.so.9).
   - Installing one version via pip can overwrite the other and cause library mismatch.

2. Synchronous downloads at runtime block the asyncio event loop:
   - torch.hub.load() for Silero VAD and huggingface token downloads used in warmup blocked the main thread and caused WS timeouts (code 1006).

3. Pod DNS (Docker internal resolver 127.0.0.11) was unreliable on the RunPod node:
   - Prevented pip/huggingface downloads at container start and during warmup.

4. Browser audio capture defaults (AEC/NS/AGC) removed playback when testing with speaker playback (it returned digital silence), causing confusion that server was "not receiving audio".


## 5. Fixes applied (ephemeral, then baked where needed)

A. Immediate pod fixes (applied to running container to recover quickly):
- Overwrote /etc/resolv.conf with public resolvers (8.8.8.8, 1.1.1.1) so pip/hf downloads succeed.
- pip install nvidia-cudnn-cu12==8.9.7.29 (cuDNN 8) — allowed WhisperX to run without crashing.
- Exported LD_LIBRARY_PATH in /app/start.sh so uvicorn’s subprocesses saw cuDNN libs.
- Pulled updated static/index.html (client) that disables echoCancellation/noiseSuppression/autoGainControl and logs client RMS; this fixed the “digital silence” issue when playing audio through speakers.

B. Code fixes (committed to repo):
- server.py: wrap _chunk_has_speech call in asyncio.to_thread(...) to avoid blocking event loop when Silero VAD / torch.hub.load() downloads happen.
- static/index.html: disable AEC/NS/AGC constraints, add client-side RMS logging and zero-gain destination to ensure AudioWorklet fires across browsers.
- start.sh: write resolv.conf on start, pre-cache NLLB tokenizer (if network available), use curl fallback to fetch certain repo files (git might not be present on runtime image), and include safer LD_LIBRARY_PATH fallbacks.

C. Dockerfile (committed) — permanent fixes to bake into image:
- Install both cuDNN 8 and cuDNN 9 in builder stage:
  * Install cuDNN 8, copy its .so files to /usr/local/lib/cudnn8/.
  * Install cuDNN 9 (matching torch) as required.
  * Create /etc/ld.so.conf.d/zzz-nvidia-cudnn.conf listing both directories and run ldconfig so both cuDNN versions are available at runtime.
- Copy cudnn8 libs and linker conf into runtime stage and run ldconfig.


## 6. What to run on a live pod (commands)

If you have a single terminal on the pod and cuDNN 8 already downloaded, run this one-liner to make both cuDNN versions available and refresh the linker cache:

mkdir -p /usr/local/lib/cudnn8 && \
cp /usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib/libcudnn*.so* /usr/local/lib/cudnn8/ && \
pip install nvidia-cudnn-cu12==9.1.0.70 && \
printf '/usr/local/lib/cudnn8\n/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib\n' > /etc/ld.so.conf.d/zzz-nvidia-cudnn.conf && \
ldconfig && \
ldconfig -p | grep -E "libcudnn\.so\.9|libcudnn_ops_infer\.so\.8"

Expected output: both libraries should be listed and point to their respective files.

After that, update start.sh from GitHub and restart uvicorn:

curl -fsSL "https://raw.githubusercontent.com/GedeonNzemba/whisperx-api/main/start.sh" -o /app/start.sh && chmod +x /app/start.sh && kill $(ps -ef | grep uvicorn | grep -v grep | awk '{print $2}')

Then hard reload the browser and retry S2S.


## 7. Why WebSocket closed (1006) and container restarted

- The server was hard-aborting at native layer when libcudnn_ops_infer.so.8 was missing. This kills the Python process (not an exception), so uvicorn / PID 20 dies and the container runtime restarts the process. The client sees an abnormal close (1006). By installing and registering cuDNN 8 and 9 reliably, the native abort no longer occurs.


## 8. Frontend checklist for reliable S2S testing

- Hard reload the page (Cmd+Shift+R)
- Open DevTools → Console
- Confirm these lines appear after Start:
  - mic acquired (raw: AEC/NS/AGC disabled)
  - mic track settings: {"echoCancellation":false,...}
  - mic stats: rms=... (values > 1e-3 for detectable speech)
- If RMS remains 0: check OS microphone device selection or browser privacy permissions.


## 9. Permanent work recommended

1. Rebuild and push a new Docker image from the updated Dockerfile so the RunPod image includes both cuDNN versions — avoids downloading at runtime.
2. Pre-cache heavy models (NLLB, Silero, pyannote) into the image or into the mounted /models network volume during CI so first-run downloads don't block live traffic.
3. Add a small startup script that sets resolv.conf if the internal resolver fails (already done in start.sh); consider federated / cloud DNS alternatives if RunPod's resolver remains flaky.
4. Move any heavy hub downloads to background threads or to build-time to avoid blocking the event loop. (server.py already wraps VAD call in to_thread.)
5. Add healthchecks and more robust retry/backoff with timeouts around HF downloads. Consider a preflight bootstrap step for images with limited network.
6. Add CI step building the Docker image and uploading to a registry; deploy that image to RunPod so debugging downloads are not needed on first run.


## 10. Current status (live)

- Server code updated and pushed to GitHub (commits: static/index.html, server.py, start.sh, Dockerfile fixes).
- Runtime pod: cuDNN 8/9 install may be in-flight depending on timing; if both libs are registered, S2S works end-to-end.
- Client-side audio capture fixed (disable browser AEC/NS/AGC).
- VAD model (Silero) will be cached on first run; server now loads it in a thread to keep WS alive.


## 11. Troubleshooting checklist (if things break again)

1. Check container logs: `docker logs <id>` or RunPod console; look for `Could not load library libcudnn_ops_infer.so.8` or `libcudnn.so.9` errors.
2. Verify ldconfig: `ldconfig -p | grep cudnn`
3. Verify /etc/resolv.conf has public nameservers if downloads are failing.
4. Check /models/torch/hub/ for Silero zip or extracted folder.
5. Confirm client console shows mic RMS > 0.
6. If a crash occurs while transcribing, check if it’s a native abort (likely cuDNN mismatch) vs. an exception (traceback in logs).


## 12. Next actions (short-term)

- If you want me to, rebuild the Docker image and push it to a container registry (requires CI/access) — I can provide a one-line build and push script.
- If you're still seeing failures now, paste the most recent `all logs.txt` tail (last 200 lines) and I’ll diagnose the current live state.

---

Last updated: 2026-05-12 21:52 UTC

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>
