#!/usr/bin/env python3
"""
Test client for the /ws/s2s (speech-to-speech) WebSocket endpoint.

Usage:
    python test_s2s.py --url wss://abc123-8000.proxy.runpod.net \
                       --audio audioShort.m4a \
                       --target fr

The script:
  1. Connects to /ws/s2s
  2. Streams the audio file as float32 PCM at 16 kHz
  3. Prints every JSON message received
  4. Saves synthesised TTS audio to s2s_output.wav
"""
from __future__ import annotations

import argparse
import asyncio
import json
import struct
import subprocess
import sys
import wave
from pathlib import Path


def decode_to_pcm_f32(audio_path: Path, sample_rate: int = 16000) -> bytes:
    """Decode any audio file → raw float32 PCM at sample_rate Hz, mono."""
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(audio_path),
        "-ac", "1",
        "-ar", str(sample_rate),
        "-f", "f32le",
        "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True, check=True)
    return result.stdout


async def run(url: str, audio_path: Path, target_lang: str, source_lang: str,
              chunk_ms: int, out_path: Path) -> None:
    try:
        import websockets
    except ImportError:
        sys.exit("Install websockets:  pip install websockets")

    import ssl as _ssl
    ssl_ctx = _ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = _ssl.CERT_NONE

    sample_rate = 16_000
    chunk_samples = sample_rate * chunk_ms // 1000
    chunk_bytes = chunk_samples * 4  # float32 = 4 bytes

    ws_url = f"{url.rstrip('/')}/ws/s2s"
    print(f"Connecting to {ws_url} ...")

    collected_audio: list[bytes] = []
    tts_sample_rate = 24_000  # default; updated from server 'audio' message

    async with websockets.connect(ws_url, max_size=64 * 1024 * 1024, ssl=ssl_ctx) as ws:
        # 0. Peek: server may immediately send an error and close (e.g. S2S
        #    disabled, TTS unavailable). Wait briefly before sending start.
        try:
            peek = await asyncio.wait_for(ws.recv(), timeout=2.0)
            msg = json.loads(peek)
            if msg.get("type") == "error":
                print(f"\033[91m[error] Server rejected connection: {msg}\033[0m")
                return
            print(f"[pre-start msg] {peek}")
        except asyncio.TimeoutError:
            pass  # no pre-start message — good, proceed normally

        # 1. Send start
        await ws.send(json.dumps({
            "type": "start",
            "language": source_lang,
            "target_language": target_lang,
        }))

        # 2. Decode audio in background, stream PCM
        print(f"Decoding {audio_path} → f32le 16 kHz ...")
        pcm_bytes = decode_to_pcm_f32(audio_path, sample_rate)
        total_samples = len(pcm_bytes) // 4
        duration_s = total_samples / sample_rate
        print(f"Audio duration: {duration_s:.2f}s  "
              f"({len(pcm_bytes)} bytes, {chunk_ms}ms chunks)")

        async def send_audio() -> None:
            offset = 0
            sent_chunks = 0
            while offset < len(pcm_bytes):
                chunk = pcm_bytes[offset: offset + chunk_bytes]
                await ws.send(chunk)
                offset += chunk_bytes
                sent_chunks += 1
                # Real-time pacing: sleep as long as the chunk would play
                await asyncio.sleep(chunk_ms / 1000 * 0.9)
            # Signal end of stream
            await ws.send(json.dumps({"type": "end"}))
            print(f"\n[sender] Sent {sent_chunks} chunks + end signal")

        send_task = asyncio.create_task(send_audio())

        # 3. Receive messages
        async for raw in ws:
            if isinstance(raw, bytes):
                collected_audio.append(raw)
                # Print a progress dot for each audio frame
                n_samples = len(raw) // 4
                dur = n_samples / tts_sample_rate
                print(f"  [audio frame] {len(raw)} bytes ({dur:.2f}s PCM received)")
                continue

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                print(f"  [raw] {raw!r}")
                continue

            mtype = msg.get("type")
            color = {
                "ready":       "\033[92m",   # green
                "final":       "\033[96m",   # cyan
                "translation": "\033[93m",   # yellow
                "audio":       "\033[95m",   # magenta
                "audio_skipped": "\033[91m", # red
                "done":        "\033[92m",   # green
                "error":       "\033[91m",   # red
            }.get(mtype, "")
            reset = "\033[0m" if color else ""
            print(f"{color}[{mtype}]{reset} {json.dumps(msg, ensure_ascii=False)}")

            if mtype == "audio":
                tts_sample_rate = int(msg.get("sample_rate", 24_000))

            if mtype in ("done", "error"):
                break

        await send_task

    # 4. Save collected TTS audio
    if collected_audio:
        combined = b"".join(collected_audio)
        n_samples = len(combined) // 4
        floats = struct.unpack(f"{n_samples}f", combined)
        # Convert float32 → int16 for a standard .wav file
        int16_data = bytes(
            struct.pack("h", max(-32768, min(32767, int(f * 32767))))
            for f in floats
        )
        with wave.open(str(out_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(tts_sample_rate)
            wf.writeframes(int16_data)
        duration = n_samples / tts_sample_rate
        print(f"\n✅ TTS audio saved → {out_path}  ({duration:.2f}s, {tts_sample_rate} Hz)")
    else:
        print("\n⚠️  No TTS audio received (check logs above for audio_skipped / error)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the /ws/s2s endpoint")
    parser.add_argument("--url", required=True,
                        help="RunPod base URL, e.g. https://abc123-8000.proxy.runpod.net")
    parser.add_argument("--audio", type=Path, default=Path("audioShort.m4a"),
                        help="Audio file to stream (any ffmpeg-supported format)")
    parser.add_argument("--target", default="fr",
                        help="Target language ISO 639-1 code (default: fr)")
    parser.add_argument("--source", default="en",
                        help="Source language ISO 639-1 code (default: en)")
    parser.add_argument("--chunk-ms", type=int, default=500,
                        help="PCM chunk size in milliseconds (default: 500)")
    parser.add_argument("--out", type=Path, default=Path("s2s_output.wav"),
                        help="Output WAV file for synthesised audio")
    args = parser.parse_args()

    # Normalise URL: replace http(s) with ws(s)
    url = args.url.replace("https://", "wss://").replace("http://", "ws://")

    if not args.audio.is_file():
        sys.exit(f"Audio file not found: {args.audio}")

    asyncio.run(run(url, args.audio, args.target, args.source, args.chunk_ms, args.out))


if __name__ == "__main__":
    main()
