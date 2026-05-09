#!/usr/bin/env python3
"""Minimal client for the WhisperX API server.

Usage:
    python client_test.py path/to/audio.mp3
    python client_test.py path/to/audio.mp3 --no-diarize --formats json,srt --language en
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--language", default=None, help="ISO 639-1 code, e.g. 'en', 'fr'.")
    parser.add_argument("--diarize", dest="diarize", action="store_true", default=None)
    parser.add_argument("--no-diarize", dest="diarize", action="store_false")
    parser.add_argument("--formats", default="json,srt,vtt,txt")
    parser.add_argument("--min-speakers", type=int, default=None)
    parser.add_argument("--max-speakers", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path("./output"))
    args = parser.parse_args()

    if not args.audio.is_file():
        print(f"File not found: {args.audio}", file=sys.stderr)
        return 1

    print(f"-> Health: {args.url}/health")
    h = requests.get(f"{args.url}/health", timeout=10)
    print(json.dumps(h.json(), indent=2))

    data = {"output_format": args.formats}
    if args.language:
        data["language"] = args.language
    if args.diarize is not None:
        data["diarize"] = str(args.diarize).lower()
    if args.min_speakers is not None:
        data["min_speakers"] = args.min_speakers
    if args.max_speakers is not None:
        data["max_speakers"] = args.max_speakers

    print(f"\n-> Transcribing {args.audio} ...")
    with open(args.audio, "rb") as f:
        r = requests.post(
            f"{args.url}/transcribe",
            files={"file": (args.audio.name, f, "application/octet-stream")},
            data=data,
            timeout=60 * 60,
        )
    if not r.ok:
        print(f"HTTP {r.status_code}: {r.text}", file=sys.stderr)
        return 2

    payload = r.json()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "result.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    for ext in ("txt", "srt", "vtt"):
        if ext in payload.get("files", {}):
            (args.out / f"transcript.{ext}").write_text(payload["files"][ext])

    print(f"\nLanguage: {payload.get('language')}")
    print(f"Duration: {payload.get('duration_seconds')}s  "
          f"Processing: {payload.get('processing_time_seconds')}s  "
          f"RTF: {payload.get('realtime_factor')}x")
    print(f"Diarized: {payload.get('diarized')}")
    print(f"Segments: {len(payload.get('segments', []))}")
    print(f"Outputs written to: {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
