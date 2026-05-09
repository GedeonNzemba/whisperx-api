"""Smoke test for the two-tier diarization stack.

Verifies that the FoxNoseTech `diarize` library produces ≥ 2 distinct
speakers on a stereo / multi-speaker audio file, and that the host's
returning speech is not collapsed into a single SPEAKER_00 cluster.

Usage:
    python test_diarization.py path/to/audio.wav
    python test_diarization.py path/to/audio.wav --min 2 --max 2

Falls back to ``audio.m4a`` in the project root if no path is given.
Exit code 0 = pass, 1 = fail.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "audio",
        nargs="?",
        default=str(Path(__file__).parent / "audio.m4a"),
        help="Path to audio file (any format soundfile/ffmpeg can read).",
    )
    parser.add_argument("--min", type=int, default=None, dest="min_speakers")
    parser.add_argument("--max", type=int, default=None, dest="max_speakers")
    parser.add_argument(
        "--num", type=int, default=None, dest="num_speakers",
        help="Exact number of speakers (overrides min/max)."
    )
    args = parser.parse_args()

    audio_path = Path(args.audio).expanduser().resolve()
    if not audio_path.is_file():
        print(f"ERROR: audio file not found: {audio_path}", file=sys.stderr)
        return 1

    try:
        from diarize import diarize as _diarize
    except ImportError as exc:
        print(f"ERROR: `diarize` not installed: {exc}", file=sys.stderr)
        print("Run: pip install diarize", file=sys.stderr)
        return 1

    kwargs = {}
    if args.num_speakers:
        kwargs["num_speakers"] = args.num_speakers
    else:
        if args.min_speakers:
            kwargs["min_speakers"] = args.min_speakers
        if args.max_speakers:
            kwargs["max_speakers"] = args.max_speakers

    print(f"→ Diarizing {audio_path.name} (kwargs={kwargs or 'auto'}) ...")
    result = _diarize(str(audio_path), **kwargs)

    seg_count = len(result.segments)
    speakers = sorted({s.speaker for s in result.segments})
    counts = Counter(s.speaker for s in result.segments)
    durations = {
        spk: round(sum(s.end - s.start for s in result.segments if s.speaker == spk), 2)
        for spk in speakers
    }
    total_speech = sum(durations.values())

    print()
    print(f"Audio duration:    {result.audio_duration:.1f} s")
    print(f"Total speech:      {total_speech:.1f} s")
    print(f"Number of speakers: {result.num_speakers}")
    print(f"Speakers found:    {speakers}")
    print(f"Segments per spk:  {dict(counts)}")
    print(f"Seconds per spk:   {durations}")
    print()

    # First 10 turns for visual sanity check
    print("First 10 turns:")
    for s in result.segments[:10]:
        print(f"  [{s.start:6.2f} – {s.end:6.2f}] {s.speaker}")
    if seg_count > 10:
        print(f"  ... ({seg_count - 10} more)")
    print()

    # Pass/fail criteria
    failures = []
    if args.num_speakers and result.num_speakers != args.num_speakers:
        failures.append(
            f"expected exactly {args.num_speakers} speakers, got {result.num_speakers}"
        )
    elif args.min_speakers and result.num_speakers < args.min_speakers:
        failures.append(
            f"expected at least {args.min_speakers} speakers, got {result.num_speakers}"
        )

    # Generic dominance check — flag if one speaker covers > 90% of speech
    # and we expected more than one.
    if total_speech > 0 and len(speakers) > 1:
        top_spk, top_dur = max(durations.items(), key=lambda kv: kv[1])
        dom = top_dur / total_speech
        print(f"Dominance ratio:   {dom:.2f}  (top: {top_spk})")
        if dom > 0.90:
            failures.append(
                f"speaker {top_spk} dominates {dom:.0%} of speech — "
                "likely collapsed clustering"
            )

    if failures:
        print("FAIL:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("PASS: diarization output looks plausible.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
