"""LocalAgreement-2 streaming ASR core (ported from ufal/whisper_streaming, MIT).

Upstream: https://github.com/ufal/whisper_streaming
    Macháček, Dabre, Bojar: "Turning Whisper into Real-Time Transcription
    System", IJCNLP-AACL 2023 demo. MIT license. Core classes
    (HypothesisBuffer, OnlineASRProcessor) adapted here with attribution;
    logic kept faithful to upstream so future fixes can be diffed.

Why: the previous streaming design transcribed fixed 5-second chunks and
stitched the seams with language-specific dedup heuristics, giving 6-8 s
first-word latency and English-only hallucination filters. LocalAgreement-2
instead re-transcribes a growing audio buffer and COMMITS the longest common
prefix of two consecutive passes — a principled, language-agnostic stability
rule. It yields sub-second *partial* hypotheses and ~2-3 s *confirmed* text,
matching the architecture used by production streaming STT systems.

Design notes for this port:
* Works directly on a raw ``faster_whisper.WhisperModel`` (the object that
  whisperx's FasterWhisperPipeline wraps as ``.model``), so the model already
  loaded by the server is reused — no extra VRAM.
* ``word_timestamps=True`` and ``condition_on_previous_text=True`` are
  required for the algorithm (word-level LCP + prompt continuity).
* Buffer trimming uses the "segment" strategy (no sentence-tokenizer
  dependency, which upstream needs only for the optional "sentence" mode).
* Words inside segments with ``no_speech_prob > 0.9`` are dropped — this
  replaces the old English-only hallucination phrase lists.
"""

from __future__ import annotations

import logging
import time
from typing import Any, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("streaming-asr")

SAMPLE_RATE = 16000


class HypothesisBuffer:
    """Tracks consecutive ASR hypotheses; commits their longest common prefix.

    Word tuples are ``(start_s, end_s, text)`` with absolute stream times.
    """

    def __init__(self) -> None:
        self.commited_in_buffer: List[Tuple[float, float, str]] = []
        self.buffer: List[Tuple[float, float, str]] = []   # previous pass
        self.new: List[Tuple[float, float, str]] = []      # current pass
        self.last_commited_time = 0.0
        self.last_commited_word: Optional[str] = None

    def insert(self, new: List[Tuple[float, float, str]], offset: float) -> None:
        """Insert the current pass's words (shifted by ``offset`` seconds).

        Only words that extend past the already-committed time are kept.
        The 1..5-gram boundary check removes words that duplicate the tail of
        the committed text (Whisper often re-emits the last few words when the
        window scrolls)."""
        new = [(a + offset, b + offset, t) for a, b, t in new]
        self.new = [(a, b, t) for a, b, t in new if a > self.last_commited_time - 0.1]

        if len(self.new) >= 1:
            a, b, t = self.new[0]
            if abs(a - self.last_commited_time) < 1.0 and self.commited_in_buffer:
                # Check if the head of `new` repeats the tail of committed
                # (compare 1..5-grams; drop the duplicated words from `new`).
                cn = len(self.commited_in_buffer)
                nn = len(self.new)
                for i in range(1, min(min(cn, nn), 5) + 1):
                    # last i committed words, restored to chronological order
                    c = " ".join(
                        [self.commited_in_buffer[-j][2] for j in range(1, i + 1)][::-1]
                    )
                    # first i words of the new hypothesis
                    tail = " ".join(self.new[j - 1][2] for j in range(1, i + 1))
                    if c == tail:
                        for _ in range(i):
                            self.new.pop(0)
                        break

    def flush(self) -> List[Tuple[float, float, str]]:
        """Commit and return the longest common prefix of the previous and
        current pass (LocalAgreement-2)."""
        commit: List[Tuple[float, float, str]] = []
        while self.new:
            na, nb, nt = self.new[0]
            if not self.buffer:
                break
            if nt.strip().lower() == self.buffer[0][2].strip().lower():
                commit.append((na, nb, nt))
                self.last_commited_word = nt
                self.last_commited_time = nb
                self.buffer.pop(0)
                self.new.pop(0)
            else:
                break
        self.buffer = self.new
        self.new = []
        self.commited_in_buffer.extend(commit)
        return commit

    def pop_commited(self, t: float) -> None:
        """Drop committed words that end before stream-time ``t`` (buffer scroll)."""
        while self.commited_in_buffer and self.commited_in_buffer[0][1] <= t:
            self.commited_in_buffer.pop(0)

    def complete(self) -> List[Tuple[float, float, str]]:
        """The current *unconfirmed* tail (last hypothesis not yet agreed)."""
        return self.buffer


class OnlineASRProcessor:
    """Grow-and-commit streaming wrapper around a faster-whisper model.

    Usage::
        proc = OnlineASRProcessor(fw_model, language="en")
        proc.insert_audio_chunk(pcm_f32)     # any size, 16 kHz mono
        committed, partial = proc.process_iter()
        ...
        committed, partial = proc.finish()
    """

    def __init__(
        self,
        model: Any,                       # faster_whisper.WhisperModel
        language: Optional[str] = None,   # None → auto-detect (updated per pass)
        buffer_trimming_sec: float = 15.0,
        initial_prompt: Optional[str] = None,
        beam_size: int = 5,
    ) -> None:
        self.model = model
        self.language = language
        self.buffer_trimming_sec = buffer_trimming_sec
        self.initial_prompt = initial_prompt
        self.beam_size = beam_size

        self.audio_buffer = np.array([], dtype=np.float32)
        self.buffer_time_offset = 0.0  # stream time of audio_buffer[0]
        self.transcript_buffer = HypothesisBuffer()
        self.commited: List[Tuple[float, float, str]] = []  # full session commits
        self.detected_language: Optional[str] = language  # updated per pass in auto mode

    # -- audio ------------------------------------------------------------

    def insert_audio_chunk(self, audio: np.ndarray) -> None:
        self.audio_buffer = np.append(self.audio_buffer, audio)

    @property
    def buffered_seconds(self) -> float:
        return len(self.audio_buffer) / SAMPLE_RATE

    # -- prompt -----------------------------------------------------------

    def _prompt(self) -> str:
        """Last ~200 chars of committed text that has scrolled OUT of the
        audio buffer — passed as initial_prompt for decoding continuity."""
        k = len(self.commited) - 1
        while k >= 0 and self.commited[k][1] > self.buffer_time_offset:
            k -= 1
        scrolled = self.commited[: k + 1]
        prompt_words: List[str] = []
        length = 0
        for _, _, t in reversed(scrolled):
            length += len(t) + 1
            if length > 200:
                break
            prompt_words.append(t)
        prompt = " ".join(reversed(prompt_words))
        if self.initial_prompt and not prompt:
            return self.initial_prompt
        return prompt

    # -- main iteration ---------------------------------------------------

    def process_iter(self) -> Tuple[List[Tuple[float, float, str]], List[Tuple[float, float, str]]]:
        """Run one ASR pass over the buffer. Returns (newly_committed, partial_tail).

        Each word tuple is (abs_start_s, abs_end_s, text)."""
        if self.buffered_seconds < 0.1:
            return [], self.transcript_buffer.complete()

        prompt = self._prompt()
        t0 = time.time()
        segments, info = self.model.transcribe(
            self.audio_buffer,
            language=self.language,
            initial_prompt=prompt or None,
            beam_size=self.beam_size,
            word_timestamps=True,
            condition_on_previous_text=True,
        )
        segments = list(segments)
        if self.language is None and getattr(info, "language", None):
            # Adopt per-pass detection so mixed-language sessions stay usable;
            # the *latest* pass wins (buffer usually holds one language at a
            # time). Exposed via .detected_language for the S2S source side.
            self.detected_language = info.language
        else:
            self.detected_language = self.language

        # Word extraction: skip probable non-speech segments (replaces the old
        # English-only hallucination phrase lists — language agnostic).
        words: List[Tuple[float, float, str]] = []
        seg_ends: List[float] = []
        for seg in segments:
            seg_ends.append(seg.end)
            if getattr(seg, "no_speech_prob", 0.0) > 0.9:
                continue
            for w in seg.words or []:
                words.append((w.start, w.end, w.word))

        self.transcript_buffer.insert(words, self.buffer_time_offset)
        committed = self.transcript_buffer.flush()
        self.commited.extend(committed)

        logger.debug(
            "process_iter: %.2fs audio, %.2fs infer, +%d committed, %d partial",
            self.buffered_seconds, time.time() - t0, len(committed),
            len(self.transcript_buffer.complete()),
        )

        # Trim the buffer at the last COMPLETED segment boundary once it grows
        # beyond the threshold ("segment" trimming strategy).
        if self.buffered_seconds > self.buffer_trimming_sec and committed:
            self._chunk_completed_segment(seg_ends)

        return committed, self.transcript_buffer.complete()

    def _chunk_completed_segment(self, seg_ends: List[float]) -> None:
        if not self.commited or len(seg_ends) <= 1:
            return
        last_commit_t = self.commited[-1][1]
        # second-to-last segment end, in absolute time
        ends = [e + self.buffer_time_offset for e in seg_ends[:-1]]
        cut = None
        for e in reversed(ends):
            if e <= last_commit_t:
                cut = e
                break
        if cut is None:
            return
        self._chunk_at(cut)

    def _chunk_at(self, t: float) -> None:
        """Scroll the audio buffer so it starts at stream-time ``t``."""
        self.transcript_buffer.pop_commited(t)
        cut_samples = int((t - self.buffer_time_offset) * SAMPLE_RATE)
        if cut_samples <= 0:
            return
        self.audio_buffer = self.audio_buffer[cut_samples:]
        self.buffer_time_offset = t
        logger.debug("buffer trimmed at %.2fs", t)

    def finish(self) -> Tuple[List[Tuple[float, float, str]], List[Tuple[float, float, str]]]:
        """Final flush at end of stream: commit whatever the last hypothesis
        holds (there is no next pass to agree with)."""
        tail = self.transcript_buffer.complete()
        self.commited.extend(tail)
        self.transcript_buffer.buffer = []
        return tail, []
