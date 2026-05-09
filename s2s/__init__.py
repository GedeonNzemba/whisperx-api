"""Speech-to-speech translation Phase 1.

Cascaded pipeline used by the /ws/s2s WebSocket endpoint:

    Mic → STT (existing WhisperX) → MT (NLLB-200) → TTS (Chatterbox-Turbo) → Speaker

Phase 1 scope (instructions_1.txt):
* English STT (existing) → arbitrary target language MT → English TTS only
  (Chatterbox-Turbo is English-only). Multilingual TTS (Qwen3-TTS 0.6B) is
  scaffolded behind ``TTS_BACKEND`` and will be wired in a later phase.
* Headphones assumed (no AEC).
* Single-GPU shared with WhisperX. Loads each model once at startup.

Modules:
* :mod:`s2s.translator` – NLLB-200 distilled wrapper.
* :mod:`s2s.tts` – pluggable TTS backend (env: ``TTS_BACKEND``).
"""

from __future__ import annotations
