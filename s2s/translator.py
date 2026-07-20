"""Machine translation for /ws/s2s — pluggable backend.

Backends (env ``MT_BACKEND``):

* ``madlad`` (default) — google/madlad400-3b-mt converted to CTranslate2 int8.
  **Apache 2.0 → commercially safe.** 400+ languages including African
  (Swahili, Yoruba, Hausa, Igbo, Zulu, Shona, Amharic, Wolof, Akan/Twi, …).
  Runs on the ctranslate2 runtime the server already ships; int8 weights are
  ~3 GB. Target language is selected with a ``<2xx>`` prefix token; the
  source language is auto-detected by the model.
  Model dir: ``MADLAD_MODEL_DIR`` (default ``/models/madlad400-3b-mt-ct2-int8``).
  A larger ``madlad400-7b-mt-bt`` conversion can be pointed at via the same
  env var for weak low-resource pairs.

* ``nllb`` — facebook/nllb-200-distilled-600M. **CC-BY-NC 4.0 — NON-COMMERCIAL
  USE ONLY.** Kept strictly as a development/research fallback; do NOT enable
  in any revenue-generating deployment. (An earlier version of this module
  mislabelled NLLB as Apache 2.0 — that was wrong.)

Validated June 2026 (CT2 int8, local CPU): fr/sw/yo/ha/ig/zu translate well;
en→ln (Lingala) is weak on the 3B model (garbage/French output) — pending
re-test on the 7B-bt conversion. Decoding uses beam=4 with
``no_repeat_ngram_size=3`` to prevent the repetition-loop pathology observed
with greedy decoding on low-resource targets.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger("s2s.translator")

MT_BACKEND = os.environ.get("MT_BACKEND", "madlad").strip().lower() or "madlad"
MADLAD_MODEL_DIR = os.environ.get(
    "MADLAD_MODEL_DIR", "/models/madlad400-3b-mt-ct2-int8"
).strip()
NLLB_MODEL_ID = os.environ.get("NLLB_MODEL_ID", "facebook/nllb-200-distilled-600M")

# ---------------------------------------------------------------------------
# Language code tables
# ---------------------------------------------------------------------------

# ISO-639-1 → NLLB FLORES-200 codes (nllb backend only).
ISO1_TO_FLORES: dict[str, str] = {
    "en": "eng_Latn",
    "fr": "fra_Latn",
    "es": "spa_Latn",
    "de": "deu_Latn",
    "it": "ita_Latn",
    "pt": "por_Latn",
    "nl": "nld_Latn",
    "pl": "pol_Latn",
    "ru": "rus_Cyrl",
    "uk": "ukr_Cyrl",
    "tr": "tur_Latn",
    "ar": "arb_Arab",
    "he": "heb_Hebr",
    "fa": "pes_Arab",
    "hi": "hin_Deva",
    "bn": "ben_Beng",
    "ur": "urd_Arab",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "zh": "zho_Hans",
    "vi": "vie_Latn",
    "th": "tha_Thai",
    "id": "ind_Latn",
    "ms": "zsm_Latn",
    "sw": "swh_Latn",
    # African languages (NLLB-200 FLORES codes)
    "ln": "lin_Latn",   # Lingala
    "yo": "yor_Latn",   # Yoruba
    "ha": "hau_Latn",   # Hausa
    "ig": "ibo_Latn",   # Igbo
    "zu": "zul_Latn",   # Zulu
    "sn": "sna_Latn",   # Shona
    "am": "amh_Ethi",   # Amharic
    "wo": "wol_Latn",   # Wolof
    "tw": "twi_Latn",   # Twi
    "af": "afr_Latn",
    "sv": "swe_Latn",
    "no": "nob_Latn",
    "da": "dan_Latn",
    "fi": "fin_Latn",
    "el": "ell_Grek",
    "cs": "ces_Latn",
    "ro": "ron_Latn",
    "hu": "hun_Latn",
    "bg": "bul_Cyrl",
    "ca": "cat_Latn",
}

# ISO-639-1 → MADLAD ``<2xx>`` token code. MADLAD uses (mostly) ISO-639-1
# directly; only the exceptions are listed. Anything not listed passes
# through unchanged and is checked against the sentencepiece vocabulary.
ISO1_TO_MADLAD_OVERRIDES: dict[str, str] = {
    "tw": "ak",   # Twi → Akan (MADLAD groups Twi under Akan)
    "tl": "fil",  # Tagalog → Filipino
    "nb": "no",
}

# Target languages whose MADLAD OUTPUT is broken despite the token existing in
# the vocabulary. Validated June 2026 on BOTH 3B and 7B-bt: en→ln produces
# French/Swahili/gibberish (contaminated Lingala web data). We refuse these
# targets with a clear error instead of silently returning wrong-language text.
# (Lingala as a SOURCE language still works — e.g. speak-Lingala → English —
# and OmniVoice TTS can still speak Lingala text.) Path to support: fine-tune
# MADLAD on clean ln corpora (JW300/Bible/AFRIDOC) — tracked on the roadmap.
MADLAD_BROKEN_TARGETS: frozenset[str] = frozenset({"ln"})


def iso1_to_flores(code: str) -> str:
    """Map ISO-639-1 → FLORES (nllb). Pass-through for FLORES-style codes."""
    code = (code or "").strip()
    if not code:
        raise ValueError("language code is empty")
    if "_" in code:
        return code
    flores = ISO1_TO_FLORES.get(code.lower())
    if not flores:
        raise ValueError(
            f"Unsupported language code: {code!r}. Pass an ISO-639-1 code "
            f"({sorted(ISO1_TO_FLORES)[:10]}…) or a FLORES-200 code."
        )
    return flores


# ---------------------------------------------------------------------------
# MADLAD backend (Apache 2.0 — commercial-safe default)
# ---------------------------------------------------------------------------


class MadladTranslator:
    """MADLAD-400 CT2 wrapper. Thread-safe; loaded lazily, warmed at startup.

    ``translate()`` keeps the same signature as the NLLB backend so server.py
    is backend-agnostic. ``source_lang`` is accepted but unused (MADLAD
    auto-detects the source)."""

    def __init__(self, model_dir: str = MADLAD_MODEL_DIR, device: Optional[str] = None) -> None:
        self.model_dir = model_dir
        self._device = device
        self._translator = None
        self._sp = None
        self._lock = threading.Lock()

    # -- lazy loading ----------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._translator is not None:
            return
        with self._lock:
            if self._translator is not None:
                return
            import ctranslate2
            import sentencepiece as spm

            if not Path(self.model_dir).is_dir():
                raise RuntimeError(
                    f"MADLAD model dir not found: {self.model_dir}. "
                    "start.sh fetches it to the Network Volume; check its logs "
                    "or set MADLAD_MODEL_DIR."
                )
            if self._device is None:
                try:
                    import torch
                    self._device = "cuda" if torch.cuda.is_available() else "cpu"
                except Exception:  # noqa: BLE001
                    self._device = "cpu"
            logger.info("Loading MADLAD CT2 from %s on %s", self.model_dir, self._device)
            self._translator = ctranslate2.Translator(
                self.model_dir,
                device=self._device,
                compute_type="int8" if self._device == "cpu" else "int8_float16",
            )
            sp = spm.SentencePieceProcessor()
            sp.load(str(Path(self.model_dir) / "spiece.model"))
            self._sp = sp
            logger.info("MADLAD ready on %s", self._device)

    def warmup(self) -> None:
        self._ensure_loaded()
        out = self.translate("Hello.", source_lang="en", target_lang="fr")
        logger.info("MADLAD warm-up complete (%r)", out[:40])

    # -- language support -------------------------------------------------

    def _madlad_code(self, code: str) -> str:
        code = (code or "").strip().lower()
        return ISO1_TO_MADLAD_OVERRIDES.get(code, code)

    def supports(self, code: str) -> bool:
        """True iff the ``<2xx>`` token exists in the model vocabulary AND the
        pair is not on the known-broken-output list."""
        try:
            if self._madlad_code(code) in MADLAD_BROKEN_TARGETS:
                return False
            self._ensure_loaded()
            tok = f"<2{self._madlad_code(code)}>"
            return self._sp.piece_to_id(tok) != self._sp.unk_id()
        except Exception:  # noqa: BLE001
            return False

    def ensure_supported_target(self, code: str) -> None:
        if not code or not code.strip():
            raise ValueError("language code is empty")
        if self._madlad_code(code) in MADLAD_BROKEN_TARGETS:
            raise ValueError(
                f"target language {code!r} is not reliably supported by the "
                "current MT model (MADLAD-400 outputs wrong-language text for "
                "it). Speaking it as a SOURCE language still works."
            )
        if not self.supports(code):
            raise ValueError(
                f"target language {code!r} is not in the MADLAD vocabulary"
            )

    # -- inference -------------------------------------------------------

    def translate(
        self,
        text: str,
        *,
        source_lang: str,   # accepted for interface compat; MADLAD auto-detects
        target_lang: str,
        max_new_tokens: int = 256,
    ) -> str:
        text = (text or "").strip()
        if not text:
            return ""
        tgt = self._madlad_code(target_lang)
        src = (source_lang or "").strip().lower()
        if src and self._madlad_code(src) == tgt:
            return text  # no-op shortcut for matching langs
        self._ensure_loaded()

        tokens = self._sp.encode(f"<2{tgt}> {text}", out_type=str)
        # beam=4 + no_repeat_ngram_size=3: prevents the greedy repetition-loop
        # pathology on low-resource targets while keeping latency reasonable.
        # Decode length is capped at ~3× the input tokens (translation length
        # ratios rarely exceed that) — curbs MADLAD's tendency to append a
        # redundant paraphrase after the real translation.
        max_len = min(max_new_tokens, max(24, 3 * len(tokens)))
        results = self._translator.translate_batch(
            [tokens],
            beam_size=4,
            no_repeat_ngram_size=3,
            max_decoding_length=max_len,
        )
        return self._sp.decode(results[0].hypotheses[0]).strip()


# ---------------------------------------------------------------------------
# NLLB backend (CC-BY-NC 4.0 — NON-COMMERCIAL, dev fallback only)
# ---------------------------------------------------------------------------


class Translator:
    """NLLB-200 wrapper. **License: CC-BY-NC 4.0 — non-commercial only.**

    Retained as a research/dev fallback (``MT_BACKEND=nllb``); never enable in
    a paid deployment."""

    def __init__(self, model_id: str = NLLB_MODEL_ID, device: Optional[str] = None) -> None:
        self.model_id = model_id
        self._device = device
        self._model = None
        self._tokenizer = None
        self._lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
            dtype = torch.float16 if device == "cuda" else torch.float32
            logger.warning(
                "Loading NLLB-200 (%s) — CC-BY-NC 4.0, NON-COMMERCIAL USE ONLY",
                self.model_id,
            )
            tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_id, torch_dtype=dtype
            ).to(device).eval()
            self._tokenizer = tokenizer
            self._model = model
            self._device = device
            logger.info("NLLB-200 ready on %s", device)

    def warmup(self) -> None:
        try:
            self._ensure_loaded()
            _ = self.translate("Hello.", source_lang="en", target_lang="fr")
            logger.info("NLLB-200 warm-up complete.")
        except Exception as exc:  # noqa: BLE001
            logger.warning("NLLB-200 warm-up failed: %s", exc)

    def supports(self, code: str) -> bool:
        try:
            iso1_to_flores(code)
            return True
        except ValueError:
            return False

    def ensure_supported_target(self, code: str) -> None:
        iso1_to_flores(code)  # raises ValueError if unknown

    def translate(
        self,
        text: str,
        *,
        source_lang: str,
        target_lang: str,
        max_new_tokens: int = 256,
    ) -> str:
        text = (text or "").strip()
        if not text:
            return ""
        self._ensure_loaded()
        import torch

        src = iso1_to_flores(source_lang)
        tgt = iso1_to_flores(target_lang)
        if src == tgt:
            return text

        tokenizer = self._tokenizer
        model = self._model
        assert tokenizer is not None and model is not None

        tokenizer.src_lang = src
        encoded = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(self._device)
        try:
            forced_bos = tokenizer.convert_tokens_to_ids(tgt)
        except Exception:  # noqa: BLE001
            forced_bos = tokenizer.lang_code_to_id[tgt]  # type: ignore[attr-defined]

        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                forced_bos_token_id=forced_bos,
                max_new_tokens=max_new_tokens,
                num_beams=1,
            )
        return tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()


# ---------------------------------------------------------------------------
# Factory + module singleton
# ---------------------------------------------------------------------------

_singleton = None
_singleton_lock = threading.Lock()


def _build(name: str):
    name = (name or "").strip().lower()
    if name in {"madlad", "madlad400", ""}:
        return MadladTranslator()
    if name == "nllb":
        return Translator()
    raise ValueError(f"Unknown MT_BACKEND={name!r}. Supported: madlad, nllb.")


def get_translator():
    """Process-wide MT singleton for the backend selected by ``MT_BACKEND``."""
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = _build(MT_BACKEND)
    return _singleton


def ensure_supported_target(code: str) -> None:
    """Backend-agnostic target-language validation (used by /ws/s2s start)."""
    get_translator().ensure_supported_target(code)
