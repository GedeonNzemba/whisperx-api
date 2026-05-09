"""Machine-translation wrapper around NLLB-200 distilled 600M.

Loaded once at server startup and reused across every /ws/s2s segment.
NLLB uses BCP-47-ish FLORES codes (e.g. ``eng_Latn``, ``fra_Latn``); we
expose a thin ISO-639-1 mapping for the most common targets so callers can
just pass ``"fr"``.

Why NLLB-200 distilled 600M:
* Apache 2.0 (commercially safe — matches our monetisation plan)
* ~2.5 GB VRAM in fp16 – fits comfortably alongside WhisperX + TTS
* 200+ languages, sub-100ms per short segment on RTX 4090

Reference: https://huggingface.co/facebook/nllb-200-distilled-600M
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Optional

logger = logging.getLogger("s2s.translator")

NLLB_MODEL_ID = os.environ.get("NLLB_MODEL_ID", "facebook/nllb-200-distilled-600M")

# Common ISO-639-1 → NLLB FLORES-200 code table. Covers Phase-1 target langs.
# Extend as needed; unknown ISO-1 codes raise ValueError so the caller can
# surface a clear error to the WebSocket client.
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


def iso1_to_flores(code: str) -> str:
    """Map an ISO-639-1 code (e.g. ``"fr"``) to NLLB FLORES (``"fra_Latn"``).

    Pass-through for codes that already look like FLORES (contain an underscore).
    """
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


class Translator:
    """Thread-safe NLLB-200 wrapper (model is GPU; tokeniser is CPU).

    Loaded lazily on first ``translate()`` call, but the server's lifespan
    triggers a warm-up call so the first user request doesn't pay the
    ~3 s download/load cost.
    """

    def __init__(self, model_id: str = NLLB_MODEL_ID, device: Optional[str] = None) -> None:
        self.model_id = model_id
        self._device = device
        self._model = None
        self._tokenizer = None
        self._lock = threading.Lock()

    # -- lazy loading ----------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            import torch  # local import keeps top-level cheap
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
            dtype = torch.float16 if device == "cuda" else torch.float32
            logger.info("Loading NLLB-200 (%s) on %s [%s]", self.model_id, device, dtype)

            tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_id, torch_dtype=dtype
            ).to(device).eval()

            self._tokenizer = tokenizer
            self._model = model
            self._device = device
            logger.info("NLLB-200 ready on %s", device)

    def warmup(self) -> None:
        """Force model load + a tiny dummy translation so the first real
        request doesn't pay the JIT/cudnn-tune cost."""
        try:
            self._ensure_loaded()
            _ = self.translate("Hello.", source_lang="en", target_lang="fr")
            logger.info("NLLB-200 warm-up complete.")
        except Exception as exc:  # noqa: BLE001
            logger.warning("NLLB-200 warm-up failed: %s", exc)

    # -- inference -------------------------------------------------------

    def translate(
        self,
        text: str,
        *,
        source_lang: str,
        target_lang: str,
        max_new_tokens: int = 256,
    ) -> str:
        """Translate a short utterance. Both args accept ISO-639-1 or FLORES."""
        text = (text or "").strip()
        if not text:
            return ""
        self._ensure_loaded()
        import torch

        src = iso1_to_flores(source_lang)
        tgt = iso1_to_flores(target_lang)
        if src == tgt:
            return text  # no-op shortcut for matching langs

        tokenizer = self._tokenizer
        model = self._model
        assert tokenizer is not None and model is not None  # for type-checkers

        # NLLB tokenisers expose src_lang as an attribute; we set it before
        # encoding so the BOS token is correct for the source side.
        tokenizer.src_lang = src
        encoded = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(self._device)

        # forced_bos_token_id selects the target language at decoding time.
        # Different transformers versions expose this differently; cover both.
        try:
            forced_bos = tokenizer.convert_tokens_to_ids(tgt)
        except Exception:  # noqa: BLE001
            forced_bos = tokenizer.lang_code_to_id[tgt]  # type: ignore[attr-defined]

        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                forced_bos_token_id=forced_bos,
                max_new_tokens=max_new_tokens,
                num_beams=1,  # greedy keeps latency low; quality is fine for short utterances
            )
        out = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        return out.strip()


# Module-level singleton — server.py imports this and calls warmup() during
# /ws/s2s startup. Multiple concurrent /ws/s2s connections share one model.
_singleton: Optional[Translator] = None
_singleton_lock = threading.Lock()


def get_translator() -> Translator:
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = Translator()
    return _singleton
