"""OmniVoice multilingual TTS sidecar package.

Runs OmniVoice (k2-fsa/OmniVoice, Apache 2.0, 646 languages incl. African) in an
ISOLATED venv because it requires transformers 5.x, which we do not want to force
on the main server's whisperx / ctranslate2 stack. The main server talks to it
over localhost HTTP via :mod:`omnivoice_client`.

Deliberately named ``omnivoice_tts`` (not ``omnivoice``) so it never shadows the
pip-installed ``omnivoice`` package inside the sidecar venv.
"""
from __future__ import annotations
