"""Tier-2 VBx (Variational-Bayes HMM) re-clustering for speaker diarization.

This module is the second-pass referee called from server.py when the Tier-1
backend (FoxNoseTech `diarize`) collapses a long-gap return-of-speaker into a
single label. It re-clusters per-window x-vectors with a Bayesian HMM whose
temporal `loopProb` prior explicitly resists AHC's long-gap merge failure.

The core `VBx()` function and `forward_backward()` are vendored verbatim from
BUTSpeechFIT/VBx (Apache 2.0, Burget & Diez 2021) — see file header in
``_VBx_apache_header`` below for the upstream license notice.

Reference:
    Landini, Profant, Diez, Burget, "Bayesian HMM clustering of x-vector
    sequences (VBx) in speaker diarization," Computer Speech & Language, 2022.
    https://github.com/BUTSpeechFIT/VBx

Constraint adaptations (deepseek brief, 2026-05-08):
    * Brief mandates re-using existing embedding stack (no new model deps).
      Upstream VBx ships a Kaldi-ResNet101 PLDA that is *not* aligned with
      WeSpeaker embeddings (the only model already loaded by `diarize`),
      so we run VBx without a PLDA: x-vectors are l2-normalised and Phi
      is set to ones. The HMM `loopProb` prior — not the PLDA — is what
      breaks the AHC long-gap failure mode for our 2–4-speaker case.
    * Sliding-window embedding extraction (1.5 s window, 0.25 s hop)
      uses the `wespeakerruntime.Speaker` ONNX session that the
      `diarize` library has already initialised, so no extra model
      download or memory cost.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


_VBx_apache_header = """\
Vendored from https://github.com/BUTSpeechFIT/VBx (commit 57466e6).
Copyright 2021 Lukas Burget, Mireia Diez (burget@fit.vutbr.cz, mireia@fit.vutbr.cz)
Licensed under the Apache License, Version 2.0.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Vendored core: VBx() and forward_backward() — Apache 2.0, BUTSpeechFIT
# Source: VBx/VBx.py (sha 22ce4e0). Kept byte-equivalent to upstream so future
# upstream fixes can be diffed cleanly. Do not modify without bumping the
# attribution comment above.
# ──────────────────────────────────────────────────────────────────────────────
def _forward_backward(lls: np.ndarray, tr: np.ndarray, ip: np.ndarray):
    from scipy.special import logsumexp

    eps = 1e-8
    ltr = np.log(tr + eps)
    lfw = np.empty_like(lls)
    lbw = np.empty_like(lls)
    lfw[:] = -np.inf
    lbw[:] = -np.inf
    lfw[0] = lls[0] + np.log(ip + eps)
    lbw[-1] = 0.0
    for ii in range(1, len(lls)):
        lfw[ii] = lls[ii] + logsumexp(lfw[ii - 1] + ltr.T, axis=1)
    for ii in reversed(range(len(lls) - 1)):
        lbw[ii] = logsumexp(ltr + lls[ii + 1] + lbw[ii + 1], axis=1)
    tll = logsumexp(lfw[-1], axis=0)
    pi = np.exp(lfw + lbw - tll)
    return pi, tll, lfw, lbw


def _VBx(
    X: np.ndarray,
    Phi: np.ndarray,
    loopProb: float = 0.9,
    Fa: float = 1.0,
    Fb: float = 1.0,
    pi: int = 10,
    gamma: Optional[np.ndarray] = None,
    maxIters: int = 40,
    epsilon: float = 1e-6,
    alphaQInit: float = 1.0,
):
    from scipy.special import logsumexp

    D = X.shape[1]
    if isinstance(pi, int):
        pi_vec = np.ones(pi) / pi
    else:
        pi_vec = np.asarray(pi, dtype=np.float64).copy()

    if gamma is None:
        gamma = np.random.gamma(alphaQInit, size=(X.shape[0], len(pi_vec)))
        gamma = gamma / gamma.sum(1, keepdims=True)

    G = -0.5 * (np.sum(X ** 2, axis=1, keepdims=True) + D * np.log(2 * np.pi))
    V = np.sqrt(Phi)
    rho = X * V
    Li: List[float] = []
    alpha = None
    invL = None
    for ii in range(maxIters):
        if ii > 0 or alpha is None or invL is None:
            invL = 1.0 / (1 + Fa / Fb * gamma.sum(axis=0, keepdims=True).T * Phi)
            alpha = Fa / Fb * invL * gamma.T.dot(rho)
        log_p_ = Fa * (rho.dot(alpha.T) - 0.5 * (invL + alpha ** 2).dot(Phi) + G)
        tr = np.eye(len(pi_vec)) * loopProb + (1 - loopProb) * pi_vec
        gamma, log_pX_, logA, logB = _forward_backward(log_p_, tr, pi_vec)
        ELBO = log_pX_ + Fb * 0.5 * np.sum(np.log(invL) - invL - alpha ** 2 + 1)
        pi_vec = gamma[0] + (1 - loopProb) * pi_vec * np.sum(
            np.exp(
                logsumexp(logA[:-1], axis=1, keepdims=True)
                + log_p_[1:]
                + logB[1:]
                - log_pX_
            ),
            axis=0,
        )
        pi_vec = pi_vec / pi_vec.sum()
        Li.append(ELBO)
        if ii > 0 and ELBO - Li[-2] < epsilon:
            if ELBO - Li[-2] < 0:
                logger.debug("VBx ELBO decreased at iter %d (harmless)", ii)
            break
    return gamma, pi_vec, Li


def _l2_norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, ord=2, keepdims=True)
    return x / np.maximum(n, 1e-12)


def _merge_adjacent(starts: np.ndarray, ends: np.ndarray, labels: np.ndarray):
    """Vendored from BUTSpeechFIT/VBx diarization_lib.merge_adjacent_labels."""
    if len(starts) == 0:
        return starts, ends, labels
    adjacent = np.logical_or(np.isclose(ends[:-1], starts[1:]), ends[:-1] > starts[1:])
    to_split = np.nonzero(np.logical_or(~adjacent, labels[1:] != labels[:-1]))[0]
    starts = starts[np.r_[0, to_split + 1]]
    ends = ends[np.r_[to_split, -1]]
    labels = labels[np.r_[0, to_split + 1]]
    overlapping = np.nonzero(starts[1:] < ends[:-1])[0]
    if len(overlapping):
        ends[overlapping] = starts[overlapping + 1] = (
            ends[overlapping] + starts[overlapping + 1]
        ) / 2.0
    return starts, ends, labels


# ──────────────────────────────────────────────────────────────────────────────
# Embedding extraction — reuses the WeSpeaker ONNX session inside the `diarize`
# library so we add zero new model dependencies (per brief constraint).
# ──────────────────────────────────────────────────────────────────────────────
def _load_wespeaker_session(diarize_lib: Any):
    """Locate the wespeaker Speaker session inside the `diarize` library.

    The library wraps `wespeakerruntime.Speaker`. Different versions expose
    it under different attribute names — try the most common, and fall back
    to constructing a fresh one if needed.
    """
    candidates = [
        "_speaker",
        "speaker",
        "_speaker_model",
        "speaker_model",
        "embedder",
        "_embedder",
    ]
    for attr in candidates:
        spk = getattr(diarize_lib, attr, None)
        if spk is not None and hasattr(spk, "extract_embedding"):
            return spk
    # Fall back to creating one (downloads model on first use, same dir).
    try:
        import wespeakerruntime  # type: ignore

        return wespeakerruntime.Speaker(lang="en")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Could not obtain wespeaker session: {exc}") from exc


def _read_audio_mono16k(audio_path: str) -> Tuple[np.ndarray, int]:
    """Decode any audio file to a mono 16 kHz float32 waveform via ffmpeg."""
    import subprocess

    sr = 16000
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-loglevel",
        "error",
        "-i",
        str(audio_path),
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        str(sr),
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=True)
    pcm = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    return pcm, sr


def _extract_window_xvectors(
    audio_path: str,
    diarize_lib: Any,
    window_s: float = 1.5,
    hop_s: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slide windows over the audio and extract one WeSpeaker embedding per window.

    Uses ``Speaker.extract_embedding_feat`` (numpy fbank batch input) so we
    don't need to write temp wavs per window. Fbank computation matches the
    wespeakerruntime defaults (80 mel bins, 25 ms frame, 10 ms shift, CMN).

    Returns:
        xvecs   - (N, D) float32 array of l2-normalised x-vectors
        starts  - (N,) window start times in seconds
        ends    - (N,) window end times in seconds
    """
    import torch
    import torchaudio.compliance.kaldi as kaldi

    spk = _load_wespeaker_session(diarize_lib)
    pcm, sr = _read_audio_mono16k(audio_path)

    win = int(round(window_s * sr))
    hop = int(round(hop_s * sr))
    if len(pcm) < win:
        pad = np.zeros(win, dtype=np.float32)
        pad[: len(pcm)] = pcm
        pcm = pad

    starts: List[float] = []
    ends: List[float] = []
    feats_list: List[np.ndarray] = []
    pos = 0
    while pos + win <= len(pcm):
        chunk = pcm[pos : pos + win]
        # Match wespeakerruntime._compute_fbank exactly: scale to int16 range
        # before fbank, no dither, hamming window, no CVN, with CMN done by
        # extract_embedding_feat.
        wav_t = torch.from_numpy(chunk).unsqueeze(0).float() * (1 << 15)
        mat = kaldi.fbank(
            wav_t,
            num_mel_bins=80,
            frame_length=25,
            frame_shift=10,
            dither=0.0,
            sample_frequency=sr,
            window_type="hamming",
            use_energy=False,
        ).numpy()
        feats_list.append(mat)
        starts.append(pos / sr)
        ends.append((pos + win) / sr)
        pos += hop

    if not feats_list:
        return (np.zeros((0, 0), dtype=np.float32), np.zeros(0), np.zeros(0))

    feats_batch = np.stack(feats_list, axis=0).astype(np.float32)  # [B, T, D]
    feats_batch = feats_batch - feats_batch.mean(axis=1, keepdims=True)
    embs = spk.extract_embedding_feat(feats_batch, cmn=False)  # [B, emb_dim]
    X = np.asarray(embs, dtype=np.float32)
    X = _l2_norm(X)
    return X, np.asarray(starts, dtype=np.float64), np.asarray(ends, dtype=np.float64)


# ──────────────────────────────────────────────────────────────────────────────
# Initialisation from Tier-1 segmentation
# ──────────────────────────────────────────────────────────────────────────────
def _initial_labels_from_tier1(
    starts: np.ndarray,
    ends: np.ndarray,
    tier1: List[Dict[str, Any]],
) -> Tuple[np.ndarray, List[str]]:
    """Map each window to its max-overlap Tier-1 speaker.

    Returns (label_ids, id_to_speaker_name).
    """
    speakers = sorted({r["speaker"] for r in tier1 if r.get("speaker")})
    name_to_idx = {s: i for i, s in enumerate(speakers)}
    K = len(speakers)
    if K == 0:
        return np.zeros(len(starts), dtype=int), []

    labels = np.zeros(len(starts), dtype=int)
    for i, (ws, we) in enumerate(zip(starts, ends)):
        best_ov = 0.0
        best_spk = 0
        for r in tier1:
            ov = max(0.0, min(we, r["end"]) - max(ws, r["start"]))
            if ov > best_ov:
                best_ov = ov
                best_spk = name_to_idx.get(r["speaker"], 0)
        labels[i] = best_spk
    return labels, speakers


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────
def vbx_resegment(
    audio_path: str,
    tier1_records: List[Dict[str, Any]],
    diarize_lib: Any,
    *,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    loopProb: float = 0.95,
    Fa: float = 0.4,
    Fb: float = 11.0,
    window_s: float = 1.5,
    hop_s: float = 0.25,
) -> Optional[List[Dict[str, Any]]]:
    """Run a VBx VB-HMM re-clustering pass on top of Tier-1 segmentation.

    Args:
        audio_path: source media (any format ffmpeg can decode).
        tier1_records: list of {"start","end","speaker"} dicts from Tier-1.
        diarize_lib: the loaded `diarize` module (provides WeSpeaker session).
        min_speakers / max_speakers: hard hints (max becomes the VBx HMM
            state count cap; defaults to max(K_tier1, 4)).
        loopProb: HMM stay-in-state probability — the parameter that fixes
            AHC's long-gap failure. Higher = stickier within speaker turn.
        Fa / Fb: VBx scaling parameters (BUTSpeechFIT defaults for the
            no-PLDA, l2-normalised case).
        window_s / hop_s: embedding window geometry.

    Returns:
        Refined records in the same {"start","end","speaker"} schema as
        input, or None if VBx could not run (caller falls back to Tier-1).
    """
    if not tier1_records:
        return None

    try:
        X, w_starts, w_ends = _extract_window_xvectors(
            audio_path, diarize_lib, window_s=window_s, hop_s=hop_s
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("VBx: embedding extraction failed: %s", exc)
        return None

    if X.shape[0] < 4:
        logger.info("VBx: too few windows (%d) — skipping", X.shape[0])
        return None

    init_labels, init_speakers = _initial_labels_from_tier1(w_starts, w_ends, tier1_records)
    K_init = max(1, len(init_speakers))

    # Pick the maximum-state count for the HMM. Brief says "use min/max as
    # hard constraints"; we honour `max_speakers` strictly and otherwise
    # allow up to max(K_init, 4) to give VBx room to *split* the collapsed
    # Tier-1 cluster (its whole purpose).
    if max_speakers is not None and max_speakers >= 1:
        K_max = int(max_speakers)
    else:
        K_max = max(K_init, 4)
    K_max = max(K_max, 2)

    # When Tier-1 collapsed (init_speakers == 1) but caller wants ≥2 speakers,
    # seed gamma_init via k-means over X so VBx has a non-degenerate starting
    # point. Otherwise, seed from Tier-1 hard labels (smoothed) so VBx mostly
    # *resegments* Tier-1's existing partition.
    if K_init < K_max:
        try:
            from sklearn.cluster import KMeans  # type: ignore

            km = KMeans(n_clusters=K_max, n_init=10, random_state=0).fit(X)
            seed_labels = km.labels_
        except Exception:
            # Deterministic fallback: split by half-window index
            seed_labels = (np.arange(len(init_labels)) * K_max // len(init_labels)).astype(int)
    else:
        seed_labels = np.minimum(init_labels, K_max - 1)

    gamma_init = np.full((len(init_labels), K_max), 0.05 / max(K_max - 1, 1))
    rows = np.arange(len(init_labels))
    gamma_init[rows, seed_labels] = 0.95
    gamma_init = gamma_init / gamma_init.sum(axis=1, keepdims=True)

    # No PLDA available for WeSpeaker embeddings (see module docstring).
    # Use empirical per-dim variance as Phi (proxy for between-class spread).
    # Dimensions with more variance across the recording carry more speaker
    # discrimination — without this, VBx with Phi=ones tends to collapse to
    # a single cluster on l2-normalised x-vectors.
    Phi = np.var(X.astype(np.float64), axis=0) + 1e-6
    Phi = Phi / Phi.mean()  # normalise so Fa/Fb defaults remain in range

    try:
        gamma, pi_vec, _Li = _VBx(
            X=X.astype(np.float64),
            Phi=Phi,
            loopProb=loopProb,
            Fa=Fa,
            Fb=Fb,
            pi=K_max,
            gamma=gamma_init,
            maxIters=40,
            epsilon=1e-6,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("VBx: solver failed: %s", exc)
        return None

    # Hard-assign each window to its argmax speaker.
    new_labels = np.argmax(gamma, axis=1)

    # Honour min_speakers if supplied: if VBx returned fewer distinct labels
    # than the minimum, flag it (caller can decide to keep Tier-1).
    n_found = len(np.unique(new_labels))
    if min_speakers is not None and n_found < min_speakers:
        logger.info(
            "VBx: returned %d speakers, min_speakers=%d — caller may discard",
            n_found,
            min_speakers,
        )

    # Merge adjacent identical-label windows back into segments.
    starts, ends, labels = _merge_adjacent(
        np.asarray(w_starts, dtype=np.float64),
        np.asarray(w_ends, dtype=np.float64),
        new_labels.astype(np.int64),
    )

    # Renumber labels to SPEAKER_NN compactly in chronological-first-appearance
    # order so output is deterministic and matches the rest of the stack.
    seen: Dict[int, str] = {}
    out: List[Dict[str, Any]] = []
    for s, e, lab in zip(starts, ends, labels):
        if lab not in seen:
            seen[int(lab)] = f"SPEAKER_{len(seen):02d}"
        out.append(
            {"start": float(s), "end": float(e), "speaker": seen[int(lab)]}
        )
    return out


def map_diar_to_whisperx_segments(
    whisperx_segments: List[Dict[str, Any]],
    diar_records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Assign a speaker label to every WhisperX segment by max-overlap.

    Critical contract from the deepseek brief: NEVER touch the WhisperX
    segment ``start``/``end`` fields. Only the ``speaker`` field is added
    (or overwritten if the existing label has lower overlap with diar).

    For segments with zero overlap against any diarization turn (e.g. a
    short segment falling in a silence gap that VAD didn't cut), we fall
    back to the chronologically nearest turn — the same fallback strategy
    server.py's existing nearest-by-time pass uses.
    """
    if not diar_records or not whisperx_segments:
        return whisperx_segments

    diar_sorted = sorted(diar_records, key=lambda r: r["start"])

    def _nearest(s_start: float, s_end: float) -> Optional[str]:
        best: Optional[Tuple[float, str]] = None
        for r in diar_sorted:
            if s_end < r["start"]:
                gap = r["start"] - s_end
            elif s_start > r["end"]:
                gap = s_start - r["end"]
            else:
                gap = 0.0
            if best is None or gap < best[0]:
                best = (gap, r["speaker"])
        return best[1] if best else None

    for seg in whisperx_segments:
        s_start = float(seg.get("start", 0.0) or 0.0)
        s_end = float(seg.get("end", s_start) or s_start)
        best_ov = 0.0
        best_spk: Optional[str] = None
        for r in diar_sorted:
            ov = max(0.0, min(s_end, r["end"]) - max(s_start, r["start"]))
            if ov > best_ov:
                best_ov = ov
                best_spk = r["speaker"]
        if best_spk is not None:
            seg["speaker"] = best_spk
        elif not seg.get("speaker"):
            fallback = _nearest(s_start, s_end)
            if fallback is not None:
                seg["speaker"] = fallback

        # Propagate label down to word-level so SRT/transcript renderers
        # (which look at words first) stay consistent. NEVER change w[start/end].
        spk = seg.get("speaker")
        if spk:
            for w in seg.get("words", []) or []:
                if not w.get("speaker"):
                    w["speaker"] = spk

    return whisperx_segments
