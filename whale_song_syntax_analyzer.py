#!/usr/bin/env python3
"""Thalassa whale-song decoder: bioacoustic language pipeline.

End-to-end bioacoustic language pipeline:
  hydrophone WAV (44.1 kHz) -> bandpass 50-4000 Hz + spectral-gating denoise
  -> adaptive energy VAD segmentation -> 128-bin log-mel
  -> PyTorch VQ-VAE (K=64) unit discovery -> token sequences
  -> small causal GPT language model -> perplexity + Shannon entropy
  -> song_tokens.json + syntax report (PNG + PDF, matplotlib only).

Usage:
  python whale_song_syntax_analyzer.py --input underwater_hydrophone.wav --outdir outputs
  python whale_song_syntax_analyzer.py --input missing.wav --outdir outputs --synthetic-if-missing
  python whale_song_syntax_analyzer.py --help
"""

import argparse
import json
import math
import os
import random
import sys

import numpy as np

DEFAULT_SR = 44100
DEFAULT_N_MELS = 128
DEFAULT_K = 64
DEFAULT_LOW = 50.0
DEFAULT_HIGH = 4000.0
# Bump when payload/artifact schema changes so old job caches go stale.
CODEC_VERSION = 9

# Filterbank cache: identical (bank, sr, n_fft, n_mels, fmin, fmax) builds the
# same matrix, so reuse it across segments instead of rebuilding per segment.
_FB_CACHE = {}

import threading as _threading
# Serializes seed + model training across run_batch threads. set_seed writes
# process-global torch RNG state, so concurrent pipelines would cross their
# random streams (and share the intra-op pool); numpy denoise/VAD/features
# stay parallel since they use local RNGs or none.
_TRAIN_LOCK = _threading.Lock()


def _cached_filterbank(bank, sr, n_fft, n_mels, fmin, fmax):
    key = (bank, int(sr), int(n_fft), int(n_mels), float(fmin),
           float(fmax))
    fb = _FB_CACHE.get(key)
    if fb is None:
        if bank == "gammatone":
            fb = _gammatone_filterbank(sr, n_fft, n_mels, fmin, fmax)
        else:
            fb = _mel_filterbank(sr, n_fft, n_mels, fmin, fmax)
        _FB_CACHE[key] = fb
    return fb


def theme_syntax(theme_seq):
    """Theme-level syntax: sequence + bigram + entropy (hierarchy top)."""
    import math as _math
    seq = [int(t) for t in theme_seq]
    n = len(seq)
    counts = {}
    for a, b in zip(seq, seq[1:]):
        counts[(a, b)] = counts.get((a, b), 0) + 1
    tot = sum(counts.values()) or 1
    ent = -sum((c / tot) * _math.log2(c / tot) for c in counts.values()) \
        if counts else 0.0
    return {"sequence": seq,
            "n_transitions": int(sum(counts.values())),
            "n_unique_transitions": int(len(counts)),
            "entropy_bits": round(float(ent), 3),
            "bigram_counts": [[int(a), int(b), int(c)]
                              for (a, b), c in
                              sorted(counts.items(), key=lambda kv: -kv[1])[:64]]}

PCEN_PRESETS = {
    "quiet": {"pcen_tau": 0.06, "pcen_s": 0.05, "pcen_alpha": 0.98},
    "noisy": {"pcen_tau": 0.10, "pcen_s": 0.04, "pcen_alpha": 0.95},
    "harsh": {"pcen_tau": 0.15, "pcen_s": 0.03, "pcen_alpha": 0.90},
}


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass


# ----------------------------------------------------------------------------
# Audio IO + synthetic fallback (numpy + scipy only)
# ----------------------------------------------------------------------------

def load_wav(path, target_sr=DEFAULT_SR):
    from scipy.io import wavfile
    from scipy.signal import resample_poly
    from math import gcd
    sr, data = wavfile.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float32) / float(max(1, np.iinfo(data.dtype).max))
    else:
        data = data.astype(np.float32)
    if sr != target_sr:
        g = gcd(int(sr), int(target_sr))
        data = resample_poly(data, target_sr // g, sr // g).astype(np.float32)
        sr = target_sr
    return data, sr


def generate_synthetic_wav(path, sr=DEFAULT_SR, duration_s=12.0, seed=0):
    """Synthetic hydrophone-like fixture: chirps + harmonics + noise + silences."""
    from scipy.io import wavfile
    from scipy.signal import chirp
    rng = np.random.default_rng(seed)
    n = int(sr * duration_s)
    t = np.arange(n) / sr
    x = np.zeros(n, dtype=np.float64)
    # Whale-like phrases scattered across the duration with silence gaps for VAD.
    n_phrases = max(2, int(duration_s / 2.0))
    starts = [float(duration_s * (i + 0.5) / (n_phrases + 1))
              for i in range(n_phrases)]
    for i, s0 in enumerate(starts):
        dur = rng.uniform(0.5, 1.1)
        f0 = float(rng.uniform(120, 500))
        f1 = float(rng.uniform(600, 3200))
        m = int(sr * dur)
        s1 = min(n, int(s0 * sr) + m)
        if s1 - int(s0 * sr) < 8:
            continue
        tt = np.arange(s1 - int(s0 * sr)) / sr
        if rng.random() < 0.5:
            base = chirp(tt, f0=f0, f1=f1, t1=tt[-1] if len(tt) else 1.0, method="logarithmic")
        else:
            base = chirp(tt, f0=f1, f1=f0, t1=tt[-1] if len(tt) else 1.0, method="linear")
        env = np.exp(-2.0 * tt / (tt[-1] + 1e-9)) * np.minimum(1.0, tt / 0.05)
        seg = base * env + 0.35 * np.sin(2 * np.pi * 2 * f0 * tt + 1.0) * env
        x[int(s0 * sr):s1] += 0.6 * seg
    # Ocean background: pinkish noise + low rumble + a couple of hiss bursts.
    x += 0.05 * rng.standard_normal(n)
    x += 0.08 * np.sin(2 * np.pi * 30.0 * t + 0.3)
    for b0 in (duration_s * 0.33, duration_s * 0.8):
        if b0 >= duration_s:
            continue
        m = int(0.3 * sr)
        s1 = min(n, int(b0 * sr) + m)
        if s1 > int(b0 * sr):
            x[int(b0 * sr):s1] += 0.12 * rng.standard_normal(s1 - int(b0 * sr))
    x /= max(1e-9, np.abs(x).max())
    wavfile.write(path, sr, (0.9 * x * 32767).astype(np.int16))
    return path


# ----------------------------------------------------------------------------
# Denoise: Butterworth bandpass + stationary spectral gating
# ----------------------------------------------------------------------------

def bandpass(x, sr, low=DEFAULT_LOW, high=DEFAULT_HIGH, order=4):
    from scipy.signal import butter, sosfiltfilt
    sos = butter(order, [low, high], btype="bandpass", fs=sr, output="sos")
    return sosfiltfilt(sos, x).astype(np.float32)


def spectral_gate_denoise(x, sr, n_fft=2048, hop=512, prop_decrease=0.9,
                          thresh_mult=1.5, profile_s=0.75,
                          noise_profile="flat"):
    """Stationary spectral gating (noisereduce-style, hand-rolled with scipy).

    noise_profile="sea" weights the gate by a Wenz-like ambient prior:
    extra suppression of low shipping rumble and high wind hiss, so the
    gate bites harder where the ocean is loudest.
    """
    from scipy.signal import stft, istft
    from scipy.ndimage import uniform_filter1d
    _, _, Z = stft(x, fs=sr, window="hann", nperseg=n_fft,
                   noverlap=n_fft - hop, nfft=n_fft, boundary="zeros")
    mag = np.abs(Z)
    phase = np.exp(1j * np.angle(Z))
    energy = (mag ** 2).mean(axis=0)
    n_prof = max(1, int(profile_s * sr / hop))
    prof_idx = np.argsort(energy)[:n_prof]
    noise = mag[:, prof_idx].mean(axis=1, keepdims=True)
    if str(noise_profile).lower() == "sea":
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)[:, None]
        w = (1.0 + 0.8 * np.exp(-freqs / 120.0)
             + 0.6 * (freqs / 4000.0) ** 2)
        noise = noise * w
    thresh = noise * thresh_mult
    gain = np.clip((mag - thresh) / (mag + 1e-10), 0.0, 1.0)
    gain = uniform_filter1d(uniform_filter1d(gain, 3, axis=0), 5, axis=1,
                            mode="nearest")
    mag_d = mag * ((1.0 - prop_decrease) + prop_decrease * gain)
    _, x_d = istft(mag_d * phase, fs=sr, window="hann", nperseg=n_fft,
                   noverlap=n_fft - hop, nfft=n_fft)
    x_d = x_d[:len(x)]
    return x_d.astype(np.float32)


def denoise_backend_available(backend="gating"):
    if str(backend).lower() == "noisereduce":
        try:
            import noisereduce  # noqa: F401
            return True
        except Exception:
            return False
    return True


def denoise(x, sr, backend="gating", noise_profile="flat", **kwargs):
    """Denoise dispatcher: hand-rolled gating (default) or noisereduce.

    backend="noisereduce" tries the ``noisereduce`` package (non-stationary
    mode, explicit noise profile) and falls back to the hand-rolled
    stationary gate when the package is absent, so scipy-only installs
    keep working. noise_profile="sea" weights the hand-rolled gate by a
    Wenz-like ambient prior. Returns (audio, used_fallback_bool).
    """
    if str(backend).lower() == "noisereduce":
        try:
            import noisereduce as nr
            out = nr.reduce_noise(y=np.asarray(x, dtype=np.float32), sr=sr,
                                  stationary=False).astype(np.float32)
            return out, False
        except Exception:
            pass
        return spectral_gate_denoise(x, sr,
                                     noise_profile=noise_profile,
                                     **kwargs), True
    return spectral_gate_denoise(x, sr, noise_profile=noise_profile,
                                 **kwargs), False


# ----------------------------------------------------------------------------
# VAD: adaptive band-limited energy gate
# ----------------------------------------------------------------------------

def _frame_harmonicity(frames, sr, fmin=40.0, fmax=1000.0,
                         chunk=2048):
    """Voicing strength per frame via FFT autocorrelation peak (0..~1).

    Tonal/FM whale units correlate strongly at pitch lags; broadband
    boat/wind noise does not. Chunked so long songs stay in memory.
    """
    n = frames.shape[1]
    lo = max(2, int(sr / fmax))
    hi = min(n // 2 - 1, int(sr / fmin))
    out = np.zeros(frames.shape[0], dtype=np.float64)
    for s in range(0, frames.shape[0], chunk):
        f = frames[s:s + chunk].astype(np.float64)
        f = f - f.mean(axis=1, keepdims=True)
        F = np.fft.rfft(f, n=4096, axis=1)
        ac = np.fft.irfft(np.abs(F) ** 2, n=4096, axis=1)[:, :n // 2]
        ac /= (ac[:, :1] + 1e-12)
        out[s:s + chunk] = ac[:, lo:hi + 1].max(axis=1)
    return np.clip(out, 0.0, 1.0)


def vad_segments(x, sr, frame_len=2048, hop=512, offset_db=7.0,
                 medfilt_k=7, hangover_s=0.25, min_unit_s=0.25,
                 merge_gap_s=0.20, harmonic_gate=True, harm_thresh=0.30,
                 harm_bypass_db=6.0):
    from scipy.signal import medfilt
    n_frames = 1 + max(0, (len(x) - frame_len) // hop)
    if n_frames < 1:
        return []
    idx = (np.arange(n_frames) * hop)[:, None] + np.arange(frame_len)[None, :]
    frames = x[idx]
    rms = np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1) + 1e-12)
    db = 20.0 * np.log10(rms)
    floor = np.percentile(db, 30)
    dec = (db > floor + offset_db).astype(np.float32)
    if harmonic_gate:
        # Two-stage gate: keep loud frames regardless; quiet frames must
        # also look voiced (tonal/FM), which rejects boat/wind hiss while
        # keeping faint song units.
        harm = _frame_harmonicity(frames, sr)
        loud = db > floor + offset_db + harm_bypass_db
        dec = (dec.astype(bool) & ((harm > harm_thresh) | loud)
               ).astype(np.float32)
    k = int(medfilt_k) if int(medfilt_k) % 2 == 1 else int(medfilt_k) + 1
    if len(dec) >= k:
        dec = medfilt(dec, kernel_size=k)
    dec = (dec > 0.5)
    # Hangover: dilate speech regions by hangover frames.
    h = max(1, int(hangover_s * sr / hop))
    dil = np.convolve(dec.astype(np.float32), np.ones(h, dtype=np.float32),
                      mode="same") > 0
    # Contiguous regions -> (start, end) in seconds.
    bounds, s = [], None
    for i, v in enumerate(dil):
        if v and s is None:
            s = i
        elif not v and s is not None:
            bounds.append((s, i - 1))
            s = None
    if s is not None:
        bounds.append((s, len(dil) - 1))
    segs = []
    for a, b in bounds:
        t0, t1 = a * hop / sr, min(len(x) / sr, (b * hop + frame_len) / sr)
        if t1 - t0 >= min_unit_s:
            segs.append([t0, t1])
    # Merge gaps shorter than merge_gap_s.
    merged = []
    for s0, s1 in segs:
        if merged and s0 - merged[-1][1] < merge_gap_s:
            merged[-1][1] = s1
        else:
            merged.append([s0, s1])
    return merged


# ----------------------------------------------------------------------------
# 128-bin log-mel (torchaudio/librosa optional accelerators, scipy fallback)
# ----------------------------------------------------------------------------

def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def _mel_filterbank(sr, n_fft, n_mels, fmin, fmax):
    n_bins = n_fft // 2 + 1
    mel_pts = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2)
    hz_pts = _mel_to_hz(mel_pts)
    bin_pts = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
    fb = np.zeros((n_mels, n_bins), dtype=np.float64)
    for m in range(1, n_mels + 1):
        f0, f1, f2 = bin_pts[m - 1], bin_pts[m], bin_pts[m + 1]
        if f1 > f0:
            fb[m - 1, f0:f1] = (np.arange(f0, f1) - f0) / max(1, f1 - f0)
        if f2 > f1:
            fb[m - 1, f1:f2] = (f2 - np.arange(f1, f2)) / max(1, f2 - f1)
    return fb


def _hz_to_erb(f):
    return 24.7 * (4.37 * np.asarray(f, dtype=np.float64) / 1000.0 + 1.0)


def _erb_to_hz(e):
    return (np.asarray(e, dtype=np.float64) / 24.7 - 1.0) * 1000.0 / 4.37


def _gammatone_filterbank(sr, n_fft, n_bands, fmin, fmax):
    """ERB-spaced asymmetric (cochlea-like) filterbank energies.

    Magnitude approximation of 4th-order gammatone filters: Gaussian
    skirts with a narrower high side, bandwidth = ERB(fc). Finer low
    resolution than mel triangles — better for low baleen moans.
    """
    n_bins = n_fft // 2 + 1
    freqs = np.linspace(0, sr / 2.0, n_bins)
    erb_pts = np.linspace(_hz_to_erb(fmin), _hz_to_erb(fmax), n_bands)
    fc = _erb_to_hz(erb_pts)
    bw = np.maximum(20.0, _hz_to_erb(fc))
    fb = np.zeros((n_bands, n_bins), dtype=np.float64)
    for i in range(n_bands):
        d = freqs - fc[i]
        sigma = np.where(d < 0, 0.6 * bw[i], bw[i])
        fb[i] = np.exp(-0.5 * (d / sigma) ** 2)
        fb[i] /= max(fb[i].sum(), 1e-12)
    return fb


def pcen_frontend(mel_power, sr=DEFAULT_SR, hop_ms=10.0, eps=1e-6,
                    s=0.05, alpha=0.98, delta=2.0, r=0.5, tau=0.06):
    """Simple per-channel energy normalization (PCEN-lite, numpy only).

    mel_power: (n_mels, T) linear-power mel energies. Returns PCEN values
    in a roughly log-like range without needing librosa. Defaults follow
    Lostanlen bioacoustics practice (tau=0.06 s AGC, alpha~0.98).
    tau may be a scalar or a per-channel array (longer smoothing for
    low-frequency channels, which carry slow baleen moans).
    """
    hop_s = hop_ms / 1000.0
    tau_arr = np.asarray(tau, dtype=np.float64)
    coeff = 1.0 - np.exp(-hop_s / np.maximum(1e-6, tau_arr))
    M = np.zeros_like(mel_power, dtype=np.float64)
    run = mel_power[:, 0].astype(np.float64)
    for t in range(mel_power.shape[1]):
        run = (1.0 - coeff) * run + coeff * mel_power[:, t].astype(np.float64)
        M[:, t] = run
    smooth = s * M + eps
    return ((mel_power / smooth) ** alpha + delta) ** r - delta ** r


def _linear_mel_power(x, sr, n_mels=DEFAULT_N_MELS, n_fft=2048,
                      win_ms=25.0, hop_ms=10.0, fmin=DEFAULT_LOW,
                      fmax=DEFAULT_HIGH, bank="mel"):
    hop = max(1, int(sr * hop_ms / 1000.0))
    win = max(16, int(sr * win_ms / 1000.0))
    from scipy.signal import stft
    _, _, Z = stft(x, fs=sr, window="hann", nperseg=win,
                   noverlap=win - hop, nfft=n_fft, boundary="zeros")
    power = np.abs(Z) ** 2
    fb = _cached_filterbank(bank, sr, n_fft, n_mels, fmin, fmax)
    return (fb @ power).astype(np.float64)


def add_deltas(mel, width=9):
    """Append delta + delta-delta channels (HTK-style simple differences)."""
    w = max(3, int(width) | 1)
    half = w // 2
    denom = 2.0 * sum(i * i for i in range(1, half + 1))
    def _delta(feat):
        padded = np.pad(feat, ((0, 0), (half, half)), mode="edge")
        d = np.zeros_like(feat, dtype=np.float64)
        for i in range(1, half + 1):
            d += i * (padded[:, half + i:half + i + feat.shape[1]]
                      - padded[:, half - i:half - i + feat.shape[1]])
        return d / max(denom, 1e-9)
    d1 = _delta(mel.astype(np.float64))
    d2 = _delta(d1)
    return np.concatenate([mel, d1.astype(np.float32),
                           d2.astype(np.float32)], axis=0)


def log_mel(x, sr, n_mels=DEFAULT_N_MELS, n_fft=2048,
            win_ms=25.0, hop_ms=10.0, fmin=DEFAULT_LOW, fmax=DEFAULT_HIGH,
            use_pcen=False, use_deltas=False, pcen_tau=0.06, pcen_s=0.05,
            pcen_alpha=0.98, frontend="mel"):
    """Return (n_mels[, x3], T) mel/gammatone spectrogram, ~100 fps.

    frontend="gammatone" uses an ERB-spaced asymmetric filterbank
    (cochlea-like, finer low-frequency resolution for baleen moans) and
    forces the dependency-free scipy path.
    """
    hop = max(1, int(sr * hop_ms / 1000.0))
    win = max(16, int(sr * win_ms / 1000.0))
    if frontend == "gammatone":
        # Longer AGC smoothing on low channels (slow moans) — per-band tau.
        tau_b = pcen_tau * np.linspace(1.5, 0.8, n_mels)
        lin = _linear_mel_power(x, sr, n_mels, n_fft, win_ms, hop_ms,
                                fmin, fmax, bank="gammatone")
        if use_pcen:
            base = pcen_frontend(lin, sr=sr, hop_ms=hop_ms, tau=tau_b,
                                 s=pcen_s, alpha=pcen_alpha).astype(np.float32)
        else:
            base = np.log(lin + 1e-10).astype(np.float32)
        return add_deltas(base) if use_deltas else base
    if use_pcen:
        # PCEN needs linear power; use the dependency-free scipy path so
        # results are identical with or without torchaudio/librosa.
        lin = _linear_mel_power(x, sr, n_mels, n_fft, win_ms, hop_ms,
                                fmin, fmax)
        base = pcen_frontend(lin, sr=sr, hop_ms=hop_ms, tau=pcen_tau,
                             s=pcen_s, alpha=pcen_alpha).astype(np.float32)
        return add_deltas(base) if use_deltas else base
    # Optional torchaudio fast path.
    try:
        import torch
        import torchaudio
        wav = torch.from_numpy(np.asarray(x, dtype=np.float32)).unsqueeze(0)
        mel_t = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=n_fft, win_length=win, hop_length=hop,
            n_mels=n_mels, f_min=fmin, f_max=fmax, power=2.0)(wav)
        base = torch.log(mel_t.squeeze(0) + 1e-10).numpy().astype(np.float32)
        return add_deltas(base) if use_deltas else base
    except Exception:
        pass
    # Optional librosa path.
    try:
        import librosa
        S = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop,
                                win_length=win, window="hann")) ** 2
        mel = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels,
                                  fmin=fmin, fmax=fmax)
        base = np.log(mel @ S + 1e-10).astype(np.float32)
        return add_deltas(base) if use_deltas else base
    except Exception:
        pass
    # Scipy fallback: STFT power + hand-rolled filterbank.
    from scipy.signal import stft
    _, _, Z = stft(x, fs=sr, window="hann", nperseg=win,
                   noverlap=win - hop, nfft=n_fft, boundary="zeros")
    power = np.abs(Z) ** 2
    if frontend == "gammatone":
        fb = _gammatone_filterbank(sr, n_fft, n_mels, fmin, fmax)
    else:
        fb = _mel_filterbank(sr, n_fft, n_mels, fmin, fmax)
    base = np.log(fb @ power + 1e-10).astype(np.float32)
    return add_deltas(base) if use_deltas else base


# ----------------------------------------------------------------------------
# VQ-VAE (K=64): conv encoder/decoder, STE, commitment loss, optional EMA
# ----------------------------------------------------------------------------

def _get_torch():
    import torch
    import torch.nn as nn
    return torch, nn


class VQVAE:
    """Conv1d VQ-VAE over log-mel frames. Downsamples time by 4."""

    def __init__(self, n_mels=DEFAULT_N_MELS, K=DEFAULT_K, dim=64,
                 beta=0.25, ema=True, ema_decay=0.99, seed=0):
        torch, nn = _get_torch()
        self.torch = torch
        self.K, self.dim, self.beta = K, dim, beta
        self.ema = ema
        g = torch.Generator().manual_seed(seed)

        def conv(c_in, c_out):
            m = nn.Conv1d(c_in, c_out, 4, 2, 1)
            nn.init.xavier_uniform_(m.weight, generator=g)
            nn.init.zeros_(m.bias)
            return m

        def deconv(c_in, c_out):
            m = nn.ConvTranspose1d(c_in, c_out, 4, 2, 1)
            nn.init.xavier_uniform_(m.weight, generator=g)
            nn.init.zeros_(m.bias)
            return m

        self.enc = nn.Sequential(conv(n_mels, 256), nn.ReLU(),
                                 conv(256, dim), nn.ReLU())
        self.dec = nn.Sequential(deconv(dim, 256), nn.ReLU(),
                                 deconv(256, n_mels))
        bound = 1.0 / K
        self.codebook = nn.Embedding(K, dim)
        nn.init.uniform_(self.codebook.weight, -bound, bound, generator=g)
        if ema:
            self.register = True
            self.cluster = torch.zeros(K)
            self.emb_avg = self.codebook.weight.data.clone()
            self.ema_decay = ema_decay

    def parameters(self):
        ps = list(self.enc.parameters()) + list(self.dec.parameters())
        if not self.ema:
            ps += list(self.codebook.parameters())
        return ps

    def quantize(self, ze):
        """ze: (B, dim, T) -> indices (B, T), quantized (B, dim, T)."""
        torch = self.torch
        B, _, T = ze.shape
        flat = ze.permute(0, 2, 1).reshape(-1, self.dim)
        d = (flat ** 2).sum(1, keepdim=True) - 2 * flat @ self.codebook.weight.t() \
            + (self.codebook.weight ** 2).sum(1)
        idx = d.argmin(1)
        zq = self.codebook(idx).view(B, T, self.dim).permute(0, 2, 1)
        return idx.view(B, T), zq

    def forward(self, x):
        ze = self.enc(x)
        idx, zq = self.quantize(ze)
        zq_ste = ze + (zq - ze).detach()
        recon = self.dec(zq_ste)
        return recon, ze, zq, idx

    def encode_tokens(self, mel):
        """mel: (n_mels, T) -> list of code indices (time/4 resolution)."""
        torch = self.torch
        self.enc.eval()
        with torch.no_grad():
            T = mel.shape[1]
            pad = (-T) % 4
            if pad:
                mel = np.concatenate(
                    [mel, np.repeat(mel[:, -1:], pad, axis=1)], axis=1)
            x = torch.from_numpy(mel[None].astype(np.float32))
            ze = self.enc(x)
            idx, _ = self.quantize(ze)
            return idx[0].tolist()


def _mel_windows(mels, win=32, stride=16):
    wins = []
    for m in mels:
        T = m.shape[1]
        if T < win:
            pad = win - T
            m = np.concatenate([m, np.repeat(m[:, -1:], pad, axis=1)], axis=1)
            T = win
        for s in range(0, T - win + 1, stride):
            wins.append(m[:, s:s + win])
    return wins


def train_vqvae(mels, K=DEFAULT_K, epochs=5, beta=0.25, ema=True,
                batch_size=32, lr=1e-3, seed=0, n_mels=None):
    torch, nn = _get_torch()
    set_seed(seed)
    n_mels = int(n_mels or mels[0].shape[0])
    model = VQVAE(n_mels=n_mels, K=K, beta=beta, ema=ema, seed=seed)
    wins = _mel_windows(mels)
    if not wins:
        raise ValueError("no mel frames to train VQ-VAE on")
    data = np.stack(wins).astype(np.float32)
    mu, sd = data.mean(), data.std() + 1e-6
    data = (data - mu) / sd
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = []
    n = len(data)
    for ep in range(max(1, epochs)):
        perm = np.random.permutation(n)
        tot, nb = 0.0, 0
        for s in range(0, n, batch_size):
            b = torch.from_numpy(data[perm[s:s + batch_size]])
            recon, ze, zq, idx = model.forward(b)
            loss_recon = ((recon - b) ** 2).mean()
            loss_commit = ((ze - zq.detach()) ** 2).mean()
            if model.ema:
                loss = loss_recon + beta * loss_commit
            else:
                loss_embed = ((ze.detach() - zq) ** 2).mean()
                loss = loss_recon + loss_embed + beta * loss_commit
            opt.zero_grad()
            loss.backward()
            opt.step()
            if model.ema:
                with torch.no_grad():
                    flat = ze.detach().permute(0, 2, 1).reshape(-1, model.dim)
                    oh = torch.zeros(flat.shape[0], model.K)
                    oh.scatter_(1, idx.view(-1, 1), 1.0)
                    d = model.ema_decay
                    model.cluster.mul_(d).add_(oh.sum(0), alpha=1 - d)
                    model.emb_avg.mul_(d).add_(oh.t() @ flat, alpha=1 - d)
                    norm = model.cluster + 1e-5
                    model.codebook.weight.data.copy_(model.emb_avg / norm[:, None])
            tot += float(loss.item()) * len(b)
            nb += len(b)
        # Dead-code revival: reseed codes the encoder stopped using so
        # one code cannot swallow the whole song (collapse seen on real
        # humpback audio with plain EMA).
        with torch.no_grad():
            use = model.cluster / (model.cluster.sum() + 1e-9)
            dead = torch.where(use * model.K < 1.0)[0]
            if len(dead):
                pick = flat[torch.randint(0, flat.shape[0], (len(dead),))]
                jitter = 0.01 * torch.randn_like(pick)
                model.codebook.weight.data[dead] = pick + jitter
                model.cluster[dead] = 1.0
                model.emb_avg[dead] = model.codebook.weight.data[dead]
        hist.append(tot / max(1, nb))
    stats = {"loss_history": hist, "mean": float(mu), "std": float(sd)}
    return model, stats


class KMeansUnits:
    """Zero-training baseline: MiniBatchKMeans-64 on standardized mel frames."""

    def __init__(self, K=DEFAULT_K, seed=0):
        self.K = K
        self.seed = seed
        self.km = None
        self.mean = 0.0
        self.std = 1.0

    def fit(self, mels):
        from sklearn.cluster import MiniBatchKMeans
        frames = np.concatenate([m.T for m in mels], axis=0)
        self.mean = float(frames.mean())
        self.std = float(frames.std() + 1e-6)
        normed = (frames - self.mean) / self.std
        self.km = MiniBatchKMeans(n_clusters=self.K, n_init=3,
                                 batch_size=1024, random_state=self.seed)
        self.km.fit(normed)
        return {"loss_history": [0.0],
                "mean": self.mean, "std": self.std}

    def encode_tokens(self, mel):
        labels = self.km.predict(((mel.T - self.mean) / self.std))
        # Match VQ-VAE time resolution (stride 4) so LM block sizes compare.
        return dedup([int(t) for t in labels[::4]])


class FSQUnits:
    """Zero-training FSQ-lite: fixed 8x8 grid (d=2, product 64, no collapse).

    Projects standardized mel frames to 2 dims with a seeded random matrix,
    squashes with tanh, and quantizes each dim to 8 levels. Mixed-radix index
    gives exactly K=64 implicit codes with ~100% utilization pressure.
    """

    LEVELS = (8, 8)

    def __init__(self, K=DEFAULT_K, seed=0):
        if int(K) != 64:
            raise ValueError("FSQ-lite maps exactly to K=64 (8x8 levels)")
        self.K = 64
        self.seed = seed
        self.proj = None
        self.mean = 0.0
        self.std = 1.0

    def fit(self, mels):
        rng = np.random.default_rng(self.seed)
        frames = np.concatenate([m.T for m in mels], axis=0)
        self.mean = float(frames.mean())
        self.std = float(frames.std() + 1e-6)
        n_feat = frames.shape[1]
        self.proj = rng.standard_normal((n_feat, 2)) / math.sqrt(n_feat)
        return {"loss_history": [0.0],
                "mean": self.mean, "std": self.std}

    def _quantize_frames(self, mel):
        z = ((mel.T - self.mean) / self.std) @ self.proj  # (T, 2)
        z = np.tanh(z / 2.0)  # (-1, 1)
        toks = np.zeros(z.shape[0], dtype=int)
        for d, levels in enumerate(self.LEVELS):
            q = np.clip(((z[:, d] + 1.0) / 2.0 * levels).astype(int),
                        0, levels - 1)
            toks = toks * levels + q
        return toks

    def encode_tokens(self, mel):
        return dedup([int(t) for t in self._quantize_frames(mel)[::4]])

    def codebook_matrix(self):
        grid = np.array([[i, j] for i in range(8) for j in range(8)],
                        dtype=np.float64)
        return (grid / 7.0 * 2.0 - 1.0).tolist()


class SSLEmbeddingUnits:
    """Pretrained bioacoustics frontend + k-means (optional weights).

    When ``weights_path`` points at local AVES/Animal2Vec-style weights
    and a loader (timm/transformers) is installed, embeddings come from
    the encoder; otherwise the backend falls back to stride-4 mean-pooled
    mel with ``used_fallback=True`` recorded in the run metadata, so the
    option stays runnable offline and the token cadence stays identical
    to VQ-VAE/KMeans/FSQ (dedup + stride 4).
    """

    def __init__(self, K=DEFAULT_K, seed=0, weights_path=None):
        self.K = int(K)
        self.seed = seed
        self.weights_path = weights_path
        self.used_fallback = weights_path is None
        self.km = None
        self.mean = 0.0
        self.std = 1.0

    def _embed(self, mels):
        import torch
        if self.weights_path:
            try:
                import timm
                enc = timm.create_model(
                    "aves", pretrained=False, num_classes=0,
                    checkpoint_path=self.weights_path)
                enc.eval()
                feats = []
                with torch.no_grad():
                    for m in mels:
                        t = torch.from_numpy(
                            np.asarray(m, dtype=np.float32)).unsqueeze(0)
                        feats.append(enc(t).squeeze(0))
                self.used_fallback = False
                return torch.cat(feats, dim=0).numpy()
            except Exception:
                self.used_fallback = True
        feats = []
        for m in mels:
            t = torch.from_numpy(np.asarray(m, dtype=np.float32))
            T = t.shape[1]
            pad = (-T) % 4
            if pad:
                t = torch.cat([t, t[:, -1:].repeat(1, pad)], dim=1)
            feats.append(t.reshape(t.shape[0], -1, 4).mean(-1))
        return torch.cat(feats, dim=1).t().numpy()

    def fit(self, mels):
        from sklearn.cluster import MiniBatchKMeans
        frames = self._embed(mels)
        self.mean = float(frames.mean())
        self.std = float(frames.std() + 1e-6)
        normed = (frames - self.mean) / self.std
        self.km = MiniBatchKMeans(n_clusters=self.K, n_init=3,
                                 batch_size=1024, random_state=self.seed)
        self.km.fit(normed)
        return {"loss_history": [0.0],
                "mean": self.mean, "std": self.std}

    def encode_tokens(self, mel):
        import torch
        t = torch.from_numpy(np.asarray(mel, dtype=np.float32))
        T = t.shape[1]
        pad = (-T) % 4
        if pad:
            t = torch.cat([t, t[:, -1:].repeat(1, pad)], dim=1)
        pooled = t.reshape(t.shape[0], -1, 4).mean(-1).t().numpy()
        labels = self.km.predict((pooled - self.mean) / self.std)
        return dedup([int(x) for x in labels])


# ----------------------------------------------------------------------------
# Token sequences, bigram baseline, Shannon entropy
# ----------------------------------------------------------------------------

def dedup(seq):
    out = []
    for t in seq:
        if not out or t != out[-1]:
            out.append(int(t))
    return out


def split_sequences(seqs, val_frac=0.2):
    n = len(seqs)
    n_val = max(1 if n > 1 else 0, int(round(n * val_frac)))
    if n_val >= n:
        n_val = n - 1 if n > 1 else 0
    return seqs[:n - n_val] if n_val else seqs, seqs[n - n_val:] if n_val else []


def bigram_stats(seqs, K):
    C = np.zeros((K, K), dtype=np.float64)
    for s in seqs:
        for a, b in zip(s[:-1], s[1:]):
            C[a, b] += 1.0
    C += 1.0  # Laplace smoothing, alpha=1
    P = C / C.sum(axis=1, keepdims=True)
    return C, P


def seq_nll_bigram(seqs, P):
    nll, n = 0.0, 0
    for s in seqs:
        for a, b in zip(s[:-1], s[1:]):
            nll -= math.log(max(P[a, b], 1e-12))
            n += 1
    return nll / max(1, n)


def unigram_entropy_bits(seqs, K):
    counts = np.zeros(K, dtype=np.float64)
    for s in seqs:
        for t in s:
            counts[t] += 1.0
    p = counts / counts.sum()
    p = p[p > 0]
    h_bits = float(-(p * np.log2(p)).sum())
    return h_bits, math.exp(h_bits * math.log(2.0))


def codebook_utilization(seqs, K):
    used = np.zeros(K, dtype=bool)
    for s in seqs:
        for t in s:
            used[t] = True
    n_used = int(used.sum())
    counts = np.zeros(K)
    for s in seqs:
        for t in s:
            counts[t] += 1
    p = counts / max(1, counts.sum())
    p = p[p > 0]
    perp = float(np.exp(-(p * np.log(p)).sum()))
    return n_used, K - n_used, perp


def trigram_nll_gap(seqs, K, alpha=1.0):
    """1st-vs-2nd-order Markov gap: bigram NLL minus trigram NLL (nats).

    Positive gap = second-order context helps = evidence for structure
    beyond adjacent pairs (Suzuki et al. motivation for moving past
    first-order models). Laplace-smoothed counts, same train data.
    """
    bigr = {}
    tri = {}
    for s in seqs:
        for a, b in zip(s[:-1], s[1:]):
            bigr[(a, b)] = bigr.get((a, b), 0.0) + 1.0
        for a, b, c in zip(s[:-2], s[1:-1], s[2:]):
            tri[(a, b, c)] = tri.get((a, b, c), 0.0) + 1.0
    nll1 = nll2 = n = 0.0
    for s in seqs:
        for i in range(1, len(s)):
            a, b = s[i - 1], s[i]
            tot1 = sum(bigr.get((a, x), 0.0) for x in range(K)) + alpha * K
            nll1 -= math.log((bigr.get((a, b), 0.0) + alpha) / tot1)
            if i >= 2:
                ctx = (s[i - 2], s[i - 1])
                tot2 = sum(tri.get((ctx[0], ctx[1], x), 0.0)
                           for x in range(K)) + alpha * K
                nll2 -= math.log((tri.get((ctx[0], ctx[1], b), 0.0)
                                  + alpha) / tot2)
            n += 1
    n = max(1, n)
    return nll1 / n - nll2 / n


def zipf_slope(seqs, K):
    """Zipf slope of ranked token frequencies (log-log linear fit).

    Returns (slope, r_squared). Slopes near -1 resemble natural-language
    word distributions; reported per recording for cross-year comparison.
    """
    counts = np.zeros(K, dtype=np.float64)
    for s in seqs:
        for t in s:
            counts[t] += 1.0
    counts = counts[counts > 0]
    if len(counts) < 3:
        return 0.0, 0.0
    ranks = np.arange(1, len(counts) + 1)
    freq = np.sort(counts)[::-1] / counts.sum()
    x = np.log(ranks)
    y = np.log(np.maximum(freq, 1e-12))
    slope = float(np.polyfit(x, y, 1)[0])
    r2 = float(np.corrcoef(x, y)[0, 1] ** 2)
    return slope, r2


def find_phrases(seqs, min_len=3, min_repeats=2):
    """Repeat-sequence phrase grouping over deduped token sequences.

    Finds token n-grams (len>=min_len) occurring >=min_repeats times
    across segments — the phrase layer above Payne & McVay units.
    Returns [{phrase:[ids], count, segments:[ids]}] sorted by count.
    """
    loc = {}
    for si, s in enumerate(seqs):
        for L in range(min_len, min(len(s), 12) + 1):
            for i in range(len(s) - L + 1):
                key = tuple(s[i:i + L])
                loc.setdefault(key, set()).add(si)
    out = [{"phrase": list(k), "count": len(v), "segments": sorted(v)}
           for k, v in loc.items() if len(v) >= min_repeats]
    # Drop phrases fully contained in a longer phrase with equal count.
    out.sort(key=lambda d: (-d["count"], -len(d["phrase"])))
    kept = []
    for d in out:
        if not any(set(d["phrase"]) <= set(o["phrase"]) and
                   d["count"] == o["count"] for o in kept):
            kept.append(d)
    return kept[:64]


def shuffled_control_gap(seqs, K, alpha=1.0, seed=0):
    """GmSLM-style zero-shot check: real vs time-shuffled NLL gap (nats).

    Shuffles tokens inside each segment (fixed seed), scores both with the
    same Laplace bigram model. Positive gap = order carries syntax beyond
    unigram frequency. Returns float gap (real_shuffled_NLL - real_NLL
    sign flipped so positive means structure).
    """
    rng = np.random.default_rng(seed)
    shuf = []
    for s in seqs:
        s = list(s)
        rng.shuffle(s)
        shuf.append(s)
    _, P = bigram_stats(seqs or shuf, K)
    nll_real = seq_nll_bigram(seqs, P)
    nll_shuf = seq_nll_bigram(shuf, P)
    return float(nll_shuf - nll_real)


def find_themes(P, K=None, seed=0):
    """Theme layer above phrases: Louvain communities on the bigram graph.

    Symmetrizes the KxK transition matrix into an undirected weighted
    graph and runs Louvain modularity optimization (Blondel et al. 2008):
    themes are groups of units that transition densely among themselves.
    Returns [{theme_id, codes:[ids], size}] sorted by size.
    """
    P = np.asarray(P, dtype=np.float64)
    K = int(K or P.shape[0])
    W = P + P.T
    np.fill_diagonal(W, 0.0)
    thr = float(np.quantile(W, 0.5))
    W = np.where(W >= thr, W, 0.0)
    rng = np.random.default_rng(seed)
    comm = np.arange(K)
    deg = W.sum(axis=1)
    m2 = float(deg.sum())
    if m2 <= 0:
        return [{"theme_id": 0, "codes": list(range(K)), "size": K}]
    tot = {}
    for c in range(K):
        tot[c] = float(deg[comm == c].sum())
    for _ in range(10):
        moved = False
        for i in rng.permutation(K):
            ci = int(comm[i])
            tot[ci] -= float(deg[i])
            neigh = set(int(c) for c in comm[W[i] > 0])
            neigh.add(ci)
            best_c, best_g = ci, -1e18
            for c in neigh:
                kin = float(W[i][comm == c].sum())
                g = kin - tot.get(c, 0.0) * float(deg[i]) / m2
                if g > best_g:
                    best_g, best_c = g, c
            comm[i] = best_c
            tot[best_c] = tot.get(best_c, 0.0) + float(deg[i])
            if best_c != ci:
                moved = True
        if not moved:
            break
    groups = {}
    for i in range(K):
        groups.setdefault(int(comm[i]), []).append(i)
    if len(groups) == 1:
        return [{"theme_id": 0, "codes": sorted(next(iter(groups.values()))),
                 "size": K}]
    return [{"theme_id": i, "codes": sorted(v), "size": len(v)}
            for i, v in enumerate(sorted(groups.values(), key=len,
                                        reverse=True))]


def code_to_theme_map(themes):
    m = {}
    for t in themes:
        for c in t["codes"]:
            m[int(c)] = int(t["theme_id"])
    return m


def levenshtein(a, b):
    a, b = list(a), list(b)
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (0 if ca == cb else 1))
        prev = cur
    return int(prev[-1])


def needleman_wunsch_sim(a, b, match=2.0, mismatch=-1.0, gap=-1.0):
    """Global-alignment similarity in [0, 1] (Needleman-Wunsch, 1970).

    Rewards shared ordered motifs (Garland transmission idea) instead of
    raw edit counts: two songs sharing whole phrases score higher than
    two sharing scattered units. Normalized by the perfect-match score.
    """
    a = [int(t) for t in a][:512]
    b = [int(t) for t in b][:512]
    if not a or not b:
        return 1.0 if len(a) == len(b) else 0.0
    prev = [i * gap for i in range(len(b) + 1)]
    for x in a:
        cur = [(0.0)] * (len(b) + 1)
        cur[0] = prev[0] + gap
        for j, y in enumerate(b, 1):
            sub = prev[j - 1] + (match if x == y else mismatch)
            cur[j] = max(sub, prev[j] + gap, cur[j - 1] + gap)
        prev = cur
    best = match * max(len(a), len(b))
    return float(max(0.0, min(1.0, prev[len(b)] / best)))


def population_similarity(all_seqs):
    """Cross-file similarity matrix (Garland-style comparison).

    all_seqs: list of token-sequence lists (one entry per file, flattened
    per file). Uses Needleman-Wunsch global alignment so shared ordered
    motifs count more than scattered matches. Returns symmetric matrix.
    """
    flat = []
    for seqs in all_seqs:
        toks = []
        for s in seqs:
            toks.extend(int(t) for t in s)
        flat.append(toks)
    n = len(flat)
    M = np.eye(n, dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            sim = needleman_wunsch_sim(flat[i], flat[j])
            M[i, j] = M[j, i] = float(sim)
    return M.tolist()


def change_points(xs, penalty=2.0, min_size=2):
    """Binary-segmentation change points (PELT-style, L2 cost + penalty).

    Returns sorted indices where the series mean shifts. Used for
    revolution detection on per-file complexity instead of a fixed dip.
    """
    import numpy as _np
    xs = [float(v) for v in xs]
    n = len(xs)
    if n < 2 * min_size:
        return []

    def cost(s, e):
        seg = xs[s:e]
        mu = sum(seg) / len(seg)
        return sum((v - mu) ** 2 for v in seg)

    cps, stack = [], [(0, n)]
    while stack:
        s, e = stack.pop()
        if e - s < 2 * min_size:
            continue
        base = cost(s, e)
        best_k, best_gain = -1, penalty
        for k in range(s + min_size, e - min_size + 1):
            gain = base - (cost(s, k) + cost(k, e))
            if gain > best_gain:
                best_gain, best_k = gain, k
        if best_k > 0:
            cps.append(best_k)
            stack += [(s, best_k), (best_k, e)]
    return sorted(cps)


def parse_raven_table(path):
    """Raven round-trip import: parse selection_table.txt Begin/End times."""
    segs = []
    with open(path) as f:
        lines = [l.rstrip("\n") for l in f if l.strip()]
    if not lines:
        return segs
    head = lines[0].lower()
    has_head = "begin" in head and ("time" in head or "begin" in head)
    body = lines[1:] if has_head else lines
    for line in body:
        parts = line.replace(",", "\t").split()
        if len(parts) < 3:
            continue
        try:
            nums = [float(x) for x in parts[1:3]]
            segs.append([nums[0], nums[1]])
        except Exception:
            continue
    return segs


class EncodecRVQUnits:
    """Experimental EnCodec-RVQ-style flag: 2-stage residual FSQ-lite.

    Coarse 8x8 grid plus residual 8x8 grid, mixed into one K=64 index
    (coarse*8 + residual_dim0 blended). Zero-training, numpy-only,
    low-bitrate experiment — not a real EnCodec checkpoint.
    """

    def __init__(self, K=DEFAULT_K, seed=0):
        if int(K) != 64:
            raise ValueError("encodec-rvq maps exactly to K=64")
        self.K = 64
        self.seed = seed
        self.base = FSQUnits(K=64, seed=seed)

    def fit(self, mels):
        return self.base.fit(mels)

    def encode_tokens(self, mel):
        toks = self.base._quantize_frames(mel)
        # Residual pass: re-quantize tanh projection remainder.
        z = ((mel.T - self.base.mean) / self.base.std) @ self.base.proj
        z = np.tanh(z / 2.0)
        res = np.clip((((z[:, 0] - z[:, 1]) + 1.0) / 2.0 * 8).astype(int),
                      0, 7)
        mixed = (np.asarray(toks, dtype=int) + res[:len(toks)]) % 64
        return dedup([int(t) for t in mixed[::4]])

    def codebook_matrix(self):
        return self.base.codebook_matrix()


# ----------------------------------------------------------------------------
# Small causal GPT (hand-rolled, nanoGPT-style)
# ----------------------------------------------------------------------------

class TinyGPT:
    def __init__(self, K=DEFAULT_K, n_layer=4, n_head=4, n_embd=256,
                 block_size=128, dropout=0.0, seed=0):
        torch, nn = _get_torch()
        self.torch = torch
        self.nn = nn
        self.block_size = block_size
        g = torch.Generator().manual_seed(seed)

        class Block(nn.Module):
            def __init__(self, d, heads, drop):
                super().__init__()
                self.ln1 = nn.LayerNorm(d)
                self.attn = nn.MultiheadAttention(d, heads, dropout=drop,
                                                  batch_first=True)
                self.ln2 = nn.LayerNorm(d)
                self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(),
                                         nn.Linear(4 * d, d), nn.Dropout(drop))

            def forward(self, h, mask):
                a, _ = self.attn(self.ln1(h), self.ln1(h), self.ln1(h),
                                 attn_mask=mask, need_weights=False)
                h = h + a
                return h + self.mlp(self.ln2(h))

        self.tok_emb = nn.Embedding(K, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, dropout) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, K, bias=False)
        for m in list(self.blocks.modules()) + [self.tok_emb, self.head]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, generator=g)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.tok_emb.weight, std=0.02, generator=g)
        nn.init.normal_(self.pos_emb.weight, std=0.02, generator=g)
        causal = torch.full((block_size, block_size), float("-inf"))
        self.causal = torch.triu(causal, diagonal=1)

    def parameters(self):
        ps = list(self.tok_emb.parameters()) + list(self.pos_emb.parameters()) \
            + list(self.blocks.parameters()) + list(self.ln.parameters()) \
            + list(self.head.parameters())
        return ps

    def forward(self, idx):
        B, T = idx.shape
        pos = self.torch.arange(T, device=idx.device)
        h = self.tok_emb(idx) + self.pos_emb(pos)[None]
        mask = self.causal[:T, :T].to(idx.device)
        for b in self.blocks:
            h = b(h, mask)
        return self.head(self.ln(h))


def _concat(seqs):
    out = []
    for s in seqs:
        out.extend(int(t) for t in s)
    return out


def train_gpt(train_seqs, val_seqs, K=DEFAULT_K, epochs=5, block_size=64,
              n_layer=2, n_head=4, n_embd=128, lr=3e-4, batch_size=16,
              seed=0):
    torch, nn = _get_torch()
    set_seed(seed)
    model = TinyGPT(K=K, n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                    block_size=block_size, seed=seed)
    tr = _concat(train_seqs)
    va = _concat(val_seqs) if val_seqs else []
    if len(tr) < block_size + 2:
        raise ValueError("not enough tokens to train GPT "
                         f"({len(tr)} tokens, block {block_size})")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(1, epochs))
    ce = nn.CrossEntropyLoss()
    train_hist, val_hist = [], []
    steps = max(10, len(tr) // max(1, batch_size))
    for ep in range(max(1, epochs)):
        model.blocks.train()
        tot = 0.0
        for _ in range(steps):
            i = int(np.random.randint(0, len(tr) - block_size))
            x = torch.tensor([tr[i:i + block_size]], dtype=torch.long)
            y = torch.tensor([tr[i + 1:i + block_size + 1]], dtype=torch.long)
            logits = model.forward(x)
            loss = ce(logits.view(-1, K), y.view(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.item())
        sched.step()
        train_hist.append(tot / steps)
        with torch.no_grad():
            model.blocks.eval()
            if len(va) > block_size + 1:
                tot, nb = 0.0, 0
                for i in range(0, len(va) - block_size - 1, block_size):
                    x = torch.tensor([va[i:i + block_size]], dtype=torch.long)
                    y = torch.tensor([va[i + 1:i + block_size + 1]],
                                     dtype=torch.long)
                    l = ce(model.forward(x).view(-1, K), y.view(-1))
                    tot += float(l.item()) * len(y.view(-1))
                    nb += len(y.view(-1))
                val_hist.append(tot / max(1, nb))
            else:
                val_hist.append(train_hist[-1])
    return model, {"train_loss": train_hist, "val_loss": val_hist}


# ----------------------------------------------------------------------------
# Report (matplotlib only): codebook PCA scatter, transition heatmap, curves
# ----------------------------------------------------------------------------

def make_report(outdir, codebook, freq, P, vq_hist, gpt_hist,
                metrics, K=DEFAULT_K, seqs=None, themes=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    Z = np.asarray(codebook, dtype=np.float64)
    n_clu = max(2, min(8, K // 8))
    labels = KMeans(n_clusters=n_clu, n_init=10,
                    random_state=0).fit_predict(Z)
    xy = PCA(n_components=2, random_state=0).fit_transform(Z)
    try:
        import umap
        xy_u = umap.UMAP(n_neighbors=8, min_dist=0.3,
                         random_state=0).fit_transform(Z)
        umap_ok = True
    except Exception:
        xy_u, umap_ok = xy, False

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    # Panel 1: codebook embedding scatter.
    ax = axes[0, 0]
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=labels, s=20 + 180 * freq,
                    cmap="tab10", alpha=0.85, edgecolors="k", linewidths=0.4)
    top = np.argsort(freq)[-5:]
    for i in top:
        ax.annotate(str(i), (xy[i, 0], xy[i, 1]), fontsize=8)
    title = "Codebook PCA scatter (K=%d)" % K
    if umap_ok:
        ax2 = ax.inset_axes([0.62, 0.62, 0.36, 0.36])
        ax2.scatter(xy_u[:, 0], xy_u[:, 1], c=labels, s=8, cmap="tab10")
        ax2.set_xticks([]), ax2.set_yticks([])
        ax2.set_title("UMAP", fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    # Panel 2: bigram transition heatmap.
    ax = axes[0, 1]
    im = ax.imshow(P, norm=LogNorm(vmin=max(P.min(), 1e-4), vmax=1.0),
                   cmap="magma", aspect="equal", origin="upper")
    ax.set_title("Bigram transition P(next|cur), log scale")
    ax.set_xlabel("next unit")
    ax.set_ylabel("current unit")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    # Panel 3: loss / perplexity curves.
    ax = axes[1, 0]
    ep_vq = np.arange(1, len(vq_hist) + 1)
    ax.plot(ep_vq, vq_hist, "o-", label="VQ-VAE train loss")
    ep_lm = np.arange(1, len(gpt_hist["train_loss"]) + 1)
    ax.plot(ep_lm, gpt_hist["train_loss"], "s-", label="GPT train CE")
    ax.plot(ep_lm, gpt_hist["val_loss"], "^-", label="GPT val CE")
    ax.axhline(metrics["bigram_nll"], color="gray", ls="--",
               label="bigram val NLL (%.2f)" % metrics["bigram_nll"])
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss (nats)")
    ax.set_title("Training curves (GPT val PPL=%.1f)" % metrics["gpt_ppl"])
    ax.legend(fontsize=8)
    # Panel 4: recurrence plot of first long segment (Bioacoustics 2021 idea).
    ax = axes[1, 1]
    try:
        long_s = max(seqs or [], key=len) if seqs else []
        if len(long_s) >= 8:
            L = min(128, len(long_s))
            R = np.zeros((L, L))
            for i in range(L):
                for j in range(L):
                    R[i, j] = 1.0 if long_s[i] == long_s[j] else 0.0
            ax.imshow(R, cmap="Greys", origin="upper", aspect="equal")
            ax.set_title("Recurrence: segment reuse (len=%d)" % L)
        else:
            ax.text(0.5, 0.5, "recurrence: need len>=8", ha="center")
            ax.set_title("Recurrence (insufficient tokens)")
        ax.set_xlabel("token index")
        ax.set_ylabel("token index")
    except Exception:
        ax.set_title("Recurrence (unavailable)")
    n_themes = len(themes or [])
    fig.suptitle("Thalassa whale song syntax analysis  |  " +
                 "unigram H=%.2f bits  bigram PPL=%.1f  GPT PPL=%.1f  "
                 "codes %d/%d themes %d" % (metrics["unigram_bits"],
                                            metrics["bigram_ppl"],
                                            metrics["gpt_ppl"],
                                            metrics["codes_used"], K,
                                            n_themes))
    fig.tight_layout()
    png = os.path.join(outdir, "syntax_analysis_report.png")
    pdf = os.path.join(outdir, "syntax_analysis_report.pdf")
    fig.savefig(png, dpi=175)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


# ----------------------------------------------------------------------------
# Full pipeline
# ----------------------------------------------------------------------------

def _emit(progress_cb, stage, frac):
    if progress_cb is not None:
        try:
            progress_cb(str(stage), float(frac))
        except Exception:
            pass


def job_cache_dir(base_outdir, input_path, params):
    import hashlib
    h = hashlib.sha256()
    try:
        with open(input_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
    except OSError:
        h.update(os.path.basename(input_path).encode())
    h.update(json.dumps(params, sort_keys=True).encode())
    return os.path.join(base_outdir, "jobs", h.hexdigest()[:16])


def mel_cache_path(outdir, input_path, params):
    import hashlib
    h = hashlib.sha256()
    try:
        with open(input_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
    except OSError:
        h.update(os.path.basename(input_path).encode())
    h.update(json.dumps(params, sort_keys=True).encode())
    return os.path.join(outdir, "mel_cache",
                        h.hexdigest()[:16] + ".npz")


def embedding_cosine_matrix(vecs):
    import numpy as np
    X = np.asarray(vecs, dtype=np.float64)
    n = X - X.mean(axis=1, keepdims=True)
    denom = np.sqrt((n ** 2).sum(axis=1, keepdims=True)) + 1e-12
    Xn = n / denom
    return (Xn @ Xn.T).tolist()


def segment_spectral_stats(x, sr, bounds):
    """Measured per-segment acoustics: duration, peak and centroid freq."""
    import numpy as np
    stats = []
    for s0, s1 in bounds:
        a = max(0, int(s0 * sr))
        b = min(len(x), int(s1 * sr))
        seg = x[a:b]
        if len(seg) < 64:
            continue
        seg = seg - seg.mean()
        spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
        freqs = np.fft.rfftfreq(len(seg), 1.0 / sr)
        m = freqs >= 10.0
        spec, freqs = spec[m], freqs[m]
        tot = float(spec.sum()) + 1e-12
        stats.append({"dur_s": round(float(s1 - s0), 3),
                      "peak_hz": round(float(freqs[int(np.argmax(spec))]), 1),
                      "centroid_hz": round(float((spec * freqs).sum() / tot),
                                           1)})
    return stats


# Literature-grounded acoustic ranges (broad, cite-before-use):
# humpback: broadband 20Hz-4kHz+ patterned song, units->phrases->themes
#   (Payne & McVay 1971 hierarchy; Cholewiak 2013 review).
# blue: low-frequency 10-100Hz moans, long B-calls (Watkins/NOAA literature).
# fin: ~20Hz short regular pulses (Watkins fin-whale literature).
# NOTE: the pipeline bandpass low-cut (default 50Hz, 10Hz in low-freq
# mode) trims deep fundamentals, so blue/fin calls are judged on surviving
# harmonics + duration + pattern unless low-frequency mode is on.
def clip_band_energy(x, sr):
    """Clip-level energy fractions below 200/500Hz + centroid (numpy)."""
    import numpy as np
    x = np.asarray(x[: sr * 120], dtype=np.float64)
    x = x - x.mean()
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    freqs = np.fft.rfftfreq(len(x), 1.0 / sr)
    tot = float(spec.sum()) + 1e-12
    return {"e200": round(float(spec[freqs < 200].sum() / tot), 3),
            "e500": round(float(spec[freqs < 500].sum() / tot), 3),
            "centroid_hz": round(float((spec * freqs).sum() / tot), 1)}


def match_species(stats, meta, metrics, n_phrases, n_themes, clip_e=None):
    """Evidence-matched species assessment from measured acoustics.

    NOT a trained classifier: counts how many literature checks agree and
    reports the measurements. Confidence = checks passed / checks run.
    """
    import numpy as np
    if not stats:
        return {"label": "unassigned", "confidence": 0.0,
                "note": "no voiced segments found",
                "evidence": []}
    if clip_e and clip_e.get("e500", 1.0) < 0.35:
        return {"label": "unassigned",
                "confidence": 0.2,
                "note": ("broadband noise dominates "
                         f"(only {clip_e['e500'] * 100:.0f}% energy below "
                         "500Hz); species verdict unreliable"),
                "evidence": [
                    f"[n] only {clip_e['e500'] * 100:.0f}% clip energy "
                    f"below 500Hz (clip centroid {clip_e['centroid_hz']:.0f}Hz)"
                ],
                "measured": {"n_units": len(stats),
                             "clip_e500": clip_e["e500"]}}
    peaks = np.array([s["peak_hz"] for s in stats])
    cents = np.array([s["centroid_hz"] for s in stats])
    durs = np.array([s["dur_s"] for s in stats])
    med_peak = float(np.median(peaks))
    med_cent = float(np.median(cents))
    med_dur = float(np.median(durs))
    frac_low = float((cents < 200.0).mean())
    frac_vlow = float((peaks < 80.0).mean())
    gap = float(metrics.get("shuffle_gap_nats", 0.0))
    checks = {"humpback": [], "blue": [], "fin": []}
    # Weights: phrase repetition is near-definitional for humpback song.
    checks["humpback"].append(
        (med_cent >= 300.0, 1.0,
         f"median centroid {med_cent:.0f}Hz "
         f"({'broadband, song-like' if med_cent >= 300 else 'low, call-like'})"))
    checks["humpback"].append(
        (n_phrases >= 5, 1.5,
         f"{n_phrases} repeat phrases, {n_themes} themes "
         f"(patterned song structure)"))
    checks["humpback"].append(
        (gap > 0.3, 1.0,
         f"shuffle gap +{gap:.2f} nats (order carries syntax)"))
    checks["blue"].append(
        (frac_low >= 0.6, 1.2,
         f"{frac_low * 100:.0f}% of units centroid below 200Hz "
         f"(low-frequency moans)"))
    checks["blue"].append(
        (med_dur >= 3.0, 0.8,
         f"median unit {med_dur:.1f}s "
         f"({'long moans' if med_dur >= 3 else 'short units'})"))
    checks["fin"].append(
        (frac_vlow >= 0.6 and med_dur <= 2.5, 1.0,
         f"{frac_vlow * 100:.0f}% of peaks below 80Hz, "
         f"median unit {med_dur:.1f}s (pulse-like)"))
    structured = n_phrases >= 5
    # Bayesian log-odds: flat prior, each check +/-w*1.4 nats.
    import math as _math
    logodds, hits = {}, {}
    for k, v in checks.items():
        lo = sum((w * 1.4 if ok else -w * 1.4) for ok, w, _ in v)
        if k == "humpback" and not structured:
            lo = float("-inf")
        logodds[k] = lo
        hits[k] = sum(1 for ok, _, _ in v if ok)
    mx = max(logodds.values())
    post = {k: _math.exp(lo - mx) for k, lo in logodds.items()
            if lo > float("-inf")}
    tot = sum(post.values()) or 1.0
    post = {k: v / tot for k, v in post.items()}
    label = max(post, key=lambda k: post[k]) if post else "unassigned"
    conf = post.get(label, 0.0)
    evidence = [f"[{'Y' if ok else 'n'}] {txt}"
                for k, v in checks.items() for ok, _, txt in v]
    if not structured:
        evidence.append("[gate] no repeat-phrase structure found — "
                        "humpback ruled out; blue/fin need clean "
                        "sub-50Hz data this pipeline filters out")
    if hits.get(label, 0) == 0:
        # No supporting evidence for any class: refuse to label.
        label, conf = "unassigned", 0.3 if not structured else 0.15
    note = ("evidence-matched acoustics, not a trained classifier; "
            f"{meta.get('lowcut', 50.0):.0f}Hz low-cut applied in pipeline")
    return {"label": f"{label}-like" if label != "unassigned" else label,
            "confidence": round(float(conf), 2),
            "note": note, "evidence": evidence,
            "measured": {"n_units": len(stats), "med_peak_hz": med_peak,
                         "med_centroid_hz": med_cent,
                         "med_dur_s": med_dur}}


def provisional_species_hint(meta, metrics):
    fmax = float(meta.get("fmax", 4000.0))
    gap = float(metrics.get("shuffle_gap_nats", 0.0))
    if fmax >= 7000.0 and gap > 0.3:
        return {"label": "humpback-like", "confidence": 0.6,
                "note": "provisional: high-band + ordered syntax"}
    if gap > 0.3:
        return {"label": "humpback-like", "confidence": 0.55,
                "note": "provisional: ordered syntax"}
    return {"label": "unassigned", "confidence": 0.3,
            "note": "provisional: needs Perch probe"}


def file_vector(metrics):
    """Aggregate file embedding used for population/library comparison."""
    import math as _math
    return [float(metrics["unigram_bits"]),
            _math.log(max(metrics["bigram_ppl"], 1e-9)),
            _math.log(max(metrics["gpt_ppl"], 1e-9)),
            float(metrics.get("zipf_slope", 0.0)),
            float(metrics.get("trigram_gap_nats", 0.0)),
            float(metrics.get("shuffle_gap_nats", 0.0)),
            float(metrics["codes_used"]) / 64.0]


def match_library(vec, library):
    """Rank library entries by cosine similarity to a file vector.

    library: [{label, file, vec}]. Returns matches sorted best-first with
    a nearest-neighbor verdict when the margin is clear.
    """
    import numpy as _np
    v = _np.asarray(vec, dtype=float)
    scored = []
    for e in library:
        u = _np.asarray(e["vec"], dtype=float)
        denom = (float((v ** 2).sum()) ** 0.5
                 * float((u ** 2).sum()) ** 0.5 + 1e-12)
        sim = float((v @ u) / denom)
        scored.append({"label": e.get("label", e.get("file", "?")),
                       "file": e.get("file", "?"),
                       "similarity": round(sim, 4)})
    scored.sort(key=lambda d: d["similarity"], reverse=True)
    verdict = "unassigned"
    if len(scored) >= 2 and scored[0]["similarity"] - scored[1]["similarity"] > 0.002:
        verdict = scored[0]["label"]
    elif scored and scored[0]["similarity"] > 0.995:
        verdict = scored[0]["label"]
    return {"matches": scored[:5], "verdict": verdict}


def provisional_meaning_tags(segments, metrics, phrases=None):
    """Per-segment meaning from measured phrase membership.

    Segments inside repeated phrases are patterned song units; lone
    segments match the profile of social/call fragments (Dunlop/Stimpert:
    unpatterned, both sexes). File-level context still needs labels.
    """
    in_phrase = {}
    for ph in phrases or []:
        for sid in ph.get("segments", []):
            in_phrase[sid] = in_phrase.get(sid, 0) + 1
    gap = float(metrics.get("shuffle_gap_nats", 0.0))
    tags = []
    for i, s in enumerate(segments):
        sid = int(s.get("id", i))
        n = in_phrase.get(sid, 0)
        if n >= 2:
            ctx = (f"patterned song unit — in {n} shared phrases "
                   f"(breeding-display-like)")
            conf = 0.7
        elif n == 1:
            ctx = "song unit — in 1 shared phrase"
            conf = 0.55
        else:
            ctx = ("unpatterned unit — social/call-fragment candidate "
                   "(needs audit)")
            conf = 0.4 if gap > 0.15 else 0.35
        tags.append({"segment": sid, "context": ctx, "confidence": conf})
    return tags


def run_pipeline(input_path, outdir, epochs_vq=5, epochs_lm=5, K=DEFAULT_K,
                 seed=0, vq_ema=True, gpt_layers=2, gpt_dim=128,
                 block_size=64, synthetic_if_missing=False,
                 synth_duration_s=12.0, units="vqvae", use_pcen=False,
                 use_deltas=False, fmax=DEFAULT_HIGH, progress_cb=None,
                 use_cache=True, denoise_backend="gating",
                 pcen_tau=0.06, pcen_s=0.05, pcen_alpha=0.98,
                 ssl_weights=None, pcen_preset=None,
                 vad_offset_db=7.0, vad_hangover_s=0.25,
                 vad_min_unit_s=0.25, vad_merge_gap_s=0.20,
                 raven_import_path=None, lowcut=DEFAULT_LOW,
                 frontend="mel", denoise_profile="flat"):
    import torch
    import time as _time
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    set_seed(seed)
    os.makedirs(outdir, exist_ok=True)
    _t0 = _time.perf_counter()
    _timings = {}
    _emit(progress_cb, "start", 0.02)
    if pcen_preset in PCEN_PRESETS:
        pcen_tau = PCEN_PRESETS[pcen_preset]["pcen_tau"]
        pcen_s = PCEN_PRESETS[pcen_preset]["pcen_s"]
        pcen_alpha = PCEN_PRESETS[pcen_preset]["pcen_alpha"]
    if use_cache and os.path.isfile(input_path):
        params = {"K": K, "epochs_vq": epochs_vq, "epochs_lm": epochs_lm,
                  "seed": seed, "units": str(units).lower(),
                  "use_pcen": bool(use_pcen), "use_deltas": bool(use_deltas),
                  "fmax": fmax, "gpt_layers": gpt_layers, "gpt_dim": gpt_dim,
                  "block_size": block_size, "denoise": str(denoise_backend),
                  "pcen_tau": pcen_tau, "pcen_s": pcen_s,
                  "pcen_alpha": pcen_alpha, "pcen_preset": str(pcen_preset),
                  "vad_offset_db": vad_offset_db,
                  "vad_hangover_s": vad_hangover_s,
                  "vad_min_unit_s": vad_min_unit_s,
                  "vad_merge_gap_s": vad_merge_gap_s,
                  "lowcut": float(lowcut),
                  "frontend": str(frontend),
                  "denoise_profile": str(denoise_profile),
                  "code_v": CODEC_VERSION}
        cached = job_cache_dir(outdir, input_path, params)
        hit = os.path.join(cached, "song_tokens.json")
        if os.path.isfile(hit):
            import shutil
            for name in ("song_tokens.json", "syntax_analysis_report.png",
                         "syntax_analysis_report.pdf", "selection_table.txt"):
                src = os.path.join(cached, name)
                if os.path.isfile(src):
                    shutil.copyfile(src, os.path.join(outdir, name))
            _emit(progress_cb, "cache-hit", 1.0)
            with open(hit) as f:
                payload = json.load(f)
            m = payload["lm"]
            return {"json": os.path.join(outdir, "song_tokens.json"),
                    "png": os.path.join(outdir, "syntax_analysis_report.png"),
                    "pdf": os.path.join(outdir, "syntax_analysis_report.pdf"),
                    "table": os.path.join(outdir, "selection_table.txt"),
                    "metrics": {"bigram_nll": float(m.get("bigram_nll", 0.0)),
                                "bigram_ppl": float(m["bigram_perplexity"]),
                                "unigram_bits": float(m["entropy_bits_per_token"]),
                                "unigram_ppl": float(m["unigram_perplexity"]),
                                "gpt_ppl": float(m["perplexity"]),
                                "gpt_bits_per_token": float(m["bits_per_token"]),
                                "codes_used": int(payload["codebook"]["utilization"]["used"]),
                                "dead_codes": int(payload["codebook"]["utilization"]["dead"]),
                                "codebook_perplexity": float(payload["codebook"]["utilization"]["perplexity"]),
                                "trigram_gap_nats": float(m.get("trigram_gap_nats", 0.0)),
                                "zipf_slope": float(m.get("zipf_slope", 0.0)),
                                "shuffle_gap_nats": float(m.get("shuffle_gap_nats", 0.0)),
                                "n_phrases": len(payload.get("phrases", [])),
                                "n_themes": len(payload.get("themes", []))},
                    "n_segments": len(payload["segments"]),
                    "n_phrases": len(payload.get("phrases", [])),
                    "n_themes": len(payload.get("themes", [])),
                    "cached": True}
    if not os.path.isfile(input_path):
        if synthetic_if_missing:
            input_path = os.path.join(outdir, "synthetic.wav")
            generate_synthetic_wav(input_path, duration_s=synth_duration_s,
                                   seed=seed)
        else:
            raise FileNotFoundError(f"input not found: {input_path} "
                                    "(use --synthetic-if-missing)")
    # 1) Load -> bandpass -> denoise -> VAD.
    _emit(progress_cb, "load", 0.08)
    _t = _time.perf_counter()
    x, sr = load_wav(input_path)
    dur = len(x) / sr
    _timings["load_s"] = round(_time.perf_counter() - _t, 3)
    _emit(progress_cb, "denoise", 0.2)
    _t = _time.perf_counter()
    xb = bandpass(x, sr, low=lowcut)
    xd, denoise_fallback = denoise(xb, sr, backend=denoise_backend,
                                      noise_profile=denoise_profile)
    _timings["denoise_s"] = round(_time.perf_counter() - _t, 3)
    _emit(progress_cb, "vad", 0.35)
    _t = _time.perf_counter()
    if raven_import_path and os.path.isfile(raven_import_path):
        try:
            bounds = parse_raven_table(raven_import_path)
        except Exception:
            bounds = []
    else:
        bounds = []
    if not bounds:
        bounds = vad_segments(xd, sr, offset_db=vad_offset_db,
                              hangover_s=vad_hangover_s,
                              min_unit_s=vad_min_unit_s,
                              merge_gap_s=vad_merge_gap_s)
    if not bounds:
        bounds = [[0.0, dur]]
    _timings["vad_s"] = round(_time.perf_counter() - _t, 3)
    # 2) 128-bin mel per segment (mel / pcen-mel / +deltas, fmax flag).
    _t = _time.perf_counter()
    seg_mels, seg_rms = [], []
    for s0, s1 in bounds:
        seg = xd[int(s0 * sr):max(int(s0 * sr) + 1, int(s1 * sr))]
        seg_mels.append(log_mel(seg, sr, fmax=fmax, use_pcen=use_pcen,
                                use_deltas=use_deltas, pcen_tau=pcen_tau,
                                pcen_s=pcen_s, pcen_alpha=pcen_alpha,
                                frontend=frontend))
        seg_rms.append(float(20 * np.log10(
            np.sqrt((seg.astype(np.float64) ** 2).mean()) + 1e-12)))
    # 3) Unit discovery: vqvae | kmeans | fsq | ssl-kmeans | encodec-rvq.
    units = str(units).lower().replace("_", "-")
    vq_hist_label = "VQ-VAE train loss"
    ssl_fallback = None
    if units == "kmeans":
        vq = KMeansUnits(K=K, seed=seed)
        vq_stats = vq.fit(seg_mels)
        vq_hist_label = "kmeans (no training loss)"
        import numpy as _np
        codebook = _np.asarray(vq.km.cluster_centers_,
                               dtype=_np.float64).tolist()
        seqs = [vq.encode_tokens(m) for m in seg_mels]
    elif units == "fsq":
        vq = FSQUnits(K=K, seed=seed)
        vq_stats = vq.fit(seg_mels)
        vq_hist_label = "FSQ (no training loss)"
        codebook = vq.codebook_matrix()
        seqs = [vq.encode_tokens(m) for m in seg_mels]
    elif units == "ssl-kmeans":
        vq = SSLEmbeddingUnits(K=K, seed=seed, weights_path=ssl_weights)
        vq_stats = vq.fit(seg_mels)
        ssl_fallback = bool(vq.used_fallback)
        vq_hist_label = "ssl-kmeans (fallback mel, no training loss)" \
            if ssl_fallback else "ssl-kmeans (no training loss)"
        import numpy as _np
        codebook = _np.asarray(vq.km.cluster_centers_,
                               dtype=_np.float64).tolist()
        seqs = [vq.encode_tokens(m) for m in seg_mels]
    elif units == "encodec-rvq":
        vq = EncodecRVQUnits(K=K, seed=seed)
        vq_stats = vq.fit(seg_mels)
        vq_hist_label = "encodec-rvq (experimental, no training loss)"
        codebook = vq.codebook_matrix()
        seqs = [vq.encode_tokens(m) for m in seg_mels]
    else:
        units = "vqvae"
        with _TRAIN_LOCK:
            set_seed(seed)
            vq, vq_stats = train_vqvae(seg_mels, K=K, epochs=epochs_vq,
                                       ema=vq_ema, seed=seed,
                                       n_mels=seg_mels[0].shape[0])
        with torch.no_grad():
            codebook = vq.codebook.weight.detach().cpu().numpy().tolist()
        seqs = [dedup(vq.encode_tokens((m - vq_stats["mean"]) / vq_stats["std"]))
                for m in seg_mels]
    _timings["features_s"] = round(_time.perf_counter() - _t, 3)
    _t = _time.perf_counter()
    seqs = [s if s else [0] for s in seqs]
    _emit(progress_cb, "units", 0.6)
    # 4) Bigram baseline + tiny GPT + metrics.
    train_seqs, val_seqs = split_sequences(seqs)
    _, P = bigram_stats(train_seqs, K)
    bigram_nll = seq_nll_bigram(val_seqs or train_seqs, P)
    bigram_ppl = math.exp(bigram_nll)
    h_bits, unigram_ppl = unigram_entropy_bits(train_seqs, K)
    codes_used, dead, code_perp = codebook_utilization(seqs, K)
    n_tok = min(len(_concat(train_seqs)),
                len(_concat(val_seqs or train_seqs)))
    blk = max(2, min(block_size, n_tok - 1))
    try:
        with _TRAIN_LOCK:
            set_seed(seed)
            gpt, gpt_hist = train_gpt(train_seqs, val_seqs or train_seqs,
                                      K=K, epochs=epochs_lm, block_size=blk,
                                      n_layer=gpt_layers, n_embd=gpt_dim,
                                      seed=seed)
        gpt_ppl = math.exp(min(gpt_hist["val_loss"]))
    except ValueError:
        # Too few tokens for any context window: fall back to bigram NLL so
        # tiny fixtures and silent-edge cases still produce a report.
        gpt_hist = {"train_loss": [bigram_nll], "val_loss": [bigram_nll]}
        gpt_ppl = bigram_ppl
    metrics = {"bigram_nll": bigram_nll, "bigram_ppl": bigram_ppl,
               "unigram_bits": h_bits, "unigram_ppl": unigram_ppl,
               "gpt_ppl": gpt_ppl, "gpt_bits_per_token": math.log2(gpt_ppl),
               "codes_used": codes_used, "dead_codes": dead,
               "codebook_perplexity": code_perp}
    _timings["units_lm_s"] = round(_time.perf_counter() - _t, 3)
    # Phrase/theme layer + deep-syntax diagnostics (Zipf, 2nd-order gap).
    _emit(progress_cb, "syntax", 0.75)
    _t = _time.perf_counter()
    try:
        gap_nats = float(trigram_nll_gap(train_seqs or seqs, K))
    except Exception:
        gap_nats = 0.0
    zipf_a, zipf_r2 = zipf_slope(train_seqs or seqs, K)
    phrases = find_phrases(seqs)
    try:
        shuffle_gap = float(shuffled_control_gap(train_seqs or seqs, K,
                                                seed=seed))
    except Exception:
        shuffle_gap = 0.0
    themes = find_themes(np.asarray(P), K=K)
    theme_map = code_to_theme_map(themes)
    metrics.update({"trigram_gap_nats": gap_nats,
                    "zipf_slope": zipf_a, "zipf_r2": zipf_r2,
                    "shuffle_gap_nats": shuffle_gap,
                    "n_phrases": len(phrases),
                    "n_themes": len(themes)})
    # Outputs.
    segments = [{"id": i, "start_s": round(s0, 3), "end_s": round(s1, 3),
                 "duration_s": round(s1 - s0, 3), "rms_db": round(r, 2),
                 "tokens": list(map(int, toks)),
                 "theme": int(theme_map.get(int(toks[0]), 0)) if toks else 0}
                for i, ((s0, s1), r, toks)
                in enumerate(zip(bounds, seg_rms, seqs))]
    meta = {"sr": sr, "duration_s": round(dur, 3),
            "n_segments": len(segments), "codebook_K": K,
            "epochs_vq": epochs_vq, "epochs_lm": epochs_lm,
            "seed": seed, "input": os.path.basename(input_path),
            "units": units, "use_pcen": bool(use_pcen),
            "use_deltas": bool(use_deltas), "fmax": fmax,
            "feature_dim": int(seg_mels[0].shape[0]),
            "denoise_backend": str(denoise_backend),
            "lowcut": float(lowcut),
            "frontend": str(frontend),
            "denoise_profile": str(denoise_profile),
            "denoise_fallback": bool(denoise_fallback),
            "pcen_tau": pcen_tau, "pcen_s": pcen_s,
            "pcen_alpha": pcen_alpha, "pcen_preset": str(pcen_preset),
            "vad_offset_db": vad_offset_db,
            "vad_hangover_s": vad_hangover_s,
            "vad_min_unit_s": vad_min_unit_s,
            "vad_merge_gap_s": vad_merge_gap_s}
    if ssl_fallback is not None:
        meta["ssl_fallback"] = bool(ssl_fallback)
        if ssl_weights:
            meta["ssl_weights"] = str(ssl_weights)
    metrics_for_hints = {"shuffle_gap_nats": shuffle_gap,
                         "trigram_gap_nats": gap_nats}
    try:
        spec_stats = segment_spectral_stats(xd, sr, bounds)
    except Exception:
        spec_stats = []
    for s, st in zip(segments, spec_stats):
        s["peak_hz"] = st["peak_hz"]
        s["centroid_hz"] = st["centroid_hz"]
    try:
        clip_e = clip_band_energy(xb, sr)
    except Exception:
        clip_e = None
    if spec_stats:
        species_hint = match_species(spec_stats, meta, metrics_for_hints,
                                     len(phrases), len(themes),
                                     clip_e=clip_e)
    else:
        species_hint = provisional_species_hint(meta, metrics_for_hints)
    meaning_tags = provisional_meaning_tags(segments, metrics_for_hints,
                                            phrases=phrases)
    tsyntax = theme_syntax([s.get("theme", 0) for s in segments])
    _timings["syntax_s"] = round(_time.perf_counter() - _t, 3)
    _timings["total_s"] = round(_time.perf_counter() - _t0, 3)
    meta["timings_s"] = dict(_timings)
    metrics.update({"theme_entropy_bits": tsyntax["entropy_bits"],
                    "theme_transitions": tsyntax["n_transitions"],
                    "total_s": _timings["total_s"]})
    payload = {"meta": meta,
               "segments": segments,
               "phrases": phrases,
               "themes": themes,
               "theme_syntax": tsyntax,
               "species_hint": species_hint,
               "meaning_tags": meaning_tags,
               "codebook_vectors": codebook,
               "codebook": {"K": K, "dim": len(codebook[0]),
                            "vectors": codebook,
                            "utilization": {"used": codes_used,
                                            "dead": dead,
                                            "perplexity": code_perp}},
               "lm": {"vocab_K": K, "perplexity": gpt_ppl,
                      "bits_per_token": math.log2(gpt_ppl),
                      "entropy_bits_per_token": h_bits,
                      "bigram_perplexity": bigram_ppl,
                      "bigram_nll": bigram_nll,
                      "unigram_perplexity": unigram_ppl,
                      "trigram_gap_nats": gap_nats,
                      "shuffle_gap_nats": shuffle_gap,
                      "zipf_slope": zipf_a, "zipf_r2": zipf_r2,
                      "train_loss": gpt_hist["train_loss"],
                      "val_loss": gpt_hist["val_loss"],
                      "transition_matrix_KxK": P.tolist()}}
    with open(os.path.join(outdir, "song_tokens.json"), "w") as f:
        json.dump(payload, f)
    # Raven-compatible selection table (Begin/End Time + theme per segment).
    with open(os.path.join(outdir, "selection_table.txt"), "w") as f:
        f.write("Selection\tBegin Time (s)\tEnd Time (s)\t"
                "Low Freq (Hz)\tHigh Freq (Hz)\t"
                "Duration (s)\tRMS (dB)\tTheme\tTokens\n")
        for s in segments:
            f.write("%d\t%.3f\t%.3f\t%.1f\t%.1f\t%.3f\t%.2f\t%d\t%s\n"
                    % (s["id"] + 1, s["start_s"], s["end_s"],
                       DEFAULT_LOW, fmax,
                       s["duration_s"], s["rms_db"], s.get("theme", 0),
                       "-".join(str(t) for t in s["tokens"][:32])))
    freq = np.zeros(K)
    for s in seqs:
        for t in s:
            freq[t] += 1
    freq = freq / freq.sum()
    png, pdf = make_report(outdir, codebook, freq, np.asarray(P),
                           vq_stats["loss_history"], gpt_hist, metrics, K,
                           seqs=seqs, themes=themes)
    _emit(progress_cb, "report", 0.95)
    if use_cache and os.path.isfile(input_path):
        import shutil
        params = {"K": K, "epochs_vq": epochs_vq, "epochs_lm": epochs_lm,
                  "seed": seed, "units": str(units).lower(),
                  "use_pcen": bool(use_pcen), "use_deltas": bool(use_deltas),
                  "fmax": fmax, "gpt_layers": gpt_layers, "gpt_dim": gpt_dim,
                  "block_size": block_size, "denoise": str(denoise_backend),
                  "pcen_tau": pcen_tau, "pcen_s": pcen_s,
                  "pcen_alpha": pcen_alpha, "pcen_preset": str(pcen_preset),
                  "vad_offset_db": vad_offset_db,
                  "vad_hangover_s": vad_hangover_s,
                  "vad_min_unit_s": vad_min_unit_s,
                  "vad_merge_gap_s": vad_merge_gap_s,
                  "lowcut": float(lowcut),
                  "frontend": str(frontend),
                  "denoise_profile": str(denoise_profile),
                  "code_v": CODEC_VERSION}
        cached = job_cache_dir(outdir, input_path, params)
        os.makedirs(cached, exist_ok=True)
        for name in ("song_tokens.json", "syntax_analysis_report.png",
                     "syntax_analysis_report.pdf", "selection_table.txt"):
            src = os.path.join(outdir, name)
            if os.path.isfile(src):
                shutil.copyfile(src, os.path.join(cached, name))
    _emit(progress_cb, "done", 1.0)
    return {"json": os.path.join(outdir, "song_tokens.json"),
            "png": png, "pdf": pdf, "metrics": metrics,
            "table": os.path.join(outdir, "selection_table.txt"),
            "n_segments": len(segments), "n_phrases": len(phrases),
            "n_themes": len(themes),
            "cached": False}


def make_trend_report(outdir, rows):
    """Entropy/complexity time-tracker across batch files (Allen 2018 idea)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = list(range(len(rows)))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(xs, [r["unigram_bits"] for r in rows], "o-", label="entropy bits")
    ax.plot(xs, [math.log2(max(r["bigram_ppl"], 1e-9)) for r in rows],
            "s-", label="log2 bigram PPL")
    ax.plot(xs, [math.log2(max(r["gpt_ppl"], 1e-9)) for r in rows],
            "^-", label="log2 GPT PPL")
    for i, r in enumerate(rows):
        if r.get("revolution_candidate"):
            ax.axvline(i, color="red", ls="--", alpha=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([r["file"] for r in rows], rotation=30, ha="right",
                       fontsize=7)
    ax.set_ylabel("bits")
    ax.set_title("Complexity tracker (red = revolution candidate dip)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    png = os.path.join(outdir, "trend_report.png")
    pdf = os.path.join(outdir, "trend_report.pdf")
    fig.savefig(png, dpi=150)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def run_batch(input_dir, outdir, max_workers=1, **kwargs):
    """Process every .wav in a folder; write per-file subdirs + summary."""
    import glob
    wavs = sorted(glob.glob(os.path.join(input_dir, "*.wav")))
    if not wavs:
        raise FileNotFoundError(f"no .wav files in {input_dir}")

    def _one(wav):
        name = os.path.splitext(os.path.basename(wav))[0]
        sub = os.path.join(outdir, name)
        return run_pipeline(wav, sub, **kwargs)

    if int(max_workers) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=int(max_workers)) as ex:
            results = list(ex.map(_one, wavs))
    else:
        results = [_one(w) for w in wavs]
    rows = []
    all_seqs = []
    for wav, res in zip(wavs, results):
        with open(res["json"]) as f:
            payload = json.load(f)
        m = res["metrics"]
        all_seqs.append([s["tokens"] for s in payload["segments"]])
        rows.append({"file": os.path.basename(wav),
                     "n_segments": res["n_segments"],
                     "n_phrases": res.get("n_phrases", 0),
                     "n_themes": res.get("n_themes", 0),
                     "gpt_ppl": m["gpt_ppl"],
                     "bigram_ppl": m["bigram_ppl"],
                     "trigram_gap_nats": m.get("trigram_gap_nats", 0.0),
                     "shuffle_gap_nats": m.get("shuffle_gap_nats", 0.0),
                     "zipf_slope": m.get("zipf_slope", 0.0),
                     "unigram_bits": m["unigram_bits"],
                     "codes_used": m["codes_used"],
                     "duration_s": payload["meta"]["duration_s"]})
    # Revolution detection: change-point on the entropy series (PELT-style)
    # plus an entropy drop (Allen 2018: complexity falls at revolutions).
    cps = set(change_points([r["unigram_bits"] for r in rows]))
    for i in range(len(rows)):
        drop = (rows[i - 1]["unigram_bits"] - rows[i]["unigram_bits"]
                if i > 0 else 0.0)
        rows[i]["revolution_candidate"] = bool(i in cps and drop >= 0.5)
    pop = population_similarity(all_seqs) if len(all_seqs) > 1 else [[1.0]]
    vecs = [file_vector({"unigram_bits": r["unigram_bits"],
                          "bigram_ppl": r["bigram_ppl"],
                          "gpt_ppl": r["gpt_ppl"],
                          "zipf_slope": r["zipf_slope"],
                          "trigram_gap_nats": r["trigram_gap_nats"],
                          "shuffle_gap_nats": r["shuffle_gap_nats"],
                          "codes_used": r["codes_used"]}) for r in rows]
    cos = embedding_cosine_matrix(vecs) if len(vecs) > 1 else [[1.0]]
    summary = os.path.join(outdir, "summary_by_file.json")
    with open(summary, "w") as f:
        json.dump({"n_files": len(rows), "files": rows,
                   "population_similarity": pop,
                   "population_cosine": cos,
                   "population_files": [r["file"] for r in rows]}, f,
                  indent=2)
    tpng, tpdf = make_trend_report(outdir, rows)
    with open(os.path.join(outdir, "population_matrix.json"), "w") as f:
        json.dump({"files": [r["file"] for r in rows], "matrix": pop,
                   "cosine": cos}, f, indent=2)
    return {"summary": summary, "n_files": len(rows), "files": rows,
            "trend_png": tpng, "trend_pdf": tpdf,
            "population": pop, "population_cosine": cos}


def _apply_config(args, config_path):
    with open(config_path) as f:
        cfg = json.load(f)
    for k, v in cfg.items():
        attr = k.replace("-", "_")
        if hasattr(args, attr):
            setattr(args, attr, v)
    return args


def build_parser():
    p = argparse.ArgumentParser(
        description="Thalassa whale-song decoder: bioacoustic language pipeline.")
    p.add_argument("--input", default=None, help="input hydrophone WAV path")
    p.add_argument("--input-dir", default=None,
                   help="batch mode: process every .wav in this folder")
    p.add_argument("--config", default=None,
                   help="JSON config file overriding CLI flags")
    p.add_argument("--outdir", default="outputs", help="output directory")
    p.add_argument("--epochs-vq", type=int, default=5)
    p.add_argument("--epochs-lm", type=int, default=5)
    p.add_argument("--K", type=int, default=DEFAULT_K)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--vq-ema", action="store_true", default=True)
    p.add_argument("--no-vq-ema", dest="vq_ema", action="store_false")
    p.add_argument("--gpt-layers", type=int, default=2)
    p.add_argument("--gpt-dim", type=int, default=128)
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument("--units", default="vqvae",
                   choices=["vqvae", "kmeans", "fsq", "ssl-kmeans",
                            "encodec-rvq"],
                   help="acoustic unit discovery backend")
    p.add_argument("--ssl-weights", default=None,
                   help="local AVES/Animal2Vec-style weights for ssl-kmeans; "
                        "offline fallback pools mel when absent")
    p.add_argument("--denoise-backend", default="gating",
                   choices=["gating", "noisereduce"],
                   help="denoise: hand-rolled stationary gate (default) or "
                        "noisereduce non-stationary (falls back if missing)")
    p.add_argument("--use-pcen", action="store_true",
                   help="use PCEN frontend instead of log-mel")
    p.add_argument("--pcen-tau", type=float, default=0.06,
                   help="PCEN AGC time constant in seconds")
    p.add_argument("--pcen-s", type=float, default=0.05,
                   help="PCEN smoother gain")
    p.add_argument("--pcen-alpha", type=float, default=0.98,
                   help="PCEN exponent")
    p.add_argument("--use-deltas", action="store_true",
                   help="append delta + delta-delta channels")
    p.add_argument("--fmax", type=float, default=DEFAULT_HIGH,
                   help="mel filterbank top frequency in Hz")
    p.add_argument("--pcen-preset", default=None,
                   choices=["quiet", "noisy", "harsh"],
                   help="PCEN marine preset (overrides tau/s/alpha)")
    p.add_argument("--vad-offset-db", type=float, default=7.0)
    p.add_argument("--vad-hangover-s", type=float, default=0.25)
    p.add_argument("--vad-min-unit-s", type=float, default=0.25)
    p.add_argument("--vad-merge-gap-s", type=float, default=0.20)
    p.add_argument("--raven-import", default=None,
                   help="Raven selection table to use as segment seeds")
    p.add_argument("--lowcut", type=float, default=DEFAULT_LOW,
                   help="bandpass low-cut in Hz (10 for blue/fin low-freq mode)")
    p.add_argument("--frontend", default="mel",
                   choices=["mel", "gammatone"],
                   help="auditory frontend: mel triangles or ERB gammatone")
    p.add_argument("--denoise-profile", default="flat",
                   choices=["flat", "sea"],
                   help="hand-rolled gate weighting: flat or Wenz-like sea")
    p.add_argument("--synthetic-if-missing", action="store_true",
                   help="generate a synthetic fixture WAV when --input is absent")
    p.add_argument("--synth-duration-s", type=float, default=12.0)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.config:
        args = _apply_config(args, args.config)
    extra = dict(units=args.units, use_pcen=args.use_pcen,
                   use_deltas=args.use_deltas, fmax=args.fmax,
                   denoise_backend=args.denoise_backend,
                   pcen_tau=args.pcen_tau, pcen_s=args.pcen_s,
                   pcen_alpha=args.pcen_alpha, ssl_weights=args.ssl_weights,
                   pcen_preset=args.pcen_preset,
                   vad_offset_db=args.vad_offset_db,
                   vad_hangover_s=args.vad_hangover_s,
                   vad_min_unit_s=args.vad_min_unit_s,
                   vad_merge_gap_s=args.vad_merge_gap_s,
                   raven_import_path=args.raven_import,
                   lowcut=args.lowcut, frontend=args.frontend,
                   denoise_profile=args.denoise_profile)
    if args.input_dir:
        res = run_batch(args.input_dir, args.outdir,
                        epochs_vq=args.epochs_vq, epochs_lm=args.epochs_lm,
                        K=args.K, seed=args.seed, vq_ema=args.vq_ema,
                        gpt_layers=args.gpt_layers, gpt_dim=args.gpt_dim,
                        block_size=args.block_size,
                        synthetic_if_missing=args.synthetic_if_missing,
                        synth_duration_s=args.synth_duration_s, **extra)
        print(json.dumps({"wrote": [res["summary"]],
                          "n_files": res["n_files"],
                          "files": res["files"]}, indent=2))
        return 0
    if not args.input:
        build_parser().error("--input is required (or use --input-dir)")
    res = run_pipeline(args.input, args.outdir, epochs_vq=args.epochs_vq,
                       epochs_lm=args.epochs_lm, K=args.K, seed=args.seed,
                       vq_ema=args.vq_ema, gpt_layers=args.gpt_layers,
                       gpt_dim=args.gpt_dim, block_size=args.block_size,
                       synthetic_if_missing=args.synthetic_if_missing,
                       synth_duration_s=args.synth_duration_s, **extra)
    print(json.dumps({"wrote": [res["json"], res["png"], res["pdf"],
                                 res["table"]],
                      "n_segments": res["n_segments"],
                      "n_phrases": res.get("n_phrases", 0),
                      "metrics": res["metrics"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
