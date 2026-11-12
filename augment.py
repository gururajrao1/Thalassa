"""Training augmentations (#6): SpecAugment + mixup + ship-noise injection.

Mel-level: random frequency/time masks (SpecAugment-lite) and convex
mixup pairs. Waveform-level: brown-noise + ship-like tonal lines
(harmonic engine bands with propeller amplitude modulation) mixed at a
target SNR. All deterministic given seed; the pipeline appends augmented
copies to the unit-training set only (inference is never augmented).
Usage: --aug-spec --aug-ship-snr 5.0 --aug-mixup
"""
import numpy as np


def spec_augment(mel, seed, freq_masks=1, time_masks=1, freq_w=16,
                 time_w=24):
    """Zero-mask random bands/frames. Same shape in/out, deterministic."""
    rng = np.random.default_rng(seed)
    m = np.asarray(mel, dtype=np.float64).copy()
    n, T = m.shape[0], m.shape[1]
    for _ in range(max(0, int(freq_masks))):
        if n < 2:
            break
        f0 = int(rng.integers(0, n))
        m[f0:min(n, f0 + max(1, int(freq_w * rng.random())))] = m.mean()
    for _ in range(max(0, int(time_masks))):
        if T < 2:
            break
        t0 = int(rng.integers(0, T))
        m[:, t0:min(T, t0 + max(1, int(time_w * rng.random())))] = m.mean()
    return m.astype(np.float32)


def mixup(mel_a, mel_b, lam=0.4, seed=0):
    """Convex combo of two mels (time-aligned by truncation). Returns
    (mixed, lam). lam is drawn from Uniform(0.2, 0.6) when lam is None."""
    rng = np.random.default_rng(seed)
    lam = float(rng.uniform(0.2, 0.6)) if lam is None else float(lam)
    a = np.asarray(mel_a, dtype=np.float64)
    b = np.asarray(mel_b, dtype=np.float64)
    T = min(a.shape[1], b.shape[1])
    n = min(a.shape[0], b.shape[0])
    return (lam * a[:n, :T] + (1.0 - lam) * b[:n, :T]).astype(np.float32), lam


def ship_noise(n, sr, seed, snr_db=5.0, ref_rms=0.1):
    """Ship-like interferer: brown noise + 60/90/120 Hz engine lines with
    ~8 Hz propeller modulation, scaled so 20*log10(ref_rms/noise_rms)=snr.
    Returns float32 noise of length n."""
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(n)
    brown = np.cumsum(w)
    brown = brown / max(np.abs(brown).max(), 1e-9)
    t = np.arange(n) / sr
    tonal = sum(np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28))
                for f in (60.0, 90.0, 120.0, 180.0))
    tonal = tonal / max(np.abs(tonal).max(), 1e-9)
    mod = 0.6 + 0.4 * np.sin(2 * np.pi * 8.0 * t)
    mix = 0.6 * brown + 0.4 * tonal * mod
    mix = mix / max(np.sqrt((mix ** 2).mean()), 1e-12)
    scale = ref_rms / (10.0 ** (float(snr_db) / 20.0))
    return (mix * scale).astype(np.float32)


def augment_clip(x, sr, seed, ship_snr=None):
    """Waveform-level augmentation: optional ship-noise injection."""
    x = np.asarray(x, dtype=np.float32)
    if ship_snr is None:
        return x
    ref = float(np.sqrt((x.astype(np.float64) ** 2).mean()) + 1e-12)
    return (x + ship_noise(len(x), sr, seed, float(ship_snr),
                           ref_rms=ref)).astype(np.float32)


def augment_mel(mel, seed, use_spec=True):
    """Mel-level augmentation: SpecAugment view (or clean copy)."""
    if not use_spec:
        return np.asarray(mel, dtype=np.float32)
    return spec_augment(mel, seed)
