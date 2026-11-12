"""Denoise benchmark (#5): gating vs learned U-Net mask on held-out pairs.

Builds held-out synthetic noisy-clean pairs (seeds disjoint from the
training script), runs each denoise backend, and reports input/output SNR
per input-SNR level plus the mean lift. Skips the U-Net rows (flagged)
when models/unet_denoiser.pt is absent.
Usage: python eval_denoise.py
"""
import json
import os

import numpy as np

SNRS = (-5.0, 0.0, 5.0)
OUTDIR = "outputs_denoise"


def time_pairs(n_clean=2, seed=1000, sr=44100, duration_s=6.0):
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from whale_song_syntax_analyzer import generate_synthetic_wav
    from scipy.io import wavfile
    os.makedirs(OUTDIR, exist_ok=True)
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n_clean):
        cp = os.path.join(OUTDIR, f"heldout_{i}.wav")
        generate_synthetic_wav(cp, duration_s=duration_s, seed=seed + i)
        _, clean = wavfile.read(cp)
        clean = clean.astype(np.float64) / 32768.0
        for snr in SNRS:
            noise = rng.standard_normal(len(clean))
            scale = np.sqrt((clean ** 2).mean() /
                            max(noise.var(), 1e-12) / (10.0 ** (snr / 10.0)))
            out.append((clean.astype(np.float32),
                        (clean + scale * noise).astype(np.float32), snr))
    return out


def snr_db(clean, est):
    clean = np.asarray(clean, dtype=np.float64)
    est = np.asarray(est, dtype=np.float64)[:len(clean)]
    return float(10 * np.log10((clean ** 2).sum() /
                               max(((clean - est) ** 2).sum(), 1e-12)))


def main():
    from whale_song_syntax_analyzer import (denoise_backend_available,
                                            spectral_gate_denoise)
    from unet_denoise import unet_denoise
    pairs = time_pairs()
    have_unet = denoise_backend_available("unet")
    rows = []
    for clean, noisy, snr_in in pairs:
        row = {"snr_in": snr_in,
               "in": round(snr_db(clean, noisy), 2)}
        g, _ = spectral_gate_denoise(noisy, 44100), None
        row["gating"] = round(snr_db(clean, g), 2)
        if have_unet:
            u, fb = unet_denoise(noisy, 44100)
            row["unet"] = round(snr_db(clean, u), 2)
            row["unet_fallback"] = bool(fb)
        else:
            row["unet"] = None
            row["unet_fallback"] = True
        rows.append(row)
    print("backend SNR (dB), held-out synthetic pairs:")
    for snr in SNRS:
        sub = [r for r in rows if r["snr_in"] == snr]
        line = (f"  in {snr:+.0f} dB: noisy {np.mean([r['in'] for r in sub]):.2f} "
                f"gating {np.mean([r['gating'] for r in sub]):.2f}")
        if have_unet:
            line += f" unet {np.mean([r['unet'] for r in sub]):.2f}"
        print(line)
    if have_unet:
        lift_g = float(np.mean([r["gating"] - r["in"] for r in rows]))
        lift_u = float(np.mean([r["unet"] - r["in"] for r in rows]))
        print(f"mean lift: gating {lift_g:+.2f} dB, unet {lift_u:+.2f} dB")
    else:
        print("unet weights absent: unet rows skipped (fallback flagged)")
        lift_g = float(np.mean([r["gating"] - r["in"] for r in rows]))
        lift_u = None
    os.makedirs(OUTDIR, exist_ok=True)
    json.dump({"rows": rows, "mean_lift_gating_db": round(lift_g, 2),
               "mean_lift_unet_db": (round(lift_u, 2)
                                     if lift_u is not None else None),
               "have_unet": have_unet},
              open(os.path.join(OUTDIR, "denoise_eval.json"), "w"), indent=1)
    return rows


if __name__ == "__main__":
    main()
