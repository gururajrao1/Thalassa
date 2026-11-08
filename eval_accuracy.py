"""Ground-truth species probes: controlled stimuli with known labels.

Measures species-hint accuracy honestly: humpback-like patterned song must
score humpback-like; noise must score unassigned; low-frequency moans and
pulse trains must never score humpback-like.
"""
import os

import numpy as np
from scipy.io import wavfile

from app import PRESETS, run_job

SR = 44100
CASES = {}


def _save(name, x):
    os.makedirs("outputs_eval", exist_ok=True)
    p = os.path.join("outputs_eval", name)
    x = x / (abs(x).max() + 1e-9)
    wavfile.write(p, SR, (0.9 * x * 32767).astype(np.int16))
    return p


def blue_moan(dur=8.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(SR * dur)) / SR
    f0 = 55 - 25 * t / dur
    x = np.sin(2 * np.pi * np.cumsum(f0) / SR)
    x += 0.4 * np.sin(2 * np.pi * np.cumsum(2 * f0) / SR)
    x += 0.05 * rng.standard_normal(len(t))
    return _save("probe_blue.wav", x.astype(np.float32))


def fin_pulses(dur=10.0):
    t = np.arange(int(SR * dur)) / SR
    x = np.zeros_like(t)
    for on in np.arange(0.5, dur, 2.0):
        i, n = int(on * SR), int(0.5 * SR)
        x[i:i + n] = np.sin(2 * np.pi * 22.0 * t[:n]) * np.hanning(n)
    x += 0.03 * np.random.default_rng(1).standard_normal(len(t))
    return _save("probe_fin.wav", x.astype(np.float32))


def humpbackish(seed=2):
    # Proper FM upsweep 300->520Hz + weak 2nd harmonic: keeps clip e500
    # high and centroid song-like (~500Hz), unlike the old phase-bug
    # motif whose derivative sprayed energy to ~2.2kHz.
    n = 22050  # 0.5 s unit
    t = np.arange(n) / SR
    f_inst = 300.0 + 220.0 * t / (n / SR)
    phase = 2 * np.pi * np.cumsum(f_inst) / SR
    motif = np.sin(phase) + 0.3 * np.sin(2 * phase)
    motif = motif * np.hanning(n)
    x = np.zeros(SR * 9, dtype=np.float64)
    for k in range(5):
        i = k * 70560  # exact grid: identical framing, identical tokens
        x[i:i + n] += motif
    return _save("probe_humpback.wav", x.astype(np.float32))


def noise_bed(dur=6.0, seed=3):
    x = np.random.default_rng(seed).standard_normal(int(SR * dur)) * 0.3
    return _save("probe_noise.wav", x.astype(np.float32))


def main():
    p = dict(PRESETS["Fast (kmeans, instant)"])
    low = dict(p, lowcut=10.0)
    jobs = [("blue-like moan", blue_moan(), low, {"blue-like", "unassigned"}),
            ("fin-like pulses", fin_pulses(), low,
             {"fin-like", "unassigned"}),
            ("humpback-like song", humpbackish(), p, {"humpback-like"}),
            ("noise bed", noise_bed(), p, {"unassigned"})]
    ok, rows = True, []
    for title, wav, params, allowed in jobs:
        tag = os.path.splitext(os.path.basename(wav))[0]
        r = run_job(wav, os.path.join("outputs_eval", tag), params)
        import json
        sp = json.load(open(r["json"]))["species_hint"]
        good = sp["label"] in allowed
        ok &= good
        rows.append((title, sp["label"], round(sp["confidence"], 2),
                     "PASS" if good else "FAIL"))
        print(f"{title}: {sp['label']} {sp['confidence']:.2f} "
              f"({'PASS' if good else 'FAIL, want one of ' + str(allowed)})")
    print("ALL PASS" if ok else "FAILURES PRESENT")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
