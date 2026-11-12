"""Pit model backends/frontends head-to-head on efficiency.

Runs every contender on the same 4 ground-truth probes (blue/fin/humpback/noise)
plus the short real bluewhale clip, then ranks by efficiency = accuracy / time.
Usage: python bench_models.py
"""
import json
import os
import time

from app import run_job
import eval_accuracy as EA

CONFIGS = [
    ("kmeans/mel", {"units": "kmeans", "frontend": "mel", "use_pcen": False}),
    ("fsq/mel", {"units": "fsq", "frontend": "mel", "use_pcen": False}),
    ("fsq/gamma+pcen", {"units": "fsq", "frontend": "gammatone", "use_pcen": True}),
    ("kmeans/gamma+pcen", {"units": "kmeans", "frontend": "gammatone", "use_pcen": True}),
    ("vqvae/mel-1ep", {"units": "vqvae", "frontend": "mel", "use_pcen": False}),
    ("ssl-kmeans/mel", {"units": "ssl-kmeans", "frontend": "mel", "use_pcen": False}),
    ("encodec-rvq/mel", {"units": "encodec-rvq", "frontend": "mel", "use_pcen": False}),
    ("hubert-iter/mel", {"units": "hubert-iter", "frontend": "mel", "use_pcen": False}),
    ("fsq/aug-ship", {"units": "fsq", "frontend": "mel", "use_pcen": False,
                       "aug_spec": True, "aug_ship_snr": 5.0}),
]

BASE = {"epochs_vq": 1, "epochs_lm": 1, "gpt_layers": 1, "gpt_dim": 32,
        "block_size": 8, "use_deltas": False}


def ensure_probes():
    blue = EA.blue_moan()
    fin = EA.fin_pulses()
    hum = EA.humpbackish()
    noise = EA.noise_bed()
    return [(blue, {"blue-like", "unassigned"}, 10.0),
            (fin, {"fin-like", "unassigned"}, 10.0),
            (hum, {"humpback-like"}, 50.0),
            (noise, {"unassigned"}, 50.0)]


def main():
    probes = ensure_probes()
    real = os.path.join("samples", "bluewhale.wav")
    rows = []
    for name, over in CONFIGS:
        t0 = time.perf_counter()
        passes, times, phrases, codes, gaps = 0, [], [], [], []
        detail = []
        ok_all = True
        for wav, allowed, lowcut in probes:
            tag = f"bench_{name.replace('/', '_').replace('+', '')}_{os.path.splitext(os.path.basename(wav))[0]}"
            params = dict(BASE, lowcut=lowcut, **over)
            try:
                r = run_job(wav, os.path.join("outputs_bench", tag), params)
                p = json.load(open(r["json"]))
                label = p["species_hint"]["label"]
                good = label in allowed
                passes += good
                ts = p["meta"].get("timings_s", {}).get("total_s", 0)
                times.append(ts)
                phrases.append(len(p.get("phrases", [])))
                codes.append(p["codebook"]["utilization"]["used"])
                gaps.append(p["lm"].get("shuffle_gap_nats", 0.0))
                detail.append(f"{os.path.basename(wav)}->{label}:{'PASS' if good else 'FAIL'}")
            except Exception as e:  # noqa: BLE001 - bench must report, not crash
                ok_all = False
                detail.append(f"{os.path.basename(wav)}->ERROR:{e}")
        # Short real clip for realism (not scored on accuracy).
        real_t, real_ph = 0.0, 0
        if os.path.isfile(real):
            try:
                tag = f"bench_{name.replace('/', '_').replace('+', '')}_bluewhale"
                r = run_job(real, os.path.join("outputs_bench", tag),
                            dict(BASE, lowcut=10.0, **over))
                p = json.load(open(r["json"]))
                real_t = p["meta"].get("timings_s", {}).get("total_s", 0)
                real_ph = len(p.get("phrases", []))
            except Exception as e:  # noqa: BLE001
                ok_all = False
                detail.append(f"bluewhale->ERROR:{e}")
        wall = time.perf_counter() - t0
        acc = passes / 4.0
        mean_t = sum(times) / len(times) if times else wall
        eff = acc / max(mean_t, 1e-9)
        rows.append({"config": name, "accuracy": round(acc, 3),
                     "passes": f"{passes}/4", "mean_probe_s": round(mean_t, 2),
                     "real_clip_s": round(real_t, 2),
                     "mean_phrases": round(sum(phrases) / len(phrases), 1) if phrases else 0,
                     "mean_codes": round(sum(codes) / len(codes), 1) if codes else 0,
                     "efficiency": round(eff, 4), "ok": ok_all, "detail": detail})
        print(f"{name}: acc {passes}/4 meant {mean_t:.2f}s real {real_t:.2f}s "
              f"eff {eff:.4f} {' '.join(detail)}")
    rows.sort(key=lambda r: (-r["efficiency"], -r["accuracy"], r["mean_probe_s"]))
    os.makedirs("outputs_bench", exist_ok=True)
    json.dump(rows, open("outputs_bench/bench_ranking.json", "w"), indent=1)
    print("\nRANKING (efficiency = accuracy / mean_probe_s):")
    for i, r in enumerate(rows, 1):
        print(f"{i}. {r['config']}: acc {r['passes']} meant {r['mean_probe_s']}s "
              f"eff {r['efficiency']} phrases {r['mean_phrases']} codes {r['mean_codes']}")
    print(f"\nWinner: {rows[0]['config']}")
    return rows


if __name__ == "__main__":
    main()
