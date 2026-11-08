"""Wild-clip eval protocol: confusion matrix + P/R/F1 + threshold sweep.

Replaces boolean ALL-PASS with BirdNET-style measurement: per-class
precision/recall/F1, confusion matrix (unassigned is an explicit predicted
column), confidence-threshold sweep with max-F1 operating point, and a
Platt-scaling calibration stub that refuses to fit until N>=30 wild clips.

Clip set = 4 synthetic probes (exact ground truth) + wild clips with
ASSUMED labels (flagged as such; provenance in LIB_NOTES). Probes alone
(n=4) are statistically meaningless — reported separately, not pooled.
Usage: python eval_matrix.py
"""
import csv
import json
import os

from app import PRESETS, run_job
import eval_accuracy as EA

CHAMPION = "Champion (encodec-rvq, efficient)"

# (clip path, true label, assumed?, lowcut)
LIB_NOTES = {
    "probe_blue.wav": "synthetic ground truth",
    "probe_fin.wav": "synthetic ground truth",
    "probe_humpback.wav": "synthetic ground truth",
    "probe_noise.wav": "synthetic ground truth",
    "mbari_2152.wav": "ASSUMED humpback: same deployment family as library.json",
    "mbari_2133.wav": "ASSUMED humpback: same deployment family as library.json",
    "mbari_2143.wav": "ASSUMED humpback: same deployment family as library.json",
    "bluewhale.wav": "ASSUMED blue-like: unverified archive.org field clip, noisy",
}

CLIPS = [
    ("probe_blue", "blue-like", False, 10.0),
    ("probe_fin", "fin-like", False, 10.0),
    ("probe_humpback", "humpback-like", False, 50.0),
    ("probe_noise", "unassigned", False, 50.0),
    ("mbari_2152", "humpback-like", True, 50.0),
    ("mbari_2133", "humpback-like", True, 50.0),
    ("mbari_2143", "humpback-like", True, 50.0),
    ("bluewhale", "blue-like", True, 10.0),
]

LABELS = ["humpback-like", "blue-like", "fin-like", "unassigned"]


def ensure_probes():
    EA.blue_moan()
    EA.fin_pulses()
    EA.humpbackish()
    EA.noise_bed()


def predict(name, lowcut):
    base = dict(PRESETS[CHAMPION])
    params = dict(base, lowcut=lowcut)
    wav = (os.path.join("outputs_eval", name + ".wav") if name.startswith("probe_")
           else os.path.join("samples", name + ".wav"))
    r = run_job(wav, os.path.join("outputs_matrix", name), params)
    sp = json.load(open(r["json"]))["species_hint"]
    return sp["label"], float(sp["confidence"])


def prf(rows, thresh):
    """rows: (true, pred_label, conf). conf < thresh -> abstain to unassigned."""
    eff = [(t, "unassigned" if c < thresh else p) for t, p, c in rows]
    out = {}
    for lab in LABELS:
        tp = sum(1 for t, p in eff if t == lab and p == lab)
        fp = sum(1 for t, p in eff if t != lab and p == lab)
        fn = sum(1 for t, p in eff if t == lab and p != lab)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[lab] = {"precision": round(prec, 3), "recall": round(rec, 3),
                    "f1": round(f1, 3), "tp": tp, "fp": fp, "fn": fn}
    acc = sum(1 for t, p in eff if t == p) / len(eff)
    macro = sum(v["f1"] for v in out.values()) / len(out)
    return out, round(acc, 3), round(macro, 3), eff


def main():
    ensure_probes()
    rows, probe_rows = [], []
    for name, true, assumed, lowcut in CLIPS:
        pred, conf = predict(name, lowcut)
        rows.append((true, pred, conf, assumed, name))
        if not assumed:
            probe_rows.append((true, pred, conf))
        print(f"{name}: true={true} pred={pred} conf={conf:.2f} "
              f"({'assumed' if assumed else 'ground truth'})")
    # Confusion matrix over all rows (wild rows flagged).
    matrix = {t: {p: 0 for p in LABELS} for t in LABELS}
    for t, p, _, _, _ in rows:
        matrix[t][p if p in LABELS else "unassigned"] += 1
    # Metrics on ground-truth probes only + all rows.
    bare = [(t, p, c) for t, p, c, _, _ in rows]
    out, acc, macro, _ = prf(bare, 0.0)
    pout, pacc, pmacro, _ = prf(probe_rows, 0.0)
    # Threshold sweep (abstention): BirdNET-style operating points.
    sweep = []
    for i in range(1, 10):
        t = i / 10.0
        o, a, m, _ = prf(bare, t)
        sweep.append({"threshold": t, "accuracy": a, "macro_f1": m,
                      "per_class": o})
    best = max(sweep, key=lambda s: (s["macro_f1"], s["accuracy"]))
    # Calibration guard: Platt scaling needs real N.
    n_wild = sum(1 for r in rows if r[3])
    calib = ("REFUSED: Platt scaling needs >=30 wild labeled clips "
             f"(have {n_wild}); confidences remain uncalibrated placeholders."
             if n_wild < 30 else "fit allowed")
    os.makedirs("outputs_matrix", exist_ok=True)
    report = {
        "preset": CHAMPION,
        "confusion_matrix": matrix,
        "all_rows_macro_f1@0.0": macro,
        "all_rows_accuracy@0.0": acc,
        "probes_only": {"accuracy": pacc, "macro_f1": pmacro,
                        "note": "n=4, statistically meaningless, smoke only"},
        "threshold_sweep": sweep,
        "best_operating_point": {"threshold": best["threshold"],
                                 "macro_f1": best["macro_f1"],
                                 "accuracy": best["accuracy"]},
        "calibration": calib,
        "clip_notes": {k: v for k, v in LIB_NOTES.items()},
    }
    json.dump(report, open("outputs_matrix/report.json", "w"), indent=1)
    with open("outputs_matrix/confusion.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true \\ pred"] + LABELS)
        for t in LABELS:
            w.writerow([t] + [matrix[t][p] for p in LABELS])
    print(f"\nprobes-only: acc {pacc} macro-F1 {pmacro} (n=4, smoke only)")
    print(f"all rows: acc {acc} macro-F1 {macro}")
    print(f"best operating point: t={best['threshold']} "
          f"macro-F1={best['macro_f1']} acc={best['accuracy']}")
    print(f"calibration: {calib}")
    return report


if __name__ == "__main__":
    main()
