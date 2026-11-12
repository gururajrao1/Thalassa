"""Event-based detection metrics (#7): collars, per-class PR, temporal OOD.

True events come from the DCLDE 2013 Raven log: each manifest clip anchors
on its note detection (species + start ISO); the clip window is
[anchor - pre, anchor - pre + duration] with pre=10 s for FIWH else 0 s
(fetch_wild.py rule), and every same-species log row overlapping that
window is a true event. Clips whose anchor row is missing or whose audio
is shorter than the anchor offset are UNMAPPED: excluded from F1, counted
separately — never silently scored.

Predicted events are time-frequency boxes (payload["detections"], computed
fresh from audio when a payload predates them) versus VAD segments
(payload["segments"]) as the ablation. Matching is sed_eval-style:
greedy by score, TP when onset is within collar_s and intervals overlap;
one true event matches at most one prediction.

Slices: per-class F1, score-threshold sweep with max-F1 operating point
per class, per-day F1 (temporal OOD proxy: different days, same site),
and a confuser box-rate diagnostic (RIWH clips are out-of-scope: any box
there is counted, never scored as TP).
Usage: python eval_events.py [--collar 0.5]
"""
import csv
import datetime as dt
import json
import os
import wave

import numpy as np

WILD = "wild_dclde"
OUT = "outputs_events"
PRE_S = {"FIWH": 10.0}
SP_OF = {"fin-like": "FIWH", "humpback-like": "HUWH",
         "blue-like": "BLWH", "unassigned": "RIWH"}


def load_log():
    for name in ("detections_2025.csv", "detections.csv"):
        p = os.path.join(WILD, name)
        if os.path.isfile(p):
            rows = list(csv.DictReader(open(p)))
            return rows, name
    return [], None


def parse_note(note):
    """'DCLDE2013 FIWH 2009-03-28T00:03:08-05:00 70.7-174.2Hz' -> (sp, iso)."""
    parts = str(note).split()
    if len(parts) >= 3 and parts[0] == "DCLDE2013":
        return parts[1], parts[2]
    return None, None


def clip_duration(path):
    with wave.open(path, "rb") as w:
        return w.getnframes() / float(w.getframerate())


def true_events(clip_file, true_label, note, log_rows):
    """(events, day, status): events = [(t0, t1)] in clip seconds."""
    sp = SP_OF.get(true_label)
    anchor_sp, anchor_iso = parse_note(note)
    if not sp or not anchor_iso:
        return [], None, "unmapped-note"
    anchor = None
    for r in log_rows:
        if (r.get("Species") == (anchor_sp or sp) and
                r.get("Start_DateTime_ISO8601") == anchor_iso):
            anchor = r
            break
    if anchor is None:
        return [], None, "unmapped-anchor"
    try:
        a0 = dt.datetime.fromisoformat(anchor["Start_DateTime_ISO8601"])
    except ValueError:
        return [], None, "unmapped-time"
    pre = PRE_S.get(anchor_sp or "", 0.0)
    path = os.path.join(WILD, clip_file)
    if not os.path.isfile(path):
        return [], None, "unmapped-missing-audio"
    dur = clip_duration(path)
    win0 = -pre
    win1 = win0 + dur
    evs = []
    for r in log_rows:
        if r.get("Species") != sp:
            continue
        try:
            s = dt.datetime.fromisoformat(r["Start_DateTime_ISO8601"])
            e = dt.datetime.fromisoformat(r["End_DateTime_ISO8601"])
        except ValueError:
            continue
        t0 = (s - a0).total_seconds() + pre
        t1 = (e - a0).total_seconds() + pre
        if t1 < win0 or t0 > win1:
            continue
        evs.append((max(0.0, t0), min(dur, t1)))
    evs = [(a, b) for a, b in evs if b > a]
    day = a0.strftime("%Y-%m-%d")
    if not evs:
        return [], day, "unmapped-no-events"
    return evs, day, "mapped"


def event_f1(pred, true, collar_s=0.5):
    """Greedy collar matching. pred: [(t0, t1, score)]; true: [(t0, t1)].
    Returns (precision, recall, f1, tp, fp, fn)."""
    order = sorted(range(len(pred)), key=lambda i: -pred[i][2])
    used = [False] * len(true)
    tp = 0
    for i in order:
        p0, p1, _ = pred[i]
        best, best_on = -1, None
        for j, (t0, t1) in enumerate(true):
            if used[j]:
                continue
            if abs(p0 - t0) <= collar_s and min(p1, t1) > max(p0, t0):
                if best < 0 or abs(p0 - t0) < best_on:
                    best, best_on = j, abs(p0 - t0)
        if best >= 0:
            used[best] = True
            tp += 1
    fp = len(pred) - tp
    fn = len(true) - tp
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return round(prec, 3), round(rec, 3), round(f1, 3), tp, fp, fn


def sweep(pred, true, collar_s=0.5, grid=None):
    """Score-threshold sweep -> PR curve + max-F1 operating point."""
    grid = grid if grid is not None else [round(x * 0.5, 1)
                                          for x in range(0, 21)]
    curve, best = [], None
    for th in grid:
        kept = [p for p in pred if p[2] >= th]
        pr, rc, f1, tp, fp, fn = event_f1(kept, true, collar_s)
        row = {"thresh": th, "precision": pr, "recall": rc, "f1": f1,
               "tp": tp, "fp": fp, "fn": fn}
        curve.append(row)
        if best is None or (f1, -th) > (best["f1"], -best["thresh"]):
            best = row
    return curve, best


def segments_of(clip_file):
    """VAD segments from the eval_matrix payload as the ablation pred set
    (uniform score 1.0). Returns [] when the payload is absent."""
    stem = os.path.splitext(clip_file)[0]
    jf = os.path.join("outputs_matrix", "wild_" + stem, "song_tokens.json")
    if not os.path.isfile(jf):
        return []
    try:
        p = json.load(open(jf))
    except Exception:
        return []
    return [(float(s.get("start_s", 0.0)), float(s.get("end_s", 0.0)), 1.0)
            for s in p.get("segments", [])]


def main(collar=0.5, offset=1.0, min_dur=0.20):
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from whale_song_syntax_analyzer import detect_boxes, log_mel
    from scipy.io import wavfile
    log_rows, log_name = load_log()
    print(f"detection log: {log_name} ({len(log_rows)} rows)")
    man = os.path.join(WILD, "manifest.csv")
    clips = list(csv.DictReader(open(man)))
    per_class_pred, per_class_true = {}, {}
    per_class_seg = {}
    per_day = {}
    confuser_clips, confuser_boxed = 0, 0
    unmapped = []
    for c in clips:
        f, lab, note = c["file"], c["true_label"], c["note"]
        evs, day, status = true_events(f, lab, note, log_rows)
        if status != "mapped":
            unmapped.append((f, status))
            continue
        path = os.path.join(WILD, f)
        sr, data = wavfile.read(path)
        if data.ndim > 1:
            data = data.mean(axis=1)
        m = log_mel((data / 32768.0).astype(np.float32), sr,
                    fmax=min(4000.0, sr / 2 - 1))
        boxes = detect_boxes(m, sr, fmin=50.0,
                             fmax=min(4000.0, sr / 2 - 1),
                             offset=offset, min_dur_s=min_dur)
        pred = [(b["t0_s"], b["t1_s"], b["score"]) for b in boxes]
        seg_pred = segments_of(f)
        if seg_pred:
            per_class_seg.setdefault(lab, ([], []))
            per_class_seg[lab][0].extend(seg_pred)
            per_class_seg[lab][1].extend(evs)
        if lab == "unassigned":
            confuser_clips += 1
            confuser_boxed += 1 if pred else 0
            continue
        per_class_pred.setdefault(lab, []).extend(pred)
        per_class_true.setdefault(lab, []).extend(evs)
        d = per_day.setdefault(day, {"pred": [], "true": []})
        d["pred"].extend(pred)
        d["true"].extend(evs)
    print(f"mapped clips: {len(clips) - len(unmapped)}/{len(clips)}; "
          f"unmapped: {len(unmapped)}")
    for f, st in unmapped:
        print(f"  UNMAPPED {f}: {st}")
    report = {"collar_s": collar, "det_offset": offset,
              "det_min_dur_s": min_dur, "log": log_name, "classes": {},
              "days": {}, "confuser_box_rate": None,
              "unmapped": [{"file": f, "status": s} for f, s in unmapped]}
    for lab in sorted(per_class_true):
        pr, rc, f1, tp, fp, fn = event_f1(per_class_pred[lab],
                                         per_class_true[lab], collar)
        curve, best = sweep(per_class_pred[lab], per_class_true[lab],
                            collar)
        report["classes"][lab] = {
            "n_true": len(per_class_true[lab]),
            "n_pred": len(per_class_pred[lab]),
            "precision": pr, "recall": rc, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn,
            "operating_point": best,
            "pr_curve": [(r["thresh"], r["precision"], r["recall"], r["f1"])
                         for r in curve]}
        print(f"{lab}: TP {tp} FP {fp} FN {fn} P {pr} R {rc} F1 {f1} "
              f"op@thresh {best['thresh']} (F1 {best['f1']})")
        if lab in per_class_seg:
            sp_, st_ = per_class_seg[lab]
            spr, src, sf1, stp, sfp, sfn = event_f1(sp_, st_, collar)
            report["classes"][lab]["segments"] = {
                "precision": spr, "recall": src, "f1": sf1,
                "tp": stp, "fp": sfp, "fn": sfn}
            print(f"  VAD-segments ablation: P {spr} R {src} F1 {sf1} "
                  f"(TP {stp} FP {sfp} FN {sfn})")
        else:
            report["classes"][lab]["segments"] = None
            print("  VAD-segments ablation: no payload segments")
    for day in sorted(per_day):
        d = per_day[day]
        pr, rc, f1, tp, fp, fn = event_f1(d["pred"], d["true"], collar)
        report["days"][day] = {"f1": f1, "tp": tp, "fp": fp, "fn": fn,
                               "n_true": len(d["true"])}
        print(f"day {day}: F1 {f1} (TP {tp} FP {fp} FN {fn})")
    if confuser_clips:
        rate = confuser_boxed / confuser_clips
        report["confuser_box_rate"] = round(rate, 3)
        print(f"confuser box rate: {rate:.3f} "
              f"({confuser_boxed}/{confuser_clips} RIWH clips with any box)")
    os.makedirs(OUT, exist_ok=True)
    json.dump(report, open(os.path.join(OUT, "event_eval.json"), "w"),
              indent=1)
    return report


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--collar", type=float, default=0.5)
    ap.add_argument("--offset", type=float, default=1.0)
    ap.add_argument("--min-dur", type=float, default=0.20)
    main(**vars(ap.parse_args()))
