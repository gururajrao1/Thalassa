"""Multi-whale species recognizer: pit model families, ship the winner.

Classes the data actually supports: humpback, blue, fin, unassigned
(noise), other (right-whale confuser). Labels come from synthetic
ground truth, DCLDE 2013 Raven logs, and flagged assumed clips —
never from the rule-based hint being evaluated.

Contenders (all leave-one-out evaluated):
  A. file-vector + multinomial logistic (7 syntax numbers, ~free)
  B. token-histogram + multinomial logistic (64 code counts)
  D. dual-path (8 waveform stats + pooled mel) + logistic (numpy, ~free)
  C. frozen foundation embeddings + multinomial logistic (AST, else
     mel-pool fallback flagged; ships class centroids for per-call scoring)
Winner (best LOO accuracy, ties -> cheapest) is saved to
models/species_head.pkl with scaler + metadata; run_pipeline loads it
when present and reports payload["species_model"] next to the rules.
Usage: python train_species.py
"""
import json
import os

import numpy as np

LABELS = ["humpback", "blue", "fin", "unassigned", "other"]
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "species_head.pkl")
META_PATH = os.path.join(MODEL_DIR, "species_head.json")


def true_of(name, true, flag):
    if name.startswith("wild/"):
        if "RIWH" in str(flag):
            return "other"
    t = str(true)
    if t.endswith("-like"):
        t = t[:-5]
    return t if t in LABELS else "unassigned"


def load_rows():
    import eval_matrix as EM
    rows = []
    for name, true, flag, lowcut in list(EM.CLIPS) + list(EM.WILD):
        sub = os.path.join("outputs_matrix", name.replace("/", "_"))
        jf = os.path.join(sub, "song_tokens.json")
        if not os.path.isfile(jf):
            print(f"SKIP {name}: no payload (run eval_matrix.py first)")
            continue
        p = json.load(open(jf))
        rows.append((name, true_of(name, true, flag), bool(flag), p))
    return rows


def feat_filevec(p):
    m = p["lm"]
    return [m["entropy_bits_per_token"], m["bigram_perplexity"],
            m["perplexity"], m.get("zipf_slope", 0.0),
            m.get("trigram_gap_nats", 0.0),
            m.get("shuffle_gap_nats", 0.0),
            p["codebook"]["utilization"]["used"]]


def feat_tokhist(p, K=64):
    h = np.zeros(K)
    for s in p["segments"]:
        for t in s["tokens"]:
            if 0 <= t < K:
                h[t] += 1
    return (h / max(h.sum(), 1)).tolist()


def clip_wav_path(name):
    if name.startswith("probe_"):
        return os.path.join("outputs_eval", name + ".wav")
    if name.startswith("wild/"):
        return os.path.join("wild_dclde", name.split("/")[1] + ".wav")
    return os.path.join("samples", name + ".wav")


def segment_agreement(rows, y, centroids, backend):
    """Per-call scoring benchmark: nearest-centroid label per VAD segment.

    Only clips with a real species label count; unassigned/other clips
    report abstention (no segment may claim a whale species). Segments
    beyond the 12-window cap are uniformly subsampled, mirroring inference.
    Returns (macro_agreement, per_class_dict, abstention, n_abst_clips).
    """
    from train_probe import embed_segments, centroid_scores
    per_lab, other_ok, other_n = {}, [], 0
    for (name, true, _flag, p), yi in zip(rows, y):
        path = clip_wav_path(name)
        if not os.path.isfile(path):
            continue
        bounds = [(s.get("start_s", 0.0), s.get("end_s", 0.0))
                  for s in p.get("segments", [])]
        if not bounds:
            continue
        try:
            Xs, _ = embed_segments(path, bounds, backend=backend)
        except Exception:
            continue
        if len(Xs) == 0:
            continue
        pred, _ = centroid_scores(Xs, centroids, LABELS)
        lab_pred = [LABELS[i] for i in pred]
        if LABELS[yi] in ("unassigned", "other"):
            other_n += 1
            other_ok.append(all(l in ("unassigned", "other")
                                for l in lab_pred))
        else:
            agree = sum(1 for l in lab_pred if l == LABELS[yi]) / len(lab_pred)
            per_lab.setdefault(LABELS[yi], []).append(agree)
    per_class = {lab: round(float(np.mean(v)), 3)
                 for lab, v in per_lab.items() if v}
    scored = [v for vs in per_lab.values() for v in vs]
    macro = round(float(np.mean(scored)), 3) if scored else 0.0
    abst = round(float(np.mean(other_ok)), 3) if other_ok else None
    return macro, per_class, abst, other_n


def loo_score(X, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneOut
    from sklearn.preprocessing import StandardScaler
    Xs = StandardScaler().fit_transform(np.asarray(X, dtype=float))
    pred = np.zeros_like(y)
    for tr, te in LeaveOneOut().split(Xs):
        clf = LogisticRegression(max_iter=2000).fit(Xs[tr], y[tr])
        pred[te] = clf.predict(Xs[te])
    return pred, float((pred == y).mean())


def main():
    rows = load_rows()
    names = [r[0] for r in rows]
    y = np.array([LABELS.index(r[1]) for r in rows])
    print(f"dataset: {len(rows)} clips, " +
          ", ".join(f"{lab}={int((y == i).sum())}"
                    for i, lab in enumerate(LABELS)))
    results = {}
    Xa = [feat_filevec(r[3]) for r in rows]
    pa, aa = loo_score(Xa, y)
    results["A:file-vector+logreg"] = (pa, aa)
    print(f"A file-vector+logreg LOO acc: {aa:.3f}")
    Xb = [feat_tokhist(r[3]) for r in rows]
    pb, ab = loo_score(Xb, y)
    results["B:token-histogram+logreg"] = (pb, ab)
    print(f"B token-histogram+logreg LOO acc: {ab:.3f}")
    from whale_song_syntax_analyzer import load_wav, dual_path_features
    Xd = []
    for n in names:
        try:
            x, sr = load_wav(clip_wav_path(n))
            Xd.append(dual_path_features(x, sr))
        except Exception:
            Xd.append([0.0] * (8 + 2 * 128))
    pd, ad = loo_score(Xd, y)
    results["D:dual-path+logreg"] = (pd, ad)
    print(f"D dual-path(waveform+mel)+logreg LOO acc: {ad:.3f}")
    emb_backend, emb_fallback, centroids = "none", True, {}
    try:
        from train_probe import embed
        paths = [clip_wav_path(n) for n in names]
        Xc, emb_backend, emb_fallback = embed(paths)
        pc, ac = loo_score(Xc, y)
        results["C:AST-frozen+logreg"] = (pc, ac)
        print(f"C foundation-frozen+logreg LOO acc: {ac:.3f} "
              f"(backend={emb_backend} fallback={emb_fallback})")
        for i, lab in enumerate(LABELS):
            if (y == i).any():
                centroids[lab] = Xc[y == i].mean(axis=0)
        macro, per_class, abst, n_abst = segment_agreement(
            rows, y, centroids, emb_backend)
        print(f"C per-call segment agreement (clip-label match): "
              f"macro {macro:.3f} {per_class}")
        if abst is not None:
            print(f"C confuser abstention: {abst:.3f} (n={n_abst})")
    except Exception as e:
        print(f"C foundation skipped ({e})")
    order = ["A:file-vector+logreg", "B:token-histogram+logreg",
             "D:dual-path+logreg", "C:AST-frozen+logreg"]
    avail = [(k, results[k][1]) for k in order if k in results]
    winner = max(avail, key=lambda kv: kv[1])[0]
    print(f"winner: {winner}")
    print("confusion (rows true, cols pred):")
    pred = results[winner][0]
    print("  " + " ".join(f"{lab[:5]:>5}" for lab in LABELS))
    for i, lab in enumerate(LABELS):
        print(f"  {lab[:5]:>5} " +
              " ".join(f"{int(((pred[y == i]) == j).sum()):>5}"
                       for j in range(len(LABELS))))
    # Ship the winner: refit on all rows, save scaler+model+metadata.
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    Xw = {"A:file-vector+logreg": Xa, "B:token-histogram+logreg": Xb,
          "D:dual-path+logreg": Xd,
          "C:AST-frozen+logreg": Xc}[winner]
    sc = StandardScaler().fit(np.asarray(Xw, dtype=float))
    clf = LogisticRegression(max_iter=2000).fit(sc.transform(
        np.asarray(Xw, dtype=float)), y)
    os.makedirs(MODEL_DIR, exist_ok=True)
    import joblib
    ship_centroids = (centroids if winner.startswith("C") and centroids
                      else {})
    joblib.dump({"scaler": sc, "clf": clf, "labels": LABELS,
                 "features": winner.split(":")[0],
                 "centroids": {k: np.asarray(v, dtype=float)
                               for k, v in ship_centroids.items()},
                 "centroid_backend": emb_backend if ship_centroids else "none"},
                MODEL_PATH)
    json.dump({"features": winner.split(":")[0], "model": "logreg",
               "loo_accuracy": round(results[winner][1], 3),
               "labels": LABELS, "n": len(rows),
               "embedding_backend": emb_backend,
               "embedding_fallback": bool(emb_fallback),
               "has_centroids": bool(ship_centroids),
               "note": "supervised head; rule-based match_species stays "
                       "as the evidence baseline"}, open(META_PATH, "w"),
              indent=1)
    print(f"saved {MODEL_PATH}")
    return winner


if __name__ == "__main__":
    main()
