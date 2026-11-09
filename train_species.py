"""Multi-whale species recognizer: pit model families, ship the winner.

Classes the data actually supports: humpback, blue, fin, unassigned
(noise), other (right-whale confuser). Labels come from synthetic
ground truth, DCLDE 2013 Raven logs, and flagged assumed clips —
never from the rule-based hint being evaluated.

Contenders (all leave-one-out evaluated):
  A. file-vector + multinomial logistic (7 syntax numbers, ~free)
  B. token-histogram + multinomial logistic (64 code counts)
  C. frozen AST embeddings + multinomial logistic (needs weights)
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
    try:
        from train_probe import embed
        paths = []
        for n in names:
            if n.startswith("probe_"):
                paths.append(os.path.join("outputs_eval", n + ".wav"))
            elif n.startswith("wild/"):
                paths.append(os.path.join("wild_dclde", n.split("/")[1] + ".wav"))
            else:
                paths.append(os.path.join("samples", n + ".wav"))
        Xc = embed(paths)
        pc, ac = loo_score(Xc, y)
        results["C:AST-frozen+logreg"] = (pc, ac)
        print(f"C AST-frozen+logreg LOO acc: {ac:.3f}")
    except Exception as e:
        print(f"C AST skipped ({e})")
    order = ["A:file-vector+logreg", "B:token-histogram+logreg",
             "C:AST-frozen+logreg"]
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
          "C:AST-frozen+logreg": Xc}[winner]
    sc = StandardScaler().fit(np.asarray(Xw, dtype=float))
    clf = LogisticRegression(max_iter=2000).fit(sc.transform(
        np.asarray(Xw, dtype=float)), y)
    os.makedirs(MODEL_DIR, exist_ok=True)
    import joblib
    joblib.dump({"scaler": sc, "clf": clf, "labels": LABELS,
                 "features": winner.split(":")[0]},
                MODEL_PATH)
    json.dump({"features": winner.split(":")[0], "model": "logreg",
               "loo_accuracy": round(results[winner][1], 3),
               "labels": LABELS, "n": len(rows),
               "note": "supervised head; rule-based match_species stays "
                       "as the evidence baseline"}, open(META_PATH, "w"),
              indent=1)
    print(f"saved {MODEL_PATH}")
    return winner


if __name__ == "__main__":
    main()
