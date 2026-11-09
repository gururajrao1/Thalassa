"""Grow library.json from verified wild clips (Raven-log labels, not predictions).

Usage: python append_library.py outputs_wildlib
Reads wild_dclde/manifest.csv for ground-truth labels, pulls file_vectors
from the batch subdirs, merges into library.json with provenance tags.
"""
import csv
import json
import os
import sys

from whale_song_syntax_analyzer import file_vector

LABEL_OF = {"fin-like": "fin", "humpback-like": "humpback",
            "unassigned": "right-whale-confuser"}


def main():
    batch = sys.argv[1] if len(sys.argv) > 1 else "outputs_wildlib"
    man = {r["file"]: r for r in
           csv.DictReader(open("wild_dclde/manifest.csv"))}
    lib = json.load(open("library.json")) if os.path.isfile("library.json") \
        else {"entries": []}
    have = {e["file"] for e in lib["entries"]}
    added = 0
    for f, r in sorted(man.items()):
        if f in have:
            continue
        jf = os.path.join(batch, os.path.splitext(f)[0], "song_tokens.json")
        if not os.path.isfile(jf):
            print(f"SKIP {f}: no batch output (run batch first)")
            continue
        p = json.load(open(jf))
        m = p["lm"]
        vec = file_vector({
            "unigram_bits": m["entropy_bits_per_token"],
            "bigram_ppl": m["bigram_perplexity"],
            "gpt_ppl": m["perplexity"],
            "zipf_slope": m.get("zipf_slope", 0.0),
            "trigram_gap_nats": m.get("trigram_gap_nats", 0.0),
            "shuffle_gap_nats": m.get("shuffle_gap_nats", 0.0),
            "codes_used": p["codebook"]["utilization"]["used"]})
        lib["entries"].append({
            "label": LABEL_OF.get(r["true_label"], r["true_label"]),
            "file": f, "vec": vec,
            "n_segments": p["meta"]["n_segments"],
            "provenance": "DCLDE2013 Stellwagen Bank Raven log, "
                          "DOI 10.25921/zaea-1s39 (" + r["note"] + ")"})
        added += 1
    json.dump(lib, open("library.json", "w"), indent=1)
    print(f"library.json: {len(lib['entries'])} entries (+{added} wild)")


if __name__ == "__main__":
    main()
