"""Build library.json from verified batch outputs (no re-analysis).

Usage: python build_library.py [summary_dir] [out]
Defaults: outputs_mbari/all4 -> library.json
Labels: file names starting with mbari -> humpback; everything else is
labeled from the LABELS map below (edit when adding verified clips).
"""
import json
import os
import sys

from whale_song_syntax_analyzer import file_vector

LABELS = {
    "bluewhale.wav": "unverified field clip (archive.org blue whale)",
}

def main():
    sdir = sys.argv[1] if len(sys.argv) > 1 else "outputs_mbari/all4"
    out = sys.argv[2] if len(sys.argv) > 2 else "library.json"
    entries = []
    for name in sorted(os.listdir(sdir)):
        sub = os.path.join(sdir, name)
        jf = os.path.join(sub, "song_tokens.json")
        if not os.path.isdir(sub) or not os.path.isfile(jf):
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
        wav = p["meta"]["input"]
        label = ("humpback" if wav.startswith("mbari")
                 else LABELS.get(wav, "unassigned"))
        entries.append({"label": label, "file": wav, "vec": vec,
                        "n_segments": p["meta"]["n_segments"]})
    json.dump({"entries": entries}, open(out, "w"), indent=2)
    print(f"wrote {out} with {len(entries)} entries: "
          + ", ".join(f"{e['file']}->{e['label']}" for e in entries))


if __name__ == "__main__":
    main()
