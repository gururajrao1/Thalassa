"""App-level test: background job entry + progress hook + cache reuse."""

import json
import os

from app import PRESETS, run_batch_job, run_job


def test_app_job_fast_cached(tmp_path):
    params = dict(PRESETS["Fast (kmeans, instant)"])
    outdir = str(tmp_path / "out")
    stages = []
    res = run_job(str(tmp_path / "missing.wav"), outdir, params,
                  progress_cb=lambda s, f: stages.append((s, f)))
    assert os.path.isfile(res["json"])
    order = ["start", "load", "denoise", "vad", "units", "syntax",
             "report", "done", "cache-hit"]
    assert [order.index(s) for s, _ in stages] == sorted(
        order.index(s) for s, _ in stages)
    with open(res["json"]) as f:
        assert len(json.load(f)["segments"]) >= 1
    synth = os.path.join(outdir, "synthetic.wav")
    first = run_job(synth, str(tmp_path / "job"), params)
    assert first.get("cached") is not True
    second = run_job(synth, str(tmp_path / "job"), params)
    assert second.get("cached") is True


def test_species_model_wired(tmp_path):
    """Shipped learned head names the clip next to the rules."""
    import eval_accuracy as EA
    wav = EA.humpbackish()
    params = dict(PRESETS["Champion (encodec-rvq, efficient)"])
    res = run_job(wav, str(tmp_path / "m"), params)
    with open(res["json"]) as f:
        payload = json.load(f)
    sm = payload.get("species_model")
    assert sm and sm["label"] == "humpback"
    assert sm["confidence"] > 0.5


def test_batch_parallel_parity(tmp_path):
    """run_batch workers=2 must match workers=1 exactly (torch RNG lock)."""
    from whale_song_syntax_analyzer import generate_synthetic_wav
    indir = str(tmp_path / "wavs")
    os.makedirs(indir, exist_ok=True)
    for i in (0, 1):
        generate_synthetic_wav(os.path.join(indir, f"s{i}.wav"),
                               duration_s=3.0, seed=i)
    params = dict(PRESETS["Champion (encodec-rvq, efficient)"])
    a = run_batch_job(indir, str(tmp_path / "b1"), dict(params))
    b = run_batch_job(indir, str(tmp_path / "b2"),
                      dict(params, max_workers=2))
    ka = [(r["file"], r["n_segments"], r["n_phrases"],
           round(r["gpt_ppl"], 6)) for r in a["files"]]
    kb = [(r["file"], r["n_segments"], r["n_phrases"],
           round(r["gpt_ppl"], 6)) for r in b["files"]]
    assert ka == kb
