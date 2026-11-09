"""Thalassa API server (FastAPI) + static dashboard.

Run:  uvicorn server:app --port 8000
Then: open http://localhost:8000 in a browser.
"""

import json
import os
import uuid

from fastapi import (BackgroundTasks, FastAPI, File, Form, HTTPException,
                     UploadFile)
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from app import PRESETS, run_batch_job, run_job

ROOT = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(ROOT, "web")
JOBS_DIR = os.path.join(ROOT, "jobs")
os.makedirs(JOBS_DIR, exist_ok=True)

ARTIFACTS = ("song_tokens.json", "syntax_analysis_report.png",
             "syntax_analysis_report.pdf", "selection_table.txt",
             "trend_report.png", "trend_report.pdf",
             "population_matrix.json", "summary_by_file.json")

PROGRESS: dict = {}


def _library():
    try:
        with open(os.path.join(ROOT, "library.json")) as f:
            return json.load(f).get("entries", [])
    except Exception:
        return []

app = FastAPI(title="Thalassa")


def _parse_params(preset: str, raw: str | None) -> dict:
    base = PRESETS.get(preset or "", PRESETS["Fast (kmeans, instant)"])
    params = dict(base)
    if raw:
        try:
            params.update(json.loads(raw))
        except Exception as e:
            raise HTTPException(400, f"bad params JSON: {e}")
    return params


def _save(upload: UploadFile, path: str) -> None:
    with open(path, "wb") as f:
        f.write(upload.file.read())


@app.get("/api/health")
def health():
    return {"ok": True, "presets": sorted(PRESETS)}


@app.get("/api/presets")
def presets():
    return {"presets": PRESETS}


@app.get("/api/samples")
def samples():
    import wave
    sdir = os.path.join(ROOT, "samples")
    out = []
    if os.path.isdir(sdir):
        for name in sorted(os.listdir(sdir)):
            if not name.lower().endswith(".wav"):
                continue
            p = os.path.join(sdir, name)
            try:
                with wave.open(p, "rb") as w:
                    dur = w.getnframes() / float(w.getframerate() or 1)
            except Exception:
                try:
                    from scipy.io import wavfile as _wv
                    _sr, _x = _wv.read(p)
                    dur = len(_x) / float(_sr or 1)
                except Exception:
                    dur = 0.0
            out.append({"name": name, "bytes": os.path.getsize(p),
                        "duration_s": round(dur, 1)})
    return {"samples": out}


@app.post("/api/analyze")
async def analyze(background: BackgroundTasks,
                  wav: UploadFile | None = File(None),
                  preset: str = Form("Champion (encodec-rvq, efficient)"),
                  params: str | None = Form(None),
                  raven: UploadFile | None = File(None),
                  sample: str | None = Form(None)):
    job = uuid.uuid4().hex[:12]
    jdir = os.path.join(JOBS_DIR, job)
    indir = os.path.join(jdir, "in")
    outdir = os.path.join(jdir, "out")
    os.makedirs(indir, exist_ok=True)
    if sample:
        base = os.path.basename(sample)
        spath = os.path.join(ROOT, "samples", base)
        if not os.path.isfile(spath):
            raise HTTPException(404, "unknown sample")
        wpath = os.path.join(indir, base)
        import shutil
        shutil.copyfile(spath, wpath)
    elif wav is not None:
        wpath = os.path.join(indir, wav.filename or "upload.wav")
        _save(wav, wpath)
    else:
        raise HTTPException(400, "upload a WAV or pick a sample")
    p = _parse_params(preset, params)
    if raven is not None:
        rpath = os.path.join(indir, "raven_in.txt")
        _save(raven, rpath)
        p["raven_import_path"] = rpath

    PROGRESS[job] = {"stage": "queued", "frac": 0.0}
    background.add_task(_do_analyze, job, wpath, outdir, p)
    return {"job": job}


def _do_analyze(job, wpath, outdir, p):
    from whale_song_syntax_analyzer import file_vector, match_library
    rfile = os.path.join(JOBS_DIR, job, "result.json")
    try:
        def _cb(stage, frac):
            PROGRESS[job] = {"stage": str(stage), "frac": float(frac)}
        res = run_job(wpath, outdir, p, progress_cb=_cb)
        with open(res["json"]) as f:
            payload = json.load(f)
        try:
            lib = match_library(file_vector({
                "unigram_bits": res["metrics"]["unigram_bits"],
                "bigram_ppl": res["metrics"]["bigram_ppl"],
                "gpt_ppl": res["metrics"]["gpt_ppl"],
                "zipf_slope": res["metrics"].get("zipf_slope", 0.0),
                "trigram_gap_nats": res["metrics"].get("trigram_gap_nats",
                                                       0.0),
                "shuffle_gap_nats": res["metrics"].get("shuffle_gap_nats",
                                                       0.0),
                "codes_used": res["metrics"]["codes_used"]}), _library())
        except Exception:
            lib = {"matches": [], "verdict": "unassigned"}
        out = {"job": job, "metrics": res["metrics"],
               "n_segments": res["n_segments"],
               "n_phrases": res.get("n_phrases", 0),
               "n_themes": res.get("n_themes", 0),
               "cached": res.get("cached", False),
               "payload": payload,
               "library": lib,
               "files": {n: f"/api/files/{job}/{n}" for n in ARTIFACTS
                         if os.path.isfile(os.path.join(outdir, n))}}
    except Exception as e:
        out = {"error": str(e)}
    with open(rfile, "w") as f:
        json.dump(out, f)
    PROGRESS[job] = {"stage": "done", "frac": 1.0}


@app.get("/api/result/{job}")
def result(job: str):
    rfile = os.path.join(JOBS_DIR, job, "result.json")
    if not os.path.isfile(rfile):
        raise HTTPException(404, "result not ready")
    return JSONResponse(json.load(open(rfile)))


@app.get("/api/progress/{job}")
def progress(job: str):
    return PROGRESS.get(job, {"stage": "queued", "frac": 0.0})


@app.get("/api/clip/{job}/{seg}")
def clip(job: str, seg: int):
    indir = os.path.join(JOBS_DIR, job, "in")
    outdir = os.path.join(JOBS_DIR, job, "out")
    wavs = [f for f in os.listdir(indir) if f.lower().endswith(".wav")] \
        if os.path.isdir(indir) else []
    if not wavs or not os.path.isfile(os.path.join(outdir,
                                                   "song_tokens.json")):
        raise HTTPException(404, "clip not ready")
    payload = json.load(open(os.path.join(outdir, "song_tokens.json")))
    segs = [s for s in payload["segments"] if s["id"] == seg]
    if not segs:
        raise HTTPException(404, "unknown segment")
    s = segs[0]
    from scipy.io import wavfile as _wv
    import numpy as _np
    sr, x = _wv.read(os.path.join(indir, wavs[0]))
    if x.ndim > 1:
        x = x.mean(axis=1)
    a, b = int(s["start_s"] * sr), int(s["end_s"] * sr)
    cdir = os.path.join(JOBS_DIR, job, "clips")
    os.makedirs(cdir, exist_ok=True)
    cp = os.path.join(cdir, f"seg{seg}.wav")
    if not os.path.isfile(cp):
        part = _np.asarray(x[a:b], dtype=_np.float64)
        part = part / (abs(part).max() + 1e-9)
        _wv.write(cp, sr, (0.9 * part * 32767).astype(_np.int16))
    return FileResponse(cp)


@app.post("/api/batch")
async def batch(files: list[UploadFile] = File(...),
                preset: str = Form("Champion (encodec-rvq, efficient)"),
                params: str | None = Form(None)):
    if len(files) < 2:
        raise HTTPException(400, "upload 2+ WAV files for comparison")
    job = uuid.uuid4().hex[:12]
    jdir = os.path.join(JOBS_DIR, job)
    indir = os.path.join(jdir, "batch_in")
    outdir = os.path.join(jdir, "batch_out")
    os.makedirs(indir, exist_ok=True)
    for i, up in enumerate(files):
        _save(up, os.path.join(indir, f"f{i:02d}_{up.filename or 'x.wav'}"))
    p = _parse_params(preset, params)

    def _run():
        return run_batch_job(indir, outdir, p)
    try:
        res = await run_in_threadpool(_run)
    except FileNotFoundError as e:
        raise HTTPException(400, str(e))
    import glob as _glob
    summaries = {}
    sfile = os.path.join(outdir, "summary_by_file.json")
    if os.path.isfile(sfile):
        with open(sfile) as f:
            summaries = json.load(f)
    pngs = sorted(_glob.glob(os.path.join(outdir, "*", "*.png")))
    return {"job": job, "n_files": res["n_files"], "files": res["files"],
            "summary": summaries,
            "trend": f"/api/files/{job}/trend_report.png"
            if os.path.isfile(os.path.join(outdir, "trend_report.png"))
            else None,
            "population": f"/api/files/{job}/population_matrix.json"
            if os.path.isfile(
                os.path.join(outdir, "population_matrix.json")) else None,
            "previews": [f"/api/files/{job}/{os.path.basename(d)}/"
                         f"syntax_analysis_report.png"
                         for d in sorted(_glob.glob(
                             os.path.join(outdir, "f*")))
                         if os.path.isfile(os.path.join(
                             d, "syntax_analysis_report.png"))][:8]}


@app.get("/api/files/{job}/{name}")
def files(job: str, name: str):
    if "/" in name or "\\" in name or name not in ARTIFACTS:
        raise HTTPException(404, "unknown artifact")
    for cand in (os.path.join(JOBS_DIR, job, "out", name),
                 os.path.join(JOBS_DIR, job, "batch_out", name)):
        if os.path.isfile(cand):
            return FileResponse(cand)
    raise HTTPException(404, "artifact not ready")


@app.get("/api/files/{job}/{sub}/{name}")
def subfiles(job: str, sub: str, name: str):
    if name not in ARTIFACTS or "/" in sub or "\\" in sub:
        raise HTTPException(404, "unknown artifact")
    cand = os.path.join(JOBS_DIR, job, "batch_out", sub, name)
    if os.path.isfile(cand):
        return FileResponse(cand)
    raise HTTPException(404, "artifact not ready")


if os.path.isdir(WEB_DIR):
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True),
              name="web")


@app.get("/api", include_in_schema=False)
def api_root():
    return JSONResponse({"use": "open / in a browser"})
