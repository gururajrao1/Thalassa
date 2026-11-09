"""Thalassa Streamlit fallback (Streamlit MVP, local use).

Run: streamlit run app.py  ->  open http://localhost:8501 in a browser.
"""

import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor

_EXEC = ThreadPoolExecutor(max_workers=1)

PRESETS = {
    "Champion (encodec-rvq, efficient)": {
        "units": "encodec-rvq", "epochs_vq": 1, "epochs_lm": 1,
        "gpt_layers": 1, "gpt_dim": 32, "block_size": 8,
        "use_pcen": False, "use_deltas": False,
    },
    "Balanced (fsq, robust)": {
        "units": "fsq", "epochs_vq": 1, "epochs_lm": 1,
        "gpt_layers": 1, "gpt_dim": 32, "block_size": 16,
        "use_pcen": False, "use_deltas": False,
    },
    "Fast (kmeans, instant)": {
        "units": "kmeans", "epochs_vq": 1, "epochs_lm": 1,
        "gpt_layers": 1, "gpt_dim": 32, "block_size": 8,
        "use_pcen": False, "use_deltas": False,
    },
    "Full (vqvae + GPT)": {
        "units": "vqvae", "epochs_vq": 5, "epochs_lm": 5,
        "gpt_layers": 2, "gpt_dim": 128, "block_size": 64,
        "use_pcen": False, "use_deltas": False,
    },
}


def run_job(wav_path, outdir, params, progress_cb=None):
    from whale_song_syntax_analyzer import run_pipeline
    return run_pipeline(
        wav_path, outdir,
        epochs_vq=params["epochs_vq"], epochs_lm=params["epochs_lm"],
        K=64, seed=params.get("seed", 0),
        gpt_layers=params["gpt_layers"], gpt_dim=params["gpt_dim"],
        block_size=params["block_size"], units=params["units"],
        use_pcen=params.get("use_pcen", False),
        use_deltas=params.get("use_deltas", False),
        fmax=params.get("fmax", 4000.0),
        denoise_backend=params.get("denoise_backend", "gating"),
        pcen_tau=params.get("pcen_tau", 0.06),
        pcen_s=params.get("pcen_s", 0.05),
        pcen_alpha=params.get("pcen_alpha", 0.98),
        pcen_preset=params.get("pcen_preset"),
        vad_offset_db=params.get("vad_offset_db", 7.0),
        vad_hangover_s=params.get("vad_hangover_s", 0.25),
        vad_min_unit_s=params.get("vad_min_unit_s", 0.25),
        vad_merge_gap_s=params.get("vad_merge_gap_s", 0.20),
        ssl_weights=params.get("ssl_weights"),
        raven_import_path=params.get("raven_import_path"),
        lowcut=params.get("lowcut", 50.0),
        frontend=params.get("frontend", "mel"),
        denoise_profile=params.get("denoise_profile", "flat"),
        synthetic_if_missing=True, synth_duration_s=6.0,
        progress_cb=progress_cb, use_cache=True,
    )


def run_batch_job(input_dir, outdir, params, progress_cb=None):
    from whale_song_syntax_analyzer import run_batch
    kw = dict(epochs_vq=params["epochs_vq"], epochs_lm=params["epochs_lm"],
              K=64, seed=params.get("seed", 0),
              gpt_layers=params["gpt_layers"],
              gpt_dim=params["gpt_dim"],
              block_size=params["block_size"], units=params["units"],
              use_pcen=params.get("use_pcen", False),
              use_deltas=params.get("use_deltas", False),
              fmax=params.get("fmax", 4000.0),
              denoise_backend=params.get("denoise_backend", "gating"),
              pcen_tau=params.get("pcen_tau", 0.06),
              pcen_s=params.get("pcen_s", 0.05),
              pcen_alpha=params.get("pcen_alpha", 0.98),
              pcen_preset=params.get("pcen_preset"),
              vad_offset_db=params.get("vad_offset_db", 7.0),
              vad_hangover_s=params.get("vad_hangover_s", 0.25),
              vad_min_unit_s=params.get("vad_min_unit_s", 0.25),
              vad_merge_gap_s=params.get("vad_merge_gap_s", 0.20),
              lowcut=params.get("lowcut", 50.0),
              frontend=params.get("frontend", "mel"),
              denoise_profile=params.get("denoise_profile", "flat"),
              ssl_weights=params.get("ssl_weights"),
              max_workers=params.get("max_workers", 1))
    if progress_cb is not None:
        try:
            progress_cb("start", 0.05)
        except Exception:
            pass
    res = run_batch(input_dir, outdir, **kw)
    if progress_cb is not None:
        try:
            progress_cb("done", 1.0)
        except Exception:
            pass
    return res


def main():
    import streamlit as st

    st.set_page_config(page_title="Thalassa", layout="wide")
    st.markdown(
        "<style>"
        ":root{--abyss:#062631;--panel:#0a3646;--line:#14505f;"
        "--foam:#e8f4f1;--sonar:#3ee6c4;--krill:#ffb454;--alert:#ff6b6b;}"
        ".stApp{background:var(--abyss);color:var(--foam);}"
        ".stApp h1{letter-spacing:.02em;font-weight:750;}"
        ".stApp h2,.stApp h3{letter-spacing:.04em;text-transform:uppercase;"
        "font-size:1.02rem;}"
        ".stApp [data-testid='stMetricValue'],.stApp code,.stApp table{"
        "font-variant-numeric:tabular-nums;font-feature-settings:'tnum';}"
        ".stApp [data-testid='stMetric']{background:var(--panel);"
        "border:1px solid var(--line);border-left:3px solid var(--sonar);"
        "border-radius:10px;padding:10px 12px;}"
        ".stApp [data-testid='stSidebar']{background:#08222c;"
        "border-right:1px solid var(--line);}"
        ".stApp button[kind='primary']{background:var(--sonar);color:#04262e;"
        "font-weight:700;}"
        "@media (prefers-reduced-motion:reduce){.stApp *{animation:none;}}"
        "</style>"
        "<div>HYDROPHONE STATION // NIGHT WATCH</div>",
        unsafe_allow_html=True,
    )
    st.title("Thalassa — Field Console")
    st.caption("Cetacean acoustic watch: detections on the left, "
               "sonar report on the right. Numbers are tabular.")

    with st.sidebar:
        st.header("Run settings")
        preset_name = st.selectbox("Preset", list(PRESETS), index=0)
        params = dict(PRESETS[preset_name])
        params["units"] = st.selectbox(
            "Acoustic units",
            ["kmeans", "fsq", "vqvae", "ssl-kmeans", "encodec-rvq"],
            index=["kmeans", "fsq", "vqvae", "ssl-kmeans",
                   "encodec-rvq"].index(params.get("units", "kmeans")))
        params["denoise_backend"] = st.selectbox(
            "Denoise", ["gating", "noisereduce"], index=0)
        params["use_pcen"] = st.checkbox("PCEN frontend (noisy recordings)", value=False)
        if params["use_pcen"]:
            preset = st.selectbox("PCEN preset", ["custom", "quiet",
                                                 "noisy", "harsh"], index=0)
            params["pcen_preset"] = None if preset == "custom" else preset
            params["pcen_tau"] = st.slider("PCEN tau (s)", 0.02, 0.4, 0.06, 0.01)
            params["pcen_s"] = st.slider("PCEN gain s", 0.01, 0.2, 0.05, 0.01)
            params["pcen_alpha"] = st.slider("PCEN alpha", 0.5, 1.0, 0.98, 0.01)
        else:
            params["pcen_preset"] = None
        params["use_deltas"] = st.checkbox("Delta features (experiment)", value=False)
        params["lowcut"] = 10.0 if st.checkbox(
            "Low-frequency mode (10Hz, blue/fin)", value=False) else 50.0
        params["frontend"] = st.selectbox("Auditory frontend",
                                          ["mel", "gammatone"], index=0)
        params["fmax"] = st.slider("Mel top frequency (Hz)", 2000.0, 8000.0, 4000.0, 500.0)
        with st.expander("Detector (VAD) thresholds"):
            params["vad_offset_db"] = st.slider("VAD offset dB", 3.0, 15.0, 7.0, 0.5)
            params["vad_hangover_s"] = st.slider("Hangover (s)", 0.05, 0.6, 0.25, 0.05)
            params["vad_min_unit_s"] = st.slider("Min unit (s)", 0.1, 0.8, 0.25, 0.05)
            params["vad_merge_gap_s"] = st.slider("Merge gap (s)", 0.05, 0.5, 0.20, 0.05)
        cfg_json = json.dumps(params, indent=2)
        st.download_button("Download config JSON", cfg_json, "whale_config.json")
        go = st.button("Analyze", type="primary")

    up = st.file_uploader("Hydrophone WAV (44.1 kHz)", type=["wav"])
    raven_up = st.file_uploader("Raven selection table (optional seeds)",
                                type=["txt", "tsv"])
    batch_ups = st.file_uploader("Batch WAVs (optional, 2+ for comparison)",
                                 type=["wav"], accept_multiple_files=True)
    cfg_up = st.file_uploader("Config JSON (optional)", type=["json"])
    if cfg_up is not None:
        try:
            loaded = json.load(cfg_up)
            for k, v in loaded.items():
                params[k] = v
            st.caption("Config loaded.")
        except Exception as e:
            st.warning(f"Config load failed: {e}")
    use_demo = st.checkbox("Use synthetic demo when no file is uploaded", value=True)

    if not go:
        st.info("Choose a preset, upload a WAV or use the demo, then press Analyze.")
        return

    tmpdir = tempfile.mkdtemp(prefix="whale_app_")
    if up is not None:
        wav_path = os.path.join(tmpdir, "upload.wav")
        with open(wav_path, "wb") as f:
            f.write(up.getbuffer())
    elif use_demo:
        wav_path = os.path.join(tmpdir, "missing.wav")
    else:
        st.warning("Upload a WAV file or enable the synthetic demo.")
        return
    if raven_up is not None:
        raven_path = os.path.join(tmpdir, "raven_in.txt")
        with open(raven_path, "wb") as f:
            f.write(raven_up.getbuffer())
        params["raven_import_path"] = raven_path
    outdir = os.path.join(tmpdir, "out")
    if batch_ups and len(batch_ups) >= 2:
        batch_dir = os.path.join(tmpdir, "batch_in")
        os.makedirs(batch_dir, exist_ok=True)
        for i, b in enumerate(batch_ups):
            with open(os.path.join(batch_dir, f"f{i:02d}_{b.name}"),
                      "wb") as f:
                f.write(b.getbuffer())
        batch_out = os.path.join(tmpdir, "batch_out")
        progress_b = st.progress(0.0, text="Batch…")

        def _cb2(stage, frac):
            try:
                progress_b.progress(min(1.0, float(frac)), text=str(stage))
            except Exception:
                pass

        fut_b = _EXEC.submit(run_batch_job, batch_dir, batch_out,
                             dict(params), _cb2)
        with st.spinner("Batch analyzing…"):
            bres = fut_b.result()
        progress_b.progress(1.0, text="Done")
        st.subheader("Batch comparison")
        st.write(bres["files"])
        if os.path.isfile(os.path.join(batch_out, "trend_report.png")):
            st.image(os.path.join(batch_out, "trend_report.png"),
                     use_column_width=True)
        if os.path.isfile(os.path.join(batch_out, "population_matrix.json")):
            with open(os.path.join(batch_out,
                                   "population_matrix.json"), "rb") as f:
                st.download_button("Download population matrix", f,
                                   "population_matrix.json")

    progress = st.progress(0.0, text="Starting…")
    state = {"stage": "start"}

    def _cb(stage, frac):
        state["stage"] = stage
        try:
            progress.progress(min(1.0, float(frac)), text=str(stage))
        except Exception:
            pass

    fut = _EXEC.submit(run_job, wav_path, outdir, dict(params), _cb)
    with st.spinner("Analyzing…"):
        res = fut.result()
    progress.progress(1.0, text="Done")

    with open(res["json"]) as f:
        payload = json.load(f)
    m = res["metrics"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Segments", res["n_segments"])
    c2.metric("Bigram PPL", f"{m['bigram_ppl']:.2f}")
    c3.metric("GPT PPL", f"{m['gpt_ppl']:.2f}")
    c4.metric("Codes used", f"{m['codes_used']}/64")
    st.caption(f"Unigram entropy {m['unigram_bits']:.2f} bits · "
               f"trigram gap {m.get('trigram_gap_nats', 0.0):+.3f} nats · "
               f"shuffle gap {m.get('shuffle_gap_nats', 0.0):+.3f} nats · "
               f"Zipf slope {m.get('zipf_slope', 0.0):.2f} · "
               f"phrases {res.get('n_phrases', 0)} · "
               f"themes {res.get('n_themes', 0)} · "
               f"cached={res.get('cached', False)}")
    if up is not None:
        st.audio(up.getbuffer())

    left, right = st.columns([7, 5])
    with left:
        st.subheader("Detections")
        for seg in payload["segments"][:24]:
            toks = " ".join(f"{t:02d}" for t in seg["tokens"][:48])
            hz = (f" · peak {seg['peak_hz']:.0f}Hz/ctr "
                  f"{seg['centroid_hz']:.0f}Hz" if "peak_hz" in seg else "")
            st.write(f"#{seg['id']} {seg['start_s']:.2f}–{seg['end_s']:.2f}s "
                     f"· {seg['rms_db']:.1f} dB · theme "
                     f"{seg.get('theme', 0)}{hz} · `{toks}`")
        if len(payload["segments"]) > 24:
            st.caption(f"…and {len(payload['segments']) - 24} more.")

        phrases = payload.get("phrases", [])[:12]
        if phrases:
            st.subheader("Repeated phrases")
            for ph in phrases:
                st.write(f"{ph['count']}× segments {ph['segments']} · "
                         f"`{' '.join(f'{t:02d}' for t in ph['phrase'])}`")
        themes = payload.get("themes", [])[:12]
        if themes:
            st.subheader("Themes (provisional)")
            for th in themes:
                st.write(f"theme {th['theme_id']}: {th['size']} codes · "
                         f"`{' '.join(f'{t:02d}' for t in th['codes'][:32])}`")

        sp = payload.get("species_hint", {})
        if sp:
            st.subheader("Species")
            st.write(f"{sp.get('label', 'unassigned')} · "
                     f"confidence {float(sp.get('confidence', 0.0)):.2f} · "
                     f"{sp.get('note', '')}")
            for e in sp.get("evidence", []):
                st.write(f"- {e}")
        mt = payload.get("meaning_tags", [])[:24]
        if mt:
            st.subheader("Meaning")
            for t in mt:
                st.write(f"segment {t.get('segment')}: {t.get('context')} · "
                         f"{float(t.get('confidence', 0.0)):.2f}")
            st.caption("Provisional tags — requires human audit.")
    with right:
        st.subheader("Sonar report")
        st.image(res["png"], use_column_width=True)
        with open(res["json"], "rb") as f:
            st.download_button("Download song_tokens.json", f,
                               "song_tokens.json")
        with open(res["pdf"], "rb") as f:
            st.download_button("Download report PDF", f,
                               "syntax_analysis_report.pdf")
        if os.path.isfile(res.get("table", "")):
            with open(res["table"], "rb") as f:
                st.download_button("Download Raven selection table", f,
                                   "selection_table.txt")


if __name__ == "__main__":
    main()
