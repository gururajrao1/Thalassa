const $ = (id) => document.getElementById(id);
const PRESETS = {
  "Champion (encodec-rvq, efficient)": {epochs_vq: 1, epochs_lm: 1, gpt_layers: 1,
    gpt_dim: 32, block_size: 8, units: "encodec-rvq"},
  "Balanced (fsq, robust)": {epochs_vq: 1, epochs_lm: 1, gpt_layers: 1,
    gpt_dim: 32, block_size: 16, units: "fsq"},
  "Fast (kmeans, instant)": {epochs_vq: 1, epochs_lm: 1, gpt_layers: 1,
    gpt_dim: 32, block_size: 8, units: "kmeans"},
  "Full (vqvae + GPT)": {epochs_vq: 5, epochs_lm: 5, gpt_layers: 2,
    gpt_dim: 128, block_size: 64, units: "vqvae"},
};
const PRESET_HINTS = {
  "Champion (encodec-rvq, efficient)": "Bench winner — 4/4 probes, deepest real-song syntax.",
  "Fast (kmeans, instant)": "Quick look — instant unit inventory.",
  "Balanced (fsq, robust)": "Standard — stable units plus syntax scores.",
  "Full (vqvae + GPT)": "Deep — trained units and full language model.",
};
const psel = $("preset");
Object.keys(PRESETS).forEach((k) => psel.add(new Option(k, k)));
function syncPreset() {
  $("units").value = PRESETS[psel.value].units;
  $("preset_hint").textContent = PRESET_HINTS[psel.value] || "";
}
psel.onchange = syncPreset; syncPreset();

let chosenSample = null;
fetch("/api/samples").then((r) => r.json()).then((d) => {
  const box = $("samples");
  if (!d.samples.length) {
    box.innerHTML = "<p class='hint'>No samples bundled yet.</p>";
    return;
  }
  box.innerHTML = d.samples.map((s) =>
    `<button class="ghost" data-s="${s.name}">${s.name} · ` +
    `${s.duration_s}s</button>`).join("");
  box.querySelectorAll("button").forEach((b) => {
    b.onclick = () => {
      chosenSample = b.dataset.s;
      box.querySelectorAll("button").forEach((x) =>
        x.style.borderColor = "");
      b.style.borderColor = "var(--krill)";
      $("status").textContent = `Sample ready: ${chosenSample} — press Analyze song.`;
    };
  });
}).catch(() => {
  $("samples").innerHTML = "<p class='hint'>Samples unavailable.</p>";
});
$("fmax").oninput = (e) => { $("fmaxv").textContent = e.target.value; };
$("vad").oninput = (e) => { $("vadv").textContent = e.target.value; };
$("workers").oninput = (e) => { $("workersv").textContent = e.target.value; };

function params() {
  const p = Object.assign({}, PRESETS[psel.value]);
  p.units = $("units").value;
  p.denoise_backend = $("denoise").value;
  p.use_pcen = $("pcen").checked;
  p.pcen_preset = $("pcen_preset").value || null;
  p.use_deltas = $("deltas").checked;
  p.lowcut = $("lowfreq").checked ? 10.0 : 50.0;
  p.frontend = $("frontend").value;
  p.denoise_profile = $("denoise_profile").value;
  p.fmax = parseFloat($("fmax").value);
  p.vad_offset_db = parseFloat($("vad").value);
  p.max_workers = parseInt($("workers").value, 10) || 1;
  return p;
}
$("dlcfg").onclick = () => {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify(params(),
    null, 2)], {type: "application/json"}));
  a.download = "whale_config.json"; a.click();
};
const fmt = (x, d = 2) => (x === null || x === undefined) ? "–"
  : Number(x).toFixed(d);

function tokline(s, job, i) {
  const toks = s.tokens.slice(0, 48).map((t) => String(t).padStart(2, "0"))
    .join(" ");
  const hz = (s.peak_hz !== undefined) ?
    ` · peak ${s.peak_hz}Hz / ctr ${s.centroid_hz}Hz` : "";
  const play = (job !== null && i < 12) ?
    `<br><audio controls preload="none" style="width:220px;height:28px" ` +
    `src="/api/clip/${job}/${s.id}"></audio>` : "";
  return `<div class="seg">#${s.id} ${s.start_s.toFixed(2)}–` +
    `${s.end_s.toFixed(2)}s · ${s.rms_db.toFixed(1)} dB · ` +
    `theme ${s.theme ?? 0}${hz} · <code>${toks}</code>${play}</div>`;
}

function kv(obj) {
  return Object.entries(obj).map(([k, v]) =>
    `<div><span>${k}</span><b>${v}</b></div>`).join("");
}
function render(res) {
  const m = res.metrics, p = res.payload;
  $("m_seg").textContent = res.n_segments;
  $("m_bi").textContent = fmt(m.bigram_ppl);
  $("m_gpt").textContent = fmt(m.gpt_ppl);
  $("m_codes").textContent = `${m.codes_used}/64`;
  $("m_gap").textContent = (m.shuffle_gap_nats >= 0 ? "+" : "") +
    fmt(m.shuffle_gap_nats, 2);
  $("m_phr").textContent = res.n_phrases;
  const sp0 = p.species_hint;
  const v = $("verdict");
  if (sp0) {
    const abst = sp0.label === "unassigned";
    v.className = "verdict " + (abst ? "abstain" : "found");
    $("v_value").textContent = sp0.label +
      (abst ? " — no call" : "");
    $("v_bar").style.width = Math.round(sp0.confidence * 100) + "%";
    $("v_note").textContent = sp0.note +
      ((sp0.evidence || []).length ?
        " · " + sp0.evidence.filter((e) => e.startsWith("[Y]")).length +
        " supporting checks" : "");
  }
  const th = (p.themes || []).slice(0, 24);
  const mx = Math.max(1, ...th.map((t) => t.size));
  $("tstrip").innerHTML = th.map((t) =>
    `<i class="${t.size === mx ? "dom" : t.size * 2 >= mx ? "mid" : ""}" ` +
    `style="flex:${t.size}" title="theme ${t.theme_id}: ${t.size} codes"></i>`).join("");
  const meta = p.meta || {};
  $("runmeta").innerHTML = kv({
    file: (meta.input || "–").slice(-22), units: meta.units || "–",
    segs: res.n_segments, dur: (meta.duration_s || 0) + "s"});
  const ts = meta.timings_s || {};
  $("timings").innerHTML = Object.keys(ts).length ? kv(ts) : "–";
  const tsz = p.theme_syntax;
  $("tsyntax").innerHTML = tsz ? kv({
    entropy: tsz.entropy_bits + " bits",
    transitions: tsz.n_transitions, motifs: tsz.n_unique_transitions}) : "–";
  const counts = {};
  p.segments.forEach((s) => s.tokens.forEach((t) => {
    counts[t] = (counts[t] || 0) + 1;
  }));
  const top = Object.entries(counts).sort((a, b) => b[1] - a[1])
    .slice(0, 8).map(([t, c]) => `${t}×${c}`).join("  ");
  const total = Object.values(counts).reduce((a, b) => a + b, 0);
  const topShare = total ? Math.round(100 * Object.entries(counts)
    .sort((a, b) => b[1] - a[1]).slice(0, 8)
    .reduce((a, [, c]) => a + c, 0) / total) : 0;
  $("caption").innerHTML =
    `<p><b>What this answers.</b> Unit inventory (which sounds recur) · ` +
    `phrases and themes (does it repeat) · syntax gaps (is order ` +
    `meaningful) · batch comparison (whose song is it, is it changing).</p>` +
    `<p>Top units: <code>${top}</code> covering ${topShare}% of sounds · ` +
    `trigram gap ${fmt(m.trigram_gap_nats, 3)} nats · shuffle gap ` +
    `${fmt(m.shuffle_gap_nats, 3)} nats · Zipf ${fmt(m.zipf_slope)} · ` +
    `unigram entropy ${fmt(m.unigram_bits)} bits · ` +
    `phrases ${res.n_phrases} · themes ${res.n_themes}</p>`;
  const segs = p.segments.slice(0, 24).map((s, i) => tokline(s, res.job, i)).join("") +
    (p.segments.length > 24 ?
      `<p>…and ${p.segments.length - 24} more.</p>` : "");
  const phrases = (p.phrases || []).slice(0, 12).map((ph) =>
    `<div class="seg">${ph.count}× [${ph.segments}] · ` +
    `<code>${ph.phrase.join(" ")}</code></div>`).join("");
  const themes = (p.themes || []).slice(0, 12).map((th) =>
    `<div class="seg">theme ${th.theme_id}: ${th.size} codes · ` +
    `<code>${th.codes.slice(0, 32).join(" ")}</code></div>`).join("");
  const sp = p.species_hint ? `<div class="card"><h3>Species</h3>` +
    `${sp.label} · conf ${fmt(sp.confidence)} · ${sp.note}<br>` +
    ((sp.evidence || []).map((e) => `<div class="seg">${e}</div>`)
      .join("")) + `</div>` : "";
  const sm = p.species_model ? `<div class="card"><h3>Learned model</h3>` +
    `${sm.label} · conf ${fmt(sm.confidence)} · ${sm.model}` +
    `<div class="seg">top-3: ${(sm.top3 || []).map((t) =>
      `${t[0]} ${t[1]}`).join(" · ")}</div>` +
    `<small>Supervised head; rules above stay the evidence baseline.</small></div>` : "";
  const lib = res.library && res.library.matches &&
    res.library.matches.length ?
    `<div class="card"><h3>Library match</h3>verdict: ` +
    `<b>${res.library.verdict}</b>` +
    res.library.matches.map((m) =>
      `<div class="seg">${m.label} — ${m.file} · ${m.similarity}</div>`)
      .join("") +
    `<small>Grows as verified clips are added to library.json.</small></div>`
    : "";
  const mt = (p.meaning_tags || []).slice(0, 24).map((t) =>
    `<div class="seg">seg ${t.segment}: ${t.context} · ` +
    `${fmt(t.confidence)}</div>`).join("");
  $("left").innerHTML =
    `<div class="card"><h3>Detections</h3>${segs}</div>` +
    (phrases ? `<div class="card"><h3>Phrases</h3>${phrases}</div>` : "") +
    (themes ? `<div class="card"><h3>Themes</h3>${themes}</div>` : "") +
    sp + sm + lib +
    (mt ? `<div class="card"><h3>Meaning</h3>${mt}</div>` : "");
  const F = res.files || {};
  const dl = (k, label) => F[k] ?
    `<a href="${F[k]}" download><button class="ghost">${label}</button></a>`
    : "";
  $("right").innerHTML = `<div class="card"><h3>Sonar report</h3>` +
    (F["syntax_analysis_report.png"] ?
      `<img class="report" src="${F["syntax_analysis_report.png"]}">` :
      "<p>No report image.</p>") +
    `<div>${dl("song_tokens.json", "tokens JSON")}` +
    `${dl("syntax_analysis_report.pdf", "PDF")}` +
    `${dl("selection_table.txt", "Raven table")}</div></div>`;
}

$("go").onclick = async () => {
  const wav = $("wav").files[0];
  const batch = $("batch").files;
  const btn = $("go");
  btn.disabled = true;
  try {
    if (batch.length >= 2) { await runBatch(batch); }
    if (wav) { await runSingle(null); }
    else if (chosenSample) { await runSingle(chosenSample); }
    else if (batch.length < 2) {
      $("status").textContent =
        "Upload a WAV, pick a sample, or add 2+ batch files first.";
    }
  } finally { btn.disabled = false; }
};

function wavBlob(mono, sr) {
  const n = mono.length, buf = new ArrayBuffer(44 + n * 2),
    v = new DataView(buf);
  const wstr = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  wstr(0, "RIFF"); v.setUint32(4, 36 + n * 2, true); wstr(8, "WAVEfmt ");
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, sr, true); v.setUint32(28, sr * 2, true);
  v.setUint16(32, 2, true); v.setUint16(34, 16, true); wstr(36, "data");
  v.setUint32(40, n * 2, true);
  for (let i = 0; i < n; i++) {
    const s = Math.max(-1, Math.min(1, mono[i]));
    v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
  }
  return new Blob([buf], {type: "audio/wav"});
}

async function runStreamed(file) {
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  const au = await ctx.decodeAudioData(await file.arrayBuffer());
  await ctx.close();
  const sr = au.sampleRate, win = 30 * sr, total = Math.ceil(au.length / win);
  const ch0 = au.getChannelData(0);
  const ch1 = au.numberOfChannels > 1 ? au.getChannelData(1) : null;
  const {session} = await (await fetch("/api/stream/start",
    {method: "POST"})).json();
  let found = 0;
  for (let i = 0; i < total; i++) {
    const a = i * win, b = Math.min(au.length, a + win), n = b - a;
    const mono = new Float32Array(n);
    for (let k = 0; k < n; k++)
      mono[k] = ch1 ? (ch0[a + k] + ch1[a + k]) / 2 : ch0[a + k];
    const fd = new FormData();
    fd.append("session", session);
    fd.append("index", i); fd.append("total", total);
    fd.append("file", wavBlob(mono, sr), `chunk_${i}.wav`);
    const r = await fetch("/api/stream/chunk", {method: "POST", body: fd});
    if (!r.ok) throw new Error(await r.text());
    const part = await r.json();
    found += part.n_segments;
    $("status").textContent =
      `Streaming chunk ${i + 1}/${total} — ${part.n_segments} ` +
      `detections here, ${found} so far (live VAD scan).`;
    $("probe").style.width = Math.round(((i + 1) / total) * 50) + "%";
    $("lamp").style.background = "var(--amber)";
  }
  const ff = new FormData();
  ff.append("session", session);
  ff.append("preset", psel.value);
  ff.append("params", JSON.stringify(params()));
  const fin = await fetch("/api/stream/finish", {method: "POST", body: ff});
  if (!fin.ok) throw new Error(await fin.text());
  return (await fin.json()).job;
}

async function runSingle(useSample) {
  const up = !useSample && $("wav").files[0];
  if (up && up.size > 25 * 1024 * 1024) {
    $("status").textContent = "Large file — streaming in 30 s slices…";
    const job = await runStreamed(up);
    await pollJob(job, 50);
    return;
  }
  $("status").textContent = "Analyzing… (up to a minute on CPU)";
  const fd = new FormData();
  if (useSample) fd.append("sample", useSample);
  else fd.append("wav", up);
  fd.append("preset", psel.value);
  fd.append("params", JSON.stringify(params()));
  const rv = $("raven").files[0];
  if (rv) fd.append("raven", rv);
  const r = await fetch("/api/analyze", {method: "POST", body: fd});
  if (!r.ok) throw new Error(await r.text());
  const {job} = await r.json();
  await pollJob(job);
}

async function pollJob(job, base = 0) {
  for (;;) {
    await new Promise((w) => setTimeout(w, 1000));
    const pr = await (await fetch(`/api/progress/${job}`)).json();
    $("status").textContent =
      `Analyzing… ${pr.stage} ${Math.round(pr.frac * 100)}%`;
    $("probe").style.width = base +
      Math.round(pr.frac * (100 - base)) + "%";
    $("lamp").style.background = "var(--amber)";
    if (pr.stage === "done") break;
  }
  const res = await (await fetch(`/api/result/${job}`)).json();
  if (res.error) throw new Error(res.error);
  render(res);
  $("probe").style.width = "100%";
  $("lamp").style.background = "var(--sig)";
  $("status").textContent = "Done.";
}

async function runBatch() {
  $("status").textContent = "Batch analyzing…";
  const fd = new FormData();
  for (const f of $("batch").files) fd.append("files", f);
  fd.append("preset", psel.value);
  fd.append("params", JSON.stringify(params()));
  const r = await fetch("/api/batch", {method: "POST", body: fd});
  if (!r.ok) throw new Error(await r.text());
  const b = await r.json();
  const rows = b.files.map((f) =>
    `<tr><td>${f.file}</td><td>${f.n_segments}</td>` +
    `<td>${fmt(f.bigram_ppl)}</td><td>${fmt(f.gpt_ppl)}</td>` +
    `<td>${fmt(f.unigram_bits)}</td><td>${f.codes_used}</td>` +
    `<td>${f.revolution_candidate ? "dip" : ""}</td></tr>`).join("");
  let cosline = "";
  try {
    const C = b.summary.population_cosine, F = b.summary.population_files;
    let best = [-1, -1, -2];
    for (let i = 0; i < C.length; i++) for (let j = i + 1; j < C.length; j++)
      if (C[i][j] > best[2]) best = [i, j, C[i][j]];
    if (best[2] > -2) cosline = `<p>Closest pair: ${F[best[0]]} ↔ ` +
      `${F[best[1]]} (cosine ${best[2].toFixed(3)}) — same aggregate ` +
      `syntax profile; exact sequences differ.</p>`;
  } catch (e) { cosline = ""; }
  $("batch").innerHTML = `<table class="cmp"><tr><th>file</th><th>seg</th>` +
    `<th>biPPL</th><th>gptPPL</th><th>entropy</th><th>codes</th>` +
    `<th>revo</th></tr>${rows}</table>` + cosline +
    (b.trend ? `<img class="report" src="${b.trend}">` : "") +
    (b.population ? `<p><a href="${b.population}" download>` +
      `<button class="ghost">population matrix</button></a></p>` : "");
  $("status").textContent = `Batch done: ${b.n_files} files.`;
}
setInterval(() => {
  $("clock").textContent = new Date().toISOString().slice(11, 19);
}, 1000);
window.addEventListener("unhandledrejection", (e) => {
  $("status").textContent = "Error: " + e.reason;
});
