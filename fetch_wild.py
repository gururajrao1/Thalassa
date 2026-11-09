"""Fetch first wild labeled clips from DCLDE 2013 (NOAA NCEI GCS).

Picks FIN + HUMPBACK + right-whale detections from the NEFSC Raven log and
range-fetches only the needed seconds from the 15-min 2kHz WAVs.
Provenance is kept per clip; species map: FIWH->fin-like, HUWH->humpback-like,
RIWH->out-of-scope confuser (expect unassigned: no matching class).
"""
import csv
import datetime as dt
import os
import struct
import urllib.request

BUCKET = "noaa-passive-bioacoustic"
PREFIX = "dclde/2013/nefsc_sbnms_200903_nopp6_ch10/source-audio"
OUT = "wild_dclde"
BASE = ("https://storage.googleapis.com/download/storage/v1/b/"
        f"{BUCKET}/o/")


def url(name, start=None, end=None):
    import urllib.parse
    u = BASE + urllib.parse.quote(name, safe="") + "?alt=media"
    req = urllib.request.Request(u)
    if start is not None:
        req.add_header("Range", f"bytes={start}-{end}")
    return req


def wav_params(blob):
    assert blob[0:4] == b"RIFF" and blob[8:12] == b"WAVE"
    fmt = blob.index(b"fmt ")
    audio, ch, sr, _, _, bits = struct.unpack("<HHIIHH", blob[fmt + 8:fmt + 24])
    assert audio == 1 and bits == 16
    data_at = blob.index(b"data")
    dlen = struct.unpack("<I", blob[data_at + 4:data_at + 8])[0]
    return sr, ch, data_at + 8, dlen


def want(targets=None):
    """{clip_name: (row, cap_s)}; spread across time, distinct rows."""
    targets = targets or {"FIWH": 8, "HUWH": 8, "RIWH": 6}
    log = os.path.join(OUT, "detections_2025.csv")
    if not os.path.isfile(log):
        log = os.path.join(OUT, "detections.csv")
    rows = [r for r in csv.DictReader(open(log))
            if r["Detection_Confidence"] == "Detected" or
            (r["Species"] == "BLWH" and
             r["Detection_Confidence"] == "Possibly_Detected")]
    picks, used, last_t, n = {}, set(), {}, {sp: 0 for sp in targets}
    for r in rows:
        sp = r["Species"]
        if sp not in targets or n[sp] >= targets[sp]:
            continue
        key = r["Start_DateTime_ISO8601"] + sp
        if key in used:
            continue
        t0 = dt.datetime.fromisoformat(r["Start_DateTime_ISO8601"])
        t1 = dt.datetime.fromisoformat(r["End_DateTime_ISO8601"])
        dur = (t1 - t0).total_seconds()
        if sp == "RIWH" and not (0.5 <= dur <= 3.0):
            continue
        if sp == "FIWH" and dur < 1.0:
            continue
        if sp == "HUWH" and dur < 8.0:
            continue
        if sp == "BLWH" and dur < 3.0:
            continue
        if sp in last_t and (t0 - last_t[sp]).total_seconds() < 1800:
            continue  # spread >=30 min apart
        n[sp] += 1
        picks[f"{sp}{n[sp]}"] = (r, 30.0 if sp == "FIWH" else min(dur, 30.0))
        used.add(key)
        last_t[sp] = t0
        if all(n[sp] >= targets[sp] for sp in targets):
            break
    return picks


LABEL_OF = {"FIWH": "fin-like", "HUWH": "humpback-like",
              "RIWH": "unassigned", "BLWH": "blue-like"}
LOWCUT_OF = {"FIWH": 10.0, "HUWH": 50.0, "RIWH": 50.0, "BLWH": 10.0}


def main(targets=None):
    picks = want(targets)
    print({k: (v[0]["Species"], v[0]["Start_DateTime_ISO8601"],
             v[0]["Low.Freq..Hz."], v[0]["High.Freq..Hz."]) for k, v in picks.items()})
    # header from first audio file to learn sr/channels
    first = "NOPP6_EST_20090328_000000_CH10.wav"
    head = urllib.request.urlopen(url(f"{PREFIX}/{first}", 0, 4095)).read()
    sr, ch, data_off, _ = wav_params(head)
    print(f"audio: {sr}Hz ch={ch} data_off={data_off}")
    block = 2 * ch
    got = {}
    for clip, (r, cap) in picks.items():
        t0 = dt.datetime.fromisoformat(r["Start_DateTime_ISO8601"])
        day = t0.strftime("%Y%m%d")
        chunk = (t0.hour * 3600 + t0.minute * 60 + t0.second) // 900 * 900
        hh, mm = divmod(chunk // 60, 60)
        mm -= mm % 15
        fname = f"NOPP6_EST_{day}_{hh:02d}{mm:02d}00_CH10.wav"
        off_s = (t0.hour * 3600 + t0.minute * 60 + t0.second
                 + t0.microsecond / 1e6) - chunk
        pre = 10.0 if clip.startswith("FIWH") else 0.0
        off_s = max(0.0, off_s - pre)
        a = data_off + int(off_s * sr) * block
        b = a + int(cap * sr) * block - 1
        try:
            raw = urllib.request.urlopen(
                url(f"{PREFIX}/{fname}", a, b)).read()
        except Exception as e:  # missing chunk file: skip clip honestly
            print(f"{clip}: SKIP {fname} ({e})")
            continue
        import wave
        p = os.path.join(OUT, f"wild_{clip.lower()}.wav")
        with wave.open(p, "wb") as w:
            w.setnchannels(ch)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(raw)
        got[clip] = (p, len(raw) / block / sr)
        print(f"{clip}: {fname} +{off_s:.1f}s x{cap:.0f}s -> {p}")
    import csv as _csv
    with open(os.path.join(OUT, "manifest.csv"), "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["file", "true_label", "note", "lowcut"])
        for clip, (r, cap) in picks.items():
            if clip not in got:
                continue  # skipped fetch: no row
            sp = r["Species"]
            w.writerow([f"wild_{clip.lower()}.wav", LABEL_OF[sp],
                        f"DCLDE2013 {sp} {r['Start_DateTime_ISO8601']} "
                        f"{r['Low.Freq..Hz.']}-{r['High.Freq..Hz.']}Hz",
                        LOWCUT_OF[sp]])
    return got


if __name__ == "__main__":
    main()
