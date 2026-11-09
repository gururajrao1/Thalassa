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


def want():
    """(clip_name, species, want_seconds_cap)."""
    rows = [r for r in csv.DictReader(open(f"{OUT}/detections.csv"))
            if r["Detection_Confidence"] == "Detected"]
    picks = {}
    used = set()
    for r in rows:
        sp = r["Species"]
        if sp in picks or sp not in ("FIWH", "HUWH", "RIWH"):
            continue
        t0 = dt.datetime.fromisoformat(r["Start_DateTime_ISO8601"])
        t1 = dt.datetime.fromisoformat(r["End_DateTime_ISO8601"])
        dur = (t1 - t0).total_seconds()
        if sp == "RIWH" and not (0.5 <= dur <= 3.0):
            continue
        if sp == "FIWH" and not (1.0 <= dur <= 40.0):
            continue
        if sp == "HUWH" and dur < 8.0:
            continue
        picks[sp] = (r, 30.0 if sp == "FIWH" else min(dur, 30.0))
        used.add(r["Start_DateTime_ISO8601"] + sp)
        if len(picks) == 4 and sum(1 for k in picks if k != "RIWH") >= 3:
            break
    # need 2 fin + 2 humpback: second pass for extras, distinct rows
    for r in rows:
        sp = r["Species"]
        if sp not in ("FIWH", "HUWH"):
            continue
        key = sp + "2"
        if key in picks:
            continue
        if r["Start_DateTime_ISO8601"] + sp in used:
            continue
        t0 = dt.datetime.fromisoformat(r["Start_DateTime_ISO8601"])
        t1 = dt.datetime.fromisoformat(r["End_DateTime_ISO8601"])
        dur = (t1 - t0).total_seconds()
        if dur < 5.0:
            continue
        picks[key] = (r, min(dur, 30.0))
        used.add(r["Start_DateTime_ISO8601"] + sp)
        if "FIWH2" in picks and "HUWH2" in picks:
            break
    return picks


def main():
    picks = want()
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
        raw = urllib.request.urlopen(
            url(f"{PREFIX}/{fname}", a, b)).read()
        import wave
        p = os.path.join(OUT, f"wild_{clip.lower()}.wav")
        with wave.open(p, "wb") as w:
            w.setnchannels(ch)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(raw)
        got[clip] = (p, len(raw) / block / sr)
        print(f"{clip}: {fname} +{off_s:.1f}s x{cap:.0f}s -> {p}")
    return got


if __name__ == "__main__":
    main()
