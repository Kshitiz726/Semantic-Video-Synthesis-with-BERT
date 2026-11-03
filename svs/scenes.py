"""Multimodal scene segmentation: group shots into semantically coherent scenes.

For every cut between consecutive units (shots, or KTS sub-shots of very long shots) a
boundary score combines
  * visual contrast  - 1 - cos between duration-weighted mean CLIP embeddings of the
                       k units before and after the cut,
  * dialogue contrast - 1 - cos between Sentence-BERT embeddings of the speech on each
                       side (only where both sides contain speech),
  * speech continuity - a penalty when one ASR sentence spans the cut.
Scores are robust-z normalised per video, and boundaries are selected greedily
(strongest first) subject to minimum / maximum scene durations.
"""

import numpy as np

from .config import FEATURE_FPS
from .features import text_embeddings
from .kts import kts


def _robust_z(x):
    x = np.asarray(x, dtype=np.float64)
    ok = ~np.isnan(x)
    if ok.sum() < 3:
        return np.where(ok, 0.0, np.nan)
    med = np.median(x[ok])
    mad = 1.4826 * np.median(np.abs(x[ok] - med)) + 1e-6
    std = x[ok].std() + 1e-6
    return (x - med) / max(mad, 0.25 * std)


def _frame_slice(start, end, n):
    a = int(np.floor(start * FEATURE_FPS))
    b = int(np.ceil(end * FEATURE_FPS))
    a = min(max(a, 0), n - 1)
    return slice(a, max(min(b, n), a + 1))


def build_units(shots, frame_feats, max_unit_len=20.0, min_sub_len=3.0):
    """Shots, with shots longer than `max_unit_len` split into KTS sub-shots."""
    units = []
    for sid, s in enumerate(shots):
        dur = s["end"] - s["start"]
        if dur <= max_unit_len:
            units.append({"start": s["start"], "end": s["end"], "shot": sid})
            continue
        sl = _frame_slice(s["start"], s["end"], len(frame_feats))
        feats = frame_feats[sl]
        lmin = max(1, int(min_sub_len * FEATURE_FPS))
        cps = kts(feats, max_ncp=max(1, int(dur / min_sub_len)), lmin=lmin) if len(feats) > 2 * lmin else []
        # guarantee pieces no longer than max_unit_len even when KTS finds no change
        times = [s["start"]] + [s["start"] + c / FEATURE_FPS for c in cps] + [s["end"]]
        pieces = []
        for a, b in zip(times[:-1], times[1:]):
            k = int(np.ceil((b - a) / max_unit_len))
            pieces += [(a + (b - a) * j / k, a + (b - a) * (j + 1) / k) for j in range(k)]
        units += [{"start": round(a, 3), "end": round(b, 3), "shot": sid} for a, b in pieces]
    return units


def _speech_in(start, end, asr):
    return " ".join(seg["text"] for seg in asr
                    if min(end, seg["end"]) - max(start, seg["start"]) > 0.2).strip()


def _spans_cut(t, asr, margin=0.3):
    return any(seg["start"] < t - margin and seg["end"] > t + margin for seg in asr)


def _pool(vectors, weights):
    w = np.asarray(weights, dtype=np.float64)[:, None]
    v = (np.asarray(vectors) * w).sum(0) / max(w.sum(), 1e-8)
    return v / (np.linalg.norm(v) + 1e-8)


def segment_scenes(shots, frame_feats, asr, duration, window=2, text_weight=0.6,
                   continuity_penalty=1.5, threshold=0.75, min_scene_len=None, max_scene_len=120.0):
    units = build_units(shots, frame_feats)
    U = len(units)
    if min_scene_len is None:
        min_scene_len = float(np.clip(duration / 40.0, 4.0, 15.0))

    n = len(frame_feats)
    vis = np.stack([_pool(frame_feats[_frame_slice(u["start"], u["end"], n)],
                          np.ones(len(frame_feats[_frame_slice(u["start"], u["end"], n)])))
                    for u in units])
    dur = np.array([u["end"] - u["start"] for u in units])
    texts = [_speech_in(u["start"], u["end"], asr) for u in units]
    has_text = np.array([bool(t) for t in texts])
    txt = np.zeros((U, 384), np.float32)
    if has_text.any():
        txt[has_text] = text_embeddings([t for t in texts if t])

    v_raw, t_raw, spans = [], [], []
    for i in range(U - 1):
        L = slice(max(0, i - window + 1), i + 1)
        R = slice(i + 1, min(U, i + 1 + window))
        v_raw.append(1 - float(_pool(vis[L], dur[L]) @ _pool(vis[R], dur[R])))
        lt, rt = has_text[L], has_text[R]
        if lt.any() and rt.any():
            t_raw.append(1 - float(_pool(txt[L][lt], dur[L][lt]) @ _pool(txt[R][rt], dur[R][rt])))
        else:
            t_raw.append(np.nan)
        spans.append(_spans_cut(units[i]["end"], asr))

    v_z, t_z = _robust_z(v_raw), _robust_z(t_raw)
    score = v_z + text_weight * np.nan_to_num(t_z, nan=0.0) - continuity_penalty * np.array(spans, float)
    cut_times = [units[i]["end"] for i in range(U - 1)]

    def allowed(idx, chosen):
        t = cut_times[idx]
        edges = sorted([0.0, duration] + [cut_times[c] for c in chosen])
        k = np.searchsorted(edges, t)
        return t - edges[k - 1] >= min_scene_len and edges[k] - t >= min_scene_len

    chosen = []
    for idx in np.argsort(-score):
        if score[idx] < threshold:
            break
        if allowed(idx, chosen):
            chosen.append(int(idx))

    # Split scenes that are still too long at their strongest admissible cut.
    changed = True
    while changed:
        changed = False
        edges = [0.0] + sorted(cut_times[c] for c in chosen) + [duration]
        for a, b in zip(edges[:-1], edges[1:]):
            if b - a <= max_scene_len:
                continue
            inside = [i for i in range(U - 1) if a < cut_times[i] < b and i not in chosen and allowed(i, chosen)]
            if inside:
                chosen.append(max(inside, key=lambda i: score[i]))
                changed = True
                break

    chosen = sorted(chosen)
    starts = [0] + [c + 1 for c in chosen]
    ends = [c for c in chosen] + [U - 1]
    scenes = []
    for a, b in zip(starts, ends):
        scenes.append({"start": units[a]["start"], "end": units[b]["end"],
                       "units": list(range(a, b + 1)),
                       "shots": sorted({units[j]["shot"] for j in range(a, b + 1)})})
    scenes[-1]["end"] = round(duration, 3)

    boundaries = [{"time": round(cut_times[i], 3), "visual": round(float(v_raw[i]), 4),
                   "text": None if np.isnan(t_raw[i]) else round(float(t_raw[i]), 4),
                   "speech_continues": bool(spans[i]), "score": round(float(score[i]), 3),
                   "chosen": i in chosen} for i in range(U - 1)]
    refine_boundaries(scenes, asr)
    return scenes, units, boundaries


def refine_boundaries(scenes, asr, max_shift=4.0, min_len=2.0):
    """Move scene boundaries that split a sentence to the nearest sentence break.

    A boundary inside an ASR segment is shifted (by at most `max_shift` s) to the
    closest segment end, preferring ends with terminal punctuation. Neighbouring
    scenes share the moved boundary, so scenes stay contiguous and never overlap.
    """
    breaks = [(g["end"], g["text"].rstrip().endswith((".", "?", "!"))) for g in asr]
    for k in range(len(scenes) - 1):
        t = scenes[k]["end"]
        if not _spans_cut(t, asr):
            continue
        lo, hi = scenes[k]["start"] + min_len, scenes[k + 1]["end"] - min_len
        cands = [(abs(b - t) + (0 if punct else 1.5), b) for b, punct in breaks
                 if abs(b - t) <= max_shift and lo <= b <= hi and not _spans_cut(b, asr)]
        if cands:
            new_t = round(min(cands)[1], 3)
            scenes[k]["cut_time"] = t
            scenes[k]["end"] = scenes[k + 1]["start"] = new_t
