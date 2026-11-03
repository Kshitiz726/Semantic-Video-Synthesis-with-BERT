"""0/1 knapsack summary selection (the standard video-summarisation protocol)."""

import numpy as np


def knapsack(values, weights, capacity):
    """Indices of items maximising sum(values) with sum(weights) <= capacity (integer weights)."""
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=int)
    n, cap = len(values), int(capacity)
    if n == 0 or cap <= 0:
        return []
    best = np.zeros(cap + 1)
    take = np.zeros((n, cap + 1), dtype=bool)
    for i in range(n):
        w = weights[i]
        if w > cap:
            continue
        cand = best[: cap + 1 - w] + values[i]
        better = cand > best[w:]
        take[i, w:] = better
        best[w:] = np.where(better, cand, best[w:])
    sel, c = [], cap
    for i in range(n - 1, -1, -1):
        if take[i, c]:
            sel.append(i)
            c -= weights[i]
    return sorted(sel)


def select_summary(units, scores, budget_seconds, min_len=1.5):
    """Pick shot-level units maximising sum(score * duration) within the time budget.

    Weighting by duration selects the most important *time* rather than many tiny
    fragments. Adjacent picks are merged into continuous segments.
    """
    idx = [i for i, u in enumerate(units) if u["end"] - u["start"] >= min_len] or list(range(len(units)))
    dur = [units[i]["end"] - units[i]["start"] for i in idx]
    picked = knapsack([scores[i] * d for i, d in zip(idx, dur)], [max(1, int(round(d * 10))) for d in dur],
                      int(budget_seconds * 10))
    if not picked:  # every unit longer than the budget: take the best one, trimmed to the budget
        best = max(idx, key=lambda i: scores[i])
        u = units[best]
        mid = 0.5 * (u["start"] + u["end"])
        half = min(budget_seconds, u["end"] - u["start"]) / 2
        return [{"start": round(mid - half, 3), "end": round(mid + half, 3)}]
    segments = []
    for i in sorted(idx[p] for p in picked):
        u = units[i]
        if segments and abs(segments[-1]["end"] - u["start"]) < 1e-3:
            segments[-1]["end"] = u["end"]
        else:
            segments.append({"start": u["start"], "end": u["end"]})
    return segments


def render_summary(video_path, segments, out_path):
    """Cut each segment and concatenate them into one MP4."""
    import shutil
    from pathlib import Path

    from .video_io import concat_clips, cut_clip

    tmp = Path(out_path).with_suffix(".parts")
    tmp.mkdir(parents=True, exist_ok=True)
    parts = []
    for k, seg in enumerate(segments):
        parts.append(tmp / f"{k:03d}.mp4")
        cut_clip(video_path, seg["start"], seg["end"], parts[-1])
    concat_clips(parts, out_path)
    shutil.rmtree(tmp)


def segment_summary(frame_scores, segments, budget=0.15):
    """Binary frame mask of the knapsack summary over [start, end] (inclusive) segments."""
    n = len(frame_scores)
    vals = [float(frame_scores[s:e + 1].mean()) for s, e in segments]
    lens = [e - s + 1 for s, e in segments]
    mask = np.zeros(n, dtype=bool)
    for i in knapsack(vals, lens, int(budget * n)):
        s, e = segments[i]
        mask[s:e + 1] = True
    return mask
