"""TVSum evaluation metrics.

* Rank correlation (Otani et al., CVPR 2019): Kendall tau / Spearman rho between the
  predicted frame-level scores and each annotator's scores, averaged over annotators.
* F1 @ 15% budget: knapsack summaries over KTS segments, compared with each
  annotator's summary and averaged (TVSum convention).
"""

import numpy as np
from scipy.stats import kendalltau, spearmanr

from svs.summarize import segment_summary


def upsample(pred, picks, n_frames):
    """Nearest-pick upsampling of step-level predictions to every annotated frame."""
    frame_idx = np.arange(n_frames)
    pos = np.clip(np.searchsorted(picks, frame_idx, side="right") - 1, 0, len(picks) - 1)
    return np.asarray(pred)[pos]


def rank_correlation(frame_pred, user_scores):
    taus, rhos = [], []
    for u in user_scores:
        taus.append(kendalltau(frame_pred, u)[0])
        rhos.append(spearmanr(frame_pred, u)[0])
    return float(np.nanmean(taus)), float(np.nanmean(rhos))


def f1(machine, user):
    overlap = np.logical_and(machine, user).sum()
    if overlap == 0:
        return 0.0
    p, r = overlap / machine.sum(), overlap / user.sum()
    return float(2 * p * r / (p + r))


def summary_f1(frame_pred, user_scores, segments, budget=0.15):
    machine = segment_summary(frame_pred, segments, budget)
    return float(np.mean([f1(machine, segment_summary(u, segments, budget)) for u in user_scores]))


def evaluate_video(frame_pred, user_scores, segments):
    tau, rho = rank_correlation(frame_pred, user_scores)
    return {"kendall": tau, "spearman": rho, "f1": summary_f1(frame_pred, user_scores, segments)}


def human_pairwise(user_scores, segments):
    """Inter-annotator agreement as in Otani et al.: every annotator vs. every other one."""
    out = {"kendall": [], "spearman": [], "f1": []}
    n = len(user_scores)
    summaries = [segment_summary(u, segments) for u in user_scores]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if i < j:
                out["kendall"].append(kendalltau(user_scores[i], user_scores[j])[0])
                out["spearman"].append(spearmanr(user_scores[i], user_scores[j])[0])
            out["f1"].append(f1(summaries[i], summaries[j]))
    return {k: float(np.nanmean(v)) for k, v in out.items()}


def human_baseline(user_scores, segments):
    """Leave-one-annotator-out agreement: each annotator vs. the mean of the others."""
    out = {"kendall": [], "spearman": [], "f1": []}
    for i in range(len(user_scores)):
        others = np.delete(user_scores, i, axis=0)
        mean_others = others.mean(0)
        u = user_scores[i]
        out["kendall"].append(kendalltau(u, mean_others)[0])
        out["spearman"].append(spearmanr(u, mean_others)[0])
        out["f1"].append(summary_f1(u, others, segments))
    return {k: float(np.nanmean(v)) for k, v in out.items()}
