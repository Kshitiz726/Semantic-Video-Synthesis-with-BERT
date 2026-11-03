"""Kernel Temporal Segmentation (Potapov et al., ECCV 2014).

Used to (a) split long shots into sub-shots and (b) build the temporal
segments required by the standard TVSum F1 / knapsack protocol.
"""

import numpy as np


def _scatters(K):
    n = K.shape[0]
    K1 = np.concatenate([[0.0], np.cumsum(np.diag(K))])
    K2 = np.zeros((n + 1, n + 1))
    K2[1:, 1:] = np.cumsum(np.cumsum(K, 0), 1)
    i = np.arange(n).reshape(-1, 1)
    j = np.arange(n).reshape(1, -1)
    d2 = np.diag(K2)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (K1[1:].reshape(1, -1) - K1[:-1].reshape(-1, 1)
             - (d2[1:].reshape(1, -1) + d2[:-1].reshape(-1, 1) - K2[1:, :-1].T - K2[:-1, 1:])
             / ((j - i + 1).astype(float) + (j == i - 1)))
    s[j < i] = 0
    return s


def _cpd_nonlin(J, m, lmin=1, backtrack=True):
    n = J.shape[0]
    INF = 1e101
    I = np.full((m + 1, n + 1), INF)
    I[0, lmin:] = J[0, lmin - 1:]
    p = np.zeros((m + 1, n + 1), dtype=int) if backtrack else None
    for k in range(1, m + 1):
        for l in range((k + 1) * lmin, n + 1):
            tmin, tmax = k * lmin, l - lmin + 1
            c = J[tmin:tmax, l - 1] + I[k - 1, tmin:tmax]
            a = int(np.argmin(c))
            I[k, l] = c[a]
            if backtrack:
                p[k, l] = a + tmin
    cps = np.zeros(m, dtype=int)
    if backtrack:
        cur = n
        for k in range(m, 0, -1):
            cps[k - 1] = p[k, cur]
            cur = cps[k - 1]
    scores = I[:, n].copy()
    scores[scores > 1e99] = np.inf
    return cps, scores


def kts(features, max_ncp=None, vmax=1.0, lmin=1, center=True):
    """Change points (indices where a new segment starts) for an (N, D) feature sequence.

    Features are centred per sequence by default so the linear kernel measures within-video
    variation (CLIP embeddings share a large common component).
    """
    X = np.asarray(features, dtype=np.float64)
    if center:
        X = X - X.mean(0)
        X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    n = len(X)
    if n < 2 * lmin + 1:
        return np.zeros(0, dtype=int)
    K = X @ X.T
    J = _scatters(K)
    m = min(max_ncp or n - 1, n // lmin - 1)
    _, scores = _cpd_nonlin(J, m, lmin, backtrack=False)
    ncp = np.arange(1, m + 1)
    penalties = np.zeros(m + 1)
    penalties[1:] = (vmax * ncp / (2.0 * n)) * (np.log(n / ncp) + 1)
    best = int(np.argmin(scores / n + penalties))
    if best == 0:
        return np.zeros(0, dtype=int)
    cps, _ = _cpd_nonlin(J, best, lmin, backtrack=True)
    return np.sort(cps)
