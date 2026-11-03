"""Scene-level importance from three modalities, and their fusion.

visual - mean of the TVSum-trained frame importance model over the scene (learned).
audio  - loudness + onset strength relative to the rest of the video (heuristic).
text   - semantic centrality: cosine between the scene's dialogue embedding and the
         centroid of all dialogue in the video, scaled by how much of the scene has
         speech (heuristic, cf. centroid-based extractive summarisation).
"""

import numpy as np

from .config import FEATURE_FPS
from .features import text_embeddings

DEFAULT_WEIGHTS = {"visual": 0.6, "audio": 0.15, "text": 0.25}


def _slice(start, end, n, rate):
    a = min(max(int(np.floor(start * rate)), 0), max(n - 1, 0))
    return slice(a, max(min(int(np.ceil(end * rate)), n), a + 1))


def visual_scores(scenes, frame_importance):
    n = len(frame_importance)
    return [float(frame_importance[_slice(s["start"], s["end"], n, FEATURE_FPS)].mean()) if n else 0.0
            for s in scenes]


def audio_scores(scenes, audio, sr=16000):
    if len(audio) < sr:
        return None
    import librosa

    hop = sr // 10  # 100 ms
    rms = librosa.feature.rms(y=audio, frame_length=2 * hop, hop_length=hop)[0]
    loud = 20 * np.log10(rms + 1e-5)
    onset = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=hop)
    if loud.max() - loud.min() < 1.0:  # effectively silent / constant track
        return None
    rate = sr / hop
    z = lambda x: (x - x.mean()) / (x.std() + 1e-8)
    loud_z, onset_z = z(loud), z(onset[: len(loud)])
    return [float(0.5 * loud_z[_slice(s["start"], s["end"], len(loud_z), rate)].mean()
                  + 0.5 * onset_z[_slice(s["start"], s["end"], len(onset_z), rate)].mean()) for s in scenes]


def text_scores(scenes, asr):
    texts, coverage = [], []
    for s in scenes:
        segs = [g for g in asr if min(s["end"], g["end"]) - max(s["start"], g["start"]) > 0.2]
        texts.append(" ".join(g["text"] for g in segs))
        spoken = sum(min(s["end"], g["end"]) - max(s["start"], g["start"]) for g in segs)
        coverage.append(min(1.0, spoken / max(s["end"] - s["start"], 1e-3)))
    if sum(bool(t) for t in texts) < 2:
        return None, texts
    idx = [i for i, t in enumerate(texts) if t]
    emb = text_embeddings([texts[i] for i in idx])
    centroid = emb.mean(0)
    centroid /= np.linalg.norm(centroid) + 1e-8
    scores = [0.0] * len(scenes)
    for k, i in enumerate(idx):
        scores[i] = float(emb[k] @ centroid) * (0.5 + 0.5 * coverage[i])
    return scores, texts


def minmax(x):
    x = np.asarray(x, dtype=np.float64)
    rng = x.max() - x.min()
    return np.full_like(x, 0.5) if rng < 1e-9 else (x - x.min()) / rng


def fuse(raw, weights=None):
    """raw: {modality: list | None}. Unavailable modalities are dropped and weights renormalised."""
    weights = weights or DEFAULT_WEIGHTS
    norm = {m: minmax(v) for m, v in raw.items() if v is not None}
    total = sum(weights[m] for m in norm) or 1.0
    fused = sum(weights[m] * norm[m] for m in norm) / total
    return fused, {m: v.tolist() for m, v in norm.items()}
