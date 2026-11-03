"""Does adding the audio and dialogue scores to the visual score help on TVSum?

Visual: out-of-fold predictions of the cross-validated model (seed 0), so every video is
scored by a model that never saw it. Audio and dialogue: the same heuristics the app uses,
computed per 0.5 s step. Weights are fixed in advance (no tuning on TVSum).

    python -m training.eval_fusion --data data/tvsum_clip.h5 --tvsum_dir path/to/ydata-tvsum50-v1_1
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from svs.config import FEATURE_FPS
from training.metrics import rank_correlation, upsample

RES = Path(__file__).resolve().parent / "results"
CONFIGS = {
    "Visual only": {"visual": 1.0},
    "Audio only": {"audio": 1.0},
    "Dialogue only": {"text": 1.0},
    "Visual + audio (0.8 / 0.2)": {"visual": 0.8, "audio": 0.2},
    "Visual + dialogue (0.8 / 0.2)": {"visual": 0.8, "text": 0.2},
    "All three, app default (0.6 / 0.15 / 0.25)": {"visual": 0.6, "audio": 0.15, "text": 0.25},
    "All three, equal": {"visual": 1 / 3, "audio": 1 / 3, "text": 1 / 3},
}


def minmax(x):
    x = np.asarray(x, float)
    r = np.ptp(x)
    return np.full_like(x, 0.5) if r < 1e-9 else (x - x.min()) / r


def audio_steps(audio, T, sr=16000):
    import librosa

    if len(audio) < sr:
        return None
    hop = sr // 10
    rms = librosa.feature.rms(y=audio, frame_length=2 * hop, hop_length=hop)[0]
    loud = 20 * np.log10(rms + 1e-5)
    if np.ptp(loud) < 1.0:
        return None
    onset = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=hop)[: len(loud)]
    z = lambda x: (x - x.mean()) / (x.std() + 1e-8)
    s = 0.5 * z(loud) + 0.5 * z(onset)
    per = int(round((sr / hop) / FEATURE_FPS))  # 5 hops per 0.5 s step
    return np.array([s[i * per:(i + 1) * per].mean() if i * per < len(s) else s[-per:].mean() for i in range(T)])


def text_steps(segments, T):
    from svs.features import text_embeddings

    segs = [g for g in segments if g["text"]]
    if len(segs) < 2:
        return None
    emb = text_embeddings([g["text"] for g in segs])
    c = emb.mean(0)
    c /= np.linalg.norm(c) + 1e-8
    out = np.zeros(T)
    for g, e in zip(segs, emb):
        a, b = int(g["start"] * FEATURE_FPS), int(np.ceil(g["end"] * FEATURE_FPS))
        out[a:min(b, T)] = np.maximum(out[a:min(b, T)], float(e @ c))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/tvsum_clip.h5")
    ap.add_argument("--tvsum_dir", required=True)
    args = ap.parse_args()

    from svs import asr
    from svs.video_io import load_audio

    cv = json.loads((RES / "tvsum_cv_results.json").read_text())
    oof = cv["oof_predictions"]
    cache_path = RES / "tvsum_asr.json"
    cache = json.loads(cache_path.read_text("utf-8")) if cache_path.exists() else {}

    h5 = h5py.File(args.data, "r")
    rows = {name: [] for name in CONFIGS}
    speech_videos = 0
    for i, v in enumerate(sorted(h5)):
        g = h5[v]
        T, picks, n, users = len(oof[v]), g["picks"][()], int(g.attrs["n_frames"]), g["user_scores"][()]
        audio = load_audio(Path(args.tvsum_dir) / "video" / f"{v}.mp4")
        if v not in cache:
            cache[v] = asr.transcribe(audio)[0]
            cache_path.write_text(json.dumps(cache, ensure_ascii=False), "utf-8")
        mods = {"visual": np.asarray(oof[v]), "audio": audio_steps(audio, T), "text": text_steps(cache[v], T)}
        speech_videos += mods["text"] is not None
        for name, w in CONFIGS.items():
            avail = {m: wt for m, wt in w.items() if mods[m] is not None}
            if not avail:
                continue
            fused = sum(wt * minmax(mods[m]) for m, wt in avail.items()) / sum(avail.values())
            tau, rho = rank_correlation(upsample(fused, picks, n), users)
            rows[name].append({"video": v, "kendall": tau, "spearman": rho})
        print(f"[{i + 1}/50] {v} speech={'yes' if mods['text'] is not None else 'no'}", flush=True)

    result = {"protocol": "Frame-level rank correlation with each of the 20 annotators, averaged. Visual = "
                          "out-of-fold predictions. Fusion weights fixed in advance.",
              "videos_with_speech": speech_videos, "rows": []}
    for name, r in rows.items():
        result["rows"].append({"config": name, "n_videos": len(r),
                               "kendall": float(np.mean([x["kendall"] for x in r])),
                               "spearman": float(np.mean([x["spearman"] for x in r])), "per_video": r})
        print(f"{name:45s} n={len(r):2d}  tau={result['rows'][-1]['kendall']:.3f}  rho={result['rows'][-1]['spearman']:.3f}")
    (RES / "fusion_eval.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
