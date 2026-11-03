"""Build the TVSum training file: CLIP features + 20-annotator frame scores + KTS segments.

Usage:
    python -m training.prepare_tvsum --tvsum_dir /path/to/ydata-tvsum50-v1_1 --out data/tvsum_clip.h5

`tvsum_dir` must contain `video/*.mp4` and `data/ydata-tvsum50-anno.tsv`
(download: https://people.csail.mit.edu/yalesong/tvsum/tvsum50_ver_1_1.tgz).
"""

import argparse
import time
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

from svs.config import FEATURE_FPS
from svs.features import clip_features
from svs.kts import kts
from svs.video_io import probe


def load_annotations(tsv_path):
    anno = defaultdict(list)
    cats = {}
    for line in Path(tsv_path).read_text().splitlines():
        vid, cat, scores = line.split("\t")
        anno[vid].append(np.array(scores.split(","), dtype=np.float32))
        cats[vid] = cat
    return {k: np.stack(v) for k, v in anno.items()}, cats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tvsum_dir", required=True)
    ap.add_argument("--out", default="data/tvsum_clip.h5")
    args = ap.parse_args()

    root = Path(args.tvsum_dir)
    users, cats = load_annotations(root / "data" / "ydata-tvsum50-anno.tsv")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(out, "a") as h5:
        for i, vid in enumerate(sorted(users)):
            if vid in h5:
                continue
            t0 = time.time()
            info = probe(root / "video" / f"{vid}.mp4")
            us = users[vid]  # (20, n_frames)
            n_frames = us.shape[1]
            fps = n_frames / info.duration  # annotation frame rate implied by length

            feats = clip_features(info.path, FEATURE_FPS)
            T = len(feats)
            picks = np.minimum(np.round(np.arange(T) / FEATURE_FPS * fps).astype(int), n_frames - 1)

            gt = us.mean(0)[picks]
            gt = (gt - gt.min()) / (gt.max() - gt.min() + 1e-8)

            cps = kts(feats, max_ncp=min(T - 1, 250), lmin=2)
            bounds = np.concatenate([[0], picks[cps], [n_frames]]).astype(int)
            segments = np.stack([bounds[:-1], bounds[1:] - 1], 1)
            segments = segments[segments[:, 1] >= segments[:, 0]]

            g = h5.create_group(vid)
            g["features"] = feats
            g["picks"] = picks
            g["gtscore"] = gt.astype(np.float32)
            g["user_scores"] = us
            g["segments"] = segments
            g.attrs.update(n_frames=n_frames, fps=fps, duration=info.duration, category=cats[vid])
            print(f"[{i + 1}/{len(users)}] {vid} {info.duration:6.1f}s T={T} segs={len(segments)} "
                  f"({time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
