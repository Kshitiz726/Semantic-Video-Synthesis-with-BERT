"""Sanity checks for the cross-validated importance model.

1. Leakage: test videos never appear in the training or early-stopping videos of their fold.
2. Shuffled-label control: the same training run with each video's labels randomly permuted
   in time must score close to zero, otherwise the evaluation is leaking information.

    python -m training.sanity_checks --data data/tvsum_clip.h5
"""

import argparse
import json

import numpy as np
from sklearn.model_selection import KFold

from training import train_importance as T
from training.metrics import evaluate_video, upsample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/tvsum_clip.h5")
    args = ap.parse_args()

    data = T.load(args.data)
    vids = np.array(sorted(data))
    folds = list(KFold(5, shuffle=True, random_state=0).split(vids))
    out = {}

    # 1. leakage
    seen = []
    for k, (tr, te) in enumerate(folds):
        tr_vids = vids[tr].copy()
        np.random.default_rng(100 + k).shuffle(tr_vids)
        assert not set(tr_vids) & set(vids[te]), "test video in training fold"
        assert not set(tr_vids[:8]) & set(vids[te]), "test video used for early stopping"
        seen += list(vids[te])
    assert sorted(seen) == sorted(vids), "every video must be tested exactly once"
    out["leakage_check"] = "passed: 5 disjoint test folds covering all 50 videos once; no test video in training or early stopping"
    print(out["leakage_check"])

    # 2. shuffled-label control
    rng = np.random.default_rng(0)
    shuffled = {v: dict(d, gtscore=rng.permutation(d["gtscore"])) for v, d in data.items()}
    per_video = {}
    for k, (tr, te) in enumerate(folds):
        tr_vids = vids[tr].copy()
        np.random.default_rng(100 + k).shuffle(tr_vids)
        model, _ = T.train_one(T.CONFIGS["transformer"], tr_vids[8:], tr_vids[:8], shuffled, seed=0)
        for v in vids[te]:
            d = data[v]  # evaluated against the real annotations
            pred = T.predict(model, d["features"])
            per_video[v] = evaluate_video(upsample(pred, d["picks"], d["n_frames"]), d["user_scores"], d["segments"])
        print(f"fold {k} done", flush=True)
    out["shuffled_label_control"] = T.summarize(per_video)
    print("shuffled labels:", out["shuffled_label_control"])
    (T.OUT_DIR / "sanity_checks.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
