"""5-fold cross-validated training and evaluation of the visual importance model on TVSum.

Every video is tested exactly once per seed (5 folds x 10 videos). Inside each fold,
8 of the 40 training videos are held out for early stopping, so test videos are never
used for model selection. All models share the same folds.

Usage:
    python -m training.train_importance --data data/tvsum_clip.h5
"""

import argparse
import copy
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

from svs.importance import build_model
from training.metrics import evaluate_video, human_pairwise, upsample

OUT_DIR = Path(__file__).resolve().parent / "results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Fixed a priori (not tuned on test folds).
CONFIGS = {
    "transformer": dict(model="transformer", kwargs={}, corr_weight=0.5),
    "transformer_mse": dict(model="transformer", kwargs={}, corr_weight=0.0),
    "transformer_nopos": dict(model="transformer", kwargs={"use_pos": False}, corr_weight=0.5),
    "frame_mlp": dict(model="frame_mlp", kwargs={}, corr_weight=0.5),
    "segment_mlp": dict(model="segment_mlp", kwargs={}, corr_weight=0.0),
}
TRAIN = dict(lr=1e-4, weight_decay=1e-2, max_epochs=150, patience=20, feat_dropout=0.1)


def load(path):
    data = {}
    with h5py.File(path, "r") as h5:
        for vid in sorted(h5):
            g = h5[vid]
            data[vid] = {k: g[k][()] for k in ("features", "picks", "gtscore", "user_scores", "segments")}
            data[vid].update(dict(g.attrs))
    return data


def pearson_loss(pred, target):
    p, t = pred - pred.mean(), target - target.mean()
    return 1 - (p * t).sum() / (p.norm() * t.norm() + 1e-8)


def predict(model, feats):
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(feats).unsqueeze(0).to(DEVICE)
        return model(x).squeeze(0).cpu().numpy()


def train_one(cfg, train_vids, val_vids, data, seed, keep_last=False):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = build_model(cfg["model"], **cfg["kwargs"]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=TRAIN["lr"], weight_decay=TRAIN["weight_decay"])
    tensors = {v: (torch.from_numpy(data[v]["features"]).to(DEVICE),
                   torch.from_numpy(data[v]["gtscore"]).to(DEVICE)) for v in train_vids}

    best, best_state, best_epoch, wait = -np.inf, None, 0, 0
    log = {"train_loss": [], "val_spearman": []}
    for epoch in range(1, TRAIN["max_epochs"] + 1):
        model.train()
        losses = []
        for v in rng.permutation(train_vids):
            x, y = tensors[v]
            x = torch.nn.functional.dropout(x, TRAIN["feat_dropout"]).unsqueeze(0)
            pred = model(x).squeeze(0)
            loss = torch.nn.functional.mse_loss(pred, y)
            if cfg["corr_weight"]:
                loss = loss + cfg["corr_weight"] * pearson_loss(pred, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())

        val = float(np.mean([spearmanr(predict(model, data[v]["features"]), data[v]["gtscore"])[0]
                             for v in val_vids]))
        log["train_loss"].append(float(np.mean(losses)))
        log["val_spearman"].append(val)
        if val > best:
            best, best_state, best_epoch, wait = val, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            wait += 1
            if wait >= TRAIN["patience"]:
                break
    if not keep_last:
        model.load_state_dict(best_state)
    log["best_epoch"] = best_epoch
    return model, log


def summarize(per_video):
    keys = ("kendall", "spearman", "f1")
    return {k: float(np.mean([m[k] for m in per_video.values()])) for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/tvsum_clip.h5")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    ap.add_argument("--final_config", default="transformer")
    args = ap.parse_args()

    data = load(args.data)
    vids = np.array(sorted(data))
    (OUT_DIR / "logs").mkdir(parents=True, exist_ok=True)
    folds = list(KFold(5, shuffle=True, random_state=0).split(vids))
    results = {"protocol": "TVSum 5-fold CV, each video tested once per seed; inner 8-video "
                           "validation split for early stopping; metrics averaged over 20 annotators",
               "train": TRAIN, "folds": [vids[te].tolist() for _, te in folds], "models": {}}

    # Reference points computed under the identical protocol.
    rng = np.random.default_rng(0)
    human, rand = {}, {}
    for v in vids:
        d = data[v]
        human[v] = human_pairwise(d["user_scores"], d["segments"])
        rand[v] = evaluate_video(rng.random(d["n_frames"]), d["user_scores"], d["segments"])
    results["models"]["human"] = {"mean": summarize(human), "per_video": human}
    results["models"]["random"] = {"mean": summarize(rand), "per_video": rand}
    print("human ", summarize(human))
    print("random", summarize(rand))

    best_epochs = []
    for name in args.configs:
        cfg = CONFIGS[name]
        seed_means, oof = [], {}
        for seed in args.seeds:
            per_video = {}
            for k, (tr, te) in enumerate(folds):
                tr_vids = vids[tr].copy()
                np.random.default_rng(100 + k).shuffle(tr_vids)
                val_vids, fit_vids = tr_vids[:8], tr_vids[8:]
                t0 = time.time()
                model, log = train_one(cfg, fit_vids, val_vids, data, seed)
                (OUT_DIR / "logs" / f"{name}_seed{seed}_fold{k}.json").write_text(json.dumps(log))
                if name == args.final_config:
                    best_epochs.append(log["best_epoch"])
                for v in vids[te]:
                    d = data[v]
                    pred = predict(model, d["features"])
                    frame_pred = upsample(pred, d["picks"], d["n_frames"])
                    per_video[v] = evaluate_video(frame_pred, d["user_scores"], d["segments"])
                    if seed == args.seeds[0]:
                        oof[v] = pred.round(4).tolist()
                print(f"{name} seed={seed} fold={k} best_epoch={log['best_epoch']} "
                      f"({time.time() - t0:.0f}s)", flush=True)
            m = summarize(per_video)
            seed_means.append(m)
            print(f"== {name} seed={seed}: {m}", flush=True)
            results["models"].setdefault(name, {"seeds": []})["seeds"].append(
                {"seed": seed, "mean": m, "per_video": per_video})
        results["models"][name]["mean"] = {k: float(np.mean([s[k] for s in seed_means])) for k in seed_means[0]}
        results["models"][name]["std"] = {k: float(np.std([s[k] for s in seed_means])) for k in seed_means[0]}
        if name == args.final_config:
            results["oof_predictions"] = oof
        (OUT_DIR / "tvsum_cv_results.json").write_text(json.dumps(results, indent=1))

    # Release model: train on all 50 videos for the median early-stopping epoch.
    if best_epochs:
        epochs = int(np.median(best_epochs))
        cfg = CONFIGS[args.final_config]
        torch.manual_seed(0)
        final = dict(TRAIN, max_epochs=epochs, patience=epochs + 1)
        TRAIN.update(final)
        model, _ = train_one(cfg, vids, vids[:8], data, seed=0, keep_last=True)
        ckpt = {"model": cfg["model"], "kwargs": cfg["kwargs"], "state_dict": model.state_dict(),
                "trained_on": "TVSum (all 50 videos)", "epochs": epochs, "config": args.final_config}
        out = Path(__file__).resolve().parent.parent / "weights" / "importance_transformer.pt"
        out.parent.mkdir(exist_ok=True)
        torch.save(ckpt, out)
        print(f"Saved release model ({epochs} epochs) -> {out}")


if __name__ == "__main__":
    main()
