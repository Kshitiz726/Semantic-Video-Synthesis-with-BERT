"""Turn the cross-validation output into the web-app benchmark file, paper tables and figures.

    python -m training.export_results --data data/tvsum_clip.h5 --tvsum_dir /path/to/ydata-tvsum50-v1_1
"""

import argparse
import csv
import glob
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from training.metrics import human_pairwise  # noqa: E402

RES = Path(__file__).resolve().parent / "results"
FIG = Path(__file__).resolve().parent.parent / "docs" / "figures"

ROWS = [
    ("human", "Human (inter-annotator)", "pairwise agreement between the 20 annotators (Otani et al. protocol)", "ref"),
    ("random", "Random", "uniform random frame scores", "ref"),
    ("segment_mlp", "Segment MLP (original design)", "gated MLP on 20 s mean-pooled CLIP segments", "base"),
    ("frame_mlp", "Frame MLP", "per-frame MLP, no temporal context", "base"),
    ("transformer_nopos", "Transformer w/o position", "ours without positional encoding", "base"),
    ("transformer_mse", "Transformer (MSE only)", "ours without the correlation loss term", "base"),
    ("transformer", "Transformer (ours)", "2-layer temporal Transformer, MSE + Pearson loss", "ours"),
]
METRICS = [("kendall", "Kendall τ"), ("spearman", "Spearman ρ"), ("f1", "F1@15%")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/tvsum_clip.h5")
    ap.add_argument("--tvsum_dir", required=True)
    args = ap.parse_args()

    cv = json.loads((RES / "tvsum_cv_results.json").read_text())
    titles = {}
    with open(Path(args.tvsum_dir) / "data" / "ydata-tvsum50-info.tsv", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            titles[r["video_id"]] = r["title"]

    h5 = h5py.File(args.data, "r")
    vids = sorted(h5)
    # Human reference under the published (pairwise) protocol.
    human = {v: human_pairwise(h5[v]["user_scores"][()], h5[v]["segments"][()]) for v in vids}
    cv["models"]["human"] = {"mean": {k: float(np.mean([h[k] for h in human.values()])) for k in ("kendall", "spearman", "f1")},
                             "per_video": human}

    rows = []
    for key, label, desc, kind in ROWS:
        if key in cv["models"]:
            m = cv["models"][key]
            rows.append({"key": key, "label": label, "description": desc, "kind": kind,
                         "mean": m["mean"], "std": m.get("std")})

    ours = cv["models"]["transformer"]["seeds"][0]["per_video"]
    videos = []
    for v in vids:
        videos.append({"id": v, "title": titles.get(v, v), "category": h5[v].attrs["category"],
                       "gt": np.round(h5[v]["gtscore"][()], 3).tolist(),
                       "pred": cv["oof_predictions"][v], "kendall": ours[v]["kendall"]})
    videos.sort(key=lambda x: -x["kendall"])
    summary = {"protocol": cv["protocol"] + ". F1 uses knapsack summaries over KTS segments; as shown by "
               "Otani et al. (CVPR 2019) it barely separates methods from random, so rank correlation is the "
               "primary metric.", "train": cv["train"], "rows": rows, "videos": videos}
    (RES / "benchmark_summary.json").write_text(json.dumps(summary))

    # ---- Markdown + LaTeX tables ----
    md = ["| Model | " + " | ".join(n for _, n in METRICS) + " |", "|---|" + "---:|" * len(METRICS)]
    tex = [r"\begin{tabular}{l" + "c" * len(METRICS) + "}", r"\toprule",
           "Model & " + " & ".join(n.replace("τ", r"$\tau$").replace("ρ", r"$\rho$").replace("%", r"\%")
                                   for _, n in METRICS) + r" \\", r"\midrule"]
    for r in rows:
        cells = [f"{r['mean'][k]:.3f}" + (f" ± {r['std'][k]:.3f}" if r["std"] else "") for k, _ in METRICS]
        md.append(f"| {'**' + r['label'] + '**' if r['kind'] == 'ours' else r['label']} | " + " | ".join(cells) + " |")
        tex.append(r["label"] + " & " + " & ".join(c.replace("±", r"$\pm$") for c in cells) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (RES / "results_table.md").write_text("\n".join(md) + "\n", "utf-8")
    (RES / "results_table.tex").write_text("\n".join(tex) + "\n", "utf-8")
    print("\n".join(md))

    # ---- Figures ----
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False})

    fig, ax = plt.subplots(figsize=(6, 3.4))
    for f in sorted(glob.glob(str(RES / "logs" / "transformer_seed*_fold*.json"))):
        log = json.loads(Path(f).read_text())
        ax.plot(range(1, len(log["val_spearman"]) + 1), log["val_spearman"], color="#3b5bdb", alpha=0.35, lw=1)
        ax.scatter([log["best_epoch"]], [log["val_spearman"][log["best_epoch"] - 1]], color="#e8590c", s=10, zorder=3)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation Spearman ρ")
    ax.set_title("Inner-validation curves, 5 folds × 3 seeds (dots: early stop)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "training_curves.png")
    plt.close(fig)

    names = {"BK": "Beekeeping", "BT": "Bike tricks", "DS": "Dog show", "FM": "Flash mob",
             "GA": "Grooming animal", "MS": "Making sandwich", "PK": "Parkour", "PR": "Parade",
             "VT": "Vehicle tire", "VU": "Vehicle unstuck"}
    cats = sorted({v["category"] for v in videos})
    per_cat = {c: [v["kendall"] for v in videos if v["category"] == c] for c in cats}
    hum_cat = {c: [human[v["id"]]["kendall"] for v in videos if v["category"] == c] for c in cats}
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    x = np.arange(len(cats))
    ax.bar(x - 0.2, [np.mean(per_cat[c]) for c in cats], 0.4, label="ours", color="#3b5bdb")
    ax.bar(x + 0.2, [np.mean(hum_cat[c]) for c in cats], 0.4, label="human", color="#adb5bd")
    ax.set_xticks(x, [names.get(c, c) for c in cats], rotation=35, ha="right")
    ax.set_ylabel("Kendall τ")
    ax.set_title("Per-category rank correlation on TVSum (held-out)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "per_category.png")
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(7, 6), sharex=False)
    for ax, v in zip(axes, [videos[0], videos[len(videos) // 2], videos[-1]]):
        t = np.arange(len(v["gt"])) / 2
        norm = lambda a: (np.asarray(a) - np.min(a)) / (np.ptp(a) + 1e-9)
        ax.plot(t, norm(v["gt"]), color="#868e96", lw=1.2, label="human mean")
        ax.plot(t, norm(v["pred"]), color="#3b5bdb", lw=1.5, label="ours")
        ax.set_title(f"{v['title'][:60]}  (τ = {v['kendall']:.2f})", fontsize=9)
        ax.set_ylabel("importance")
    axes[-1].set_xlabel("time (s)")
    axes[0].legend(frameon=False, loc="upper right")
    fig.suptitle("Best, median and worst held-out videos")
    fig.tight_layout()
    fig.savefig(FIG / "examples.png")
    plt.close(fig)
    print(f"Wrote {RES / 'benchmark_summary.json'} and figures in {FIG}")


if __name__ == "__main__":
    main()
