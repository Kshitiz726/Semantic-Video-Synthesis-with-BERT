import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import seaborn as sns

FUSED_SCORES_PATH = "fused_scores.json"
TOP_K = 5

sns.set_style("whitegrid")

with open(FUSED_SCORES_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

clips = data["ranked_clips"]

visual = np.array([c["visual"] for c in clips])
audio  = np.array([c["audio"]  for c in clips])
text   = np.array([c["text"]   for c in clips])
importance = np.array([c["importance"] for c in clips])
names = [c["clip"].replace(".mp4", "") for c in clips]

def corr(x, y):
    return pearsonr(x, y)[0], spearmanr(x, y)[0]

# =========================
# FIGURE WITH GRIDSPEC
# =========================

fig = plt.figure(figsize=(18, 5))
gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.05])

axes = [
    fig.add_subplot(gs[0, 0]),
    fig.add_subplot(gs[0, 1]),
    fig.add_subplot(gs[0, 2])
]

cax = fig.add_subplot(gs[0, 3])  # colorbar axis

plots = [
    ("Visual vs Audio", audio, visual, "Audio Importance", "Visual Importance"),
    ("Visual vs Text",  text,  visual, "Text Importance",  "Visual Importance"),
    ("Audio vs Text",   audio, text,   "Audio Importance", "Text Importance")
]

top_idx = np.argsort(importance)[-TOP_K:]

for ax, (title, x, y, xlab, ylab) in zip(axes, plots):

    sc = ax.scatter(
        x, y,
        c=importance,
        cmap="viridis",
        s=70,
        edgecolor="black",
        linewidth=0.5
    )

    sns.regplot(
        x=x, y=y,
        scatter=False,
        ax=ax,
        color="red",
        line_kws={"linewidth": 2, "alpha": 0.7}
    )

    p, s = corr(x, y)
    ax.set_title(f"{title}\nPearson={p:.2f}, Spearman={s:.2f}", fontsize=11)

    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)

    for i in top_idx:
        ax.annotate(names[i], (x[i], y[i]), fontsize=8, alpha=0.85)

# Proper colorbar
cbar = fig.colorbar(sc, cax=cax)
cbar.set_label("Final Importance Score")

plt.tight_layout()
plt.show()
