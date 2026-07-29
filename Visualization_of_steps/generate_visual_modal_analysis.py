import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from scipy.signal import savgol_filter
import seaborn as sns

# =========================
# CONFIG
# =========================

LOG_PATH = "training_logs.json"
FUSED_PATH = "fused_scores.json"
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

sns.set_style("whitegrid")

# =========================
# LOAD TRAINING LOGS
# =========================

with open(LOG_PATH) as f:
    logs = json.load(f)

train_loss = logs["train_loss"]
val_loss = logs["val_loss"]
val_spearman = logs["val_spearman"]

epochs = np.arange(1, len(train_loss) + 1)

# =========================
# FIGURE 1: LOSS CURVE
# =========================

plt.figure(figsize=(6,4))
plt.plot(epochs, train_loss, label="Train Loss")
plt.plot(epochs, val_loss, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training and Validation Loss (TVSum)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig1_loss_curve.png", dpi=300)
plt.close()

# =========================
# FIGURE 2: SPEARMAN
# =========================

plt.figure(figsize=(6,4))
plt.plot(epochs, val_spearman, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Spearman Correlation")
plt.title("Validation Rank Correlation (TVSum)")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig2_val_spearman.png", dpi=300)
plt.close()

# =========================
# LOAD FUSED SCORES
# =========================

with open(FUSED_PATH) as f:
    fused = json.load(f)

clips = fused["ranked_clips"]

visual = np.array([c["visual"] for c in clips])
audio = np.array([c["audio"] for c in clips])
text = np.array([c["text"] for c in clips])
importance = np.array([c["importance"] for c in clips])
names = [c["clip"].replace(".mp4", "") for c in clips]

# =========================
# FIGURE 3: CROSS-MODAL CORR
# =========================

fig, axes = plt.subplots(1, 3, figsize=(15,4))

pairs = [
    ("Visual vs Audio", audio, visual, "Audio", "Visual"),
    ("Visual vs Text", text, visual, "Text", "Visual"),
    ("Audio vs Text", audio, text, "Audio", "Text"),
]

for ax, (title, x, y, xl, yl) in zip(axes, pairs):
    ax.scatter(x, y, c=importance, cmap="viridis", s=60)
    sns.regplot(x=x, y=y, scatter=False, ax=ax, color="red")
    p = pearsonr(x, y)[0]
    s = spearmanr(x, y)[0]
    ax.set_title(f"{title}\nPearson={p:.2f}, Spearman={s:.2f}")
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig3_cross_modal_corr.png", dpi=300)
plt.close()

# =========================
# FIGURE 4: VISUAL DISTRIBUTION
# =========================

plt.figure(figsize=(6,4))
sns.boxplot(y=visual)
plt.ylabel("Visual Importance Score")
plt.title("Distribution of Visual Importance Scores")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig4_visual_distribution.png", dpi=300)
plt.close()

# =========================
# FIGURE 5: TEMPORAL VISUAL
# =========================

x = np.arange(len(visual))
smooth = savgol_filter(visual, window_length=5, polyorder=2, mode="nearest")

plt.figure(figsize=(8,4))
plt.plot(x, visual, label="Raw", alpha=0.6)
plt.plot(x, smooth, label="Smoothed", linewidth=2)
plt.xlabel("Scene Index (Temporal Order)")
plt.ylabel("Visual Importance")
plt.title("Temporal Evolution of Visual Importance")
plt.legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig5_temporal_visual.png", dpi=300)
plt.close()

print("✅ All visual model figures generated in ./figures/")
