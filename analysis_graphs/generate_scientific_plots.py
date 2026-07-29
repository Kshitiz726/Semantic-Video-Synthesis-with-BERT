import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import json
import numpy as np
from pathlib import Path

# Set academic style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    'axes.grid': True,
    'grid.alpha': 0.3
})

OUTPUT_DIR = Path("analysis_graphs")
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------
# 1. Parsing Training Logs (Loss & Spearman)
# ---------------------------------------------------------
LOGS_PATH = Path("importance_scoring/training_logs.json")

if LOGS_PATH.exists():
    with open(LOGS_PATH, "r") as f:
        logs = json.load(f)
    
    epochs = range(1, len(logs["train_loss"]) + 1)
    
    # Figure 1: Loss & Spearman
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    # Loss Axis
    color_loss = '#CC3333'  # Red
    ln1 = ax1.plot(epochs, logs["train_loss"], color=color_loss, label='Train MSE Loss', linewidth=2, linestyle='-')
    ln2 = ax1.plot(epochs, logs["val_loss"], color=color_loss, label='Val MSE Loss', linewidth=2, linestyle='--')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Mean Squared Error (MSE)', color=color_loss)
    ax1.tick_params(axis='y', labelcolor=color_loss)
    
    # Spearman Axis
    ax2 = ax1.twinx()
    color_acc = '#003366'  # Blue
    ln3 = ax2.plot(epochs, logs["val_spearman"], color=color_acc, label='Val Spearman', linewidth=2)
    ax2.set_ylabel('Spearman Correlation ($\\rho$)', color=color_acc)
    ax2.tick_params(axis='y', labelcolor=color_acc)
    
    # Combined Legend
    lns = ln1 + ln2 + ln3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='center right')
    
    plt.title('Training Convergence and Validation Metrics')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_convergence.png")
    print(f"Generated: {OUTPUT_DIR / 'training_convergence.png'}")
    plt.close()

# ---------------------------------------------------------
# 2. Parsing Fused Scores (Correlation Heatmap)
# ---------------------------------------------------------
SCORES_PATH = Path("importance_scoring/fused_scores.json")

if SCORES_PATH.exists():
    with open(SCORES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    clips = data["ranked_clips"]
    
    # Extract data for correlation
    df = pd.DataFrame(clips)
    correlation_data = df[["visual", "audio", "text", "importance"]]
    
    # Rename for cleaner plot
    correlation_data.columns = ["Visual ($v$)", "Audio ($a$)", "Text ($t$)", "Total Score"]
    
    # Figure 2: Pairplot / Correlation Heatmap
    plt.figure(figsize=(6, 5))
    corr = correlation_data.corr()
    
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    sns.heatmap(
        corr, 
        mask=mask, 
        annot=True, 
        fmt=".2f", 
        cmap='coolwarm', 
        vmax=1, 
        vmin=-0.2, 
        square=True, 
        linewidths=.5,
        cbar_kws={"shrink": .8}
    )
    
    plt.title('Modality Correlation Matrix')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "modality_correlation.png")
    print(f"Generated: {OUTPUT_DIR / 'modality_correlation.png'}")
    plt.close()
    
    # Figure 3: Score Distributions (KDE)
    plt.figure(figsize=(8, 5))
    sns.kdeplot(data=df['visual'], fill=True, color='#D4AF37', label='Visual', alpha=0.3)
    sns.kdeplot(data=df['audio'], fill=True, color='#009966', label='Audio', alpha=0.3)
    sns.kdeplot(data=df['text'], fill=True, color='#6A5ACD', label='Text', alpha=0.3)
    
    plt.xlabel('Normalized Importance Score')
    plt.ylabel('Density')
    plt.title('Density Distribution of Multimodal Scores')
    plt.legend(title="Modality")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "score_distributions.png")
    print(f"Generated: {OUTPUT_DIR / 'score_distributions.png'}")
    plt.close()
