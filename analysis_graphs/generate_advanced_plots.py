import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
import json
from pathlib import Path

# Set academic style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 300
})

OUTPUT_DIR = Path("analysis_graphs")
OUTPUT_DIR.mkdir(exist_ok=True)

SCORES_PATH = Path("importance_scoring/fused_scores.json")

def generate_3d_scatter(df):
    """
    Fig 4: 3D Scatter Plot of Modality Space.
    Shows the decision manifold formed by the fusion weights.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Data
    x = df['visual']
    y = df['audio']
    z = df['text']
    c = df['importance']

    img = ax.scatter(x, y, z, c=c, cmap='viridis', s=60, alpha=0.9, edgecolors='k', linewidth=0.5)

    ax.set_xlabel('Visual Score ($V$)')
    ax.set_ylabel('Audio Score ($A$)')
    ax.set_zlabel('Text Score ($T$)')
    ax.set_title('3D Feature Importance Space')

    # Colorbar
    cbar = fig.colorbar(img, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('Fused Importance Score')

    # View angle
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    save_path = OUTPUT_DIR / "3d_modality_optimization.png"
    plt.savefig(save_path)
    print(f"Generated: {save_path}")
    plt.close()

def generate_sensitivity_analysis(df):
    """
    Fig 5: Sensitivity Analysis.
    Varying 'Alpha' (Visual Weight) to see how it affects the Top-5 selection stability.
    """
    alphas = np.linspace(0, 1, 20)
    stability_scores = []
    
    # Baseline Top 5 (at alpha=0.5)
    baseline_scores = 0.5 * df['visual'] + 0.3 * df['audio'] + 0.2 * df['text']
    baseline_top5 = set(baseline_scores.nlargest(5).index)

    for alpha in alphas:
        # Re-normalize beta/gamma to sum to (1-alpha)
        remaining = 1.0 - alpha
        beta = remaining * (0.3 / 0.5) # Preserve ratio roughly
        gamma = remaining * (0.2 / 0.5)
        
        new_scores = alpha * df['visual'] + beta * df['audio'] + gamma * df['text']
        new_top5 = set(new_scores.nlargest(5).index)
        
        # Jaccard Index (Intersection over Union)
        overlap = len(baseline_top5.intersection(new_top5)) / len(baseline_top5.union(new_top5))
        stability_scores.append(overlap)

    plt.figure(figsize=(8, 5))
    plt.plot(alphas, stability_scores, color='#003366', linewidth=2, marker='o', markersize=5)
    
    plt.axvline(x=0.5, color='#CC3333', linestyle='--', label='Current $\\alpha=0.5$')
    
    plt.xlabel('Visual Weight ($\\alpha$)')
    plt.ylabel('Top-5 Selection Stability (IoU)')
    plt.title('Sensitivity Analysis of Visual Weight')
    plt.legend()
    plt.tight_layout()
    
    save_path = OUTPUT_DIR / "sensitivity_analysis.png"
    plt.savefig(save_path)
    print(f"Generated: {save_path}")
    plt.close()

if __name__ == "__main__":
    if SCORES_PATH.exists():
        with open(SCORES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        df = pd.DataFrame(data["ranked_clips"])
        
        generate_3d_scatter(df)
        generate_sensitivity_analysis(df)
    else:
        print("fused_scores.json not found!")
