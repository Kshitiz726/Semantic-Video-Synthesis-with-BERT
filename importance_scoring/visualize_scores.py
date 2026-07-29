import json
import matplotlib.pyplot as plt
import numpy as np
import re


def scene_index(name):
    """
    Extract scene index from filename like scene_003.mp4
    """
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else 0


def visualize_scores(fused_scores_path, show=True, save_path=None):
    """
    Stage 10: Temporal visualization of importance scores
    (scene order, NOT ranked)
    """

    print("[STAGE 10] Visualizing importance scores (scene order)")

    with open(fused_scores_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ranked = data["ranked_clips"]

    # 🔹 reorder by scene index (NOT by importance)
    ordered = sorted(ranked, key=lambda x: scene_index(x["clip"]))

    scenes = [item["clip"] for item in ordered]
    importance = np.array([item["importance"] for item in ordered])
    visual = np.array([item["visual"] for item in ordered]) * 100
    audio = np.array([item["audio"] for item in ordered]) * 100
    text = np.array([item["text"] for item in ordered]) * 100

    x = np.arange(len(scenes))

    plt.figure(figsize=(16, 6))

    plt.plot(x, importance, label="Final Importance (0–100)", linewidth=3)
    plt.plot(x, visual, label="Visual", linestyle="--")
    plt.plot(x, audio, label="Audio", linestyle="-.")
    plt.plot(x, text, label="Text", linestyle=":")

    plt.xticks(x, scenes, rotation=45, ha="right")
    plt.xlabel("Scenes (Temporal Order)")
    plt.ylabel("Score")
    plt.title("Scene-wise Multimodal Importance (Temporal Order)")

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"📈 Figure saved → {save_path}")

    if show:
        plt.show()
    else:
        plt.close()
