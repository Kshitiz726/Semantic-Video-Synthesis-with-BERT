import json
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

# ---------------- LOAD DATA ----------------
with open("shots_with_dialogue.json") as f:
    shots = json.load(f)

with open("scene_boundaries.json") as f:
    scenes_74 = json.load(f)

with open("final_scene_boundaries.json") as f:
    scenes_24 = json.load(f)

with open("refined_scenes_with_timestamps.json") as f:
    scenes_refined = json.load(f)

N = len(shots)
shot_y = np.arange(N)
shot_start = np.array([s["start_time"] for s in shots])
shot_dur = np.array([s["duration"] for s in shots])


def style_axis(ax):
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    ax.tick_params(axis="both", labelsize=10)


# ==================================================
# (a) Shot-level Segmentation
# ==================================================
fig, ax = plt.subplots(figsize=(8.2, 6.2))
style_axis(ax)

ax.barh(
    shot_y,
    shot_dur,
    left=shot_start,
    height=0.75,
    color="#4C72B0",
    alpha=0.85
)

ax.set_title("(a) Shot-level Segmentation using TransNetV2", fontsize=13)
ax.set_ylabel("Shots")
ax.set_xlabel("Time (seconds)")
ax.invert_yaxis()

plt.savefig("fig_a_shots.png", dpi=300, bbox_inches="tight")
plt.close()


# ==================================================
# (b) Initial Scene Segmentation (74)
# ==================================================
fig, ax = plt.subplots(figsize=(8.2, 6.2))
style_axis(ax)

ax.barh(
    shot_y,
    shot_dur,
    left=shot_start,
    height=0.4,
    color="#E6E6E6"
)

for sc in scenes_74:
    y0 = sc["shot_start"] - 1
    y1 = sc["shot_end"] - 1
    ax.add_patch(Rectangle(
        (shot_start[y0], y0 - 0.45),
        shot_start[y1] + shot_dur[y1] - shot_start[y0],
        (y1 - y0) + 0.9,
        edgecolor="#DD8452",
        facecolor="none",
        linewidth=1.4
    ))

ax.set_title("(b) Initial Scene Segmentation (74 scenes)", fontsize=13)
ax.set_ylabel("Shots")
ax.set_xlabel("Time (seconds)")
ax.invert_yaxis()

plt.savefig("fig_b_initial_scenes.png", dpi=300, bbox_inches="tight")
plt.close()


# ==================================================
# (c) Scene Merging (24) with Speech Discontinuity
# ==================================================
fig, ax = plt.subplots(figsize=(8.2, 6.2))
style_axis(ax)

ax.barh(
    shot_y,
    shot_dur,
    left=shot_start,
    height=0.4,
    color="#E6E6E6"
)

for sc in scenes_24:
    y0 = sc["shot_start"] - 1
    y1 = sc["shot_end"] - 1
    ax.add_patch(Rectangle(
        (shot_start[y0], y0 - 0.45),
        shot_start[y1] + shot_dur[y1] - shot_start[y0],
        (y1 - y0) + 0.9,
        edgecolor="#55A868",
        facecolor="none",
        linewidth=1.6
    ))

CUT = scenes_24[7]
cut_x = shot_start[CUT["shot_start"] - 1]
cut_y = CUT["shot_end"] - 1

ax.annotate(
    "...for the film, and",
    xy=(cut_x, cut_y),
    xytext=(cut_x + 120, -40),
    arrowprops=dict(arrowstyle="->", lw=1.4, color="#C44E52"),
    fontsize=10,
    color="#C44E52",
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#C44E52"),
    annotation_clip=False
)

ax.set_title("(c) Scene Merging (24 scenes): Speech Discontinuity", fontsize=13)
ax.set_ylabel("Shots")
ax.set_xlabel("Time (seconds)")
ax.invert_yaxis()

plt.savefig("fig_c_scene_merging.png", dpi=300, bbox_inches="tight")
plt.close()


# ==================================================
# (d) Refined Scene Boundaries
# ==================================================
fig, ax = plt.subplots(figsize=(8.2, 6.2))
style_axis(ax)

for sc in scenes_refined:
    ax.barh(
        sc["scene_id"],
        sc["duration"],
        left=sc["start_time"],
        height=0.6,
        color="#55A868",
        edgecolor="#333333",
        linewidth=0.4
    )

FIX = scenes_refined[7]

ax.annotate(
    "\"…for the film, and for this\"",
    xy=(FIX["start_time"] + FIX["duration"] / 2, FIX["scene_id"]),
    xytext=(FIX["start_time"] + 120, FIX["scene_id"] + 4),
    arrowprops=dict(arrowstyle="->", lw=1.2, color="#2E8B57"),
    fontsize=10,
    color="#2E8B57",
    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#2E8B57")
)

ax.set_title("(d) Refined Scene Boundaries using Dialogue Alignment", fontsize=13)
ax.set_ylabel("Scenes")
ax.set_xlabel("Time (seconds)")

plt.savefig("fig_d_refined_scenes.png", dpi=300, bbox_inches="tight")
plt.close()
