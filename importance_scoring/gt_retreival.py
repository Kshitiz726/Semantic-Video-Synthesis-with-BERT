import h5py
import numpy as np
import json

MAT_PATH = "D:\Github\Semantic-Video-Synthesis-with-BERT\Semantic-Video-Synthesis-with-BERT\importance_scoring\ydata-tvsum50-v1_1\matlab\ydata-tvsum50.mat"

SEGMENT_SECONDS = 20
FPS = 30
FRAMES_PER_SEG = SEGMENT_SECONDS * FPS

f = h5py.File(MAT_PATH, "r")
tvsum = f["tvsum50"]

videos = tvsum["video"]
gt_scores = tvsum["gt_score"]

segment_labels = {}

num_videos = videos.shape[0]

for i in range(num_videos):
    # Decode video id (YouTube ID)
    video_ref = videos[i][0]
    video_ascii = f[video_ref][()]
    vid = "".join(chr(c) for c in video_ascii.flatten())

    # Load frame-level importance
    score_ref = gt_scores[i][0]
    frame_scores = f[score_ref][()].astype(np.float32).flatten()

    # Normalize to [0,1]
    frame_scores = (frame_scores - frame_scores.min()) / (
        frame_scores.max() - frame_scores.min() + 1e-8
    )

    seg_id = 0
    for start in range(0, len(frame_scores), FRAMES_PER_SEG):
        seg = frame_scores[start:start + FRAMES_PER_SEG]

        # skip tiny tail segments
        if len(seg) < FRAMES_PER_SEG * 0.5:
            continue

        segment_labels[f"{vid}_seg_{seg_id:03d}"] = float(seg.mean())
        seg_id += 1

print("Total segments:", len(segment_labels))

with open("tvsum_segment_labels.json", "w") as f_out:
    json.dump(segment_labels, f_out, indent=2)

print("✅ tvsum_segment_labels.json created")
