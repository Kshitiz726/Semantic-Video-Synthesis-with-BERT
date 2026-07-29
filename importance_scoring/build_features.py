import os
import torch
from extract_visual import extract_visual
from extract_audio import extract_audio
from extract_text import transcribe, extract_text_embedding

SEGMENTS_ROOT = r"D:\Github\Semantic-Video-Synthesis-with-BERT\Semantic-Video-Synthesis-with-BERT\importance_scoring\ydata-tvsum50-v1_1\tvsum_segments"
OUT_DIR = "features"
TRANS_DIR = "transcripts"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TRANS_DIR, exist_ok=True)

visual_feats = {}
audio_feats = {}
text_feats = {}

for video_id in os.listdir(SEGMENTS_ROOT):
    video_dir = os.path.join(SEGMENTS_ROOT, video_id)
    if not os.path.isdir(video_dir):
        continue

    for seg in os.listdir(video_dir):
        if not seg.endswith(".mp4"):
            continue

        seg_path = os.path.join(video_dir, seg)
        key = f"{video_id}/{seg}"

        print("Processing:", key)

        visual_feats[key] = extract_visual(seg_path)
        audio_feats[key] = extract_audio(seg_path)

        txt_path = os.path.join(TRANS_DIR, seg.replace(".mp4", ".txt"))
        if not os.path.exists(txt_path):
            text = transcribe(seg_path)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)
        else:
            with open(txt_path, encoding="utf-8") as f:
                text = f.read()

        text_feats[key] = extract_text_embedding(text)

torch.save(visual_feats, f"{OUT_DIR}/visual.pt")
torch.save(audio_feats, f"{OUT_DIR}/audio.pt")
torch.save(text_feats, f"{OUT_DIR}/text.pt")

print("✅ Feature extraction complete")
