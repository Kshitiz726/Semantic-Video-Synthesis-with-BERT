import os
import torch
from extract_visual import extract_visual
from extract_audio import extract_audio
from extract_text import transcribe, extract_text_embedding

CLIPS_DIR = "clips"          # flat folder
OUT_DIR = "my_features"
TRANS_DIR = "my_transcripts"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TRANS_DIR, exist_ok=True)

visual_feats = {}
audio_feats = {}
text_feats = {}

for clip in os.listdir(CLIPS_DIR):
    if not clip.endswith(".mp4"):
        continue

    clip_path = os.path.join(CLIPS_DIR, clip)
    key = clip.replace(".mp4", "")   # 🔑 IMPORTANT

    print("Processing:", clip)

    # Visual
    visual_feats[key] = extract_visual(clip_path)

    # Audio
    audio_feats[key] = extract_audio(clip_path)

    # Text
    txt_path = os.path.join(TRANS_DIR, key + ".txt")
    if not os.path.exists(txt_path):
        text = transcribe(clip_path)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        with open(txt_path, encoding="utf-8") as f:
            text = f.read()

    text_feats[key] = extract_text_embedding(text)

# Save
torch.save(visual_feats, f"{OUT_DIR}/visual.pt")
torch.save(audio_feats, f"{OUT_DIR}/audio.pt")
torch.save(text_feats, f"{OUT_DIR}/text.pt")

print("✅ Feature extraction for YOUR clips complete")
