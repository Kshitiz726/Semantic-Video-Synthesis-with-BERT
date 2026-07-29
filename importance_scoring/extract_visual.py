import torch
import clip
import cv2
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def extract_visual(video_path, max_frames=16):
    cap = cv2.VideoCapture(video_path)

    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total // max_frames, 1)

    idx = 0
    grabbed = 0

    while cap.isOpened() and grabbed < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if idx % step == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)   # ✅ FIX
            frame = preprocess(frame).unsqueeze(0)
            frames.append(frame)
            grabbed += 1

        idx += 1

    cap.release()

    if len(frames) == 0:
        return torch.zeros(512)

    frames = torch.cat(frames).to(device)

    with torch.no_grad():
        feats = model.encode_image(frames)
        feats = feats.mean(dim=0)  # temporal pooling

    return feats.cpu()
