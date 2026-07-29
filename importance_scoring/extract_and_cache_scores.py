import os
import cv2
import torch
import librosa
import numpy as np
import whisper
import json
from sentence_transformers import SentenceTransformer
from importance_scoring.visual_model import VisualImportanceModel
import clip
from PIL import Image


device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()



def extract_and_cache_scores(
    clips_dir,
    output_json_path,
    visual_model_path,
    force_recompute=False
):
    """
    Stage 8: Extract raw importance scores (visual, audio, text)
    and cache them to JSON.
    """

    print("[STAGE 8] Extracting and caching raw importance scores")

    if os.path.exists(output_json_path) and not force_recompute:
        print(" cached_scores.json already exists — skipping extraction")
        return output_json_path

    device = "cuda" if torch.cuda.is_available() else "cpu"

    
    visual_model = VisualImportanceModel().to(device)
    visual_model.load_state_dict(
        torch.load(visual_model_path, map_location=device)
    )
    visual_model.eval()

    whisper_model = whisper.load_model("base")
    text_model = SentenceTransformer("all-MiniLM-L6-v2")

    # ==========================
    # VISUAL FEATURE
    # ==========================
    def extract_visual_feature(video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (224, 224))
            frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            frames.append(frame)

        cap.release()
        if not frames:
            return None

        frames = torch.stack(frames).mean(dim=0).view(-1)
        return frames[:512]



# def extract_visual_feature(video_path, target_fps=2):
#     cap = cv2.VideoCapture(video_path)

#     original_fps = cap.get(cv2.CAP_PROP_FPS)
#     if original_fps <= 0:
#         original_fps = 25  # fallback

#     frame_interval = int(original_fps // target_fps)
#     frame_interval = max(frame_interval, 1)

#     frame_id = 0
#     features = []

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         if frame_id % frame_interval == 0:
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             pil_img = Image.fromarray(frame_rgb)

#             img_tensor = clip_preprocess(pil_img).unsqueeze(0).to(device)

#             with torch.no_grad():
#                 feat = clip_model.encode_image(img_tensor)
#                 feat = feat / feat.norm(dim=-1, keepdim=True)

#             features.append(feat.squeeze(0))

#         frame_id += 1

#     cap.release()

#     if not features:
#         return None

    
#     scene_feature = torch.stack(features).mean(dim=0)

#     return scene_feature.cpu()   







    def audio_importance(video_path):
        y, sr = librosa.load(video_path, sr=16000)
        if len(y) < sr:
            return 0.0

        rms = np.mean(librosa.feature.rms(y=y))
        spec = np.abs(librosa.stft(y))
        flux = np.mean(np.diff(spec, axis=1).clip(min=0))

        pitches, mags = librosa.piptrack(y=y, sr=sr)
        pitch_vals = pitches[mags > np.median(mags)]
        pitch_vals = np.clip(pitch_vals, 80, 800)
        pitch_var = np.std(pitch_vals) if len(pitch_vals) else 0.0

        return 0.4 * rms + 0.4 * flux + 0.2 * pitch_var

  
    def text_importance(video_path):
        res = whisper_model.transcribe(video_path, temperature=0, best_of=1, fp16=False)
        text = res["text"].strip()

        if len(text) < 5:
            return 0.0, text

        emb = text_model.encode(text)
        semantic = np.linalg.norm(emb)
        length = len(text.split())
        sentences = text.count(".") + text.count("?") + text.count("!")

        score = 0.4 * semantic + 0.3 * length + 0.1 * sentences
        return score, text

 
    scores = {}

    for clip in sorted(os.listdir(clips_dir)):
        if not clip.endswith(".mp4"):
            continue

        path = os.path.join(clips_dir, clip)
        print("Extracting:", clip)

        v_feat = extract_visual_feature(path)
        if v_feat is None:
            continue

        with torch.no_grad():
            v_score = visual_model(v_feat.unsqueeze(0).to(device)).item()

        a_score = audio_importance(path)
        t_score, transcript = text_importance(path)

        scores[clip] = {
            "visual": float(v_score),
            "audio": float(a_score),
            "text": float(t_score),
            "transcript": transcript
        }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)

    print(f" Raw scores cached → {output_json_path}")
    return output_json_path
