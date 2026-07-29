import os
import cv2
import torch
import librosa
import numpy as np
import whisper
from sentence_transformers import SentenceTransformer
from visual_model import VisualImportanceModel

# ==========================
# USER CONTROLS
# ==========================
ALPHA = 0.5   # visual
BETA  = 0.3  # audio
GAMMA = 0.2   # text

CLIPS_DIR = r"D:\Github\Semantic-Video-Synthesis-with-BERT\Semantic-Video-Synthesis-with-BERT\importance_scoring\clips"
MODEL_PATH = "visual_importance_model.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================
# LOAD MODELS
# ==========================
visual_model = VisualImportanceModel().to(device)
visual_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
visual_model.eval()

whisper_model = whisper.load_model("base")
text_model = SentenceTransformer("all-MiniLM-L6-v2")

# ==========================
# VISUAL FEATURE EXTRACTION
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
    return frames[:512]  # ensure 512-D

# ==========================
# AUDIO IMPORTANCE
# ==========================
def audio_importance(video_path):
    try:
        y, sr = librosa.load(video_path, sr=16000)

        if len(y) < sr:
            return 0.0

        # 1. Energy
        rms = librosa.feature.rms(y=y)[0]
        energy_score = np.mean(rms)

        # 2. Spectral novelty (events / beats)
        spec = np.abs(librosa.stft(y))
        flux = np.mean(np.diff(spec, axis=1).clip(min=0))

        # 3. Pitch variation (emotion proxy)
        pitches, mags = librosa.piptrack(y=y, sr=sr)
        pitch_vals = pitches[mags > np.median(mags)]
        pitch_var = np.std(pitch_vals) if len(pitch_vals) > 0 else 0.0

        # Weighted audio importance
        audio_score = (
            0.4 * energy_score +
            0.4 * flux +
            0.2 * pitch_var
        )

        return audio_score

    except Exception as e:
        return 0.0

# ==========================
# TEXT IMPORTANCE (ROBUST)
# ==========================
KEYWORDS = {
    "kill", "dead", "fight", "love", "hate", "secret", "problem",
    "why", "because", "truth", "plan", "escape", "help",
    "money", "important", "listen", "remember", "wait"
}

def text_importance(video_path):
    result = whisper_model.transcribe(video_path, temperature=0, best_of=1, fp16=False)
    text = result["text"].strip()

    if len(text) < 5:
        return 0.0, text

    # 1. Length (dialogue density)
    length_score = len(text.split())

    # 2. Semantic richness
    emb = text_model.encode(text)
    semantic_score = np.linalg.norm(emb)

    # 3. Keyword salience
    keyword_score = sum(1 for w in KEYWORDS if w in text.lower())

    # 4. Sentence activity
    sentence_score = text.count(".") + text.count("?") + text.count("!")

    # Weighted text importance
    text_score = (
        0.4 * semantic_score +
        0.3 * length_score +
        0.2 * keyword_score +
        0.1 * sentence_score
    )

    return text_score, text


# ==========================
# NORMALIZATION
# ==========================
def normalize(scores):
    mn, mx = min(scores), max(scores)
    return [(s - mn) / (mx - mn + 1e-8) for s in scores]

# ==========================
# MAIN PIPELINE
# ==========================
results = {}

visual_scores = []
audio_scores = []
text_scores = []
cache = []

for clip in sorted(os.listdir(CLIPS_DIR)):
    if not clip.endswith(".mp4"):
        continue

    path = os.path.join(CLIPS_DIR, clip)
    print("Processing:", clip)

    # visual
    v_feat = extract_visual_feature(path)
    if v_feat is None:
        continue

    with torch.no_grad():
        v_score = visual_model(v_feat.unsqueeze(0).to(device)).item()

    # audio
    a_score = audio_importance(path)

    # text
    t_score, transcript = text_importance(path)

    visual_scores.append(v_score)
    audio_scores.append(a_score)
    text_scores.append(t_score)

    cache.append((clip, v_score, a_score, t_score, transcript))

# normalize
visual_scores = normalize(visual_scores)
audio_scores  = normalize(audio_scores)
text_scores   = normalize(text_scores)

# final fusion
for i, (clip, _, _, _, transcript) in enumerate(cache):
    final_score = (
        ALPHA * visual_scores[i]
        + BETA  * audio_scores[i]
        + GAMMA * text_scores[i]
    )

    results[clip] = {
        "final_score": round(final_score, 4),
        "visual": round(visual_scores[i], 4),
        "audio": round(audio_scores[i], 4),
        "text": round(text_scores[i], 4),
        "transcript": transcript
    }

# ==========================
# SORT & PRINT
# ==========================
results = dict(sorted(results.items(), key=lambda x: x[1]["final_score"], reverse=True))

print("\n🏆 FINAL RANKING:\n")
for k, v in results.items():
    print(f"{k} → {v['final_score']} (V:{v['visual']} A:{v['audio']} T:{v['text']})")
