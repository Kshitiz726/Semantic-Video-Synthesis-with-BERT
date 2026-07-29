# infer_on_clips.py (FINAL, CORRECT)

import os
import torch
import json
from model import RelevanceModel

FEATURE_DIR = "my_features"
MODEL_PATH = "relevance_model.pt"
OUT_JSON = "clip_scores.json"

device = "cuda" if torch.cuda.is_available() else "cpu"

visual = torch.load(os.path.join(FEATURE_DIR, "visual.pt"))
audio  = torch.load(os.path.join(FEATURE_DIR, "audio.pt"))
text   = torch.load(os.path.join(FEATURE_DIR, "text.pt"))

model = RelevanceModel().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

results = {}

with torch.no_grad():
    for key in visual.keys():
        v = visual[key].unsqueeze(0).to(device)
        a = audio[key].unsqueeze(0).to(device)
        t = text[key].unsqueeze(0).to(device)

        # modality scores
        sv = torch.sigmoid(model.score(model.v(v))).item()
        sa = torch.sigmoid(model.score(model.a(a))).item()
        st = torch.sigmoid(model.score(model.t(t))).item()

        scene = os.path.splitext(os.path.basename(key))[0]

        results[scene] = {
            "visual_score": sv,
            "audio_score": sa,
            "text_score": st
        }

with open(OUT_JSON, "w") as f:
    json.dump(results, f, indent=2)

print("✅ Saved clip_scores.json")
