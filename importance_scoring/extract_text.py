import whisper
import torch
from sentence_transformers import SentenceTransformer

asr = whisper.load_model("base")
text_model = SentenceTransformer("all-MiniLM-L6-v2")

def transcribe(video_path):
    result = asr.transcribe(video_path)
    return result["text"]

def extract_text_embedding(text):
    if text.strip() == "":
        return torch.zeros(384)

    emb = text_model.encode(text)
    return torch.tensor(emb)
