import torch
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2Model

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

def extract_audio(video_path):
    y, sr = librosa.load(video_path, sr=16000)

    if len(y) == 0:
        return torch.zeros(768)

    inputs = processor(y, sampling_rate=16000, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        feats = outputs.last_hidden_state.mean(dim=1).squeeze(0)

    return feats
