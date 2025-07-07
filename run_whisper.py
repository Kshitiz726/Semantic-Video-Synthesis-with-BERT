import whisper
import librosa
import soundfile as sf
import os
import json
import numpy as np

audio_path = "C:/Users/Asus/Desktop/BERT/Semantic-Video-Synthesis-with-BERT/editing_1.mp3"
project_dir = os.path.dirname(audio_path)
print("Project Directory:", project_dir)
output_dir = os.path.join(project_dir, "segments")
os.makedirs(output_dir, exist_ok=True)

model = whisper.load_model("base")
result = model.transcribe(audio_path)

y, sr = librosa.load(audio_path, sr=None)

metadata = []

for i, segment in enumerate(result["segments"]):
    start = int(segment["start"] * sr)
    end = int(segment["end"] * sr)
    y_seg = y[start:end]

    segment_filename = os.path.join(output_dir, f"segment_{i}.wav")
    sf.write(segment_filename, y_seg, sr)

    
    mfcc = librosa.feature.mfcc(y=y_seg, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1) 

   
    print(f"Segment {i}:")
    print(f"  Filename: {segment_filename}")
    print(f"  Text: {segment['text']}")
    print(f"  Start: {segment['start']} sec, End: {segment['end']} sec")
    print(f"  MFCC mean coefficients: {mfcc_mean}\n")

    metadata.append({
        "filename": segment_filename,
        "text": segment["text"],
        "start": segment["start"],
        "end": segment["end"],
        "mfcc_mean": mfcc_mean.tolist()  
    })

with open(os.path.join(output_dir, "segments_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print(f"Segments and metadata saved to: {output_dir}")
