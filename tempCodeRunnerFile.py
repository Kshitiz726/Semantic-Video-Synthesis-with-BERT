import whisper
import librosa
import soundfile as sf

audio_path = "C:/Users/Asus/Desktop/BERT/Semantic-Video-Synthesis-with-BERT/transcribing_1.mp3"

model = whisper.load_model("base")

result = model.transcribe(audio_path)

y, sr = librosa.load(audio_path, sr=None)
print(f"Audio length in samples: {y.shape[0]}, Sample rate: {sr}")


for i, segment in enumerate(result["segments"]):
    start = int(segment["start"] * sr)
    end = int(segment["end"] * sr)

    y_seg = y[start:end]

    
    segment_filename = f"segment_{i}.wav"
    sf.write(segment_filename, y_seg, sr)

 
    mfcc = librosa.feature.mfcc(y=y_seg, sr=sr, n_mfcc=13)
    print(f"\nSegment {i}:")
    print(f"Text: {segment['text']}")
    print(f"Start: {segment['start']}s, End: {segment['end']}s")
    print(f"MFCC shape: {mfcc.shape}")
