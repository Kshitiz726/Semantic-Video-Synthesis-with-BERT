import os
import shutil

ravdess_dir = r"C:\Users\Asus\Desktop\Major\Semantic-Video-Synthesis-with-BERT\Semantic-Video-Synthesis-with-BERT\audio_analysis\Audio_Speech_Actors_01-24"

output_dir = r"C:\Users\Asus\Desktop\Major\Semantic-Video-Synthesis-with-BERT\Semantic-Video-Synthesis-with-BERT\audio_analysis\emotiondataset"

emotion_map = {
    "01": "neutral",
    "02": "calm",  
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}   


for emotion in emotion_map.values():
    os.makedirs(os.path.join(output_dir, emotion), exist_ok=True)


for actor in os.listdir(ravdess_dir):
    actor_path = os.path.join(ravdess_dir, actor)
    if os.path.isdir(actor_path):
        for file in os.listdir(actor_path):
            if file.endswith(".wav"):
                parts = file.split("-")
                emotion_code = parts[2]
                if emotion_code in emotion_map:
                    label = emotion_map[emotion_code]
                    src = os.path.join(actor_path, file)
                    dst = os.path.join(output_dir, label, file)
                    shutil.copyfile(src, dst)

print("RAVDESS files sorted into 'emotiondataset.")
