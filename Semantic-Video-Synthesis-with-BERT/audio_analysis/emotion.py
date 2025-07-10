from pyAudioAnalysis import audioTrainTest
import os
import json


base_dir = os.path.dirname(__file__)
segment_dir = os.path.join(base_dir, "segments")
model_path = os.path.join(base_dir, "svmModelEmotion")
metadata_file = os.path.join(segment_dir, "segments_metadata.json")
model_type = "svm"


with open(os.path.join(model_path, "svmModelEmotion.classes"), "r") as f:
    class_names = f.read().splitlines()


with open(metadata_file, "r") as f:
    metadata = json.load(f)


for i, item in enumerate(metadata):
    filename = item["filename"]
    try:
        result = audioTrainTest.file_classification(filename, model_path, model_type)
        predicted_class = int(result[0])
        probs = result[1]

        print(f"\nSegment {i}:")
        print(f"Text: {item['text']}")
        print(f"Start: {item['start']}s, End: {item['end']}s")
        print(f"Predicted Emotion: {class_names[predicted_class]}")
        print(f"Probabilities: {probs}")
    except Exception as e:
        print(f"Failed to classify {filename}: {e}")



