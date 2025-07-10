import os
from pyAudioAnalysis import audioTrainTest

data_folder = r"C:\Users\Asus\Desktop\Major\Semantic-Video-Synthesis-with-BERT\Semantic-Video-Synthesis-with-BERT\audio_analysis\emotiondataset"

print("Exists:", os.path.exists(data_folder))
print("Folders:", os.listdir(data_folder))

for folder in os.listdir(data_folder):
    folder_path = os.path.join(data_folder, folder)
    if os.path.isdir(folder_path):
        files = os.listdir(folder_path)
        wavs = [f for f in files if f.lower().endswith(".wav")]
        print(f"{folder}: total files={len(files)}, wav files={len(wavs)}")


mt_win = 1.0     # mid-term window size
mt_step = 0.5    # mid-term step size
st_win = 0.05    # short-term window size
st_step = 0.025  # short-term step size

classifier_type = "svm"
model_name = "svmModelEmotion"

print(f"\nStarting training on dataset at: {data_folder}")

audioTrainTest.extract_features_and_train(
    data_folder,
    mt_win,
    mt_step,
    st_win,
    st_step,
    classifier_type,
    model_name
)

print("Model training complete.")
