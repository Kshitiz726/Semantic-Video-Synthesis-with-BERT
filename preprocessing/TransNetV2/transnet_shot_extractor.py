import os
import json
import numpy as np
import matplotlib.pyplot as plt
import whisper

from inference.transnetv2 import TransNetV2
from training.metrics_utils import predictions_to_scenes


def plot_prediction_spikes(predictions, threshold, save_path):
    plt.figure(figsize=(12, 4))
    plt.plot(predictions, label="Shot boundary probability")
    plt.axhline(y=threshold, color="r", linestyle="--", label=f"Threshold={threshold}")
    plt.xlabel("Frame")
    plt.ylabel("Probability")
    plt.title("TransNetV2 Shot Boundary Predictions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def detect_shots(video_path, threshold=0.8):
    model = TransNetV2()
    video_frames, frame_predictions, _ = model.predict_video(video_path)

    binary_preds = (frame_predictions > threshold).astype(np.uint8)
    shots = predictions_to_scenes(binary_preds)

    # ensure contiguous shots
    for i in range(1, len(shots)):
        shots[i][0] = shots[i - 1][1] + 1

    return shots, frame_predictions




def transcribe_full_video(video_path):
    print("[INFO] Running Whisper ASR on full video...")
    model = whisper.load_model("base")
    result = model.transcribe(video_path, temperature=0, best_of=1, word_timestamps=False)

    return [
        {
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "text": seg["text"].strip()
        }
        for seg in result["segments"]
    ]


def collect_dialogue_for_shot(shot_start, shot_end, asr_segments):
    texts = []
    for seg in asr_segments:
        if seg["end"] <= shot_start:
            continue
        if seg["start"] >= shot_end:
            break
        texts.append(seg["text"])
    return " ".join(texts).strip()



def extract_shots_with_dialogue(
    video_path,
    output_json_path,
    threshold=0.8,
    fps=25,
    debug_dir=None
):
    print("[STAGE 1] Shot detection + ASR")

    # 1. Detect shots
    shots, frame_predictions = detect_shots(video_path, threshold)

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        plot_prediction_spikes(
            frame_predictions,
            threshold,
            os.path.join(debug_dir, "transnet_predictions.png")
        )

    # 2. ASR
    asr_segments = transcribe_full_video(video_path)

    # 3. Build shot JSON
    shot_data = []

    for idx, (start_frame, end_frame) in enumerate(shots):
        start_time = start_frame / fps
        end_time = end_frame / fps

        if end_time <= start_time:
            continue

        dialogue = collect_dialogue_for_shot(
            start_time,
            end_time,
            asr_segments
        )

        shot_data.append({
            "shot_id": idx + 1,
            "start_time": round(start_time, 3),
            "end_time": round(end_time, 3),
            "duration": round(end_time - start_time, 3),
            "dialogue": dialogue
        })

    # 4. Save
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(shot_data, f, indent=2, ensure_ascii=False)

    print(f"[DONE] {len(shot_data)} shots saved → {output_json_path}")
    return output_json_path
