import subprocess
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
import csv

from inference.transnetv2 import TransNetV2
from training.metrics_utils import predictions_to_scenes


def detect_scenes(video_path, fps=25, threshold=0.5):
    model = TransNetV2()
    video_frames, single_frame_predictions, _ = model.predict_video(video_path)

    
    plot_prediction_spikes(single_frame_predictions, threshold)

    binary_predictions = (single_frame_predictions > threshold).astype(np.uint8)
    scenes = predictions_to_scenes(binary_predictions)

    
    for i in range(1, len(scenes)):
        scenes[i][0] = scenes[i - 1][1] + 1  

    return scenes, fps


def plot_prediction_spikes(predictions, threshold=0.5):
    plt.figure(figsize=(12, 4))
    plt.plot(predictions, label="Scene change probability")
    plt.axhline(y=threshold, color='r', linestyle='--', label=f"Threshold = {threshold}")
    plt.title("TransNetV2 Scene Change Predictions")
    plt.xlabel("Frame")
    plt.ylabel("Probability")
    plt.legend()
    plt.tight_layout()
    plt.savefig("scene_prediction_spikes.png")
    plt.close()


def format_timestamp(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}_{m:02d}_{s:06.3f}"


def extract_video_scene(video_file, output_path, start, end):
    """
    Extract video clip with re-encoding to avoid black frames
    """
    subprocess.call([
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", str(start),
        "-to", str(end),
        "-i", video_file,
        "-c:v", "libx264",
        "-c:a", "aac",
        "-preset", "fast",
        "-strict", "experimental",
        output_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def extract_audio_scene(video_file, output_path, start, end):
    subprocess.call([
        "ffmpeg", "-y", "-loglevel", "error", "-i", video_file,
        "-ss", str(start), "-to", str(end),
        "-vn", "-acodec", "mp3", output_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def extract_timestamped_frames(video_file, output_dir, start, end, fps=1):
    cap = cv2.VideoCapture(video_file)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)

    while cap.isOpened():
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        if current_time > end:
            break

        ret, frame = cap.read()
        if not ret:
            break

        timestamp_str = format_timestamp(current_time)
        filename = os.path.join(output_dir, f"frame_{timestamp_str}.jpg")
        cv2.imwrite(filename, frame)

        cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + video_fps // fps)

    cap.release()


def extract_metadata(video_file, output_dir):
    name = os.path.splitext(os.path.basename(video_file))[0]
    output_path = os.path.join(output_dir, f"{name}_metadata.json")

    os.makedirs(output_dir, exist_ok=True)

    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", video_file],
        capture_output=True, text=True, check=True
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result.stdout)


def get_video_duration(video_file):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", video_file],
        capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except Exception:
        return None


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python transnet_scene_extractor.py <video_file>")
        sys.exit(1)

    video_file = sys.argv[1]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    preprocessing_dir = os.path.abspath(os.path.join(script_dir, ".."))
    base_output_dir = os.path.join(preprocessing_dir, "extracted")
 

    video_name = os.path.splitext(os.path.basename(video_file))[0]

    audio_output_dir = os.path.join(base_output_dir, video_name, "Audio")
    frame_output_dir = os.path.join(base_output_dir, video_name, "Frames")
    metadata_output_dir = os.path.join(base_output_dir, video_name, "Metadata")
    clip_output_dir = os.path.join(base_output_dir, video_name, "Clips")
    os.makedirs(audio_output_dir, exist_ok=True)
    os.makedirs(frame_output_dir, exist_ok=True)
    os.makedirs(metadata_output_dir, exist_ok=True)
    os.makedirs(clip_output_dir, exist_ok=True) 
    print("[INFO] Detecting scenes using TransNetV2...")
    scenes, fps = detect_scenes(video_file)
    duration = get_video_duration(video_file)
    if duration is None:
        print("[WARNING] Could not determine video duration. Proceeding without clamping.")
        duration = float("inf")
    scene_data = []
    for i, (start_frame, end_frame) in enumerate(scenes):
        start_time = start_frame / fps
        end_time = end_frame / fps


        
        
        if start_time > duration:
            break
        if end_time > duration:
            end_time = duration

        
        if (end_time - start_time) < 1.0:
            continue

        scene_name = f"scene_{i+1:03d}"
        scene_video_path = os.path.join(frame_output_dir, f"{scene_name}.mp4")
        scene_audio_path = os.path.join(audio_output_dir, f"{scene_name}.mp3")
        scene_frame_dir = os.path.join(frame_output_dir, scene_name)
        os.makedirs(scene_frame_dir, exist_ok=True)

        print(f"[INFO] Processing {scene_name} from {start_time:.2f}s to {end_time:.2f}s")
        scene_data.append({
    "scene_index": i + 1,
    "scene_name": scene_name,
    "scene_duration": round(end_time - start_time, 3),
    "start_time": round(start_time, 3),
    "end_time": round(end_time, 3)
})

       
        extract_video_scene(video_file, scene_video_path, start_time, end_time)

       
        extract_audio_scene(video_file, scene_audio_path, start_time, end_time)

       
        extract_timestamped_frames(video_file, scene_frame_dir, start_time, end_time, fps=1)



    summary_csv_path = os.path.join(base_output_dir, video_name, "scene_summary.csv")
    summary_json_path = os.path.join(base_output_dir, video_name, "scene_summary.json")

    # Save JSON
    with open(summary_json_path, "w", encoding="utf-8") as jf:
        json.dump(scene_data, jf, indent=4)

    # Save CSV
    with open(summary_csv_path, "w", newline='', encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=["scene_index", "scene_name", "scene_duration", "start_time", "end_time"])
        writer.writeheader()
        writer.writerows(scene_data)


    extract_metadata(video_file, metadata_output_dir)
    print("[DONE] Scene extraction complete.")
