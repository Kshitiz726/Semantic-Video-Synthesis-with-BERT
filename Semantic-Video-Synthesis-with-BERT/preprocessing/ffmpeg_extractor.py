""" Preprocessing Step of the Project """
import subprocess
import os
import sys
import json

def extract_audio(video_file, output_dir, output_ext="mp3"):
    filename = os.path.splitext(os.path.basename(video_file))[0]

    output_path = os.path.join(output_dir, f"{filename}.{output_ext}")

    retcode = subprocess.call(
        ["ffmpeg", "-y", "-i", video_file, output_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT
    )

def extract_frames(video_file, output_frame_dir, frame_rate=1):
    video_name = os.path.splitext(os.path.basename(video_file))[0]
    
    video_frame_dir = os.path.join(output_frame_dir, video_name)

    if not os.path.exists(video_frame_dir):
        os.makedirs(video_frame_dir)

    output_pattern = os.path.join(video_frame_dir, "frame%04d.jpg")

    retcode = subprocess.call(
        ["ffmpeg", "-y", "-i", video_file, "-vf", f"fps={frame_rate}", output_pattern],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT
    )


def extract_metadata(video_file, output_dir):
    video_name = os.path.splitext(os.path.basename(video_file))[0]
    output_path = os.path.join(output_dir, f"{video_name}_metadata.json")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", video_file],
            capture_output=True,
            text=True,
            check=True
        )

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result.stdout)
        print(f"Video Processed")

    except subprocess.CalledProcessError as e:
        print(f"Failed to extract preproceess {video_file}")
        print(e)
    
    
if __name__ == "__main__":
    vf = sys.argv[1]
    output_audio_dir = sys.argv[2]
    output_frame_dir = sys.argv[3]
    output_metadata_dir = sys.argv[4]
    extract_frames(vf, output_frame_dir)
    extract_audio(vf, output_audio_dir)
    extract_metadata(vf, output_metadata_dir)
    
