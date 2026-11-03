"""Video / audio decoding and cutting via ffmpeg pipes.

All frame streams are decoded at an explicit constant frame rate so that
frame index i always corresponds to time i / fps, even for VFR input.
"""

import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import ffmpeg_bin

CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


@dataclass
class VideoInfo:
    path: str
    fps: float
    duration: float
    width: int
    height: int
    has_audio: bool

    def to_dict(self):
        return asdict(self)


def probe(path) -> VideoInfo:
    path = str(path)
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # ffmpeg -i prints container metadata to stderr (and exits non-zero: no output given)
    info = subprocess.run([ffmpeg_bin(), "-hide_banner", "-i", path], capture_output=True, text=True,
                          encoding="utf-8", errors="replace").stderr
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", info)
    duration = int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3]) if m else 0.0
    if fps <= 0 or fps > 240:
        m = re.search(r"(\d+(?:\.\d+)?) fps", info)
        fps = float(m[1]) if m else 25.0
    if duration <= 0:
        duration = n / fps if n else 0.0
    if duration <= 0:
        raise ValueError(f"Could not read video: {path}")
    return VideoInfo(path, float(fps), float(duration), w, h, "Audio:" in info)


def iter_frames(path, fps, width, height, chunk=256, crop_square=False):
    """Yield uint8 RGB frame chunks of shape (k, height, width, 3) at a constant fps."""
    if crop_square:
        vf = f"fps={fps},scale={width}:{height}:force_original_aspect_ratio=increase:flags=bicubic,crop={width}:{height}"
    else:
        vf = f"fps={fps},scale={width}:{height}:flags=bilinear"
    cmd = [ffmpeg_bin(), "-v", "error", "-i", str(path), "-vf", vf, "-an",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    frame_bytes = width * height * 3
    try:
        while True:
            buf = proc.stdout.read(frame_bytes * chunk)
            if not buf:
                break
            k = len(buf) // frame_bytes
            if k == 0:
                break
            yield np.frombuffer(bytearray(buf[: k * frame_bytes]), np.uint8).reshape(k, height, width, 3)
    finally:
        proc.stdout.close()
        proc.wait()


def read_frames(path, fps, width, height, crop_square=False):
    chunks = list(iter_frames(path, fps, width, height, crop_square=crop_square))
    if not chunks:
        return np.zeros((0, height, width, 3), np.uint8)
    return np.concatenate(chunks)


def load_audio(path, sr=16000) -> np.ndarray:
    """Decode the audio track to mono float32 at `sr` Hz (empty array if no audio)."""
    cmd = [ffmpeg_bin(), "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sr),
           "-f", "f32le", "pipe:1"]
    out = subprocess.run(cmd, capture_output=True).stdout
    return np.frombuffer(out, np.float32).copy()


def cut_clip(path, start, end, out_path, height=None):
    """Frame-accurate cut (re-encode). Input-side seek + -t keeps timestamps exact."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg_bin(), "-v", "error", "-y", "-ss", f"{start:.3f}", "-i", str(path),
           "-t", f"{max(end - start, 0.04):.3f}"]
    if height:
        cmd += ["-vf", f"scale=-2:'min({height},ih)'"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(out_path)]
    subprocess.run(cmd, check=True)


def save_thumbnail(path, t, out_path, width=320):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg_bin(), "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", str(path), "-frames:v", "1",
           "-vf", f"scale={width}:-2", "-q:v", "4", str(out_path)]
    subprocess.run(cmd, check=True)


def concat_clips(clip_paths, out_path):
    """Concatenate clips that share encoding parameters (as produced by cut_clip)."""
    out_path = Path(out_path)
    list_file = out_path.with_suffix(".txt")
    list_file.write_text("".join(f"file '{Path(p).resolve().as_posix()}'\n" for p in clip_paths))
    cmd = [ffmpeg_bin(), "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file),
           "-c", "copy", "-movflags", "+faststart", str(out_path)]
    subprocess.run(cmd, check=True)
    list_file.unlink()


def transcode_for_web(path, out_path, height=720):
    """H.264/AAC copy of the source so every browser can play it (remux only when already compatible)."""
    info = subprocess.run([ffmpeg_bin(), "-hide_banner", "-i", str(path)], capture_output=True, text=True,
                          encoding="utf-8", errors="replace").stderr
    video = re.search(r"Video: (\w+).*?, (\w+)\(?.*?, (\d+)x(\d+)", info)
    audio = re.search(r"Audio: (\w+)", info)
    if (video and video[1] == "h264" and video[2] == "yuv420p" and int(video[4]) <= height
            and (audio is None or audio[1] == "aac")):
        cmd = [ffmpeg_bin(), "-v", "error", "-y", "-i", str(path), "-map", "0:v:0", "-map", "0:a:0?",
               "-c", "copy", "-movflags", "+faststart", str(out_path)]
        if subprocess.run(cmd).returncode == 0:
            return
    cmd = [ffmpeg_bin(), "-v", "error", "-y", "-i", str(path),
           "-vf", f"scale=-2:'min({height},ih)'", "-c:v", "libx264", "-preset", "veryfast",
           "-crf", "23", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
           "-movflags", "+faststart", str(out_path)]
    subprocess.run(cmd, check=True)
