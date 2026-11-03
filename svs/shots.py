"""Shot boundary detection with TransNetV2 (PyTorch port of the official weights)."""

from functools import lru_cache

import numpy as np
import torch

from .config import DEVICE, TRANSNET_WEIGHTS
from .third_party.transnetv2 import TransNetV2
from .video_io import read_frames


@lru_cache(maxsize=1)
def _model():
    if not TRANSNET_WEIGHTS.exists():
        raise FileNotFoundError(f"{TRANSNET_WEIGHTS} missing - run `python scripts/download_weights.py`")
    model = TransNetV2()
    model.load_state_dict(torch.load(TRANSNET_WEIGHTS, map_location="cpu", weights_only=True))
    return model.eval().to(DEVICE)


@torch.no_grad()
def transition_probabilities(frames):
    """Per-frame transition probability for uint8 frames of shape (N, 27, 48, 3).

    Mirrors the official inference: windows of 100 frames, stride 50, keeping the
    central 50 predictions, with the clip padded by copies of its first/last frame.
    """
    n = len(frames)
    pad_end = 25 + 50 - (n % 50 if n % 50 else 50)
    padded = np.concatenate([np.repeat(frames[:1], 25, 0), frames, np.repeat(frames[-1:], pad_end, 0)])
    model = _model()
    preds = []
    windows = [padded[p:p + 100] for p in range(0, len(padded) - 99, 50)]
    for i in range(0, len(windows), 8):
        batch = torch.from_numpy(np.stack(windows[i:i + 8])).to(DEVICE)
        single, _ = model(batch)
        preds.append(torch.sigmoid(single[:, 25:75, 0]).reshape(-1).cpu().numpy())
    return np.concatenate(preds)[:n]


def detect_shots(video_info, threshold=0.5, min_shot_len=0.4):
    """Return (shots, probs): shots as [{start, end}] in seconds covering the whole video."""
    fps = video_info.fps
    frames = read_frames(video_info.path, fps, 48, 27)
    if len(frames) == 0:
        raise ValueError("No frames decoded from video")
    probs = transition_probabilities(frames)

    # Each run of above-threshold frames is one transition; cut at its centre so shots are contiguous.
    above = probs > threshold
    cuts, i = [], 0
    while i < len(above):
        if above[i]:
            j = i
            while j + 1 < len(above) and above[j + 1]:
                j += 1
            cuts.append((i + j + 1) // 2)
            i = j + 1
        else:
            i += 1

    min_frames = max(1, int(round(min_shot_len * fps)))
    bounds = [0]
    for c in cuts:
        if c - bounds[-1] >= min_frames and len(frames) - c >= min_frames:
            bounds.append(c)
    bounds.append(len(frames))
    duration = len(frames) / fps
    shots = [{"start": round(a / fps, 3), "end": round(min(b / fps, duration), 3)}
             for a, b in zip(bounds[:-1], bounds[1:])]
    return shots, probs
