"""Global configuration: paths, devices, model names."""

import os
import shutil
from functools import lru_cache
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = ROOT / "weights"
CACHE_DIR = Path(os.environ.get("SVS_CACHE_DIR", Path.home() / ".cache" / "svs"))

# Keep model downloads in one place, and stop `transformers` from importing TensorFlow.
os.environ.setdefault("HF_HOME", str(CACHE_DIR / "huggingface"))
os.environ.setdefault("TORCH_HOME", str(CACHE_DIR / "torch"))
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

DEVICE = os.environ.get("SVS_DEVICE") or ("cuda" if torch.cuda.is_available() and torch.cuda.device_count() else "cpu")
if os.environ.get("SVS_THREADS"):
    torch.set_num_threads(int(os.environ["SVS_THREADS"]))

# Visual backbone shared by scene segmentation and importance scoring.
CLIP_MODEL = "ViT-B/32"
FEATURE_FPS = 2.0  # frames per second sampled for CLIP features

# Sentence-BERT model used for dialogue embeddings.
TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

WHISPER_MODEL = os.environ.get("SVS_WHISPER_MODEL", "small" if DEVICE == "cuda" else "base")

TRANSNET_WEIGHTS = WEIGHTS_DIR / "transnetv2-pytorch-weights.pth"
IMPORTANCE_WEIGHTS = WEIGHTS_DIR / "importance_transformer.pt"


@lru_cache(maxsize=1)
def ffmpeg_bin() -> str:
    """Locate an ffmpeg executable (env var, PATH, then the imageio-ffmpeg wheel)."""
    if os.environ.get("SVS_FFMPEG"):
        return os.environ["SVS_FFMPEG"]
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as e:
        raise RuntimeError(
            "ffmpeg not found. Install it, add it to PATH, or `pip install imageio-ffmpeg`."
        ) from e
