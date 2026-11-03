"""CLIP visual features and Sentence-BERT text embeddings."""

from functools import lru_cache

import numpy as np
import torch

from .config import CACHE_DIR, CLIP_MODEL, DEVICE, FEATURE_FPS, TEXT_MODEL
from .video_io import CLIP_MEAN, CLIP_STD, iter_frames


@lru_cache(maxsize=1)
def _clip():
    import clip

    model, _ = clip.load(CLIP_MODEL, device=DEVICE, download_root=str(CACHE_DIR / "clip"))
    model.eval()
    return model


@lru_cache(maxsize=1)
def _sbert():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(TEXT_MODEL, device=DEVICE)


@torch.no_grad()
def clip_features(video_path, fps=FEATURE_FPS, batch=128):
    """L2-normalised CLIP image embeddings, one per 1/fps seconds. Returns (T, 512) float32."""
    model = _clip()
    size = model.visual.input_resolution
    mean = torch.tensor(CLIP_MEAN, device=DEVICE).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD, device=DEVICE).view(1, 3, 1, 1)
    out = []
    for chunk in iter_frames(video_path, fps, size, size, chunk=batch, crop_square=True):
        x = torch.from_numpy(chunk).to(DEVICE).permute(0, 3, 1, 2).float().div_(255)
        x = (x - mean) / std
        f = model.encode_image(x.to(next(model.parameters()).dtype)).float()
        out.append(torch.nn.functional.normalize(f, dim=-1).cpu())
    if not out:
        return np.zeros((0, model.visual.output_dim), np.float32)
    return torch.cat(out).numpy().astype(np.float32)


def text_embeddings(texts):
    """Normalised Sentence-BERT embeddings for a list of strings. Returns (N, 384)."""
    model = _sbert()
    if not texts:
        return np.zeros((0, model.get_sentence_embedding_dimension()), np.float32)
    return model.encode(list(texts), normalize_embeddings=True, batch_size=64,
                        show_progress_bar=False).astype(np.float32)
