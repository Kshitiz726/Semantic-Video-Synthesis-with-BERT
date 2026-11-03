"""Frame-level visual importance models (trained on TVSum, see training/train_importance.py)."""

import math
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn

from .config import DEVICE, IMPORTANCE_WEIGHTS


def sinusoidal(T, d, device):
    pos = torch.arange(T, device=device, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / d))
    pe = torch.zeros(T, d, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class ImportanceTransformer(nn.Module):
    """Temporal Transformer encoder over a CLIP feature sequence -> per-frame importance in [0, 1]."""

    def __init__(self, d_in=512, d_model=256, n_heads=4, n_layers=2, dropout=0.3, use_pos=True):
        super().__init__()
        self.use_pos = use_pos
        self.inp = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d_model), nn.GELU(), nn.Dropout(dropout))
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 2 * d_model, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, x):  # x: (B, T, d_in)
        h = self.inp(x)
        if self.use_pos:
            h = h + sinusoidal(x.shape[1], h.shape[-1], x.device)
        h = self.encoder(h)
        return torch.sigmoid(self.head(h).squeeze(-1))


class FrameMLP(nn.Module):
    """Per-frame MLP without temporal context (ablation baseline)."""

    def __init__(self, d_in=512, hidden=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, 1))

    def forward(self, x):
        return torch.sigmoid(self.net(x).squeeze(-1))


class SegmentMLP(nn.Module):
    """The project's original design: gated MLP on mean-pooled fixed-length segments.

    Applied here to correct CLIP features so it can be compared fairly.
    """

    def __init__(self, d_in=512, dropout=0.4, seg_len=40):
        super().__init__()
        self.seg_len = seg_len  # in feature steps (40 = 20 s at 2 fps)
        blocks, dims = [], [d_in, 256, 128, 64]
        for a, b in zip(dims[:-1], dims[1:]):
            blocks += [nn.Linear(a, b), nn.LayerNorm(b), nn.ReLU(), nn.Dropout(dropout)]
        self.mlp = nn.Sequential(*blocks)
        self.gate = nn.Linear(64, 64)
        self.out = nn.Linear(64, 1)

    def forward(self, x):  # (B, T, D) -> pooled per segment, broadcast back to frames
        B, T, D = x.shape
        n = math.ceil(T / self.seg_len)
        pad = n * self.seg_len - T
        mask = torch.ones(B, T, 1, device=x.device)
        xp = torch.nn.functional.pad(x, (0, 0, 0, pad)).view(B, n, self.seg_len, D)
        mp = torch.nn.functional.pad(mask, (0, 0, 0, pad)).view(B, n, self.seg_len, 1)
        pooled = (xp * mp).sum(2) / mp.sum(2).clamp(min=1)
        h = self.mlp(pooled)
        h = h * torch.sigmoid(self.gate(h))
        s = torch.sigmoid(self.out(h)).squeeze(-1)  # (B, n)
        return s.repeat_interleave(self.seg_len, dim=1)[:, :T]


MODELS = {"transformer": ImportanceTransformer, "frame_mlp": FrameMLP, "segment_mlp": SegmentMLP}


def build_model(name, **kwargs):
    return MODELS[name](**kwargs)


@lru_cache(maxsize=1)
def load_importance_model(path=str(IMPORTANCE_WEIGHTS)):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    model = build_model(ckpt["model"], **ckpt.get("kwargs", {})).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def predict_importance(features, model=None):
    """(T, 512) CLIP features -> (T,) importance scores in [0, 1]."""
    if len(features) == 0:
        return np.zeros(0, np.float32)
    model = model or load_importance_model()
    x = torch.from_numpy(np.asarray(features, np.float32)).unsqueeze(0).to(DEVICE)
    return model(x).squeeze(0).cpu().numpy()
