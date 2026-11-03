"""Speech recognition with Whisper."""

import re
from functools import lru_cache

import numpy as np

from .config import CACHE_DIR, DEVICE, WHISPER_MODEL


def _collapse_repeats(text, max_repeats=2):
    """Keep at most `max_repeats` consecutive copies of the same sentence, then of the same word."""
    def collapse(units):
        out = []
        for unit in units:
            key = unit.lower().strip(".,!? ")
            if len(out) >= max_repeats and all(u.lower().strip(".,!? ") == key for u in out[-max_repeats:]):
                continue
            out.append(unit)
        return out

    sentences = collapse(re.split(r"(?<=[.!?])\s+", text))
    return " ".join(collapse(" ".join(sentences).split()))


@lru_cache(maxsize=1)
def _whisper():
    import whisper

    return whisper.load_model(WHISPER_MODEL, device=DEVICE, download_root=str(CACHE_DIR / "whisper"))


def transcribe(audio, sr=16000):
    """Return Whisper segments [{start, end, text}] for a 16 kHz mono float32 signal."""
    if len(audio) < sr or np.abs(audio).max() < 1e-4:
        return [], None
    result = _whisper().transcribe(audio, fp16=DEVICE == "cuda", temperature=0.0,
                                   condition_on_previous_text=False, verbose=None)
    segments = []
    for s in result["segments"]:
        text = _collapse_repeats(s["text"].strip())
        # Drop likely hallucinations: silence/music, and repetition loops (Whisper's own
        # compression-ratio criterion, e.g. "yo yo yo yo ..." over a song).
        if not text or (s["no_speech_prob"] > 0.6 and s["avg_logprob"] < -1.0) or s["compression_ratio"] > 2.4:
            continue
        segments.append({"start": round(float(s["start"]), 3), "end": round(float(s["end"]), 3), "text": text})
    return segments, result.get("language")
