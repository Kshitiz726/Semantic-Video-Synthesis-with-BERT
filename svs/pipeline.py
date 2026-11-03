"""End-to-end pipeline: video -> shots -> scenes -> importance -> clips + summary.

Usage:
    python -m svs.pipeline path/to/video.mp4 --out runs/my_video
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import asr, features, importance, scenes as scene_seg, scoring, shots as shot_det, video_io
from .config import CLIP_MODEL, FEATURE_FPS, TEXT_MODEL, WHISPER_MODEL
from .summarize import render_summary, select_summary

STAGES = ["prepare", "shots", "speech", "visual_features", "scenes", "importance", "clips"]


class Pipeline:
    def __init__(self, video, out_dir, progress=None, budget=0.15, make_clips=True):
        self.src = Path(video)
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.progress = progress or (lambda stage, msg: print(f"[{stage}] {msg}", flush=True))
        self.budget = budget
        self.make_clips = make_clips
        self.timings = {}

    def _stage(self, name, fn, *args):
        self.progress(name, "started")
        t0 = time.perf_counter()
        out = fn(*args)
        self.timings[name] = round(time.perf_counter() - t0, 2)
        self.progress(name, f"done in {self.timings[name]:.1f}s")
        return out

    def _cached(self, name, compute):
        """JSON / npy cache inside the run directory so re-runs skip finished stages."""
        path = self.out / name
        if path.exists():
            return np.load(path) if path.suffix == ".npy" else json.loads(path.read_text("utf-8"))
        value = compute()
        if path.suffix == ".npy":
            np.save(path, value)
        else:
            path.write_text(json.dumps(value, ensure_ascii=False), "utf-8")
        return value

    # ---- stages -------------------------------------------------------------
    def prepare(self):
        # Analyse exactly the file the browser plays, so every timestamp matches.
        web = self.out / "video.mp4"
        if not web.exists():
            video_io.transcode_for_web(self.src, web)
        return video_io.probe(web)

    def shots(self, info):
        def run():
            s, probs = shot_det.detect_shots(info)
            return {"shots": s, "probs": np.round(probs, 3).tolist()}
        return self._cached("shots.json", run)

    def speech(self, info):
        audio = video_io.load_audio(info.path) if info.has_audio else np.zeros(0, np.float32)

        def run():
            segs, lang = asr.transcribe(audio)
            return {"segments": segs, "language": lang}
        return audio, self._cached("asr.json", run)

    def visual_features(self, info):
        return self._cached("clip_features.npy", lambda: features.clip_features(info.path))

    @staticmethod
    def score(items, frame_imp, audio, segments):
        text, texts = scoring.text_scores(items, segments)
        raw = {"visual": scoring.visual_scores(items, frame_imp),
               "audio": scoring.audio_scores(items, audio),
               "text": text}
        fused, norm = scoring.fuse(raw)
        return raw, norm, fused, texts

    def importance(self, scenes, units, feats, audio, segments):
        frame_imp = importance.predict_importance(feats)
        return frame_imp, self.score(scenes, frame_imp, audio, segments), self.score(units, frame_imp, audio, segments)

    def clips(self, info, scenes, summary):
        for s in scenes:
            mid = s["start"] + 0.5 * (s["end"] - s["start"])
            video_io.save_thumbnail(info.path, mid, self.out / "thumbs" / f"scene_{s['id']:03d}.jpg")
            if self.make_clips:
                video_io.cut_clip(info.path, s["start"], s["end"], self.out / "clips" / f"scene_{s['id']:03d}.mp4")
        if self.make_clips and summary:
            render_summary(info.path, summary, self.out / "summary.mp4")

    # ---- driver -------------------------------------------------------------
    def run(self):
        t_start = time.perf_counter()
        info = self._stage("prepare", self.prepare)
        shot_data = self._stage("shots", self.shots, info)
        audio, speech = self._stage("speech", self.speech, info)
        feats = self._stage("visual_features", self.visual_features, info)
        segs = speech["segments"]
        scenes, units, boundaries = self._stage(
            "scenes", scene_seg.segment_scenes, shot_data["shots"], feats, segs, info.duration)
        frame_imp, (raw, norm, fused, texts), (_, u_norm, u_fused, _) = self._stage(
            "importance", self.importance, scenes, units, feats, audio, segs)

        order = np.argsort(-np.asarray(fused))
        for i, s in enumerate(scenes):
            s.update(id=i + 1, duration=round(s["end"] - s["start"], 3), transcript=texts[i],
                     score=round(float(fused[i]), 4), rank=int(np.where(order == i)[0][0]) + 1,
                     modalities={m: round(float(norm[m][i]), 4) for m in norm},
                     raw={m: (None if v is None else round(float(v[i]), 4)) for m, v in raw.items()})
        for i, u in enumerate(units):
            u.update(score=round(float(u_fused[i]), 4), modalities={m: round(float(u_norm[m][i]), 4) for m in u_norm})

        # Highlight summary: shot-level knapsack under the length budget.
        summary = select_summary(units, u_fused, self.budget * info.duration)
        for s in scenes:
            s["in_summary"] = any(min(s["end"], g["end"]) - max(s["start"], g["start"]) > 0.5 for g in summary)
        self._stage("clips", self.clips, info, scenes, summary)

        result = {
            "video": info.to_dict() | {"source_name": self.src.name},
            "models": {"shots": "TransNetV2", "asr": f"whisper-{WHISPER_MODEL}", "visual": f"CLIP {CLIP_MODEL}",
                       "text": TEXT_MODEL, "importance": "ImportanceTransformer (TVSum)"},
            "feature_fps": FEATURE_FPS,
            "shots": shot_data["shots"],
            "transition_probs_fps": info.fps,
            "units": units,
            "boundaries": boundaries,
            "transcript": segs,
            "language": speech["language"],
            "frame_importance": np.round(frame_imp, 4).tolist(),
            "available_modalities": list(norm),
            "default_weights": scoring.DEFAULT_WEIGHTS,
            "budget": self.budget,
            "scenes": scenes,
            "summary": summary,
            "timings": self.timings | {"total": round(time.perf_counter() - t_start, 2)},
        }
        (self.out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), "utf-8")
        self.progress("done", f"{len(shot_data['shots'])} shots -> {len(scenes)} scenes "
                              f"in {result['timings']['total']:.0f}s")
        return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget", type=float, default=0.15, help="summary length as a fraction of the video")
    ap.add_argument("--no_clips", action="store_true")
    args = ap.parse_args()
    Pipeline(args.video, args.out, budget=args.budget, make_clips=not args.no_clips).run()


if __name__ == "__main__":
    main()
