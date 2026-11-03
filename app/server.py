"""ClipRanker web server: upload a video, follow the pipeline, explore the ranked clips.

    python -m app.server                      # local, http://127.0.0.1:8000
    SVS_PUBLIC=1 python -m app.server --host 0.0.0.0 --port 7860   # public deployment

Public mode (SVS_PUBLIC=1) keeps every visitor's runs private (a run is only reachable by
its unguessable id), limits upload size and video length, and deletes runs after a while.
"""

import json
import os
import queue
import re
import shutil
import threading
import time
import traceback
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from svs.config import DEVICE, ROOT

RUNS_DIR = Path(os.environ.get("SVS_RUNS_DIR", ROOT / "runs"))
RUNS_DIR.mkdir(parents=True, exist_ok=True)
STATIC = Path(__file__).parent / "static"
BENCHMARK = ROOT / "training" / "results"
VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}

PUBLIC = os.environ.get("SVS_PUBLIC") == "1"
MAX_UPLOAD_MB = int(os.environ.get("SVS_MAX_UPLOAD_MB", 300 if PUBLIC else 20000))
MAX_DURATION = float(os.environ.get("SVS_MAX_DURATION", 600 if PUBLIC else 1e9))  # seconds
RUN_TTL_HOURS = float(os.environ.get("SVS_RUN_TTL_HOURS", 24 if PUBLIC else 0))
CORS_ORIGINS = [o for o in os.environ.get(
    "SVS_CORS_ORIGINS", "https://kshitizrajpaudyal.com.np,https://www.kshitizrajpaudyal.com.np,"
                        "http://localhost:5173,http://127.0.0.1:8000").split(",") if o]

app = FastAPI(title="ClipRanker")
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"])
jobs: "queue.Queue[str]" = queue.Queue()
live = {}  # run_id -> {"stage", "log": [...]} for runs in progress
ready = threading.Event()


def _meta_path(run_id):
    if not re.fullmatch(r"[\w-]+", run_id):
        raise HTTPException(404, "run not found")
    return RUNS_DIR / run_id / "meta.json"


def read_meta(run_id):
    p = _meta_path(run_id)
    if not p.exists():
        raise HTTPException(404, "run not found")
    return json.loads(p.read_text("utf-8"))


def write_meta(run_id, **update):
    p = _meta_path(run_id)
    meta = json.loads(p.read_text("utf-8")) if p.exists() else {}
    meta.update(update)
    p.write_text(json.dumps(meta), "utf-8")
    return meta


def warmup():
    """Load every model once so the first upload does not pay start-up costs."""
    from svs import asr, features, importance, shots

    for load in (shots._model, features._clip, features._sbert, asr._whisper, importance.load_importance_model):
        try:
            load()
        except Exception as e:  # missing weights etc. are reported when a run fails
            print(f"warm-up: {load.__module__} failed: {e}")
    ready.set()


def worker():
    from svs.pipeline import STAGES, Pipeline
    from svs.video_io import probe

    warmup()
    while True:
        run_id = jobs.get()
        try:
            meta = read_meta(run_id)
        except HTTPException:  # deleted while queued
            continue
        state = live.setdefault(run_id, {"stage": None, "log": []})

        def progress(stage, msg):
            state["stage"] = stage
            state["log"].append(f"{time.strftime('%H:%M:%S')} [{stage}] {msg}")
            if stage in STAGES:
                write_meta(run_id, stage=stage, stage_index=STAGES.index(stage), n_stages=len(STAGES))

        write_meta(run_id, status="running", started=time.time())
        try:
            source = RUNS_DIR / run_id / meta["source"]
            duration = probe(source).duration
            if duration > MAX_DURATION:
                raise ValueError(f"Video is {duration / 60:.1f} min long; this server accepts up to "
                                 f"{MAX_DURATION / 60:.0f} min.")
            result = Pipeline(source, RUNS_DIR / run_id, progress).run()
            write_meta(run_id, status="done", finished=time.time(), n_scenes=len(result["scenes"]),
                       n_shots=len(result["shots"]), duration=result["video"]["duration"])
            if PUBLIC:
                source.unlink(missing_ok=True)  # the H.264 copy (video.mp4) is all the UI needs
        except Exception as e:  # surface the failure in the UI
            state["log"].append(traceback.format_exc())
            write_meta(run_id, status="error", error=f"{type(e).__name__}: {e}")
        finally:
            (RUNS_DIR / run_id / "log.txt").write_text("\n".join(state["log"]), "utf-8")
            live.pop(run_id, None)


def janitor():
    """Delete finished runs older than RUN_TTL_HOURS (demo runs are kept)."""
    while True:
        cutoff = time.time() - RUN_TTL_HOURS * 3600
        for p in RUNS_DIR.glob("*/meta.json"):
            try:
                meta = json.loads(p.read_text("utf-8"))
                if not meta.get("demo") and meta.get("status") in ("done", "error") \
                        and meta.get("created", 0) < cutoff:
                    shutil.rmtree(p.parent, ignore_errors=True)
            except (OSError, ValueError):
                pass
        time.sleep(1800)


@app.on_event("startup")
def startup():
    # Runs interrupted by a restart are re-queued.
    for p in sorted(RUNS_DIR.glob("*/meta.json"), key=lambda p: p.stat().st_mtime):
        meta = json.loads(p.read_text("utf-8"))
        if meta.get("status") in ("queued", "running"):
            write_meta(meta["id"], status="queued")
            jobs.put(meta["id"])
    threading.Thread(target=worker, daemon=True).start()
    if RUN_TTL_HOURS > 0:
        threading.Thread(target=janitor, daemon=True).start()


@app.get("/api/health")
def health():
    return {"ok": True, "models_loaded": ready.is_set(), "device": DEVICE, "queue": jobs.qsize(),
            "public": PUBLIC, "max_duration": MAX_DURATION, "max_upload_mb": MAX_UPLOAD_MB}


@app.post("/api/runs")
async def create_run(file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in VIDEO_EXTS:
        raise HTTPException(400, f"Unsupported file type {ext!r}")
    run_id = (time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]) if not PUBLIC else uuid.uuid4().hex
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir()
    size, limit = 0, MAX_UPLOAD_MB * 1024 * 1024
    with open(run_dir / f"source{ext}", "wb") as f:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > limit:
                f.close()
                shutil.rmtree(run_dir)
                raise HTTPException(413, f"File larger than {MAX_UPLOAD_MB} MB")
            f.write(chunk)
    name = re.sub(r"[^\w .\-()]", "_", Path(file.filename).stem)[:80]
    write_meta(run_id, id=run_id, name=name, source=f"source{ext}", status="queued", created=time.time())
    jobs.put(run_id)
    return read_meta(run_id)


@app.get("/api/runs")
def list_runs(ids: str = Query("")):
    """All runs locally; in public mode only demo runs plus the ids this browser owns."""
    wanted = {i for i in ids.split(",") if i}
    metas = []
    for p in RUNS_DIR.glob("*/meta.json"):
        meta = json.loads(p.read_text("utf-8"))
        if not PUBLIC or meta.get("demo") or meta.get("id") in wanted:
            metas.append(meta)
    return sorted(metas, key=lambda m: (not m.get("demo"), -m.get("created", 0)))


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    meta = read_meta(run_id)
    log_file = RUNS_DIR / run_id / "log.txt"
    log = live.get(run_id, {}).get("log") or (log_file.read_text("utf-8").splitlines() if log_file.exists() else [])
    return meta | {"log": log[-200:], "queue_position": list(jobs.queue).index(run_id) + 1
                   if run_id in jobs.queue else 0}


@app.delete("/api/runs/{run_id}")
def delete_run(run_id: str):
    meta = read_meta(run_id)
    if meta.get("demo"):
        raise HTTPException(403, "demo runs cannot be deleted")
    if meta.get("status") in ("queued", "running"):
        raise HTTPException(409, "run is in progress")
    shutil.rmtree(RUNS_DIR / run_id)
    return {"deleted": run_id}


class Segment(BaseModel):
    start: float
    end: float


class RenderRequest(BaseModel):
    segments: list[Segment]


@app.post("/api/runs/{run_id}/render")
def render(run_id: str, req: RenderRequest):
    """Render the summary chosen in the UI (after re-weighting) as one MP4."""
    from svs.summarize import render_summary

    if read_meta(run_id).get("status") != "done":
        raise HTTPException(409, "run not finished")
    run_dir = RUNS_DIR / run_id
    duration = json.loads((run_dir / "result.json").read_text("utf-8"))["video"]["duration"]
    segs = [s.model_dump() for s in req.segments if 0 <= s.start < s.end <= duration + 0.05]
    if not segs or len(segs) > 500:
        raise HTTPException(400, "invalid segments")
    render_summary(run_dir / "video.mp4", segs, run_dir / "summary_custom.mp4")
    return {"url": f"/runs/{run_id}/summary_custom.mp4?t={int(time.time())}"}


@app.get("/api/benchmark")
def benchmark():
    path = BENCHMARK / "benchmark_summary.json"
    if not path.exists():
        raise HTTPException(404, "run training/train_importance.py and training/export_results.py first")
    return FileResponse(path, media_type="application/json")


app.mount("/runs", StaticFiles(directory=RUNS_DIR), name="runs")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


if __name__ == "__main__":
    import argparse

    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
