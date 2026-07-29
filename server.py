import shutil
import os
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import json
import logging

# Import existing pipeline stages
# We assume main.py is in the same directory and has these functions importable
from main import (
    stage_1_shot_extraction,
    stage_2_dialogue_embedding,
    stage_3_scene_segmentation,
    stage_4_scene_merging,
    stage_5_scene_to_timestamps,
    stage_5_5_build_sentence_segments,
    stage_6_scene_refinement,
    stage_7_clip_extraction,
    stage_8_importance_scoring,
    stage_9_score_fusion,
    stage_10_visualization,
)

import uuid

app = FastAPI()

# Setup directories
WORK_DIR = Path(".")
SESSIONS_DIR = WORK_DIR / "sessions"
SESSIONS_DIR.mkdir(exist_ok=True)

# Mount static files
app.mount("/preprocessing", StaticFiles(directory="preprocessing"), name="preprocessing")
app.mount("/importance_scoring", StaticFiles(directory="importance_scoring"), name="importance_scoring")
app.mount("/sessions", StaticFiles(directory="sessions"), name="sessions")

@app.get("/")
async def read_index():
    return FileResponse('index.html')

@app.get("/style.css")
async def read_css():
    return FileResponse('style.css', media_type='text/css')


import sys
import io
import logging

# Global Log Store: {session_id: [lines]}
LOG_STORE = {}

class SessionLogger:
    def __init__(self, session_id):
        self.session_id = session_id
        self.terminal = sys.stdout

    def write(self, message):
        self.terminal.write(message)
        if self.session_id and message.strip():
            if self.session_id not in LOG_STORE:
                LOG_STORE[self.session_id] = []
            LOG_STORE[self.session_id].append(message)

    def flush(self):
        self.terminal.flush()

@app.get("/logs/{session_id}")
async def get_logs(session_id: str):
    return {"logs": LOG_STORE.get(session_id, [])}

@app.post("/update_clip")
async def update_clip(data: dict):
    # data: {session_id, clip_name, start, end}
    # For now, just a mock to show updating works. 
    # Real implementation would call ffmpeg to re-trim.
    session_id = data.get("session_id")
    clip_name = data.get("clip_name")
    print(f"Request to update clip {clip_name} in session {session_id} to {data.get('start')}-{data.get('end')}")
    return {"status": "updated", "url": f"sessions/{session_id}/importance_scoring/clips/{clip_name}"}

from fastapi import BackgroundTasks
import traceback

# Global Status Store: {session_id: {"status": "processing"|"done"|"error", "data": ...}}
STATUS_STORE = {}

def pipeline_wrapper(video_path, work_dir, session_id):
    """Runs the pipeline and updates global status."""
    logger = SessionLogger(session_id)
    original_stdout = sys.stdout
    sys.stdout = logger
    
    print(f"Starting analysis for session {session_id}...")
    STATUS_STORE[session_id] = {"status": "processing"}

    try:
        # Run stages
        stage_1_shot_extraction(video_path, work_dir)
        stage_2_dialogue_embedding(video_path, work_dir)
        stage_3_scene_segmentation(video_path, work_dir)
        stage_4_scene_merging(video_path, work_dir)
        stage_5_scene_to_timestamps(video_path, work_dir)
        stage_5_5_build_sentence_segments(video_path, work_dir)
        stage_6_scene_refinement(video_path, work_dir)
        stage_7_clip_extraction(video_path, work_dir)
        stage_8_importance_scoring(video_path, work_dir)
        stage_9_score_fusion(video_path, work_dir)
        stage_10_visualization(video_path, work_dir)
        
        # Read results
        fused_scores_path = work_dir / "importance_scoring" / "fused_scores.json"
        if not fused_scores_path.exists():
             raise Exception("fused_scores.json not found after pipeline completion")

        with open(fused_scores_path, "r", encoding="utf-8") as f:
            scores_data = json.load(f)
            scores_data["session_id"] = session_id

        STATUS_STORE[session_id] = {"status": "done", "result": scores_data}
        print("Pipeline finished successfully.")

    except Exception as e:
        err_msg = str(e)
        traceback.print_exc()
        STATUS_STORE[session_id] = {"status": "error", "message": err_msg}
        print(f"Pipeline failed: {err_msg}")
    finally:
        sys.stdout = original_stdout


@app.post("/process")
async def process_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    try:
        # 1. Setup Session
        session_id = str(uuid.uuid4())
        session_dir = SESSIONS_DIR / session_id
        
        LOG_STORE[session_id] = []
        
        # Create directories
        (session_dir / "preprocessing" / "input_video").mkdir(parents=True, exist_ok=True)
        (session_dir / "preprocessing" / "TransNetV2" / "lgss_input").mkdir(parents=True, exist_ok=True)
        (session_dir / "synthesis").mkdir(parents=True, exist_ok=True)
        (session_dir / "scene_segmentation" / "SceneSeg").mkdir(parents=True, exist_ok=True)
        (session_dir / "refine_scenes").mkdir(parents=True, exist_ok=True)
        (session_dir / "importance_scoring" / "clips").mkdir(parents=True, exist_ok=True)

        # 2. Save File
        input_video_path = session_dir / "preprocessing" / "input_video" / "input.mp4"
        with open(input_video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 3. Launch Background Task
        video_path = input_video_path.resolve()
        work_dir = session_dir.resolve()
        
        background_tasks.add_task(pipeline_wrapper, video_path, work_dir, session_id)
        
        return {"session_id": session_id, "status": "started"}

    except Exception as e:
        print(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/{session_id}")
async def get_status(session_id: str):
    return STATUS_STORE.get(session_id, {"status": "unknown"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
