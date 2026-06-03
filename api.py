import asyncio
import json
import os
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from typing import AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

load_dotenv()

APP_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(APP_DIR, "output")
INPUT_DIR = os.path.join(APP_DIR, "input")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(INPUT_DIR, exist_ok=True)

app = FastAPI(title="AutoClipper API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@dataclass
class JobState:
    status: str = "pending"  # pending | running | done | error
    logs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    process: subprocess.Popen | None = None


jobs: dict[str, JobState] = {}


class RenderRequest(BaseModel):
    video_filename: str
    clips_json: dict
    channel_name: str
    source_credit: str


@app.get("/api/inputs")
def list_inputs():
    files = sorted([
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith((".mp4", ".mkv", ".mov", ".avi"))
    ])
    return {"files": files}


@app.get("/api/inputs/{filename}")
def serve_input(filename: str):
    path = os.path.join(INPUT_DIR, filename)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, media_type="video/mp4")


@app.get("/api/outputs")
def list_outputs():
    files = sorted([
        f for f in os.listdir(OUTPUT_DIR)
        if f.lower().endswith((".mp4", ".mkv", ".mov", ".avi"))
    ])
    return {"files": files}


@app.get("/api/outputs/{filename}")
def serve_output(filename: str):
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, media_type="video/mp4")


@app.post("/api/render")
def start_render(req: RenderRequest):
    video_path = os.path.join(INPUT_DIR, req.video_filename)
    if not os.path.isfile(video_path):
        raise HTTPException(status_code=400, detail=f"Video tidak ditemukan: {req.video_filename}")

    job_id = str(uuid.uuid4())
    job = JobState()
    jobs[job_id] = job

    tmp_json = os.path.join(OUTPUT_DIR, f"_input_{job_id}.json")
    with open(tmp_json, "w", encoding="utf-8") as f:
        json.dump(req.clips_json, f, ensure_ascii=False, indent=2)

    env = os.environ.copy()
    env["AUTOCLIPPER_JSON"] = tmp_json
    env["AUTOCLIPPER_VIDEO"] = video_path
    env["AUTOCLIPPER_OUTDIR"] = OUTPUT_DIR
    env["AUTOCLIPPER_CHANNEL"] = req.channel_name
    env["AUTOCLIPPER_SOURCE_CREDIT"] = req.source_credit
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    process = subprocess.Popen(
        [sys.executable, "-u", "script.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
        cwd=APP_DIR,
    )
    job.process = process
    job.status = "running"

    # Read output in background thread
    def _drain():
        assert process.stdout is not None
        for line in process.stdout:
            for part in line.rstrip("\n").split("\r"):
                job.logs.append(part)
        rc = process.wait()
        job.status = "done" if rc == 0 else "error"
        job.outputs = sorted([
            f for f in os.listdir(OUTPUT_DIR)
            if f.lower().endswith((".mp4", ".mkv", ".mov", ".avi"))
        ])
        if os.path.exists(tmp_json):
            os.remove(tmp_json)

    import threading
    threading.Thread(target=_drain, daemon=True).start()

    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": job_id,
        "status": job.status,
        "logs": job.logs,
        "outputs": job.outputs,
    }


@app.get("/api/jobs/{job_id}/stream")
async def stream_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    async def event_generator() -> AsyncGenerator[str, None]:
        sent = 0
        while True:
            current_logs = job.logs
            while sent < len(current_logs):
                line = current_logs[sent].replace("\n", " ")
                yield f"data: {json.dumps({'log': line, 'status': job.status})}\n\n"
                sent += 1

            if job.status in ("done", "error"):
                yield f"data: {json.dumps({'log': '', 'status': job.status, 'outputs': job.outputs, 'done': True})}\n\n"
                break

            await asyncio.sleep(0.3)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


class GenerateClipsRequest(BaseModel):
    youtube_url: str
    max_clips: int = 10
    min_duration: int = 20
    max_duration: int = 90
    language: str = "id"


@app.post("/api/generate-clips")
async def generate_clips(req: GenerateClipsRequest):
    from clip_generator import generate as run_pipeline

    try:
        clips_json = await run_pipeline(
            youtube_url=req.youtube_url,
            max_clips=req.max_clips,
            min_duration=req.min_duration,
            max_duration=req.max_duration,
            language=req.language,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    return {"clips_json": clips_json}
