"""FastAPI server for Latent Space Music Mimic."""
import os
import sys
import uuid
import threading
import traceback
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pipeline import run

app = FastAPI(title="Latent Space Music Mimic", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# In-memory job store (replace with Redis for production)
jobs: dict[str, dict] = {}


class JobStatus(BaseModel):
    job_id: str
    status: str   # queued | processing | done | failed
    progress: float
    stage: str
    error: Optional[str] = None
    output_url: Optional[str] = None


def _run_job(job_id: str, input_path: str, output_path: str,
             duration: float, seed: int):
    """Background thread worker."""
    jobs[job_id]["status"] = "processing"
    jobs[job_id]["stage"] = "ingesting"
    jobs[job_id]["progress"] = 0.05

    try:
        # Monkey-patch log to update progress
        import pipeline as pl_module
        original_run = pl_module.run

        stages = [
            ("ingesting", 0.05),
            ("separating stems", 0.15),
            ("extracting features", 0.30),
            ("generating audio", 0.50),
            ("mixing", 0.90),
            ("saving", 0.98),
        ]
        stage_iter = iter(stages)

        class ProgressRun:
            def __call__(self, *args, **kwargs):
                # Just call the real run with a progress callback via prints
                return original_run(*args, **kwargs)

        pl_module.run(
            input_path=input_path,
            output_path=output_path,
            target_duration=duration,
            seed=seed,
            verbose=True,
        )

        jobs[job_id]["status"] = "done"
        jobs[job_id]["progress"] = 1.0
        jobs[job_id]["stage"] = "complete"
        jobs[job_id]["output_url"] = f"/result/{job_id}"

    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = traceback.format_exc()
        jobs[job_id]["stage"] = "error"


@app.post("/generate")
async def generate(
    file: UploadFile = File(...),
    duration: float = Form(300.0),
    seed: int = Form(42),
):
    """Upload a 30s audio clip → start generation job."""
    # Validate file type
    allowed_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_exts:
        raise HTTPException(400, f"Unsupported file type: {ext}")

    if duration < 10 or duration > 600:
        raise HTTPException(400, "Duration must be between 10 and 600 seconds")

    # Save upload
    job_id = str(uuid.uuid4())[:8]
    input_path = str(UPLOAD_DIR / f"{job_id}_input{ext}")
    output_path = str(OUTPUT_DIR / f"{job_id}_output.wav")

    content = await file.read()
    with open(input_path, "wb") as f:
        f.write(content)

    # Register job
    jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0.0,
        "stage": "queued",
        "error": None,
        "output_url": None,
        "input_path": input_path,
        "output_path": output_path,
    }

    # Start background thread
    t = threading.Thread(
        target=_run_job,
        args=(job_id, input_path, output_path, duration, seed),
        daemon=True,
    )
    t.start()

    return JSONResponse({
        "job_id": job_id,
        "status": "queued",
        "poll_url": f"/job/{job_id}/status",
        "estimated_time_seconds": int(duration / 10),
    })


@app.get("/job/{job_id}/status", response_model=JobStatus)
def job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    j = jobs[job_id]
    return JobStatus(
        job_id=job_id,
        status=j["status"],
        progress=j["progress"],
        stage=j["stage"],
        error=j.get("error"),
        output_url=j.get("output_url"),
    )


@app.get("/result/{job_id}")
def get_result(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    j = jobs[job_id]
    if j["status"] != "done":
        raise HTTPException(400, f"Job is not done yet: {j['status']}")
    if not os.path.exists(j["output_path"]):
        raise HTTPException(500, "Output file missing")
    return FileResponse(
        j["output_path"],
        media_type="audio/wav",
        filename=f"mimic_{job_id}.wav",
    )


@app.get("/health")
def health():
    return {"status": "ok", "jobs_in_memory": len(jobs)}


@app.get("/")
def root():
    return {"message": "Latent Space Music Mimic API", "docs": "/docs"}
