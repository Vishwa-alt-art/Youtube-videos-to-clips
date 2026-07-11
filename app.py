#!/usr/bin/env python3
"""
app.py — Local web server for the free YouTube -> Short clipper.

Run it, open http://localhost:8000 in your browser, paste a YouTube link (or
upload a file), describe the moment you want, and download the finished clip.

Two-phase flow (so you never wait on a render you don't want):
    1. ANALYZE  — download + transcribe (cached) + find the best moments
    2. RENDER   — you pick a moment, it renders + captions + downloads

Everything runs locally on your machine. Nothing is uploaded to any paid API.
"""

from __future__ import annotations

import os
import threading
import traceback
import uuid
from dataclasses import asdict

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

import pipeline as P

HERE = os.path.dirname(os.path.abspath(__file__))
JOBS_DIR = os.path.join(HERE, "jobs")
STATIC_DIR = os.path.join(HERE, "static")
os.makedirs(JOBS_DIR, exist_ok=True)

app = FastAPI(title="Free YouTube Clipper")

# in-memory job registry
JOBS: dict[str, dict] = {}
_LOCK = threading.Lock()


def _new_job() -> str:
    jid = uuid.uuid4().hex[:12]
    with _LOCK:
        JOBS[jid] = {
            "id": jid,
            "phase": "queued",      # queued|analyzing|analyzed|rendering|done|error
            "progress": 0.0,
            "message": "Queued...",
            "log": [],
            "candidates": [],
            "words": [],
            "source": None,
            "output": None,
            "error": None,
        }
    os.makedirs(os.path.join(JOBS_DIR, jid), exist_ok=True)
    return jid


def _progress(jid: str):
    def fn(msg: str, pct: float | None = None):
        with _LOCK:
            job = JOBS.get(jid)
            if not job:
                return
            job["message"] = msg
            if pct is not None:
                job["progress"] = round(pct, 3)
            job["log"].append(msg)
            job["log"] = job["log"][-40:]
    return fn


def _set(jid: str, **kw):
    with _LOCK:
        if jid in JOBS:
            JOBS[jid].update(kw)


# --------------------------------------------------------------------------
# Phase 1: analyze
# --------------------------------------------------------------------------
def _analyze_worker(jid: str, url: str | None, file_path: str | None,
                    prompt: str, duration: float, model: str, top_k: int):
    prog = _progress(jid)
    try:
        _set(jid, phase="analyzing")
        job_dir = os.path.join(JOBS_DIR, jid)
        if file_path:
            source = file_path
        else:
            source = os.path.join(job_dir, "source.mp4")
            P.download_video(url, source, prog)
        _set(jid, source=source)

        words = P.transcribe(source, model_size=model, progress=prog)
        cands = P.find_candidates(words, duration, prompt=prompt,
                                  top_k=top_k, progress=prog)

        with _LOCK:
            JOBS[jid]["words"] = words
            JOBS[jid]["candidates"] = [c.to_dict() for c in cands]
            JOBS[jid]["_cands"] = cands
            JOBS[jid]["phase"] = "analyzed"
            JOBS[jid]["progress"] = 0.58
            JOBS[jid]["message"] = f"Found {len(cands)} moment(s). Pick one to render."
    except Exception as e:
        _set(jid, phase="error", error=f"{type(e).__name__}: {e}")
        prog(f"ERROR: {e}", None)
        traceback.print_exc()


@app.post("/analyze")
async def analyze(
    url: str = Form(""),
    prompt: str = Form(""),
    duration: float = Form(30.0),
    model: str = Form("small"),
    top_k: int = Form(5),
    file: UploadFile | None = File(None),
):
    if not url and file is None:
        raise HTTPException(400, "Provide a YouTube URL or upload a file.")
    jid = _new_job()
    file_path = None
    if file is not None:
        file_path = os.path.join(JOBS_DIR, jid, file.filename or "upload.mp4")
        with open(file_path, "wb") as f:
            f.write(await file.read())

    t = threading.Thread(
        target=_analyze_worker,
        args=(jid, url or None, file_path, prompt, duration, model, top_k),
        daemon=True,
    )
    t.start()
    return {"job_id": jid}


# --------------------------------------------------------------------------
# Phase 2: render a chosen candidate
# --------------------------------------------------------------------------
def _render_worker(jid: str, index: int, title: str, font: str,
                   zoom: bool, fade: bool):
    prog = _progress(jid)
    try:
        with _LOCK:
            job = JOBS[jid]
            words = job["words"]
            cands = job.get("_cands", [])
            source = job["source"]
        if not (0 <= index < len(cands)):
            raise ValueError("Invalid candidate index.")
        cand = cands[index]
        if title.strip():
            cand.title = title.strip()

        _set(jid, phase="rendering", progress=0.6, message="Rendering clip...")
        out_path = os.path.join(JOBS_DIR, jid, f"short_{index}.mp4")
        P.render_clip(source, words, cand, out_path, font=font,
                      zoom=zoom, fade=fade, progress=prog,
                      workdir=os.path.join(JOBS_DIR, jid, "work"))
        _set(jid, phase="done", output=out_path, progress=1.0,
             message="Clip ready to download!")
    except Exception as e:
        _set(jid, phase="error", error=f"{type(e).__name__}: {e}")
        prog(f"ERROR: {e}", None)
        traceback.print_exc()


@app.post("/render")
async def render(
    job_id: str = Form(...),
    index: int = Form(0),
    title: str = Form(""),
    font: str = Form(P.FONT_NAME),
    zoom: bool = Form(True),
    fade: bool = Form(True),
):
    with _LOCK:
        job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found.")
    if not job.get("_cands"):
        raise HTTPException(400, "Analysis not finished yet.")
    t = threading.Thread(
        target=_render_worker,
        args=(job_id, index, title, font, zoom, fade),
        daemon=True,
    )
    t.start()
    return {"job_id": job_id}


# --------------------------------------------------------------------------
# Status + download
# --------------------------------------------------------------------------
@app.get("/status/{job_id}")
async def status(job_id: str):
    with _LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(404, "Job not found.")
        return JSONResponse({
            "id": job["id"],
            "phase": job["phase"],
            "progress": job["progress"],
            "message": job["message"],
            "log": job["log"][-8:],
            "candidates": job["candidates"],
            "error": job["error"],
            "has_output": bool(job.get("output")),
        })


@app.get("/download/{job_id}")
async def download(job_id: str):
    with _LOCK:
        job = JOBS.get(job_id)
    if not job or not job.get("output") or not os.path.exists(job["output"]):
        raise HTTPException(404, "No finished clip for this job.")
    return FileResponse(job["output"], media_type="video/mp4",
                        filename="short.mp4")


@app.get("/", response_class=HTMLResponse)
async def index():
    with open(os.path.join(STATIC_DIR, "index.html"), encoding="utf-8") as f:
        return HTMLResponse(f.read())


# serve the finished mp4 for inline <video> preview
app.mount("/files", StaticFiles(directory=JOBS_DIR), name="files")


if __name__ == "__main__":
    import uvicorn
    print("\n  Free YouTube Clipper running at:  http://localhost:8000\n")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
