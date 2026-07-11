#!/usr/bin/env python3
"""
pipeline.py — Core processing for the free, local YouTube -> Short clipper.

Everything here runs on YOUR machine, for free:
  - yt-dlp            : download the source video
  - faster-whisper    : transcribe with word-level timestamps (4x faster than
                        openai-whisper, lower memory)
  - Ollama (optional) : let a local LLM pick the best viral moments + titles
  - sentence-transformers (optional) : semantic prompt matching fallback
  - OpenCV / MediaPipe (optional)    : face-aware 9:16 reframing
  - FFmpeg            : trim, reframe, fades, Ken Burns zoom, burn captions

Every "optional" piece degrades gracefully: if it isn't installed, the
pipeline automatically falls back to a simpler free method and keeps working.

This module has NO web code — app.py drives it. A `progress` callback lets
the web layer stream status to the browser.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional

# --------------------------------------------------------------------------
# Look & feel config — tweak captions here
# --------------------------------------------------------------------------
FONT_NAME = "Arial Black"        # any bold font installed on your system
CAPTION_FONT_SIZE = 76
TITLE_FONT_SIZE = 64
HIGHLIGHT_COLOR = "&H0000FFFF"   # ASS BGR = yellow (currently-spoken word)
WHITE_COLOR = "&H00FFFFFF"
BLACK_OUTLINE = "&H00000000"
WORDS_PER_CAPTION_GROUP = 3

OUT_W, OUT_H = 1080, 1920        # 9:16 vertical target

HYPE_WORDS = [
    "never", "insane", "crazy", "million", "wow", "unbelievable", "biggest",
    "record", "win", "won", "lost", "shocking", "secret", "finally", "best",
    "worst", "first", "last", "die", "died", "kill", "money", "free", "give",
    "giveaway", "impossible", "extreme", "huge", "massive", "incredible",
    "amazing", "no one", "everyone", "everything", "literally", "actually",
]

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1")


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
ProgressFn = Callable[[str, Optional[float]], None]


def _noop(msg: str, pct: Optional[float] = None) -> None:
    print(msg)


def _run(cmd: list, timeout: Optional[int] = None) -> subprocess.CompletedProcess:
    """Run a command, raising a clean error with stderr on failure."""
    proc = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, timeout=timeout,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or "")[-1500:]
        raise RuntimeError(f"Command failed: {' '.join(cmd[:3])} ...\n{tail}")
    return proc


def ffprobe_duration(path: str) -> float:
    """Actual media duration in seconds (used to fix fade/zoom timing bugs)."""
    proc = _run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", path,
    ])
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return 0.0


def ffprobe_dimensions(path: str) -> tuple[int, int]:
    proc = _run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=s=x:p=0", path,
    ])
    w, h = proc.stdout.strip().split("x")
    return int(w), int(h)


def file_sha1(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# --------------------------------------------------------------------------
# Data types
# --------------------------------------------------------------------------
@dataclass
class Word:
    word: str
    start: float
    end: float


@dataclass
class Candidate:
    start: float
    end: float
    title: str
    reason: str
    score: float
    transcript: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["duration"] = round(self.end - self.start, 1)
        d["start"] = round(self.start, 2)
        d["end"] = round(self.end, 2)
        d["score"] = round(self.score, 2)
        return d


# --------------------------------------------------------------------------
# Step 1: download
# --------------------------------------------------------------------------
def download_video(url: str, out_path: str, progress: ProgressFn = _noop) -> str:
    progress(f"Downloading video from {url} ...", 0.05)
    _run([
        "yt-dlp",
        "-f", "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[ext=mp4]/b",
        "--no-playlist",
        "-o", out_path,
        url,
    ])
    if not os.path.exists(out_path):
        # yt-dlp may append an extension; find the produced file
        base = os.path.splitext(out_path)[0]
        for ext in (".mp4", ".mkv", ".webm"):
            if os.path.exists(base + ext):
                shutil.move(base + ext, out_path)
                break
    if not os.path.exists(out_path):
        raise RuntimeError("Download finished but no video file was produced.")
    return out_path


# --------------------------------------------------------------------------
# Step 2: transcribe (faster-whisper) with on-disk cache
# --------------------------------------------------------------------------
def transcribe(video_path: str, model_size: str = "small",
               progress: ProgressFn = _noop) -> list[Word]:
    """
    Transcribe with word timestamps. Result is cached by (file hash + model)
    so re-running with a different prompt on the same video is instant.
    """
    digest = file_sha1(video_path)
    cache_path = os.path.join(CACHE_DIR, f"{digest}_{model_size}.json")
    if os.path.exists(cache_path):
        progress("Loading cached transcript (skipping transcription)...", 0.35)
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [Word(**w) for w in data]

    progress(f"Transcribing with faster-whisper ({model_size}). "
             f"First run downloads the model...", 0.15)
    from faster_whisper import WhisperModel  # imported lazily

    # int8 keeps CPU memory low; "auto" device uses GPU if CUDA is available.
    model = WhisperModel(model_size, device="auto", compute_type="int8")
    segments, info = model.transcribe(
        video_path, word_timestamps=True, vad_filter=True,
    )

    words: list[Word] = []
    total = max(getattr(info, "duration", 0.0), 0.001)
    for seg in segments:
        if seg.words:
            for w in seg.words:
                token = w.word.strip()
                if token:
                    words.append(Word(word=token, start=float(w.start),
                                      end=float(w.end)))
        else:  # safety net if a segment lacks word timings
            words.append(Word(word=seg.text.strip(), start=float(seg.start),
                              end=float(seg.end)))
        progress(f"Transcribing... {seg.end/total*100:4.0f}%",
                 0.15 + 0.20 * min(seg.end / total, 1.0))

    if not words:
        raise RuntimeError("Whisper produced no words — is the audio silent?")

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump([asdict(w) for w in words], f)
    return words


# --------------------------------------------------------------------------
# Step 3: find the best candidate moments
# --------------------------------------------------------------------------
def _sentences(words: list[Word]) -> list[tuple[int, int]]:
    """Group word indices into (start_idx, end_idx_exclusive) sentence spans."""
    spans, start = [], 0
    for i, w in enumerate(words):
        if re.search(r"[.!?]$", w.word):
            spans.append((start, i + 1))
            start = i + 1
    if start < len(words):
        spans.append((start, len(words)))
    return spans


def _window_text(words: list[Word], a: float, b: float) -> str:
    return " ".join(w.word for w in words if a <= w.start < b).strip()


def _hype_score(words: list[Word], a: float, b: float) -> float:
    score = 0.0
    for w in words:
        if not (a <= w.start < b):
            continue
        token = w.word
        clean = token.lower().strip(".,!?\"'")
        if clean in HYPE_WORDS:
            score += 3.0
        if token.isupper() and len(token) > 1:
            score += 1.0
        if "!" in token or "?" in token:
            score += 1.0
        if re.search(r"\d", token):
            score += 1.0
    return score


def _keyword_overlap(prompt: str, text: str) -> float:
    stop = {"the", "a", "an", "of", "to", "in", "on", "and", "or", "is", "are",
            "was", "were", "clip", "find", "me", "give", "show", "about",
            "where", "that", "this", "for", "part", "moment", "he", "she",
            "they", "it", "his", "her"}
    pw = {w.lower().strip(".,!?\"'") for w in prompt.split()} - stop
    if not pw:
        return 0.0
    tw = {w.lower().strip(".,!?\"'") for w in text.split()}
    return len(pw & tw) * 2.0


def _candidate_windows(words: list[Word], duration: float) -> list[tuple[float, float]]:
    """Every window that starts on a sentence boundary and snaps its end to a
    sentence boundary near start+duration (so clips never cut mid-thought)."""
    spans = _sentences(words)
    total_end = words[-1].end
    out = []
    sentence_end_times = [words[e - 1].end for (_, e) in spans]
    for (s_idx, _) in spans:
        start_t = words[s_idx].start
        if start_t + duration > total_end + 2.0:
            continue
        target = start_t + duration
        ends = [t for t in sentence_end_times if start_t < t <= target + 3.0]
        end_t = max(ends) if ends else min(target, total_end)
        if end_t - start_t < max(3.0, duration * 0.4):
            end_t = min(start_t + duration, total_end)
        out.append((start_t, end_t))
    # de-dup near-identical windows
    uniq, seen = [], set()
    for a, b in out:
        key = (round(a, 1), round(b, 1))
        if key not in seen:
            seen.add(key)
            uniq.append((a, b))
    return uniq


def _ollama_pick(words: list[Word], prompt: str, duration: float,
                 top_k: int, progress: ProgressFn) -> Optional[list[Candidate]]:
    """Ask a local Ollama LLM to choose the best moments. Returns None if
    Ollama isn't reachable or the response can't be parsed."""
    try:
        import urllib.request
        # Build a compact timestamped transcript from sentence spans.
        spans = _sentences(words)
        lines = []
        for (s, e) in spans:
            t = words[s].start
            txt = " ".join(w.word for w in words[s:e]).strip()
            if txt:
                lines.append(f"[{t:.1f}] {txt}")
        transcript = "\n".join(lines)
        if len(transcript) > 14000:  # too long for a small local model
            return None

        instruction = (
            "You are a short-form video editor. From the timestamped "
            f"transcript, choose the {top_k} best ~{int(duration)}-second moments "
            f"to clip as a viral vertical Short. The user is looking for: "
            f"\"{prompt or 'the single most engaging, attention-grabbing moment'}\". "
            "Prefer self-contained moments with a hook. "
            "Reply ONLY with a JSON array, each item: "
            '{\"start\": <seconds>, \"end\": <seconds>, \"title\": '
            '\"<punchy ALL-CAPS title, <=6 words>\", \"reason\": \"<why>\"}. '
            "start/end must be real timestamps from the transcript.\n\n"
            f"TRANSCRIPT:\n{transcript}"
        )
        payload = json.dumps({
            "model": OLLAMA_MODEL,
            "prompt": instruction,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.4},
        }).encode()
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/generate", data=payload,
            headers={"Content-Type": "application/json"},
        )
        progress("Asking local LLM (Ollama) to pick the best moments...", 0.45)
        with urllib.request.urlopen(req, timeout=180) as resp:
            body = json.loads(resp.read().decode())
        raw = body.get("response", "").strip()
        # response is JSON (may be an object wrapping the array)
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            for v in parsed.values():
                if isinstance(v, list):
                    parsed = v
                    break
        if not isinstance(parsed, list):
            return None

        total_end = words[-1].end
        cands = []
        for item in parsed[:top_k]:
            try:
                a = max(0.0, float(item["start"]))
                b = float(item["end"])
            except (KeyError, TypeError, ValueError):
                continue
            if b <= a:
                b = a + duration
            b = min(b, total_end)
            cands.append(Candidate(
                start=a, end=b,
                title=str(item.get("title", "")).strip()[:60] or "MUST WATCH",
                reason=str(item.get("reason", "")).strip()[:200],
                score=9.0,  # LLM-chosen; ranked by order
                transcript=_window_text(words, a, b),
            ))
        # rank by order returned
        for i, c in enumerate(cands):
            c.score = round(10.0 - i * 0.5, 2)
        return cands or None
    except Exception as e:  # any failure -> graceful fallback
        progress(f"Ollama not used ({type(e).__name__}); using free scoring.", 0.45)
        return None


def _semantic_scorer(prompt: str):
    """Return a fn(text)->similarity in [0,1], or None if unavailable."""
    try:
        from sentence_transformers import SentenceTransformer, util
        model = SentenceTransformer("all-MiniLM-L6-v2")
        pe = model.encode(prompt, convert_to_tensor=True)

        def score(text: str) -> float:
            if not text.strip():
                return 0.0
            emb = model.encode(text, convert_to_tensor=True)
            return float(util.cos_sim(pe, emb)[0][0])
        return score
    except Exception:
        return None


def find_candidates(words: list[Word], duration: float, prompt: str = "",
                    top_k: int = 5, progress: ProgressFn = _noop) -> list[Candidate]:
    """
    Return up to `top_k` ranked Candidate moments.

    Selection strategy (best available wins, all free):
      1. Ollama local LLM  — reads the transcript, picks moments + titles
      2. sentence-transformers — semantic similarity to the prompt
      3. keyword overlap + hype score — zero-dependency fallback
    """
    progress("Finding the best moments...", 0.40)

    # 1) LLM
    llm = _ollama_pick(words, prompt, duration, top_k, progress)
    if llm:
        progress(f"LLM chose {len(llm)} candidate moment(s).", 0.55)
        return llm

    # 2/3) score every sentence-boundary window
    windows = _candidate_windows(words, duration)
    if not windows:
        # fall back to a single window from the start
        end = min(words[0].start + duration, words[-1].end)
        return [Candidate(words[0].start, end, "MUST WATCH", "Opening moment",
                          1.0, _window_text(words, words[0].start, end))]

    sem = _semantic_scorer(prompt) if prompt else None
    if sem:
        progress("Scoring moments with semantic matching...", 0.5)
    else:
        progress("Scoring moments with keyword + hype matching (free fallback)...", 0.5)

    scored = []
    for (a, b) in windows:
        text = _window_text(words, a, b)
        s = _hype_score(words, a, b)
        if prompt:
            s += (sem(text) * 10.0) if sem else _keyword_overlap(prompt, text)
        scored.append((s, a, b, text))

    scored.sort(key=lambda x: x[0], reverse=True)

    # Keep top_k, but avoid heavy overlap between chosen clips.
    chosen: list[Candidate] = []
    for s, a, b, text in scored:
        if any(not (b <= c.start or a >= c.end) and
               (min(b, c.end) - max(a, c.start)) > 0.5 * (b - a)
               for c in chosen):
            continue
        title = _auto_title(text)
        chosen.append(Candidate(a, b, title, "High engagement score",
                                round(s, 2), text))
        if len(chosen) >= top_k:
            break
    return chosen


def _auto_title(text: str) -> str:
    """Cheap title when no LLM: first hype word / number phrase, else opener."""
    words = text.split()
    for i, w in enumerate(words):
        if w.lower().strip(".,!?\"'") in HYPE_WORDS or re.search(r"\d", w):
            snippet = " ".join(words[max(0, i - 1):i + 4])
            return snippet.upper().strip(".,!?\"'")[:45] or "MUST WATCH"
    return " ".join(words[:6]).upper().strip(".,!?\"'")[:45] or "MUST WATCH"


# --------------------------------------------------------------------------
# Step 4: trim
# --------------------------------------------------------------------------
def trim(source_path: str, start: float, end: float, out_path: str,
         progress: ProgressFn = _noop) -> str:
    progress(f"Trimming {start:.1f}s → {end:.1f}s ...", 0.60)
    dur = max(end - start, 0.1)
    _run([
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}", "-i", source_path, "-t", f"{dur:.3f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k",
        out_path,
    ])
    return out_path


# --------------------------------------------------------------------------
# Step 4b: face-aware 9:16 reframe
# --------------------------------------------------------------------------
def _detect_face_centers(video_path: str, scaled_w: int, scaled_h: int,
                         fps: float = 2.0) -> list[tuple[float, float]]:
    """
    Sample frames (scaled to output height) and return [(t, center_x), ...]
    for the dominant face. Tries MediaPipe, then OpenCV Haar cascade.
    Returns [] if neither is available or no faces are found.
    """
    tmpdir = tempfile.mkdtemp(prefix="frames_")
    try:
        _run([
            "ffmpeg", "-y", "-i", video_path,
            "-vf", f"scale={scaled_w}:{scaled_h},fps={fps}",
            os.path.join(tmpdir, "f_%05d.jpg"),
        ])
        frames = sorted(f for f in os.listdir(tmpdir) if f.endswith(".jpg"))
        if not frames:
            return []

        try:
            import cv2
        except ImportError:
            return []

        # Prefer MediaPipe (more accurate), fall back to Haar cascade.
        detector = None
        mp_fd = None
        try:
            import mediapipe as mp
            mp_fd = mp.solutions.face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.5)
        except Exception:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            detector = cv2.CascadeClassifier(cascade_path)

        centers = []
        for i, fname in enumerate(frames):
            t = i / fps
            img = cv2.imread(os.path.join(tmpdir, fname))
            if img is None:
                continue
            cx = None
            if mp_fd is not None:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                res = mp_fd.process(rgb)
                if res.detections:
                    best = max(res.detections,
                               key=lambda d: d.location_data.relative_bounding_box.width)
                    box = best.location_data.relative_bounding_box
                    cx = (box.xmin + box.width / 2) * scaled_w
            else:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
                if len(faces):
                    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                    cx = x + w / 2
            if cx is not None:
                centers.append((t, float(cx)))
        if mp_fd is not None:
            mp_fd.close()
        return centers
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _smooth(values: list[float], window: int = 5) -> list[float]:
    if not values:
        return values
    out = []
    for i in range(len(values)):
        lo, hi = max(0, i - window // 2), min(len(values), i + window // 2 + 1)
        out.append(sum(values[lo:hi]) / (hi - lo))
    return out


def _build_x_expr(centers: list[tuple[float, float]], scaled_w: int,
                  max_x: float) -> str:
    """Build a piecewise-linear ffmpeg crop-x expression that tracks the face
    center over time. Falls back to a constant if too few points."""
    # clamp each target so the crop window stays inside the frame
    pts = []
    xs = _smooth([c[1] for c in centers])
    for (t, _), x in zip(centers, xs):
        target = min(max(x - OUT_W / 2, 0.0), max_x)
        pts.append((t, target))
    if len(pts) < 2:
        val = pts[0][1] if pts else max_x / 2
        return f"{val:.1f}"
    # nested if(): between t0..t1 interpolate, etc.
    expr = f"{pts[-1][1]:.1f}"  # default = last value (t beyond samples)
    for i in range(len(pts) - 1, 0, -1):
        t0, x0 = pts[i - 1]
        t1, x1 = pts[i]
        dt = max(t1 - t0, 1e-3)
        seg = f"({x0:.1f}+({x1 - x0:.1f})*(t-{t0:.3f})/{dt:.3f})"
        expr = f"if(lt(t,{t1:.3f}),{seg},{expr})"
    # before first sample point -> first value
    expr = f"if(lt(t,{pts[0][0]:.3f}),{pts[0][1]:.1f},{expr})"
    return expr


def reframe_vertical(input_path: str, out_path: str,
                     progress: ProgressFn = _noop) -> str:
    """Reframe to 1080x1920, keeping the speaker's face in shot when possible."""
    progress("Reframing to 9:16 (tracking the subject)...", 0.66)
    w, h = ffprobe_dimensions(input_path)

    # scale so height == 1920, preserving aspect
    scaled_h = OUT_H
    scaled_w = round(w * (OUT_H / h))
    scaled_w -= scaled_w % 2

    if scaled_w <= OUT_W:
        # Source is already vertical/narrow: fit width, blurred-fill the sides.
        vf = (
            f"scale={OUT_W}:-2,setsar=1,"
            f"split[main][bg];"
            f"[bg]scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=increase,"
            f"crop={OUT_W}:{OUT_H},gblur=sigma=25[bg2];"
            f"[bg2][main]overlay=(W-w)/2:(H-h)/2"
        )
        _run([
            "ffmpeg", "-y", "-i", input_path, "-vf", vf,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-c:a", "copy", out_path,
        ])
        return out_path

    max_x = float(scaled_w - OUT_W)
    centers = _detect_face_centers(input_path, scaled_w, scaled_h)
    if centers:
        progress(f"Tracking subject across {len(centers)} sampled frames...", 0.70)
        x_expr = _build_x_expr(centers, scaled_w, max_x)
    else:
        progress("No face detected — centering the crop.", 0.70)
        x_expr = f"{max_x/2:.1f}"

    vf = f"scale={scaled_w}:{scaled_h},crop={OUT_W}:{OUT_H}:x='{x_expr}':y=0"
    _run([
        "ffmpeg", "-y", "-i", input_path, "-vf", vf,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-c:a", "copy", out_path,
    ])
    return out_path


# --------------------------------------------------------------------------
# Step 4c: classic edits (fade + subtle zoom) — timing from REAL duration
# --------------------------------------------------------------------------
def apply_classic_edits(input_path: str, out_path: str, zoom: bool = True,
                        fade: bool = True, progress: ProgressFn = _noop) -> str:
    progress("Applying classic edits (fade + subtle zoom)...", 0.78)
    # BUG FIX: use the ACTUAL clip duration, not the requested one, so the
    # fade-out lands in the right place after sentence-snapping.
    dur = ffprobe_duration(input_path)
    if dur <= 0:
        dur = 1.0

    filters = []
    if zoom:
        filters.append(f"scale={int(OUT_W*1.1)}:{int(OUT_H*1.1)}")
        filters.append(
            f"crop={OUT_W}:{OUT_H}:"
            f"x='(in_w-{OUT_W})/2*(t/{dur:.3f})':"
            f"y='(in_h-{OUT_H})/2*(t/{dur:.3f})'"
        )
    if fade:
        fo = max(dur - 0.4, 0.0)
        filters.append(f"fade=t=in:st=0:d=0.3,fade=t=out:st={fo:.2f}:d=0.4")

    vf = ",".join(filters) if filters else "null"
    if fade:
        fo = max(dur - 0.4, 0.0)
        af = f"afade=t=in:st=0:d=0.3,afade=t=out:st={fo:.2f}:d=0.4"
    else:
        af = "anull"

    _run([
        "ffmpeg", "-y", "-i", input_path,
        "-vf", vf, "-af", af,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k",
        out_path,
    ], timeout=600)
    return out_path


# --------------------------------------------------------------------------
# Step 5: build ASS captions  (BUG FIX: no mutation of shared word list)
# --------------------------------------------------------------------------
def _sec_to_ass(t: float) -> str:
    t = max(t, 0.0)
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h:d}:{m:02d}:{s:05.2f}"


def build_ass(words: list[Word], clip_start: float, clip_end: float,
              title: str, ass_path: str, font: str = FONT_NAME) -> str:
    clip_len = clip_end - clip_start
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {OUT_W}
PlayResY: {OUT_H}
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,{font},{CAPTION_FONT_SIZE},{WHITE_COLOR},{WHITE_COLOR},{BLACK_OUTLINE},&H80000000,-1,0,0,0,100,100,0,0,1,5,2,2,60,60,300,1
Style: Title,{font},{TITLE_FONT_SIZE},&H00000000,&H00000000,&H00FFFFFF,&H00FFFFFF,-1,0,0,0,100,100,0,0,3,0,0,8,60,60,140,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = [header]

    # Title box across the whole clip
    if title:
        tw = title.upper().split()
        mid = math.ceil(len(tw) / 2)
        l1, l2 = " ".join(tw[:mid]), " ".join(tw[mid:])
        ttext = r"\N".join(x for x in (l1, l2) if x)
        lines.append(
            f"Dialogue: 0,{_sec_to_ass(0)},{_sec_to_ass(clip_len)},"
            f"Title,,0,0,0,,{{\\bord0\\shad0}}{ttext}\n"
        )

    # Bottom word-highlight captions — work on LOCAL copies (bug fix: never
    # mutate the caller's Word objects / shared list).
    local = [Word(w.word, w.start - clip_start, w.end - clip_start)
             for w in words if clip_start <= w.start < clip_end]

    for i in range(0, len(local), WORDS_PER_CAPTION_GROUP):
        group = local[i:i + WORDS_PER_CAPTION_GROUP]
        if not group:
            continue
        for active_idx, active in enumerate(group):
            seg_start = active.start
            # hold each word until the next word starts (no blank flashes)
            if active_idx + 1 < len(group):
                seg_end = group[active_idx + 1].start
            else:
                seg_end = active.end
            seg_end = max(seg_end, seg_start + 0.05)
            parts = []
            for j, w in enumerate(group):
                clean = w.word.upper().replace("{", "").replace("}", "")
                if j == active_idx:
                    parts.append(f"{{\\c{HIGHLIGHT_COLOR}}}{clean}{{\\c{WHITE_COLOR}}}")
                else:
                    parts.append(clean)
            text = " ".join(parts)
            lines.append(
                f"Dialogue: 1,{_sec_to_ass(seg_start)},{_sec_to_ass(seg_end)},"
                f"Caption,,0,0,0,,{text}\n"
            )

    with open(ass_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    return ass_path


# --------------------------------------------------------------------------
# Step 6: burn captions
# --------------------------------------------------------------------------
def burn_captions(video_path: str, ass_path: str, out_path: str,
                  progress: ProgressFn = _noop) -> str:
    progress("Rendering final captioned video...", 0.90)
    # escape path for the ass filter (Windows backslashes / colons)
    ass_arg = ass_path.replace("\\", "/").replace(":", "\\:")
    _run([
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"ass='{ass_arg}'",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-c:a", "copy",
        out_path,
    ], timeout=600)
    return out_path


# --------------------------------------------------------------------------
# Orchestration used by the web app
# --------------------------------------------------------------------------
def render_clip(source_path: str, words: list[Word], candidate: Candidate,
                out_path: str, font: str = FONT_NAME, zoom: bool = True,
                fade: bool = True, progress: ProgressFn = _noop,
                workdir: Optional[str] = None) -> str:
    """Render ONE finished clip from an already-transcribed source."""
    workdir = workdir or tempfile.mkdtemp(prefix="clip_")
    os.makedirs(workdir, exist_ok=True)
    trimmed = os.path.join(workdir, "trimmed.mp4")
    vertical = os.path.join(workdir, "vertical.mp4")
    polished = os.path.join(workdir, "polished.mp4")
    ass_path = os.path.join(workdir, "captions.ass")

    trim(source_path, candidate.start, candidate.end, trimmed, progress)
    reframe_vertical(trimmed, vertical, progress)
    if zoom or fade:
        apply_classic_edits(vertical, polished, zoom=zoom, fade=fade,
                            progress=progress)
    else:
        shutil.copy(vertical, polished)
    build_ass(words, candidate.start, candidate.end,
              candidate.title, ass_path, font=font)
    burn_captions(polished, ass_path, out_path, progress)
    progress("Done!", 1.0)
    return out_path
