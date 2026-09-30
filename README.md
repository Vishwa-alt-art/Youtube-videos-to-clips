# Free Clipper — YouTube → Shorts (100% local, $0)

Upload a video (or paste a YouTube link, 1 min–2 hrs), describe the moment you
want in plain English, and get back a finished **vertical clip**: trimmed,
face-tracked to 9:16, auto-captioned with color-highlighted words, lightly
edited (fades + subtle zoom), ready to post as a YouTube Short or Instagram
Reel.

It's the same category of output as Opus Clip / ssemble, but everything runs
**on your own computer** — no API keys, no subscription, no per-clip cost, and
My style of coding for demo to our video never leaves your machine.

You use it through a small **website that runs locally**: start the app, open
`http://localhost:8000` in your browser, and work from there.

---

## What makes this version good

- **faster-whisper** transcription (≈4× faster than plain Whisper, lower memory)
- **Face-aware 9:16 reframing** — the crop follows the speaker instead of a
  dumb center-cut, so faces don't get sliced off (auto-falls back to a smart
  center crop if the optional vision packages aren't installed)
- **Smart moment-picking** — if you have [Ollama](https://ollama.com) running,
  a local LLM reads the transcript and picks the best moments *and writes
  punchy titles*. No Ollama? It falls back to semantic matching, then to free
  keyword+hype scoring. It always works.
- **Multiple candidate moments** — it finds several and lets you pick, instead
  of guessing one
- **Transcript caching** — re-prompting the same video is instant (no
  re-transcribing)
- **Word-by-word highlight captions** + white title box (MrBeast/CapCut style)
- Bug fixes over the original script: fade/zoom timing now uses the *real*
  clip length, and caption timestamps no longer corrupt the shared transcript.

---

## Setup on Windows (one time)

1. **Install Python 3.9+** — https://www.python.org/downloads/
   During install, tick **“Add Python to PATH”**.

2. **Install FFmpeg** — download the *full* build from
   https://www.gyan.dev/ffmpeg/builds/ , unzip it, and add the `bin\` folder to
   your PATH. (Quick test: open Command Prompt and type `ffmpeg -version`.)

3. **Get this folder** onto your machine (e.g. `Desktop\yt-clipper\`).

4. **Double-click `run.bat`.**
   The first run creates a virtual environment and installs everything
   (a few minutes). Every run after that just starts the app and opens your
   browser at `http://localhost:8000`.

That's it. The first time you clip a video, faster-whisper downloads its model
once (a few hundred MB), then works offline forever after.

> **Mac / Linux:** there's no `.bat`, but it's the same three commands:
> ```
> python3 -m venv .venv && source .venv/bin/activate
> pip install -r requirements.txt
> python app.py
> ```
> (Mac: `brew install ffmpeg` first. Linux: `sudo apt install ffmpeg`.)

---

## How to use it

1. Start the app (`run.bat`) and open **http://localhost:8000**.
2. Paste a **YouTube URL** *or* drag in a **video file**.
3. Type **what moment you want** — the more specific, the better
   (name a person, a phrase, a number you remember).
4. Pick a **clip length** (8 / 30 / 60s) and click **Find the best moments**.
5. It transcribes and shows you **several candidate moments**. Pick one
   (tweak the on-screen title if you want) and click **Make this clip**.
6. Preview it, then **Download MP4**. Post it. Done.

---

## Optional upgrades (all free)

| Add-on | Gives you | How |
|---|---|---|
| **Ollama** | A local LLM that picks better moments and writes titles | Install from https://ollama.com, then `ollama pull llama3.1`. The app finds it automatically. |
| **opencv + mediapipe** | Face-tracked reframing (installed by `requirements.txt`) | Already in requirements; nothing to do. |
| **sentence-transformers** | Semantic prompt matching when Ollama is off | Already in requirements. |

To point the app at a different local model:
```
set OLLAMA_MODEL=qwen2.5
```
(before launching), or change `OLLAMA_MODEL` in `pipeline.py`.

---

## Tuning the look

Open `pipeline.py` — the config block at the top controls captions:

| Setting | Controls |
|---|---|
| `FONT_NAME` | Caption font (must be installed on your system; `Arial Black` works on Windows) |
| `CAPTION_FONT_SIZE` / `TITLE_FONT_SIZE` | Text size |
| `HIGHLIGHT_COLOR` | Color of the currently-spoken word. ASS format is `&H00BBGGRR` (hex, blue-green-red order) |
| `WORDS_PER_CAPTION_GROUP` | Words on screen at once (3 = CapCut default) |
| `HYPE_WORDS` | Words that boost a moment's score — edit for your niche |

The font and clip options are also editable right in the web UI under
**Advanced options**.

---

## Speed notes

- Transcription is the slow step. On a normal Windows laptop (CPU):
  a 1-hour video is roughly `base` ≈ 4–6 min, `small` ≈ 8–12 min.
  Have an NVIDIA GPU? It's used automatically and is far faster.
- Thanks to caching, changing your prompt or clip length on the **same video**
  skips transcription entirely — it's basically instant.
- Use a shorter Whisper model (`tiny`/`base`) for speed, `small`/`medium` for
  accuracy (accents, noisy audio, technical terms).

---

## A note on rights

Built for **your own footage** or content you have the right to reuse.
Re-downloading and republishing someone else's monetized video can run into
copyright/ToS issues — keep this for your own content or strictly personal use.

Everything runs locally: no uploads, no accounts, no cost, ever.
