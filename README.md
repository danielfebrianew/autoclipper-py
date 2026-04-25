# ✂️ AutoClipper

> Turn long-form videos into short-form vertical clips — automatically.

**AutoClipper** is a local-first tool for converting long-form video content (podcasts, interviews, talkshows, lectures) into short-form vertical clips ready for TikTok, Reels, and YouTube Shorts. Given a JSON of timestamps and viral scores, it automatically cuts each segment, transcribes it with Faster-Whisper, generates word-level animated captions, and reframes to 9:16 by tracking the active speaker's face with YOLOv8. Comes with a Streamlit dashboard for previewing clips and monitoring the render pipeline live.

**Stack:** Streamlit · Faster-Whisper · YOLOv8 · OpenCV · FFmpeg

---

## ✨ Features

- **JSON-driven workflow** — paste a JSON spec of clips (start, end, caption, viral score, etc.) and the pipeline handles the rest.
- **Automatic cutting** — slices source video into segments via FFmpeg stream copy (no re-encode at the cut step, so it's fast).
- **Word-level animated captions** — transcribes each clip with **Faster-Whisper** (`medium`, int8 CPU) and renders ASS subtitles with the currently-spoken word highlighted in yellow, Impact-font karaoke style.
- **Smart 9:16 reframing** — samples frames with **YOLOv8 face detection**, picks the active speaker, and computes a smooth crop path that:
  - locks onto the dominant face,
  - waits for confirmation before switching speakers (no flickering crop),
  - hard-cuts the crop on detected scene changes,
  - applies a deadzone + max-speed smoothing so the camera doesn't chase tiny jitter.
- **Watermark + source credit** — optional channel name watermark (center, faded) and source attribution (top, semi-transparent) baked in via FFmpeg `drawtext`.
- **Live render log** — Streamlit dashboard streams `script.py` stdout into a terminal-style panel, so you can watch each step in real time.
- **Clip preview** — built-in player that jumps to the start timestamp of any clip before you commit to rendering.
- **Hardware-accelerated encoding** — uses `h264_videotoolbox` on Apple Silicon for fast final encoding (swap to `libx264` or `h264_nvenc` if you're on a different platform).

---

## 🖼️ How It Works

```
┌────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│  Long video    │──▶│  JSON of clips   │──▶│  AutoClipper        │
│  (podcast.mp4) │    │  (from your AI)  │    │  pipeline           │
└────────────────┘    └──────────────────┘    └──────────┬──────────┘
                                                          │
                  ┌───────────────────────────────────────┼──────────┐
                  ▼                ▼                ▼                ▼
              ✂️ Cut          🎧 Whisper        👤 YOLOv8         🎞️ FFmpeg
              segment         transcribe       face track        composite
              (ffmpeg)        + ASS subs       + scene cut       9:16 + caps
                                                                      │
                                                                      ▼
                                                           📱 vertical_clip.mp4
```

For each clip in the JSON, the pipeline runs six stages: **cut → transcribe → subtitle → face-detect → reframe → composite**. Temp files are cleaned up between clips, and the final output is named after the suggested caption.

---

## 📦 Requirements

- **Python 3.10+**
- **FFmpeg** installed and on your `PATH`
- **macOS (Apple Silicon)** recommended — the script defaults to `h264_videotoolbox` for fast encoding. Easy to swap to `libx264` (CPU) or `h264_nvenc` (NVIDIA) — see [Configuration](#%EF%B8%8F-configuration).
- ~4 GB free disk for model weights on first run
- A YOLOv8 face-detection weights file: `yolov8n-face-lindevs.pt` placed in the project root

### Python dependencies

```bash
pip install streamlit faster-whisper ultralytics opencv-python numpy pandas
```

Or use the included `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## 🚀 Quick Start

1. **Clone the repo**

   ```bash
   git clone https://github.com/<your-username>/autoclipper.git
   cd autoclipper
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Download YOLOv8 face weights** and put `yolov8n-face-lindevs.pt` in the project root.

4. **Set up your HuggingFace token** (used by Faster-Whisper to download model weights on first run):

   ```bash
   # macOS / Linux
   export HF_TOKEN="hf_your_token_here"

   # Windows (PowerShell)
   $env:HF_TOKEN="hf_your_token_here"
   ```

   > **⚠️ Never commit your HF token.** Use environment variables or a `.env` file (and add it to `.gitignore`).

5. **Drop your source video** (e.g. `podcast.mp4`) into the project root.

6. **Launch the dashboard**

   ```bash
   streamlit run app.py
   ```

7. In the sidebar:
   - select your source video,
   - set the output folder,
   - fill in your channel name and source credit (watermarks),

   In the main area:
   - paste the JSON of clips,
   - click **Parse JSON**,
   - review the clips and preview,
   - hit **🚀 Render Clips**.

8. Watch the live terminal stream the render. Outputs land in your chosen output folder, one MP4 per clip.

---

## 📝 JSON Input Format

The dashboard expects a JSON object with a `clips` array. Each clip looks like:

```json
{
  "video_title": "How to Stop Procrastinating — Deep Dive",
  "video_duration": "01:42:17",
  "clips": [
    {
      "clip_id": 1,
      "start_time": "00:12:34",
      "end_time": "00:13:18",
      "duration_seconds": 44,
      "speaker": "Guest",
      "viral_score": 9,
      "category": "motivation",
      "hook": "The 2-minute rule that changed everything",
      "summary": "Guest explains why starting with just 2 minutes breaks the procrastination loop.",
      "suggested_caption": "The 2-minute rule will fix your procrastination"
    }
  ]
}
```

You can generate this JSON however you want — most users prompt an LLM (Claude, GPT, Gemini) to scan a video transcript and pick the most viral moments. The pipeline only requires `start_time`, `duration_seconds`, and `suggested_caption`. The rest is metadata for the dashboard.

---

## ⚙️ Configuration

All knobs are exposed as environment variables so you can tune without touching the code.

| Variable | Default | What it controls |
|---|---|---|
| `AUTOCLIPPER_FACE_SAMPLE_FPS` | `4` | How often (per second) to run YOLO face detection. Lower = faster, less accurate. |
| `AUTOCLIPPER_SCENE_CUT_SCORE` | `0.22` | Sensitivity for detecting hard scene cuts in the source. Lower = more cuts detected. |
| `AUTOCLIPPER_FOCUS_MIN_LOCK` | `1.50` | Minimum seconds to stay locked on one speaker before allowing a switch. |
| `AUTOCLIPPER_FOCUS_CONFIRM` | `0.85` | Seconds another speaker must dominate before the crop switches. |
| `AUTOCLIPPER_FOCUS_AREA_RATIO` | `1.35` | How much bigger a new face must be (vs current) to trigger a switch. |
| `AUTOCLIPPER_CROP_DEADZONE` | `0.07` | Crop won't move unless target moves more than this fraction of crop width. |
| `AUTOCLIPPER_CROP_SMOOTHING_TAU` | `0.45` | Higher = slower, smoother crop. Lower = snappier, more reactive. |
| `AUTOCLIPPER_CROP_MAX_SPEED` | `480` | Max pixels/sec the crop center can move (caps swing speed). |

Defaults are tuned for stable, broadcast-feeling crops on 2-person podcast/interview footage. If you find the crop too sluggish or too jittery, those four `FOCUS_*` and `CROP_*` knobs are where to start.

### Changing the encoder

`script.py` uses `h264_videotoolbox` (Apple Silicon). To run elsewhere, edit `composite()` in `script.py`:

```python
# CPU (works everywhere)
'-c:v', 'libx264', '-preset', 'medium', '-crf', '20',

# NVIDIA GPU
'-c:v', 'h264_nvenc', '-preset', 'p5', '-b:v', '5000k',
```

---

## 📂 Project Structure

```
autoclipper/
├── app.py                       # Streamlit dashboard (UI, JSON parser, live log)
├── script.py                    # Pipeline: cut → transcribe → reframe → composite
├── yolov8n-face-lindevs.pt      # YOLOv8 face-detection weights (download separately)
├── requirements.txt
├── output/                      # Rendered clips land here by default
└── README.md
```

---

## 🧠 Notes & Gotchas

- **First run is slow** — Faster-Whisper downloads the `medium` model (~1.5 GB) on first transcription.
- **Whisper language is hardcoded to Indonesian** (`language="id"` in `script.py`). Change it for other languages — or remove the argument entirely to let Whisper auto-detect.
- **CPU transcription** — the script runs Whisper on CPU with int8 quantization for portability. Switch to `device="cuda"` and `compute_type="float16"` if you have an NVIDIA GPU for a ~5–10× speedup.
- **Source video must be in the project root** — the sidebar only lists files alongside `app.py`. Drop a symlink in if your video lives elsewhere.
- **Output filenames come from `suggested_caption`** — special characters get stripped, spaces become underscores, and the name is truncated to 100 chars.

---

## 🛣️ Roadmap

- [ ] Multi-language UI (currently mixes English + Indonesian labels)
- [ ] Configurable subtitle styling from the dashboard (font, size, color, position)
- [ ] B-roll / zoom-in detection for higher-energy edits
- [ ] Optional intro/outro card overlays
- [ ] Export to `.srt` / `.vtt` alongside the burned-in subtitles
- [ ] One-click "remove silences" pass before subtitle generation

---

## 🤝 Contributing

PRs welcome. If you're tuning the face-tracking heuristics, please include a before/after video clip in your PR — the parameters interact in non-obvious ways and visual diffs are the only honest test.

---

## 📄 License

MIT — see [LICENSE](LICENSE).

---

## 🙏 Credits

- [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) for the transcription engine
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for face detection
- [lindevs/yolov8-face](https://github.com/lindevs) for the fine-tuned face weights
- [Streamlit](https://streamlit.io/) for making the UI a 30-line affair
- FFmpeg, as always, for doing the actual hard work
