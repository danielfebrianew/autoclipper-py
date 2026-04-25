import streamlit as st
import json
import html
import os
import subprocess
import sys

APP_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(APP_DIR, "output")
INPUT_DIR = os.path.join(APP_DIR, "input")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(INPUT_DIR, exist_ok=True)

st.set_page_config(page_title="AutoClipper", layout="wide", initial_sidebar_state="expanded")

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
/* Hide default chrome */
#MainMenu, footer, header { visibility: hidden; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0f0f0f;
    border-right: 1px solid #222;
    min-width: 280px !important;
    max-width: 280px !important;
}
[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stTextInput label { font-size: 11px; text-transform: uppercase; letter-spacing: .08em; color: #666 !important; }

/* Main bg */
.stApp { background: #111; }

/* Headings */
h1, h2, h3 { color: #f0f0f0 !important; }

/* Text areas — monospace */
textarea {
    font-family: "JetBrains Mono", "Fira Code", monospace !important;
    font-size: 12px !important;
    background: #1a1a1a !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 8px !important;
    color: #d4d4d4 !important;
}

/* Primary button */
div[data-testid="stButton"] button[kind="primary"] {
    background: #7c3aed !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    letter-spacing: .03em !important;
}
div[data-testid="stButton"] button[kind="primary"]:hover {
    background: #6d28d9 !important;
}

/* Ghost button (Clear) */
div[data-testid="stButton"] button[kind="secondary"] {
    background: transparent !important;
    border: 1px solid #333 !important;
    border-radius: 6px !important;
    color: #888 !important;
}
div[data-testid="stButton"] button[kind="secondary"]:hover {
    border-color: #555 !important;
    color: #ccc !important;
}

/* Tabs */
[data-testid="stTabs"] button {
    font-size: 13px !important;
    font-weight: 500 !important;
    color: #666 !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #a78bfa !important;
    border-bottom-color: #7c3aed !important;
}

/* Clip row card */
.clip-card {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 14px;
}
.clip-num {
    font-size: 11px;
    font-weight: 700;
    color: #7c3aed;
    background: #2d1b69;
    border-radius: 5px;
    padding: 2px 8px;
    white-space: nowrap;
}
.clip-caption {
    flex: 1;
    font-size: 13px;
    color: #d4d4d4;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.clip-meta {
    font-size: 11px;
    color: #555;
    white-space: nowrap;
}
.clip-bar-wrap {
    width: 80px;
    background: #222;
    border-radius: 3px;
    height: 4px;
    overflow: hidden;
}
.clip-bar-fill {
    height: 4px;
    background: #7c3aed;
    border-radius: 3px;
}
.viral-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    margin-right: 4px;
}

/* Status badge */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: .05em;
    text-transform: uppercase;
}
.badge-idle    { background:#1e1e1e; color:#555; border:1px solid #2a2a2a; }
.badge-ready   { background:#052e16; color:#4ade80; border:1px solid #166534; }
.badge-error   { background:#2d0a0a; color:#f87171; border:1px solid #7f1d1d; }
.badge-running { background:#1c1830; color:#a78bfa; border:1px solid #4c1d95; }

/* dataframe */
[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }

/* section label */
.section-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: .1em;
    color: #555;
    margin-bottom: 8px;
    margin-top: 24px;
}

/* terminal log */
.terminal-card {
    margin-top: 14px;
    border: 1px solid #2a2a2a;
    border-radius: 8px;
    overflow: hidden;
    background: #07090c;
}
.terminal-header {
    height: 34px;
    padding: 0 12px;
    display: flex;
    align-items: center;
    gap: 8px;
    background: #171717;
    border-bottom: 1px solid #2a2a2a;
    color: #8f8f8f;
    font-size: 11px;
    font-family: "JetBrains Mono", "Fira Code", monospace;
}
.terminal-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    display: inline-block;
}
.terminal-red { background: #ef4444; }
.terminal-yellow { background: #f59e0b; }
.terminal-green { background: #22c55e; }
.terminal-title {
    flex: 1;
    color: #d4d4d4;
}
.terminal-status {
    color: #777;
}
.terminal-body {
    max-height: 460px;
    overflow-y: auto;
    padding: 12px 14px;
    background: #07090c;
    color: #d7dde8;
    font-size: 12px;
    line-height: 1.45;
    font-family: "JetBrains Mono", "Fira Code", "SFMono-Regular", Consolas, monospace;
    display: flex;
    flex-direction: column-reverse;
}
.terminal-line {
    min-height: 17px;
    white-space: pre-wrap;
    word-break: break-word;
}
.terminal-empty {
    color: #6b7280;
}
</style>
""", unsafe_allow_html=True)


def render_log_card(placeholder, log_lines, status_label="running", max_lines=350):
    visible_lines = log_lines[-max_lines:]
    hidden_count = max(0, len(log_lines) - len(visible_lines))
    if hidden_count:
        visible_lines = [
            f"... {hidden_count} baris sebelumnya disembunyikan dari live view ..."
        ] + visible_lines
    if not visible_lines:
        visible_lines = ["Menunggu output dari script.py..."]

    rendered_lines = []
    for line in reversed(visible_lines):
        escaped = html.escape(line) if line else "&nbsp;"
        css_class = "terminal-line"
        if not line:
            css_class += " terminal-empty"
        rendered_lines.append(f'<div class="{css_class}">{escaped}</div>')

    placeholder.markdown(f"""
    <div class="terminal-card">
        <div class="terminal-header">
            <span class="terminal-dot terminal-red"></span>
            <span class="terminal-dot terminal-yellow"></span>
            <span class="terminal-dot terminal-green"></span>
            <span class="terminal-title">script.py live log</span>
            <span class="terminal-status">{html.escape(status_label)} · {len(log_lines)} lines</span>
        </div>
        <div class="terminal-body">
            {''.join(rendered_lines)}
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────
defaults = {
    "clips_data": None,       # parsed JSON dict
    "selected_clip_index": 0,
    "parse_error": "",
    "status": "idle",         # idle | ready | running | error
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:16px 0 20px">
        <div style="font-size:20px;font-weight:800;letter-spacing:-.02em;color:#f0f0f0">
            ✂️ AutoClipper
        </div>
        <div style="font-size:11px;color:#555;margin-top:2px">TikTok-style clip renderer</div>
    </div>
    """, unsafe_allow_html=True)

    # Status badge
    status_map = {
        "idle":    ("badge-idle",    "Menunggu"),
        "ready":   ("badge-ready",   "Siap"),
        "running": ("badge-running", "Processing"),
        "error":   ("badge-error",   "Error"),
    }
    cls, label = status_map[st.session_state.status]
    st.markdown(f'<span class="badge {cls}">{label}</span>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">Video</div>', unsafe_allow_html=True)
    video_files = sorted([
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith((".mp4", ".mkv", ".mov", ".avi"))
    ])
    if not video_files:
        st.warning("Taruh video di folder `input/`.")
        video_path = ""
    else:
        selected_video = st.selectbox("Pilih video", video_files, label_visibility="collapsed")
        video_path = os.path.join(INPUT_DIR, selected_video)
        st.caption(f"`input/{selected_video}`")

    st.markdown('<div class="section-label">Output</div>', unsafe_allow_html=True)
    output_format = st.selectbox("Format output", ["mp4", "mkv", "mov"], label_visibility="collapsed")

    save_dir = st.text_input("Folder output", value=OUTPUT_DIR, label_visibility="collapsed",
                              placeholder="Folder output...")

    st.markdown('<div class="section-label">Watermark</div>', unsafe_allow_html=True)
    channel_name = st.text_input("Nama channel kamu", value="@channelku", label_visibility="collapsed",
                                  placeholder="@namachannel")
    source_credit = st.text_input("Source credit (e.g. youtube.com/@sumber)", value="", label_visibility="collapsed",
                                   placeholder="Source: youtube.com/@sumber")

    st.markdown("---")
    always_rerun = st.checkbox("Always rerun (overwrite existing)", value=False)

# ── Main area ──────────────────────────────────────────────────
st.markdown("""
<div style="padding:8px 0 24px">
    <div style="font-size:28px;font-weight:800;letter-spacing:-.03em;color:#f0f0f0">Clip Studio</div>
    <div style="font-size:14px;color:#555;margin-top:4px">Paste JSON dari AI, review clips, render.</div>
</div>
""", unsafe_allow_html=True)

# ── Input section ──────────────────────────────────────────────
st.markdown('<div class="section-label">JSON Input</div>', unsafe_allow_html=True)

json_text = st.text_area(
    "json_input",
    height=220,
    placeholder='{\n  "video_title": "...",\n  "video_duration": "01:00:00",\n  "clips": [...]\n}',
    label_visibility="collapsed",
)

c1, c2, c3 = st.columns([1, 1, 6])
parse_clicked = c1.button("Parse JSON", type="primary")
clear_clicked = c2.button("Clear", type="secondary")

if clear_clicked:
    st.session_state.clips_data = None
    st.session_state.parse_error = ""
    st.session_state.status = "idle"
    st.rerun()

if parse_clicked:
    if not json_text.strip():
        st.session_state.parse_error = "Input kosong."
        st.session_state.status = "error"
    else:
        try:
            parsed = json.loads(json_text)
            clips = parsed.get("clips", [])
            if not clips:
                st.session_state.parse_error = "Key 'clips' tidak ditemukan atau kosong."
                st.session_state.status = "error"
            else:
                st.session_state.clips_data = parsed
                st.session_state.parse_error = ""
                st.session_state.status = "ready"
                st.session_state.selected_clip_index = 0
        except json.JSONDecodeError as e:
            st.session_state.parse_error = str(e)
            st.session_state.status = "error"

if st.session_state.parse_error:
    st.error(f"JSON tidak valid: {st.session_state.parse_error}")

# ── Output section (hanya jika sudah parsed) ───────────────────
if st.session_state.clips_data:
    data  = st.session_state.clips_data
    clips = data["clips"]

    max_dur = max((c.get("duration_seconds", 0) for c in clips), default=1)
    total_dur = sum(c.get("duration_seconds", 0) for c in clips)

    st.markdown("---")
    m1, m2, m3 = st.columns(3)
    m1.metric("Clips", len(clips))
    m2.metric("Total durasi", f"{total_dur}s")
    m3.metric("Video", data.get("video_duration", "—"))

    tab_clips, tab_preview = st.tabs(["Clips", "Preview"])

    # ── Tab: Clips ─────────────────────────────────────────────
    with tab_clips:
        viral_colors = {
            range(9, 11): "#4ade80",
            range(7, 9):  "#facc15",
            range(0, 7):  "#f87171",
        }
        def viral_color(score):
            for r, c in viral_colors.items():
                if score in r:
                    return c
            return "#555"

        clip_rows = []
        for i, clip in enumerate(clips):
            dur   = clip.get("duration_seconds", 0)
            score = clip.get("viral_score", 0)
            pct   = int(dur / max_dur * 100) if max_dur else 0
            color = viral_color(score)

            st.markdown(f"""
            <div class="clip-card">
                <span class="clip-num">#{clip.get('clip_id', i+1)}</span>
                <div style="flex:1;min-width:0">
                    <div class="clip-caption">{clip.get('suggested_caption','—')}</div>
                    <div style="margin-top:6px;display:flex;align-items:center;gap:8px">
                        <div class="clip-bar-wrap">
                            <div class="clip-bar-fill" style="width:{pct}%"></div>
                        </div>
                        <span class="clip-meta">{clip.get('start_time','?')} → {clip.get('end_time','?')} &nbsp;·&nbsp; {dur}s</span>
                    </div>
                </div>
                <div style="text-align:right">
                    <span class="clip-meta" style="color:{color}">
                        <span class="viral-dot" style="background:{color}"></span>{score}/10
                    </span>
                    <div class="clip-meta" style="margin-top:2px">{clip.get('speaker','')}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            clip_rows.append({
                "ID": clip.get("clip_id", i+1),
                "Start": clip.get("start_time", ""),
                "End": clip.get("end_time", ""),
                "Dur(s)": dur,
                "Speaker": clip.get("speaker", ""),
                "Viral": score,
                "Caption": clip.get("suggested_caption", ""),
            })

        with st.expander("Tabel lengkap"):
            import pandas as pd
            st.dataframe(pd.DataFrame(clip_rows), use_container_width=True, hide_index=True)

    # ── Tab: Preview ───────────────────────────────────────────
    with tab_preview:
        if not video_path:
            st.info("Pilih video di sidebar untuk preview.")
        else:
            clip_labels = [
                f"#{c.get('clip_id','?')} — {c.get('suggested_caption','')[:50]}"
                for c in clips
            ]
            sel = st.selectbox("Pilih clip", clip_labels,
                               index=st.session_state.selected_clip_index,
                               key="preview_select")
            st.session_state.selected_clip_index = clip_labels.index(sel)
            clip = clips[st.session_state.selected_clip_index]

            # Parse start_time ke detik untuk st.video start_time
            def ts_to_seconds(ts: str) -> int:
                parts = ts.strip().split(":")
                try:
                    if len(parts) == 3:
                        return int(parts[0])*3600 + int(parts[1])*60 + int(parts[2])
                    elif len(parts) == 2:
                        return int(parts[0])*60 + int(parts[1])
                    return int(parts[0])
                except ValueError:
                    return 0

            start_sec = ts_to_seconds(clip.get("start_time", "0"))

            col_v, col_i = st.columns([2, 1])
            with col_v:
                st.video(video_path, start_time=start_sec)
            with col_i:
                st.markdown(f"**Hook**\n\n{clip.get('hook','—')}")
                st.markdown(f"**Summary**\n\n{clip.get('summary','—')}")
                st.markdown(f"**Category:** `{clip.get('category','—')}`")
                score = clip.get('viral_score', 0)
                st.markdown(f"**Viral Score:** {score}/10")

            pc, nc = st.columns(2)
            if pc.button("◀ Prev") and st.session_state.selected_clip_index > 0:
                st.session_state.selected_clip_index -= 1
                st.rerun()
            if nc.button("Next ▶") and st.session_state.selected_clip_index < len(clips) - 1:
                st.session_state.selected_clip_index += 1
                st.rerun()

    # ── Run section ────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-label">Render</div>', unsafe_allow_html=True)

    output_json = json.dumps(data, ensure_ascii=False, indent=2)
    dl_col, run_col = st.columns([1, 3])
    dl_col.download_button("⬇️ Download JSON", data=output_json,
                           file_name="clip.json", mime="application/json")

    if not video_path:
        st.warning("Pilih video di sidebar.")
    else:
        if run_col.button("🚀 Render Clips", type="primary"):
            missing = []
            if not channel_name.strip():
                missing.append("Nama channel")
            if not source_credit.strip():
                missing.append("Source credit")
            if missing:
                st.toast(f"⚠️ Isi dulu: {', '.join(missing)}", icon="⚠️")
                st.stop()
            st.session_state.status = "running"
            tmp_json = os.path.join(save_dir, "_autoclipper_input.json")
            with open(tmp_json, "w", encoding="utf-8") as f:
                f.write(output_json)

            env = os.environ.copy()
            env["AUTOCLIPPER_JSON"]          = tmp_json
            env["AUTOCLIPPER_VIDEO"]         = video_path
            env["AUTOCLIPPER_OUTDIR"]        = save_dir
            env["AUTOCLIPPER_CHANNEL"]       = channel_name
            env["AUTOCLIPPER_SOURCE_CREDIT"] = source_credit
            env["PYTHONUNBUFFERED"]          = "1"
            env["PYTHONIOENCODING"]          = "utf-8"

            log_lines = []
            log_placeholder = st.empty()
            render_log_card(log_placeholder, log_lines)
            returncode = 1
            with st.spinner("Rendering... (sabar ya, whisper lagi kerja keras)"):
                try:
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
                    assert process.stdout is not None
                    for raw_line in process.stdout:
                        for line in raw_line.rstrip("\n").split("\r"):
                            log_lines.append(line)
                        render_log_card(log_placeholder, log_lines)
                    returncode = process.wait()
                except Exception as e:
                    log_lines.append(f"[app] ERROR: {e}")
                    render_log_card(log_placeholder, log_lines, status_label="error")

            if os.path.exists(tmp_json):
                os.remove(tmp_json)

            if returncode == 0:
                st.session_state.status = "ready"
                render_log_card(log_placeholder, log_lines, status_label="done")
                st.success("✅ Semua clip selesai dirender!")
            else:
                st.session_state.status = "error"
                render_log_card(log_placeholder, log_lines, status_label=f"error code {returncode}")
                st.error("❌ Script error.")

            if log_lines:
                with st.expander("Full render log"):
                    st.code("\n".join(log_lines))
