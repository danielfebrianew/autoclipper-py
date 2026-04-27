def get_css() -> str:
    return """
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
"""
