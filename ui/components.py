import html as _html


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
        escaped = _html.escape(line) if line else "&nbsp;"
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
            <span class="terminal-status">{_html.escape(status_label)} · {len(log_lines)} lines</span>
        </div>
        <div class="terminal-body">
            {''.join(rendered_lines)}
        </div>
    </div>
    """, unsafe_allow_html=True)


def viral_color(score: int) -> str:
    if score in range(9, 11):
        return "#4ade80"
    if score in range(7, 9):
        return "#facc15"
    return "#f87171"


def ts_to_seconds(ts: str) -> int:
    parts = ts.strip().split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        return int(parts[0])
    except ValueError:
        return 0
