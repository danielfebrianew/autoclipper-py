#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/auto-clipper-env"
VENV_PYTHON="$VENV_DIR/bin/python"
VENV_STREAMLIT="$VENV_DIR/bin/streamlit"

# Pastikan venv ada
if [[ ! -f "$VENV_PYTHON" ]]; then
  echo "ERROR: Virtual environment tidak ditemukan di $VENV_DIR"
  echo "Buat dulu dengan: python3 -m venv auto-clipper-env && source auto-clipper-env/bin/activate && pip install -r requirements.txt"
  exit 1
fi

# Pindah ke root project supaya relative path (input/, output/, script.py) benar
cd "$SCRIPT_DIR"

echo "AutoClipper — starting Streamlit..."
echo "Buka browser di http://localhost:8501"
echo "(Ctrl+C untuk stop)"
echo ""

exec "$VENV_STREAMLIT" run app.py \
  --server.port 8501 \
  --server.headless false \
  --browser.gatherUsageStats false \
  "$@"
