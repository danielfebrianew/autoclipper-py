#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/auto-clipper-env"
VENV_PYTHON="$VENV_DIR/bin/python"
VENV_UVICORN="$VENV_DIR/bin/uvicorn"
WEB_DIR="$SCRIPT_DIR/../autoclipper-web"

if [[ ! -f "$VENV_PYTHON" ]]; then
  echo "ERROR: Virtual environment tidak ditemukan di $VENV_DIR"
  echo "Buat dulu dengan: python3 -m venv auto-clipper-env && source auto-clipper-env/bin/activate && pip install -r requirements.txt"
  exit 1
fi

if [[ ! -d "$WEB_DIR" ]]; then
  echo "ERROR: Next.js project tidak ditemukan di $WEB_DIR"
  exit 1
fi

cd "$SCRIPT_DIR"

cleanup() {
  echo ""
  echo "Stopping all processes..."
  kill 0
}
trap cleanup EXIT INT TERM

echo "========================================"
echo " AutoClipper Dev Server"
echo "========================================"
echo " FastAPI  → http://localhost:8000"
echo " Next.js  → http://localhost:3000"
echo " API Docs → http://localhost:8000/docs"
echo "========================================"
echo ""

"$VENV_UVICORN" api:app --reload --port 8000 &
BACKEND_PID=$!

cd "$WEB_DIR"
npm run dev &
FRONTEND_PID=$!

wait $BACKEND_PID $FRONTEND_PID
