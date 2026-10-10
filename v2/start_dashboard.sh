#!/bin/bash
# GreenGrid AI V2 Dashboard launcher (Git Bash / Windows-compatible).
# Starts the FastAPI backend (:8000) and the Vite dev server (:5173).
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
V2_DIR="$ROOT_DIR/v2"

# Prefer the project venv (Windows path first, then Linux layout).
if [ -f "$ROOT_DIR/.venv/Scripts/python.exe" ]; then
  PY="$ROOT_DIR/.venv/Scripts/python.exe"
elif [ -f "$ROOT_DIR/.venv/bin/python" ]; then
  PY="$ROOT_DIR/.venv/bin/python"
else
  PY="python"
fi

echo "Using Python: $PY"
"$PY" -c "import fastapi, rio_tiler, uvicorn" 2>/dev/null || {
  echo "Backend dependencies missing; installing fastapi rio-tiler uvicorn ..."
  "$PY" -m pip install -q fastapi rio-tiler uvicorn
}

echo "Starting FastAPI backend on :8000 ..."
cd "$V2_DIR/backend"
"$PY" app/main.py &
BACKEND_PID=$!

echo "Starting Vite dev server on :5173 ..."
cd "$V2_DIR/frontend"
npm run dev &
FRONTEND_PID=$!

sleep 5
curl -s -m 3 -o /dev/null http://localhost:8000/api/metadata \
  && echo "OK  backend   http://localhost:8000" || echo "FAIL backend (check logs above)"
curl -s -m 3 -o /dev/null http://localhost:5173/ \
  && echo "OK  frontend  http://localhost:5173" || echo "FAIL frontend (check logs above)"

echo "Dashboard starting. Ctrl+C stops both."
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" SIGINT SIGTERM
wait $BACKEND_PID $FRONTEND_PID
