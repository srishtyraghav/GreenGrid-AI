#!/bin/bash
set -e

echo "============================================================"
echo " Starting GreenGrid AI V2 Dashboard"
echo "============================================================"

# Stop any existing processes on ports 8000, 5173, 5174
echo "Cleaning up old processes..."
for port in 8000 5173 5174; do
  pid=$(lsof -t -i:$port 2>/dev/null || true)
  if [ -n "$pid" ]; then
    echo "Killing process on port $port (PID: $pid)..."
    kill -9 $pid 2>/dev/null || true
  fi
done
sleep 1

# Define paths
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
V2_DIR="$ROOT_DIR/v2"

echo "Starting FastAPI Backend..."
cd "$V2_DIR/backend"
if [ -f "$ROOT_DIR/.venv/bin/python" ]; then
    "$ROOT_DIR/.venv/bin/python" app/main.py > /dev/null 2>&1 &
else
    python app/main.py > /dev/null 2>&1 &
fi
BACKEND_PID=$!

echo "Starting Vite Frontend..."
cd "$V2_DIR/frontend"
npm run dev > /dev/null 2>&1 &
FRONTEND_PID=$!

# Wait briefly for servers to bind
sleep 3

echo ""
echo "GreenGrid AI V2"
echo "────────────────────────────"
if lsof -t -i:8000 > /dev/null 2>&1; then
  echo "✓ Backend   http://localhost:8000"
else
  echo "✗ Backend failed to start on port 8000"
fi

if lsof -t -i:5173 > /dev/null 2>&1; then
  echo "✓ Frontend  http://localhost:5173"
else
  echo "✗ Frontend failed to start on port 5173 (it might be on 5174, check Vite logs)"
fi

echo "Dashboard ready."
echo "Press Ctrl+C to stop all services."

trap "kill -9 $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" SIGINT SIGTERM

wait $BACKEND_PID $FRONTEND_PID
