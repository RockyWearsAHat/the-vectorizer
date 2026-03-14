#!/usr/bin/env bash
# Start both the backend and frontend dev servers.
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Cairo library path for macOS (Homebrew)
export DYLD_LIBRARY_PATH="/opt/homebrew/opt/cairo/lib:${DYLD_LIBRARY_PATH:-}"

echo "Starting backend server…"
cd "$ROOT/server"
source .venv/bin/activate
uvicorn app.main:app --reload --host 127.0.0.1 --port 8100 &
BACKEND_PID=$!

echo "Starting frontend dev server…"
cd "$ROOT/client"
npm run dev &
FRONTEND_PID=$!

cleanup() {
  echo "Shutting down…"
  kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
}
trap cleanup EXIT INT TERM

wait
