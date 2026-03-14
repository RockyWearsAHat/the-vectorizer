# Copilot Instructions — SVG-gen

## Ports

- **Backend (FastAPI):** always start on port **8100** (`uvicorn … --port 8100`).
- **Frontend (Vite):** dev server on port **5173**; proxies `/api` requests to `http://127.0.0.1:8100`.
- The frontend fetches all API calls via the relative path `/api` — never hard-code a port in client-side code.

## Running Locally

```bash
# From the repo root
cd raster-to-vector/server
source .venv/bin/activate
DYLD_LIBRARY_PATH="/opt/homebrew/opt/cairo/lib" uvicorn app.main:app --reload --host 127.0.0.1 --port 8100

# Separate terminal
cd raster-to-vector/client
npm run dev
```

Or use the helper script: `./raster-to-vector/shared/scripts/dev.sh`

## Project Structure

- `raster-to-vector/server/` — Python FastAPI backend (vectorization pipeline)
- `raster-to-vector/client/` — Vite + React + TypeScript frontend
- `raster-to-vector/shared/` — Scripts and sample images
