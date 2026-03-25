# Copilot Instructions — SVG-gen

## What This Project Is

This is a **raster-to-SVG vectorization engine**. It takes raster images (PNG, JPEG) as input and produces SVG vector output. It is a one-way pipeline: **raster in → SVG out**.

**It does NOT parse, read, import, or analyze existing SVG files.** There is no SVG input path. If someone mentions "SVG" in any improvement request, they mean the **generated SVG output quality** — not SVG file parsing or reading.

## How to Interpret Requests

Any request containing words like "improve", "fix", "better", "enhance", "optimize" combined with "SVG", "output", "quality", "vectorization", "parsing", "generation", or "pipeline" means: **improve the visual quality of the raster-to-vector conversion**. This applies even when the wording is vague. Do NOT search for XML parsing code, SVG readers, or file parsers — they don't exist in this project.

## Project Goal

Output indistinguishable from hand-traced Adobe Illustrator artwork. Smooth G1-continuous Bézier curves, minimal SVG node count, zero artifacts, efficient CPU execution.

The visual standard: an artist hand-tracing the image would produce the same result — clean curves, faithful colors, proper anti-aliased edges interpreted as vector shapes.

## Ports & Running Locally

- **Backend (FastAPI):** port **8100**
- **Frontend (Vite):** port **5173**, proxies `/api` → `http://127.0.0.1:8100`

```bash
# Backend
cd raster-to-vector/server && source .venv/bin/activate
DYLD_LIBRARY_PATH="/opt/homebrew/opt/cairo/lib" uvicorn app.main:app --reload --host 127.0.0.1 --port 8100

# Frontend (separate terminal)
cd raster-to-vector/client && npm run dev
```

## Project Structure

```
SVG-gen/                          ← repo root
├── raster-to-vector/
│   ├── server/                   ← Python FastAPI backend
│   │   ├── app/core/
│   │   │   ├── multilevel/       ← MAIN PIPELINE (~1500 lines)
│   │   │   ├── curve_fitting/    ← BÉZIER ENGINE (~700 lines)
│   │   │   └── (classification, comparison, fill_reconstruction,
│   │   │        preprocess, stroke_reconstruction, svg_export)
│   │   └── .venv/                ← Python 3.12 virtualenv
│   ├── client/                   ← React 19 + Vite + TypeScript
│   └── shared/
├── compare_all.py                ← Batch validation (generates _comparisons/)
├── _inspect_closeup.py           ← Close-up crops for vision analysis
├── generate.py                   ← Single-image CLI generator
├── Ref.png, test[1-5].jpg        ← Test images
└── _comparisons/                 ← Output: comparison PNGs, SVGs, metrics
```

## Test Images

| Image     | Resolution | Subject       | Key Character                       |
| --------- | ---------- | ------------- | ----------------------------------- |
| Ref.png   | 1536×1024  | Floral logo   | Line art fast path, grayscale       |
| test2.jpg | 4016×2256  | McLaren car   | Automotive paint, reflections       |
| test3.jpg | 6124×4082  | Botanical ink | Fine ink stems, high res            |
| test4.jpg | 3310×2481  | Aerial forest | Dense texture, warm color challenge |
| test5.jpg | 3888×2592  | Street mural  | Detail/texture loss                 |
| test1.jpg | 4719×2303  | Antique map   | Full mode only, very slow           |

## Conventions

- **Pure Python + NumPy + OpenCV C extensions.** No Cython, no Numba, no compilation.
- Fast mode target: ~37s for 5 images. Do not introduce compilation dependencies.
- Frontend fetches via relative `/api` — never hard-code ports in client code.

## Knowledgebase

All KB files live in `.github/knowledge/` (git-tracked). Read them with `readFile`. Start with the index:

- `.github/knowledge/kb-index.md` — file list and access guide
- `.github/knowledge/kb-baselines.md` — current metric scores (your BEFORE numbers)
- `.github/knowledge/kb-what-failed.md` — proven dead ends, do not retry
- `.github/knowledge/kb-what-works.md` — verified techniques
- `.github/knowledge/kb-per-image.md` — per-image diagnosis and root causes
- `.github/knowledge/kb-research-queue.md` — prioritized hypotheses with status (READY / DONE / BLOCKED)
- `.github/knowledge/kb-params.md` — current parameter values (verify against code before trusting)
- `.github/knowledge/svg-vectorization-research.md` — domain research: what the best generators do and why

## Validation Workflow

```bash
# After ANY pipeline code change:
DYLD_LIBRARY_PATH="/opt/homebrew/opt/cairo/lib" python compare_all.py
python check_regression.py      # auto-diffs vs baselines, flags regressions
```
