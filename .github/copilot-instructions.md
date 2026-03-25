# Copilot Instructions — SVG-gen

## Project Goal

Raster-to-SVG vectorization engine producing output indistinguishable from hand-traced Adobe Illustrator artwork. Smooth G1-continuous Bézier curves, minimal SVG node count, zero artifacts, efficient CPU execution.

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

## Skill Loading

- **Pipeline editing:** load `svg-pipeline-knowledge` skill for architecture, and check `pipeline-edits.instructions.md` for parameters and environment.
- **New algorithm implementation:** load `math-implementation-guide` skill for NumPy/OpenCV translations of research-report math.
- **Development methodology:** load `subagent-dev-methodology` skill for iteration pattern, validation workflow, and anti-bias rules.
- **Vision tool usage:** load `vision-safety-guard` skill before any vision analysis calls.

## Default Behavior for Quality Improvement Requests

When the user asks to improve quality, fix metrics, run an experiment, or work on a specific image — follow this workflow automatically WITHOUT needing to be told to use @main-orchestrator:

1. **Read the KB** (readFile): `.github/knowledge/kb-baselines.md`, `kb-research-queue.md`, `kb-what-failed.md`, `kb-per-image.md`
2. **Identify the biggest bottleneck**: Lowest Feat%, highest WdErr, worst MnDif
3. **Pick the top READY hypothesis** from `kb-research-queue.md` that hasn't been tried
4. **Anti-stalling check**: If 3+ parameter tweaks in a row have failed → switch to an algorithmic hypothesis
5. **Implement via `@subagent-developer`** with a precise task prompt including: hypothesis, target metric, affected image, acceptance criteria, and revert condition
6. **Validate**: run `compare_all.py` then `check_regression.py`. STOP on MAJOR regression.
7. **Update KB**: move result to `kb-what-works.md` or `kb-what-failed.md`, update `kb-research-queue.md` status

This is the default workflow for any improvement task. No need to invoke `@main-orchestrator` explicitly unless you want to override this behavior.

## Creative Problem-Solving Ratchet

This project requires **genuine creative judgment**, not blind parameter tuning. Enforce this ratchet:

### Level 1 — Parameter tuning (first resort)
Try parameter changes only when you have a **specific causal hypothesis**: _"X parameter controls Y behavior, which is causing Z metric to be wrong because…"_

### Level 2 — Algorithm change (after 3 failed tunings in the same category)
If you've tried 3 parameter changes targeting the same metric and all regressed or were neutral, **stop tuning**. You've found a tuning dead end. The problem requires a structural fix.
- Read `kb-research-queue.md` for the next READY algorithmic hypothesis
- Read `svg-vectorization-research.md` for evidence about what actually works

### Level 3 — Structural rethink (when algorithm is blocked)
If both parameter tuning and algorithmic changes are blocked, step back and ask:
- **What is the actual failure mode?** (fragmentation? over-expansion? node inflation?)
- **What does the research say causes this failure?** (read `svg-vectorization-research.md`)
- **What structural change would address the ROOT CAUSE?** (not the symptom)

### Anti-patterns (never do these)
- Trying iso=0.43 after iso=0.44 failed without a new hypothesis — this is not creative, it's random walk
- Making 5 consecutive parameter tweaks to the same value
- Calling a regression "acceptable" to avoid reverting
- Adding complexity (new special cases, guards, gates) without evidence they'll help

### Creative license (encouraged)
When standard approaches are blocked, the right move is **creative structural thinking**:
- Study what the best vectorizers do (see `svg-vectorization-research.md`)
- Question pipeline assumptions (does painter's algorithm require soft-field overlap? yes — see kb-what-failed)
- Consider approaching the problem from a completely different angle
- Propose a novel hypothesis with a clear causal mechanism, even if it hasn't been tried by others

## Slash Commands

- `/run-experiment` — run one hypothesis-driven improvement cycle (pick hypothesis → implement → test → accept/reject)
- `/profile-quality <image>` — find which pipeline step causes the most quality loss for an image
- `/run-validation` — run compare_all.py and check_regression.py, report vs baselines
- `/improve-image <image>` — targeted quality improvement for a specific test image
- `/generate-single <image>` — generate SVG for one image and show metrics
