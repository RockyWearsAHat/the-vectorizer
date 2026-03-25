---
description: "Pipeline code editing context: key files, parameters, environment setup."
applyTo: "raster-to-vector/server/app/core/**"
---

# Pipeline Editing Context

## Key Files

| File                                                         | Lines | Purpose                                                           |
| ------------------------------------------------------------ | ----- | ----------------------------------------------------------------- |
| `raster-to-vector/server/app/core/multilevel/__init__.py`    | ~1500 | Full pipeline: quantize, merge, soft fields, contours, SVG output |
| `raster-to-vector/server/app/core/curve_fitting/__init__.py` | ~700  | Bézier fitting, merging, G1 continuity, reduce_nodes              |

## Current Default Parameters (verified March 2026)

```python
simplify_epsilon=1.0    # RDP tolerance (was 1.5, reduced June 2025)
max_error=1.5           # Bézier fitting max deviation (was 2.0)
line_tolerance=0.5      # Straight-line detection (was 1.2, forces more curves)
corner_threshold=55.0   # Corner angle degrees
min_contour_area=12     # Minimum area in real px
contour_scale=4         # Max superresolution factor
smooth_sigma=0.50       # Base sigma
mediator_threshold=0.3  # Mediator absorption threshold
```

⚠️ **VERIFY**: Always cross-check against `multilevel/__init__.py` line ~315 before assuming parameter values. Stale values here have caused accidental regressions.

## Environment Setup

```bash
cd /Users/alexwaldmann/Desktop/SVG-gen
source raster-to-vector/server/.venv/bin/activate
export DYLD_LIBRARY_PATH="/opt/homebrew/opt/cairo/lib"
```

## Performance Constraints

- Pure Python + NumPy + OpenCV C extensions. NO Cython, NO Numba.
- Fast mode target: ~37 seconds total for 5 images
- Per-image: Ref 3.3s, test2 7.1s, test3 9.0s, test4 8.4s, test5 9.4s

## Before Making Changes

Read relevant knowledgebase files via memory tool:

- `/memories/repo/kb-baselines.md` — current metrics (your BEFORE numbers)
- `/memories/repo/kb-what-failed.md` — don't retry proven failures
- `/memories/repo/kb-params.md` — parameter rationale

For deeper pipeline architecture details, load the `svg-pipeline-knowledge` skill.
