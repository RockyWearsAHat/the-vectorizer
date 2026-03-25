---
description: "Validation and test execution reference for compare_all.py, generate.py, and inspection tools."
applyTo: "{compare_all.py,generate.py,_inspect_closeup.py}"
---

# Validation & Testing

## Quick Test (single image, ~2-10s)

```bash
cd /Users/alexwaldmann/Desktop/SVG-gen
source raster-to-vector/server/.venv/bin/activate
DYLD_LIBRARY_PATH="/opt/homebrew/opt/cairo/lib" python generate.py Ref.png
```

## Batch Validation

```bash
# FAST MODE (default) — ~37s, skips test1 + color optimization
python compare_all.py

# FULL MODE — includes test1 + optimize_svg_colors (slower)
python compare_all.py --full
```

## Output

Generated in `_comparisons/`:

- `{name}_comparison.png` (side-by-side: original | SVG | error map)
- `{name}_output.svg`
- `{name}_metrics.txt`
- `summary.txt`

## Quality Metrics

- **feature*presence*%** — What % of dark features in original appear in SVG (higher = better)
- **width_mean_error_px** — Average stroke width difference (lower = better)
- **mean_pixel_diff** — Mean absolute pixel difference (lower = better)
- **node_count** — Total Bézier nodes in SVG (lower = more efficient)
- **svg_size_kb** — File size (lower = better, but not at cost of quality)

**Target quality**: Feature presence > 80%, width error < 1px, no visible artifacts at 200% zoom.

## Before Running Validation

Read baselines via readFile: `.github/knowledge/kb-baselines.md`
