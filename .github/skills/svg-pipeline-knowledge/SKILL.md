---
name: "svg-pipeline-knowledge"
description: "Deep knowledge of the SVG vectorization pipeline: architecture, parameters, code structure, and known issues. Load when editing pipeline code or diagnosing quality."
---

# SVG Pipeline Knowledge

> **This project converts raster images → SVG. It never parses or reads existing SVG files.** All "SVG improvement" requests mean improving the vectorization output quality.

Use this skill when editing pipeline code or diagnosing vectorization quality issues.

## Key Files

See `pipeline-edits.instructions.md` for the key files table, default parameters, and environment setup.

## Pipeline Architecture

```
Input image (BGR)
  │
  ├─ Edge-density map (_compute_edge_weight)
  ├─ Gaussian blur denoise (7×7 for km, 5×5 for dist)
  │
  ▼
Step 1: K-means quantization
  - Auto-K via _estimate_initial_k() (bins LAB at 8 ΔE)
  - Dynamic max_k: 7 for >8MP, 8 for >4MP, 12 otherwise
  - Subsamples to 250K pixels for >4MP images (500K otherwise)
  - Chunked vectorized label assignment (matrix multiply trick)
  │
  ▼
Step 1b: Merge close clusters (LAB ΔE < 30)
  │
  ▼
Step 2: Agglomerative merge + mediator absorption
  - gradient_aware_merge() with boundary contrast weighting (max 3 iterations)
  - Mediator scoring via cv2.distanceTransform (skipped for >4MP images)
  - Mediator clusters (< mediator_threshold=0.3) absorbed into neighbors
  │
  ▼
Step 2b: Line art fast path (grayscale images only)
  - Detection: mean_saturation < 20, K ≤ 6, background > 70%
  - Hysteresis thresholding: strict=min(otsu*0.82, 145), lenient=min(otsu, 170)
  - Connected component filtering (only regions touching strict mask)
  - Morph close (3×3), NO morph open (preserves thin lines)
  - Upscale by S, contour extraction at S× resolution
  - Bézier fitting with tight parameters (epsilon*0.2, max_error*0.3)
  - Returns early: single dark ink layer over detected background
  │
  ▼
Step 3: Gradient detection (only for images < 3MP)
  - _detect_gradients() finds linear color gradients across merged regions
  │
  ▼
Step 4: Per-cluster soft membership fields
  - Adaptive S: budget 500M pixels, min S=1 (>8MP), max S=2 (>4MP), up to 4 (small)
  - Nearest-two precomputation for O(1) per-cluster soft field
  - Dual-sigma Gaussian blur + edge blending
  - Upscaled via INTER_LINEAR
  │
  ▼
Step 5: Binary thresholding at adaptive iso
  - Squared-distance thresholds: 0.382 (thin) / 0.440 (non-thin)
  │
  ▼
Step 5b: Post-threshold cleanup
  - Morph close 3×3 for gap bridging
  - Morph open 3×3 ellipse for spike removal
  │
  ▼
Step 6: Contour extraction
  - cv2.findContours(RETR_CCOMP, CHAIN_APPROX_SIMPLE)
  - Hierarchy grouping: each outer + child holes → one evenodd SVG path
  - Dynamic MAX_GROUPS: 100 (K≥8) or 200 (K<8) per cluster, sorted by area
  - Micro-fragment filter: skip contours < 6 points
  │
  ▼
Step 7: Contour smoothing
  - Width-adaptive sigma: (0.6 + t*0.9) * S where t = width ratio
  - Light mode (S≤2): single heavy pass only
  - Full mode (S>2): curvature-adaptive multi-pass blend
  │
  ▼
Step 8: Artistic Bézier pipeline (_fit_contour)
  1. RDP simplification (cv2.approxPolyDP, epsilon=simplify_epsilon)
  2. fit_closed_bezier (corner-split section fitting, recursion depth max 8)
  3. reduce_nodes (tolerance=max_error * 2.5)
  4. _merge_short_curves (SHORT_THRESHOLD=12px)
  5. merge_segments_artistic (Potrace-style DP, tolerance=max_error * 2.0, MAX_MERGE=12)
  6. enforce_g1_continuity (cos(80°)=0.17 corner skip threshold)
  7. merge_segments_artistic (second pass, skipped if >200 segments)
  8. enforce_g1_continuity (final pass)
  - 2-second time budget per cluster (polygon fallback after timeout)
  │
  ▼
Step 9: SVG generation
  - Painter's algorithm: lightest cluster first, darkest last
  - optimize_svg_colors: 3 iterations (FULL mode only)
```

## Known Issues

- **Gap slivers** between adjacent color regions (iso overlap helps but not solved)
- **Tonal fidelity** on photographic images — subtle color gradients lost with low K
- **Width error** on some images (see per-image notes in `kb-per-image.md`)

## Test Images

| Image     | Resolution | Subject       | Key Character                       |
| --------- | ---------- | ------------- | ----------------------------------- |
| Ref.png   | 1536×1024  | Floral logo   | Line art fast path, grayscale       |
| test2.jpg | 4016×2256  | McLaren car   | Automotive paint, reflections       |
| test3.jpg | 6124×4082  | Botanical ink | Fine ink stems, high res            |
| test4.jpg | 3310×2481  | Aerial forest | Dense texture, warm color challenge |
| test5.jpg | 3888×2592  | Street mural  | Detail/texture loss, high Extra%    |
| test1.jpg | 4719×2303  | Antique map   | Full mode only, very slow           |
