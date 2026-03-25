# What Works — Proven Architecture & Techniques

## 4-point interpolatory subdivision (June 2025)

- **\_subdivide_4point_closed**: After RDP simplification, insert midpoints between non-corner vertices using the 4-point scheme (Dyn, Levin, Gregory 1987, ω=1/16).
- Original vertices preserved (interpolatory). Corners and their immediate neighbors skipped.
- Provides denser, geometrically optimal input for Bézier fitter in smooth sections.
- Consistent improvement across ALL 5 images: Ref Extra% -3.4pp, test2 MnDif -2.27, test3 Feat% +0.9, test4 Miss% -0.8/MnDif -1.30, test5 WdErr -2.12.
- Tradeoff: 1-10% more nodes (test3 has the biggest increase).
- Simple midpoint insertion (ω=0) also helps but slightly less than 4-point.
- Placed in \_fit_contour between cv2.approxPolyDP and fit_closed_bezier.

## Core architecture (validated, don't replace)

- **Soft field approach** beats label-mask contour tracing (~0.99 vs ~0.86 SSIM)
- **Absorbed pipeline** beats multiband (704KB/48K nodes vs 4MB/287K nodes, same visual quality)
- **Painter's algorithm** (lightest first, darkest last) for SVG layering
- **CCOMP hierarchy** contour grouping (outer + child holes → one evenodd path)
- **K-means in LAB space** with CHROMA_WEIGHT=2.0

## Upscaling & soft fields

- **4× upscale is the sweet spot** — 8× marginal and inconsistent
- **V1 (upscale soft field)** >> V2 (upscale distance maps) — faster and better
- **Halo is essential** — single iso (no halo) is terrible even at 8×
- **Nearest-two precomputation** for O(1) per-cluster soft field
- **INTER_LINEAR** (not cubic) — faster, negligible quality difference
- **Dual-sigma smoothing**: crisp=0.30*S near edges, smooth=0.55*S in flat areas

## Contour & curve pipeline

- **Geometric shape detection** (circles, ellipses, rectangles) replaces near-geometric contours with SVG primitives. Conservative thresholds: circle>0.92, ellipse>0.90, rect>0.92, aspect ratio<5:1. Only outer contours without holes in non-thin clusters. Skips line art fast path.
- **Corner-split section fitting** replaced Catmull-Rom tangents (better results)
- **Curvature-adaptive multi-pass smoothing** (heavy 1.5× + light 0.5× with curvature blend)
- **Potrace-style DP merge** (merge_segments_artistic) with MAX_MERGE=12
- **DP optimal polygon refinement** on RDP output (bounds 4-800 vertices, lam=ε²×2.5) — 3.5-6.3% node reduction, <1% Feature% loss. Vectorized closing segment. Direct DP on raw contours FAILED (increases nodes due to dense point spacing).
- ~~**G1 continuity enforcement** with cos(80°)=0.17 corner skip~~ — DEPRECATED: post-fit G1 now confirmed harmful (narrows features). Do NOT re-enable.
- **Morph open 3×3** replaces CC spike removal (ms vs 249s per image)
- **reduce_nodes curve promotion** (June 2025): Detects line-segment runs that trace curves using (a) chord deviation > 0.2 SVG units and (b) total turning angle > 0.25 rad (~14°). When detected, re-fits the run with near-zero line_tolerance (0.05) to force Bézier curve production. Runs of 3+ nearly-collinear lines are collapsed to a single line. This dramatically improves width fidelity: test2 WdErr +8.75→+0.12.

## Bézier fitter fixes (June 2025 — critical)

- **Alpha clamp 1.5× chord length**: `max_alpha = max(seg_len * 1.5, 1.0)`. Prevents control point explosion (spike artifacts). Fixed the fundamental fitter.
- **Tangent span increased to 3** (`_compute_tangent` max_k=3): Looks at 3 neighbors instead of 2. Dramatically reduces tangent jitter from pixel-grid noise.
- **line_tolerance reduced 1.2→0.5**: Old 1.2 was converting gently-curving contour sections to straight lines (L), cutting off the slight outward bulge that preserves feature width. At 0.5, more sections become cubics (C). Ref Feature% +1.8%, test2 WdErr +0.73px, MnDif improved across images.
- **simplify_epsilon reduced 1.5→1.0**: Finer RDP keeps more contour data points for better Bézier fitting. Requires FIXED fitter (alpha clamp + tangent span) — with old fitter this caused instability. test2 WdErr +2.3px, Ref WdErr halved (+2.19→+1.08), MnDif improved for test2/test3/test5. Don't go below 0.75 (Extra% triples).

## Dark cluster merge (June 2025)

- **Adaptive LAB threshold in \_merge_close_clusters**: Weber's law — JND is larger at low luminance, so very dark clusters look perceptually identical despite having LAB ΔE > 6. Modified threshold: `pair_thresh = lab_threshold * max(1.0, 2.0 - avg_L / 25.0)`. At L=3 (near-black): threshold doubles to 12. At L=25: threshold is 6 (normal). At L>50: threshold is 6.
- **Impact on test2**: WdErr -14.64→+7.81 (median +1.04), Feature% 97.3→97.5, Miss% 0.9→0.6, nodes -25%, SVG -24%. Dark background gray patches eliminated.
- **No effect on other images** — only test2 had multiple near-black clusters.
- **Note**: \_gradient_aware_merge was NOT changed. The default boundary_contrast_thresh=22 handles the remaining dark cluster merge. Only \_merge_close_clusters needed the adaptive threshold.

## Performance techniques (pure Python + NumPy + OpenCV)

- Gaussian blur replaces bilateral filter (10-100× faster)
- Chunked vectorized K-means via matrix multiply trick
- cv2.distanceTransform replaces scipy EDT (10× faster)
- 2-second time budget per cluster with polygon fallback
- Dynamic K cap and S cap based on image size
- Bézier recursion depth capped at 8, second merge skipped if >200 segments
- **Large image optimizations (>4MP):** skip hard_edge_confidence, precompute edge_weight_up outside cluster loop, flat iso (no local iso map), sequential processing when upscaled >50MP, S capped at 3 for >30MP, 4s time budget per cluster, 300-group fitting cap. Took test4 from 447s→33s.

## K-means tuning

- K-means quality matters enormously — 3 attempts vs 6 can cause 20pp Feature regressions
- **Mean-shift prefiltering** (`cv2.pyrMeanShiftFiltering(sp=8, sr=16, maxLevel=1|2)`) before K-means: flattens uniform regions while preserving edges → more coherent clusters. Biggest single improvement: test2 +10.4pp Feature. Skip for >12MP. Use maxLevel=2 for >6MP.
- K cap 16 for >4MP; allows more color diversity for complex images
- CHROMA_WEIGHT=2.0 helps separate chromatic colors (esp. yellow)
- Iters: 5 for >4MP, 8 for >2MP, 10 otherwise; attempts: 6 for >4MP, 8 otherwise

## Line art fast path (Ref.png)

- Detection: mean_saturation < 20, K ≤ 6, background > 70%
- Hysteresis thresholding (strict+lenient) is the best approach
- Optimal: strict=min(otsu\*0.82, 145), lenient=min(otsu, 170)
- NO contour smoothing (binary threshold contours are clean)
- Tight Bézier params (epsilon*0.2, max_error*0.3)
- crispEdges rendering improves Feature by ~4pp
- **STROKE_HW_MAX=1.5** is the sweet spot — 3.0 causes massive Extra% from stroke/fill overlap (1100+ strokes overlapping 450+ fills). 1.0 eliminates strokes but loses hairline fidelity.
- **DT_ERODE=0.3** — enough to prevent dark-on-dark overlap without shrinking fills too much. 1.2 kills Feature%.
- **Fill mask must use `_binary_eroded`** not `_binary_la` — the un-eroded mask causes Extra% inflation.
- **Tighter line-art contour fitting** after restoring the 1.5px stroke cap materially improves Ref. Using line-art fill fitting at `epsilon*0.14` and `max_error*0.22` plus a smaller contour area floor (`max(S*S, 8)`) recovered Ref feature coverage and far-edge detail in blind visual review better than the looser `0.20 / 0.30` setting.

## Intensity-based fringe split for line art (March 2026)

- The old DT-based erosion (`_dt_line > 0.3`) was a **complete no-op** — minimum DT at native resolution is always ≥ 0.96, so NO pixels were ever removed. The fringe layer was always empty.
- **Fix**: Replace DT erosion with intensity-based strict/fringe split. Strict mask (pixels ≤ strict_thresh) → full opacity core fill. Hysteresis additions (strict..core) → partial opacity fringe.
- **Strict threshold**: `min(otsu * 0.90, 155)` instead of `min(otsu * 0.82, 145)`. Tested 4 points: 0.82 (too light), 0.87 (good), **0.90 (optimal MnDif)**, 0.92 (re-bloating).
- **Fringe opacity**: 0.55 — matches average visual weight of AA pixels (gray 140-165 on white background).
- **Key discovery**: Upscale the strict and fringe masks separately, so contour tracing happens at 4× for both.
- **Results**: Ref Extra% **46.8 → 27.6** (-19.2pp!), WdErr **+0.98 → +0.04** (nearly perfect), MnDif **5.72 → 5.22** (-0.50), Feat% 91.8 → 90.6 (-1.2pp acceptable).
- **Visual**: Blind visual review confirmed hand-traced quality with appropriate stroke weights — the previous version was "too heavy/bold."

## Gradient detection

- 2-stop and 3-stop linear gradients supported
- Helps pixel accuracy (disabling regresses test5)
- Only runs for images < 3MP currently
- For `test5`-like mural skin tones, **small warm secondary-component radial fallback** works better than forcing everything through a linear ramp. Combined with scored region-to-path matching, it visibly reduces hand posterization without regressing `test2` in quick screening.

- For reproducible validation, **disable OpenCV threading and process clusters sequentially**. This removed severe run-to-run instability on `test2` (same command previously swung between ~83% and ~96% Feature; now stable at 95.4% across repeated runs).
- For yellow-dominant photos like `test4`, treat large saturated fill clusters as fills, not thin features, and relax their base iso slightly (`-0.008`). This recovered autumn coverage without regressing `test2` or `test5` when gated by image hue composition.
- **Cluster-aware contour-retention relaxation** (March 2026): raising per-cluster `MAX_GROUPS` for larger clusters and relaxing fragment-area thresholds for large/texture-rich clusters recovered real structure on the hard images. Current accepted effect: `test1` **89.3 -> 93.4 Feat**, `test4` **90.8 -> 91.9 Feat**, `test5` **82.2 -> 83.0 Feat** while reducing `test5` Extra **9.0 -> 8.4** after adding a separate mural-width control gate.

- **Broader warm-hue preservation in local color estimation** (`_render_color_from_samples`) helps mural-like images more than red-only protection. Expanding the warm-family gate to `hue <= 45 or hue >= 170` improved `test5` from **83.0 -> 83.6 Feat** and slightly reduced mean diff (**10.90 -> 10.88**) without regressing `test2`.

- `optimize_svg_colors` should be treated as a low-saturation cleanup pass, not a universal post-process. Gating it off for high-saturation images (`sat_frac > 0.25`) avoids severe regressions on `test3` and improves the saturated photo set overall.
- For high-saturation mural-like photos such as `test5`, conservative large-group detail overlays inside very large high-variance contour groups can improve internal mural fidelity without broad regressions. Current accepted effect: `test5` held **82.2 / 3.2 / 9.0** while improving `MnDif` from **10.59 -> 10.46**.

## Theoretical ceiling

- Pixel-perfect 5-color: SSIM 0.9985
- 1-pixel boundary shift: SSIM 0.977
- Remaining gap is from Bézier smoothing of pixel boundaries
