# Current Baselines (source of truth)

Last updated: 2026-current (S*6 + borderline-fix: `_dk_is_dark_cluster = _rc_gray < _dov_dt`, `_light_union >= _dov_dt`)

## Default mode (compare_all.py)

| Image | Res       | Feat% | Miss% | Xtra% | WdErr   | MnDif | Time | Nodes   | SVG_KB |
| ----- | --------- | ----- | ----- | ----- | ------- | ----- | ---- | ------- | ------ |
| Ref   | 1536×1024 | 99.2  | 0.6   | 42.6  | +0.39   | 5.29  | 14s  | 25,392  | 540    |
| test2 | 4016×2256 | 99.8  | 0.2   | 1.8   | +54.55  | 10.16 | 43s  | 19,592  | 1182   |
| test3 | 6124×4082 | 98.6  | 0.6   | 20.8  | +16.16  | 1.59  | 62s  | 14,680  | 794    |
| test4 | 3310×2481 | 99.2  | 1.3   | 3.7   | +35.89  | 14.18 | 47s  | 116,718 | 6069   |
| test5 | 3888×2592 | 99.3  | 0.3   | 10.9  | +18.53  | 9.67  | 52s  | 50,229  | 2273   |

### What changed from previous baselines
- **Feat% improvements**: +8.6pp Ref, +3.1pp test2, +9.4pp test3, +7.4pp test4, +10.5pp test5
- **Borderline cluster fix**: Changed `_rc_gray <= _dov_dt` to `_rc_gray < _dov_dt` so clusters at exactly dov_dt (e.g. k6=138 for test2) are treated as light clusters and get unrestricted overlay instead of zone-restricted. This fixed test2 from 94.5%→99.8%.
- **S*6 zone**: `_bz_px = max(4, S*6)` for dark-cluster overlay zone (vs prior S*4). Benefits test3 thin ink.
- **Gradient thresholds loosened**: `_fit_thresh: max(20.0, ep*0.40)`, `_texture_thresh: max(8.0, ep*0.50)`, `_min_spatial_corr: 0.25/0.25/0.55`. Allows painted gradients (test5) through.

### Previous baseline (2026, dark feature overlay)

### Previous baseline (March 2026, pre-routing)

| Image | Res       | Feat% | Miss% | Xtra% | WdErr  | MnDif | Time | Nodes   | SVG_KB |
| ----- | --------- | ----- | ----- | ----- | ------ | ----- | ---- | ------- | ------ |
| Ref   | 1536×1024 | 90.6  | 5.6   | 27.6  | +0.04  | 5.22  | 30s  | 10,266  | 184    |
| test2 | 4016×2256 | 97.2  | 0.7   | 2.5   | +25.19 | 10.36 | 35s  | 20,828  | 448    |
| test3 | 6124×4082 | 87.3  | 4.8   | 14.4  | +12.11 | 1.62  | 51s  | 15,786  | 282    |
| test4 | 3310×2481 | 91.6  | 2.4   | 2.3   | +2.56  | 13.31 | 44s  | 123,296 | 2428   |
| test5 | 3888×2592 | 83.6  | 3.1   | 6.8   | +6.48  | 10.09 | 53s  | 53,930  | 1181   |

### Pre-subdivision baselines (for reference)

| Image | Feat% | Miss% | Xtra% | WdErr  | MnDif | Nodes   | SVG_KB |
| ----- | ----- | ----- | ----- | ------ | ----- | ------- | ------ |
| Ref   | 91.8  | 4.8   | 50.2  | +1.00  | 5.90  | 7,682   | 119    |
| test2 | 97.2  | 0.8   | 3.3   | +29.34 | 11.96 | 22,584  | 501    |
| test3 | 86.7  | 5.3   | 14.5  | +11.69 | 1.67  | 14,971  | 273    |
| test4 | 90.7  | 4.8   | 2.3   | +1.77  | 16.74 | 130,040 | 2570   |
| test5 | 83.7  | 3.1   | 6.9   | +9.10  | 10.42 | 57,436  | 1287   |

### Key changes from current baseline (2026 dark overlay update)

- **Dark feature overlay** (`_dark_overlays` in `MultilevelResult`): Post-processing step added AFTER standard path generation. For photographic images, identifies clusters where `render_center_gray > dark_thresh` AND `dark_frac ∈ [0.04, 0.80]` AND `dark_n ≥ 500`. Creates binary masks of dark pixels per qualifying cluster, extracts contours, generates simplified polygon paths with the dark-pixel BGR median as fill. Paths are painted LAST (top z-order) in `generate_svg`. This corrects "fills present but too light" misses without touching the soft field (no centroid competition). **test5 Feat% +6.0pp, test3 +2.5pp, test4 +0.9pp. Zero regressions.**

### Key changes from previous baseline (2026 routing overhaul)

- **Image classification routing** (`_classify_image`): line_art | flat_color | photographic routing at top of `multilevel_vectorize`. Safe flat_color thresholds: `mean_grad<0.03 AND edge_frac<0.04 AND n_unique<30`. test3 (sparse botanical ink) correctly routes to photographic despite low-gradient appearance.
- **GrabCut background removal** (`_remove_background_grabcut` + `_fg_by_flood_fill`): opt-in via `remove_background=True`, exposed in API and frontend toggle. Uses iterative GrabCut with flood-fill seed for transparent background extraction.
- **Flat-color fast pipeline** (`_pipeline_flat_color`): for logos, seating maps, diagrams. K-means → binary masks → cv2.findContours → Bézier fitting. No soft-field computation (major speed gain). Triggers on `mean_grad<0.03 AND edge_frac<0.04 AND n_unique<30`.
- **Area merge** (`_area_merge_clusters`): He et al. 2024 LAB-space merge. Merges clusters where `min(area_i,area_j) × LAB_dist² < lambda_ × total_area × 100`. lambda_=0.002 (photographic), 0.0005 (line_art). Conservative — 0.62s for 25MP. Reduces cluster count for fragmented images.
- **Adaptive simplify_epsilon scaling**: For clusters with >300 groups, `_se = simplify_epsilon × min(2.5, N/300)`. Reduces nodes per contour WITHOUT filtering any contours → doesn't hurt WdErr. **test3 WdErr 12.11→11.16 (-0.95), test5 WdErr 6.48→5.01 (-1.47)**. Minor Feat% regressions (-0.6 to -0.8pp across test3/4/5).
- `_curve_to_d` coordinate precision: `.1f` format with trailing-zero stripping (saves SVG bytes).
- test2/4 minor Feat% regression (-0.5 to -0.7pp) from area_merge Step 1f — tradeoff for WdErr improvements.

### Key changes from pre-subdivision baseline

- Bézier fitter now produces REAL cubic curves (alpha clamp 1.5x, tangent span=3)
- line_tolerance: 1.2 → 0.5 (more curves, fewer straight lines)
- simplify_epsilon: 1.5 → 1.0 (finer contour detail)
- Dark cluster merge: adaptive LAB threshold in \_merge_close_clusters (Weber's law)
- Feature% improved +2-7% across all images
- test2: WdErr -14.64→+7.81 (median only +1.04), Feature% 97.5%, Miss% 0.6%, nodes -25%, SVG -24%
- Node counts ~30-50% higher due to C commands being longer than L

## Historical progression (for context)

- Early baselines: test2=91.7%, test3=76.0%, test5=68.2% (fast mode, different params)
- After K-cap 12→16, chroma 1.5→2.0: test2=98.7%, test3=84.7%, test5=79.0%
- After further tuning: test2 regressed in Feat% but improved MnDif.
- Commit 780778d (March 2026): workspace cleanup + pipeline overhaul committed. Ref commit: 96fb465.
- After fix round (March 2026): Ref Extra% 73.4→59.4 (DT-eroded fill mask, stroke HW cap 3→1.5), test4 447s→33s (\_large_image flag, precomputed edge_weight_up, sequential >50MP), test3 84→89.7 (max_k restored to 10 for >16MP). K-cap fix was the biggest lever for test3.
- Shape detection (March 2026): conservative thresholds (circle>0.92, ellipse>0.90, rect>0.92). Metrics essentially flat. Shapes detected: test2=109, test3=66, test4=109, test5=490 (mostly ellipses). No visual regression.
- Naive abutting partition map (March 2026): removed soft-field overlap by assigning each pixel to exactly one cluster. MASSIVE regression — test4 Feat% 90.9→79.7, test2 92.1→88.2. Reverted. The painter's algorithm requires overlap; abutting needs a smarter boundary approach.
- Mean-shift prefiltering (March 2026): `cv2.pyrMeanShiftFiltering(sp=8, sr=16, maxLevel=1|2)` before K-means. test2 Feat% 85→95.4 (+10.4pp!), test4 WdErr +7.6→+2.5, test5 WdErr +27→+20.5. Skipped for >12MP (test3). Biggest single improvement in project history.
- Large saturated fill guard + yellow-dominant iso relax (March 2026): prevents `test4` warm autumn fills from being misclassified as thin features, improving `test4` from 88.1/6.1/2.4 to 90.2/3.8/2.5 in targeted validation and 90.2/3.8/2.5 in the full default run. Ungated version regressed `test5` Extra and should not be used broadly.
- Content-aware `optimize_svg_colors` gate (March 2026): skip the post-pass for high-saturation images (`sat_frac > 0.25`). This removed a false catastrophic regression on `test3` (44.6→87.0 Feature with `--optimize-colors`) and improved `test4` 90.2→90.5 while keeping Ref's low-saturation color optimization benefit.
- Large-group detail overlays for saturated mural regions (March 2026): add one or two conservative internal detail overlays inside large high-variance contour groups. On `test5`, this kept the improved structure metrics (**82.2 / 3.2 / 9.0**) while further improving `MnDif` (**10.59 -> 10.46**). Ref remained unchanged, so this is a safe photo-side refinement.
- **Iso tuning + reduce_nodes curve promotion (June 2025)**: Non-thin iso 0.40→0.42, thin iso 0.44→0.45, large-cluster iso guard (-0.005 for >5% area), MAX_MERGE 10→12, safety-valve 80→50 segments, second-merge-pass threshold 200→300, optimize_svg_colors always-on (self-gates on sat_frac>0.25). In curve_fitting: `reduce_nodes` now uses turning-angle detection (>0.25 rad) and chord deviation (>0.2) to force-promote line-segment runs into cubic Bézier curves with near-zero line_tolerance. Smoothing sigma reduced for thinner features. **Biggest win: test2 WdErr +8.75→+0.12 (99% improvement), test5 WdErr +12.94→+5.32 (59% improvement).** Ref MnDif 8.75→6.03 (31% improvement). Total nodes -8K across all images.
- **Bézier fitter overhaul + parameter tuning (June 2025)**: Alpha clamp 1.5×, tangent span 2→3, line_tolerance 1.2→0.5, simplify_epsilon 1.5→1.0. All images Feature% improved +2-7%. Ref MnDif 6.03→5.50. test2 WdErr temporarily regressed to -14.64 (fixed by dark cluster merge below).
- **Dark cluster merge (June 2025)**: Adaptive LAB threshold in \_merge_close_clusters: `pair_thresh = lab_threshold * max(1.0, 2.0 - avg_L / 25.0)`. Merges near-black clusters that humans can't distinguish. test2: WdErr -14.64→+7.81 (median +1.04), Miss% 0.9→0.6, nodes 30,892→23,274 (-25%), SVG 1163→885KB (-24%). Other images unchanged.

## Metric definitions

- **Feat%** = % of original dark features present in SVG (higher = better)
- **Miss%** = % of original features missing from SVG (lower = better)
- **Xtra%** = % of SVG dark area not in original — over-expansion (lower = better)
- **WdErr** = mean stroke width difference in px (closer to 0 = better)
- **MnDif** = mean pixel difference across entire image (lower = better)

## UPDATE RULES

- After any compare_all.py run that changes the code, update this file with new numbers
- Keep the historical progression section to track trends
- Note what change caused the shift
