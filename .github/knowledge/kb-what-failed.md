<!-- June 2025 — Bézier fitter session -->
- **Taubin smoothing on contour points** (λ=0.5/μ=-0.53, 1-3 iterations): Meant to smooth pixel-grid noise without shrinking. Even 1 iteration degraded Feature% by 2-4% and worsened MnDif. At 3 iterations: Ref Feature% 94→90.9, test2 MnDif 8.27→14.31. Root cause: any input point movement hurts metrics because pixel boundaries ARE the ground truth for comparison.
- **Taubin with gentle params** (λ=0.15, μ=-0.16, 2 iter): Marginal — Feature% -0.4%, MnDif flat. Not worth the complexity.
- **G1 continuity enforcement** (post-fit `enforce_g1_continuity`): Moves control points to align tangents at joins. Catastrophic — Feature% -3.7% for Ref, test2 WdErr -17.66→-22.20. The weighted average tangent reduces curve bulge, systematically narrowing features.
- **Shared tangent at split points** (computing bidirectional tangent and passing to both sub-sections): Even worse than G1 post-processing. Feature% -3.7%, WdErr degraded 7px, node counts INCREASED 30%. The split point is where data maximally deviates from the curve, so a compromise tangent worsens both sub-fits.
- **Increasing max_error from 1.5 to 2.0**: WORSENED WdErr across all images (test2: -17.66→-20.16). Longer Bézier segments narrow features MORE because they have more freedom to deviate inward from the polygon.
- **Relaxing line art params** (epsilon 0.14→0.30×, max_error 0.22→0.45×): Feature% dropped 4.2% for Ref. Any relaxation of line art fitting loses detail in thin strokes. The current 0.14×/0.22× multipliers are well-tuned.
- **simplify_epsilon=0.75**: Too fine. Ref Feature% dropped 5%, test5 Extra% tripled to 22%. Causes instability in contour processing.
- **Line art contour smoothing** (_smooth_contour on line art paths): Ref Feature% 95.6→93.8, Miss% 3.2→8.2. Even very light smoothing (sigma=0.72 at S=4) smears thin text strokes and fine flower details. Binary threshold contours ARE already clean — smoothing only hurts. Same conclusion as prior KB entry: NO smoothing for line art.


<!-- June 2025 additions -->
- **SVG gap-sliver stroke**: Adding `stroke="same-color" stroke-width="X"` to fill paths to eliminate white gaps between adjacent regions. Problem: SVG viewBox is at working resolution (1/S of original), so any SVG-unit stroke-width gets multiplied by S when rendered at full resolution. Even 0.15 SVG units at S=4 = 0.6px, causing +4px WdErr regression. `vector-effect="non-scaling-stroke"` not supported by cairosvg. NOT VIABLE for this pipeline's coordinate system.
- **Uniform contour subsampling for Bézier fitting**: Replacing RDP with `np.linspace` subsampling of smoothed contour. CATASTROPHIC — test2 WdErr -35.72 (collapsed), MnDif 40.30. Uniform subsampling doesn't preserve the shape because OpenCV contour points are unevenly distributed (dense in curves, sparse in straights).
- **Reducing simplify_epsilon (RDP) to 1.2-1.3**: Caused instability with OLD fitter (before alpha clamp + tangent span fixes). At 1.3: test5 Extra% exploded to 22.2%. At 1.2: test2 WdErr flipped to -6.93. ⚠️ With FIXED fitter (alpha clamp 1.5x, tangent span=3), epsilon=1.0 works WELL — see kb-what-works.md. But 0.75 is still too aggressive (Extra% triples).
- **Stronger chroma_iso_adj/K_iso_adj**: Increasing chroma adjustment 0.015/0.0005→0.018/0.0006 and K adjustment 0.02/(K-8)*0.0025→0.025/(K-7)*0.003 caused test5 Extra% to explode to 22.9%. These relaxations are too aggressive.


# What Failed — DO NOT RETRY without new evidence

Each entry: what was tried → what happened → why it failed.

## Parameter tuning dead ends

- **iso=0.37** → test5 Feature drops 80%→42%. Catastrophic.
- **iso=0.45** → causes visible gap slivers between regions
- **iso=0.42** → over-expands features. Current 0.44/0.50 is the sweet spot.
- **max_k 14→18** → REGRESSES all images (cluster fragmentation hurts feature presence)
- **chroma weight 2→3** → REGRESSES test5 (-1.5% Feature, +1.38 MnDif)
- **Disabling gradient detection for high-K** → REGRESSES test5

## Architecture dead ends

- **Label-mask contour tracing** → much worse than soft field (~0.86 vs 0.99 SSIM)
- **Multiband pipeline** → 4MB SVG, 287K nodes, same quality as absorbed (704KB, 48K)
- **SVG feGaussianBlur filter** → hurts quality significantly. Don't use it.
- **Single iso (no halo)** → terrible even at 8× upscale
- **8× upscale** → marginal over 4×, inconsistent across images
- **INTER_CUBIC upscaling** → no quality improvement over LINEAR, just slower
- **Bilateral filter** → 10-100× slower than Gaussian, no quality benefit (quantization dominates)
- **scipy distance_transform_edt** → 10× slower than cv2.distanceTransform, same result

## Curve fitting dead ends

- **Direct DP on raw contours** (skipping RDP) → DP keeps MORE vertices than RDP because raw contour points from CHAIN_APPROX_SIMPLE are dense (~1px apart) and even gentle curves have enough SSD to justify segments. Tested with lambda 2.0, 6.0: both produced 20-70% MORE nodes than RDP→DP. Need a fundamentally different penalty metric (max-deviation instead of mean-SSD) to make direct DP work.
- **Per-edge Catmull-Rom tangents** → corner-split section fitting is better
- **Single-sigma Gaussian smoothing** → curvature-adaptive multi-pass is better
- **Raw polygons (skip curve fitting)** → same quality, but larger SVG
- **Cluster splitting (CC-based and variance-based)** → unreliable, often regresses

## Line art dead ends

- **Soft field label collapse** → doesn't work (soft field recomputes from centers)
- **Adaptive thresholding** → too much noise even with C=20 (Extra >120%)
- **Global Otsu alone** → loses ~25% of features through contour pipeline
- (Hysteresis thresholding was the winner — see kb-what-works.md)
- **Excluding stroke CCs from fill mask** (line art path) → Feature crashes 87→70%. Strokes only cover thin lines; connected components span entire features.
- **DT_ERODE=1.2** (aggressive) → Feature crashes to 65.6%. Fills shrink too much.
- **Disabling line art fast path entirely** → Feature crashes to 20.5%. Standard pipeline cannot handle line art at all.
- **STROKE_HW_MAX=1.0** (no strokes) → Extra% great (38.5%) but Feature only 86.9%. Very thin lines (hairlines) need stroke representation.

- **Per-component adaptive erosion on Ref line-art fills** (trying to create a real fringe ring only on thick fill components) → Extra% improved dramatically (**52.0 -> 26.8**) but Feature% collapsed (**85.7 -> 53.5**). Root cause: the current Ref problem is not a missing 1px fringe tweak; eroding the surviving fill geometry removes real structure. Future Ref work needs a different fast-path construction, not another erosion variant.

- **Global line-art core threshold lift on Ref** (`_core_thresh` cap `165 -> 175`) → recovered some missing coverage (**Feat 81.4 -> 82.8, Miss 12.2 -> 11.1**) but reintroduced the swollen baseline failure: **Extra 47.4 -> 58.0**, `WdErr +0.76 -> +1.04`, `MnDif 6.21 -> 6.61`. If Ref edge structure is missing, do not solve it with a global lenient-core expansion.
- **Broad-fill opacity gating on Ref** (global and component-selective) → useful for reducing center-lower heaviness, but visual analysis rejected it because it either created gray underfill in dense black regions or traded the old darkness issue for more visible right-side structure loss. Keep broad Ref fills structurally solid unless a future gate is explicitly per-component and visually validated.

## Iso tuning dead ends (large images)

- **iso base=0.37 for all images** → helps test3 Feature (+2pp) but regresses test5 Extra% (+5pp) and WdErr (+7px). The tradeoff is not worth it.
- **iso base=0.38 for all images** → test3 gains +1pp but test5 WdErr regresses +3px.
- **Edge-aware iso tightening (factor 0.12)** on large images → helps WdErr but hurts Feature% by 1-2pp across the board. Too aggressive.
- **Resolution-based iso adjustment** (lower iso for higher-res) → inconsistent, couples too many images together.

## Abutting / overlap removal dead ends

- **Naive partition map (pixel → cluster with highest soft membership)** → Used to clip per-cluster binary masks (remove pixels claimed by other clusters). Caused MASSIVE regressions: test4 Feat% 90.9→79.7, test2 85→88→back, test5 86→78. Root cause: the painter's algorithm RELIES on soft-field overlap for seamless rendering. Removing overlap creates gap slivers and undercuts feature coverage. An abutting approach requires fundamentally different SVG construction (abutting path edges, not clipped painter's layers).

## Merge/color dead ends

- **LAB ΔE guard in gradient merge** → ineffective (blocked wrong pairs, tripled node count)
- **gradient_aware_merge** is NOT the cause of test5 yellow loss (confirmed: 0 merges for test5)
- **_merge_close_clusters lab_threshold=6.0** → very tight already, not the problem

- **Ungated large-chromatic-fill thin override / iso relaxation** → helped `test4` but regressed mixed-color images (`test5` Extra 8.5→9.1, full-suite instability risk). The fix must be gated to yellow-dominant images rather than applied globally.
- **Universal `optimize_svg_colors` post-pass** → catastrophic on high-saturation images. Measured with `--optimize-colors`: `test3` collapsed 87.0→44.6 Feature while geometry remained intact. The pass must be gated by image content, not applied globally.

- **Threshold-only retries for `test5` hand gradients** (`min_region_pct` override, relaxed warm-region corr/texture thresholds, bright-to-dark fallback linear axis) → did not change the accepted gradient set. The hand candidate is being surfaced but still fails the linear ramp model itself. Further retries need a different gradient model, not looser thresholds.

## Levien area-preserving Bézier fit attempts

- **Levien as primary fitter (1.0× threshold)** → Node counts +0.1-0.4%. The area-preserving fit produces different curve shapes that merge WORSE in the downstream merge/G1 pipeline (which was tuned for Schneider-style curves).
- **Levien with relaxed threshold (1.5× max_error)** → Even WORSE — node counts +0.5-1.4%. Relaxed acceptance produces curves further from Schneider's, causing more merge failures.
- **Levien in merge step** → Added Levien as merge alternative. No benefit — merge stage already uses the same point-to-curve error metric that Schneider optimizes for.
- **Best-of-both (compute Levien + Schneider, pick lower error)** → Node counts +0.5-1.3%. When Levien has lower per-point error, the different curve geometry still disrupts downstream merge.
- **Levien for split-point guidance only** → Node counts +0.1-0.8%. Levien's error landscape identifies DIFFERENT max-error points than Schneider, but those points are worse for Schneider sub-fits.
- **Root cause**: The downstream merge/G1/merge pipeline is deeply coupled to Schneider-style curve geometry. Any change to accepted curve shapes (even ones with lower per-section error) disrupts merge patterns, causing net MORE nodes. Levien's theoretical advantage (area preservation) doesn't translate to the max point-to-curve metric used throughout the pipeline.
- **Function retained and integrated**: `_levien_fit_single` now runs as a hybrid best-of-both in `_fit_single_bezier_with_error` and for split-point guidance in the large-section path. Wins ~5-13% of fits but downstream merge coupling keeps net node counts +0-2.7%.
- **Narrow large-image thin-cluster stroke-mode re-enable** (test3-only gate on `h*w > 16MP`, `cluster_is_thin`, small-area clusters) → no measurable effect on `test3` metrics. Either the gate did not activate on the real error-driving clusters or fill broadening is happening earlier than the disabled stroke path.
- **Broad saturation-based fill-iso tightening** on all high-saturation images (`_sat_frac > 0.50`) → improved `test5` width/Extra but incorrectly pulled down `test3` Feature (**87.1 -> 86.2**). If this control is used, gate it to mural-like high-K images only.
- **Large-image dark low-chroma smoothing reduction** for supposed ink-like clusters (`h*w > 16MP`, low saturation, dark LAB gate) → no measurable effect on `test3`; metrics stayed **87.1 / 6.6 / 16.6 / +13.91 / 1.88**. The error is not coming from that late contour-smoothing branch.
- **Broader thin-cluster classification on very large low-K images** (`cluster_mean_thick < 3.2`, `interior < 0.38`) → no measurable effect on `test3`; metrics stayed **87.1 / 6.6 / 16.6 / +13.91 / 1.88**. The error-driving clusters were still not moving through a different extraction path.
- **Downscaled mean-shift prefilter for >16MP images before K-means** → materially changed `test3` geometry but was visually rejected. Stronger version reached **83.9 / 6.6 / 13.5 / +10.28 / 1.90** and a milder version reached **85.0 / 6.5 / 15.9 / +11.53 / 1.88**; blind visual comparison found no cleaner hand-traced result, only slight regression / equivalence. Do not pursue this prefilter direction without a different visual upside.


## _merge_short_curves in _fit_contour (March 2026)

- Function exists but was never called. Added to pipeline between reduce_nodes and merge_segments_artistic.
- Zero measurable effect on any image — node counts identical, metrics unchanged.
- Root cause: after 4-point subdivision densifies input, the recursive Bézier fitter produces segments >12px. The SHORT_THRESHOLD=12px never triggers.
- Not harmful, but not worth the overhead. Removed.

## Expansion-gated iso correction (March 2026)

- Attempted to tighten iso when soft field expansion exceeds 50-75% of label area.
- Inconsistent results: helped test2 WdErr (-3) and test4 WdErr (-2) but regressed test5 WdErr (+2) or Feature%.
- Background-only gating (gray>140) helped test2 but still regressed test5.
- Area-frac gating (≥3%) didn't help either.
- Root cause: painter's algorithm makes per-cluster iso correction unpredictable — tightening one cluster's iso exposes underpainting gaps from overlapping clusters.
- Reverted. Per-cluster iso correction is a dead end without fundamentally changing the stacking model.

## RULE: Before trying any parameter change, check this file first.

## Inflection-point splitting in curve fitter (June 2025)

- Tested 3 variants: (1) top-level injection in fit_bezier_path + _fit_closed_direct, (2) recursive fitter only (>60pt + error-driven redirect), (3) only >60pt inflection split.
- Variant 1: Ref Feature 95.6→94.1, test5 WdErr +3.61. Pixel-grid inflections in RDP polygon are noise.
- Variant 2: Ref 95.4/58.5, test2 WdErr +8.67. Moving splits from max-error points worsens both sub-fits.
- Variant 3: Identical to baseline — RDP already reduces contours below 60 points.
- Root cause: After RDP simplification, contours are too short for inflection detection to be meaningful. Inflection splitting only works on dense polygons (VTracer operates pre-simplification).
- **Note**: VTracer's inflection detection happens BEFORE simplification on the dense pixel contour. This distinction means a pre-simplification approach is theoretically different — but not yet tried. If trying this, it must be on the raw OpenCV contour points BEFORE approxPolyDP, not after.

## Boundary-clipped contour extraction (June 2025)

- AND soft field binary mask with dilated label mask to limit expansion. Tested 3 dilation radii.
- 1.5 native px: test2 WdErr -14.91, test4 Feature 88.1% (-4.6%), test5 Feature 76.8% (-7.7%). Catastrophic.
- 3.0 native px: test5 Feature 78.0% (-6.5%). Still terrible.
- 5.0 native px: test5 Feature 79.1% (-5.4%). Still terrible.
- Root cause: Painter's algorithm FUNDAMENTALLY REQUIRES soft-field overlap for seamless rendering. Any hard clipping creates underpainting gaps between clusters.


## If it's listed here, you need a fundamentally different approach, not a retry.
