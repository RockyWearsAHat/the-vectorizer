# Research Queue — Prioritized Hypotheses

Source: VECTORIZATION_RESEARCH_REPORT.md, project history, metric analysis, svg-vectorization-research.md

## How to Use This File

1. Pick the top-priority hypothesis with status=READY
2. Check kb-what-failed.md to confirm the approach hasn't been tried
3. Read the cited evidence before implementing
4. After testing, move finding to kb-what-works.md or kb-what-failed.md
5. Update status here to DONE or BLOCKED

---

## Priority 1: High Expected Impact

### H-RM1: Area region merging post-K-means (research-backed)

- **Status**: READY
- **Hypothesis**: K-means produces micro-fragments that survive into the contour pipeline and drive up node count. Iteratively merging adjacent connected components where `min(area_i, area_j) × ‖color_i - color_j‖² < λ` will collapse these without visual quality loss.
- **Evidence**: He et al. 2024 (arXiv 2409.15940) — measured 2418 K-means fragments → 151 clean regions after area merging (16× reduction). This is the direct root cause of test4's 123K nodes.
- **Target**: test4 nodes should drop materially (currently 123K — 5× all other images). May also reduce test5's 54K.
- **Algorithm**: After K-means labels, build connected-components per label. For each pair of adjacent small CCs, compute merge score. Merge if score < λ. Apply iteratively until convergence.
- **Risk**: MEDIUM — new post-processing step, but non-destructive (an additional merge pass after existing K-means).
- **Reference**: `.github/knowledge/svg-vectorization-research.md` §"Stage 1: Area/Scale Region Merging"
- **Anti-confusion**: This is region merging on K-means OUTPUT, not a replacement for K-means.

### H-RM4: Pre-RDP staircase removal (VTracer-verified)

- **Status**: READY
- **Hypothesis**: Diagonal or jagged edges in the binary soft-field mask produce characteristic pixel-stepping signatures (alternating signed area). Removing this staircase BEFORE RDP simplification frees the curve fitter from spending control points trying to smooth it.
- **Evidence**: VTracer uses signed-area staircase detection as a preprocessing step before all curve fitting. Contributes to their low node counts relative to other vectorizers.
- **Target**: Node count reduction across all images. Especially relevant where contour boundaries have diagonal transitions.
- **Algorithm**: Iterate the polygon, detect runs where sgn(cross(p[i-1],p[i],p[i+1])) alternates in a characteristic staircase pattern. Collapse those runs to a straight segment.
- **Risk**: LOW — purely pre-simplification preprocessing, can be toggled. Does not affect downstream pipeline.
- **Reference**: `.github/knowledge/svg-vectorization-research.md` §"Stage 4"

### H1: Abutting path generation (replace painter's algorithm)

- **Status**: READY (prior naive attempt FAILED — see kb-what-failed.md §Abutting)
- **Hypothesis**: Painter's algorithm causes overlap-dependent gap slivers and over-expansion. Abutting paths (each pixel → exactly one path) would eliminate both.
- **Why prior attempt failed**: Used soft-field clipping which breaks painter's overlap assumption. Need polygon-level clipping (Sutherland-Hodgman or Weiler-Atherton) to construct shared edges.
- **Expected impact**: Smaller SVG, no gap slivers, better WdErr (no expansion overlap)
- **Evidence**: Adobe Image Trace's "Abutting" mode produces this. See VECTORIZATION_RESEARCH_REPORT.md §2.
- **Risk**: HIGH — fundamental architecture change. Requires new SVG construction, not a parameter tweak.
- **Pre-check**: Read VECTORIZATION_RESEARCH_REPORT.md §2 fully before starting.

### H2: Visvalingam-Whyatt post-processing pass

- **Status**: READY (never tried)
- **Hypothesis**: After Bézier fitting, V-W can remove visually insignificant nodes that the merge pass misses, reducing node count 10-20% without quality loss.
- **Expected impact**: Node reduction (test4 has 123K nodes — too many)
- **Evidence**: VECTORIZATION_RESEARCH_REPORT.md §3. V-W never creates self-intersections, complementary to RDP.
- **Risk**: LOW — additive post-processing, easy to gate/revert.

### H3: Per-image metric bottleneck targeting

- **Status**: READY
- **Hypothesis**: Each image has ONE dominant quality bottleneck. Fixing that single bottleneck moves metrics more than scattered global tuning.
- **Current bottlenecks** (from kb-per-image.md + baselines):
  - Ref: Extra% 27.6 — fill geometry over-expansion, needs different line-art construction
  - test2: WdErr +25.19 — large positive width error, features expanding
  - test3: Xtra% 14.4, WdErr +12.11 — saturated cluster overlap/expansion
  - test4: Nodes 123K — excessive fragmentation from dense texture
  - test5: Feat% 83.6 — lowest feature presence, texture/detail loss
- **Approach**: Use the diagnostic profiling prompt to identify the exact pipeline step causing each bottleneck, then target that step specifically.

### H4: DP penalty metric change (max-deviation instead of mean-SSD)

- **Status**: READY (prior DP work used SSD; max-deviation not tried)
- **Hypothesis**: Current DP optimal polygon uses mean-SSD penalty which doesn't match RDP's max-deviation behavior. Switching to max-deviation would let DP work directly on denser contours.
- **Expected impact**: Node reduction without quality loss (would close the gap between RDP and DP approaches)
- **Evidence**: kb-what-failed.md §Curve fitting dead ends — "Direct DP on raw contours... Need a fundamentally different penalty metric"
- **Risk**: MEDIUM — math change to DP, but isolated to curve_fitting module.

---

## Priority 2: Medium Expected Impact

### H-RM2: Affine shortening flow for boundary smoothing (research-backed)

- **Status**: READY
- **Hypothesis**: Current Gaussian smoothing (sigma=0.5) smears contour corners and applies uniform smoothing. Affine shortening flow (ASF) ∂C/∂t = κ^(1/3)·N dampens curvature proportionally, preserving corners while eliminating pixelation better than Gaussian.
- **Evidence**: ASF is affine-invariant and preserves convex hull corners. κ^(1/3) (cube root) means high-curvature points are dampened LESS aggressively than with mean-curvature flow κ·N. This is explicitly what LIVE paper uses for smooth clean boundaries.
- **Target**: Reduce WdErr and Extra% on Ref and test3 (where over-expansion of smooth curves is the primary bottleneck).
- **Algorithm**: Discretize as: `C_new[i] = C[i] + dt * κ[i]^(1/3) * N[i]` where κ = discrete curvature, N = inward normal. Time step dt = 0.1–0.2. Run 2–5 iterations with T ≤ 1.0 total.
- **Risk**: MEDIUM — replaces smoothing step, not additive. Test on Ref first (shortest pipeline).
- **Pre-check**: Read `.github/knowledge/svg-vectorization-research.md` §"Stage 3: Contour Smoothing" before implementing.

### H5: Saliency-guided detail allocation

- **Status**: NEEDS RESEARCH
- **Hypothesis**: Allocate more nodes/clusters to foreground/salient regions, aggressively simplify background.
- **Expected impact**: Better perceived quality at same or lower node count
- **Evidence**: VECTORIZATION_RESEARCH_REPORT.md §8
- **Risk**: MEDIUM — requires some form of saliency estimation

### H6: Variable-width stroke detection for thin features

- **Status**: READY (partial stroke support exists for line art; not for photo thin features)
- **Hypothesis**: Thin curved features (test3 ink stems, test5 mural lines) would render more faithfully as stroked centerline paths with detected width variation.
- **Expected impact**: Better WdErr for thin features, fewer nodes
- **Evidence**: VECTORIZATION_RESEARCH_REPORT.md §7

---

## Priority 3: Longer-term / Higher Effort

### H7: SLIC superpixel → RAG quantization

- **Status**: READY (research complete — see svg-vectorization-research.md)
- **Hypothesis**: Replace K-means with SLIC superpixels + Region Adjacency Graph merge for more spatially coherent initial regions. The key insight from He et al. 2024: K-means can produce 2418 fragments vs 151 from RAG-based merging.
- **Algorithm**: `skimage.segmentation.slic(image, n_segments=400, compactness=10)` → build RAG → merge adjacent regions with ΔE < 15 in LAB. This produces region-consistent boundaries required for clean SVG paths.
- **Risk**: HIGH — replaces the entire quantization stage, would need parallel validation to avoid regressing mean-shift benefit. Try only AFTER H-RM1 and H-RM4 are validated.
- **Evidence**: `.github/knowledge/svg-vectorization-research.md` §"Stage 1"

### H8: DiffVG differentiable post-optimization

- **Status**: NEEDS GPU (research complete — infeasible on CPU)
- **Hypothesis**: Use differentiable rasterizer for gradient-based path refinement
- **Evidence**: `.github/knowledge/svg-vectorization-research.md` §"DiffVG". Requires GPU — not viable for this CPU pipeline unless done as optional post-pass on completed SVG.

---

## DONE (moved from this file)

- ✅ Potrace optimal polygon step → kb-what-works.md (3.5-6.3% node reduction)
- ✅ Shape detection (circles, ellipses, rectangles) → kb-what-works.md (conservative thresholds)
- ✅ Levien optimal Bézier fitting → kb-what-failed.md (downstream merge coupling)
- ✅ Mean-shift prefiltering → kb-what-works.md (biggest single improvement, test2 +10pp)
- ✅ 4-point subdivision → kb-what-works.md (consistent improvement all images)
- ✅ H-RM3 (inflection-point splice detection) → kb-what-failed.md (June 2025 — 3 variants all failed; post-RDP contours are too short; VTracer's pre-simplification timing cannot be replicated in current architecture)

## RULES

- Always read the relevant §section in svg-vectorization-research.md or VECTORIZATION_RESEARCH_REPORT.md before starting
- After testing, move finding to kb-what-works.md or kb-what-failed.md
- Status values: READY | NEEDS RESEARCH | IN PROGRESS | BLOCKED | DONE
