# SVG Vectorization Research Synthesis

Research compiled from: arXiv 2409.15940, Potrace DeepWiki, LIVE CVPR 2022, DiffVG, VTracer GitHub, community discussion.

**Purpose**: Before implementing any new algorithm, read the relevant section here. This is the "what do the best generators do?" knowledge base.

---

## Core Insight: Where Quality Comes From

The best SVG generators share a common structure:

1. **Clean region quantization** → meaningful color regions, not K-means micro-fragments
2. **Stable boundary extraction** → pixel-perfect boundaries that can be smoothed without losing structure
3. **Topology-aware curve fitting** → split at natural topology (corners, inflections) not just error maxima
4. **Post-processing without information loss** → merge/simplify in a way that preserves visual intent

The gap between "acceptable" and "hand-traced quality" is almost always in steps 1 and 3.

---

## Stage 1: Region Quantization & Merging

### The fragmentation problem (He et al. 2024 — arXiv 2409.15940)

K-means on natural images produces ~**2418 fragments** (connected components of the same label). After area/scale region merging: **151 clean regions**. This 16× reduction is the core contribution of their approach.

**Four merging criteria** (applied in priority order):

1. **BG (Background)**: Merge small regions that are likely background noise — regions touching the frame edge with area < threshold.
2. **MS (Mean-shift)**: If two adjacent regions have a mean-shift distance below threshold, merge them (they belong to the same perceptual color).
3. **Scale (Size guard)**: Merge tiny regions (area below min_size) into their dominant neighbor by color similarity.
4. **Area**: The key merge — `min(area_i, area_j) × ‖color_i - color_j‖² < λ`. A _large_ region can absorb a _small_ region if they're color-similar. This is what collapses isolated micro-fragments.

**Why this matters for SVG-gen**:

- test4's 123K nodes (vs 10K-54K for all others) are directly caused by K-means fragmentation surviving into the contour pipeline
- Area merging would operate AFTER K-means, BEFORE contour extraction — it's a post-quantization cleanup pass
- Implementation: OpenCV `cv2.connectedComponentsWithStats` per label → adjacency graph → iterative merge

**Practical λ for Area merge**: He et al. use λ tuned per-image, but a good starting point is λ = (image_area × 0.002) × max_lab_distance², where max_lab_distance is ~15 for adjacent colors.

---

## Stage 2: Contour Extraction

### What Potrace does differently

Potrace uses an **optimal polygon** as an intermediate representation between the pixel bitmap and the final Bézier curves. Key properties:

1. Trace the pixel boundary path (8-connected)
2. Find the **optimal corner placement** using dynamic programming — corners are placed at points of maximum deviation, not at arbitrary polygon vertices
3. Fit Bézier curves to the polygon segments (not directly to pixel edges)

The polygon intermediate is why Potrace curves look "cleaner" than direct pixel-to-Bézier approaches. The DP ensures corners are placed at geometrically meaningful locations.

**Our pipeline uses a similar Potrace-style DP merge** (`merge_segments_artistic`) — this is correct.

### VTracer's stacking approach

VTracer uses a **stacking/layering** strategy (similar to our painter's algorithm) where each color layer is traced independently and stacked in z-order. The key difference: VTracer builds a **color palette hierarchy** by luminance and stacks lightest-first exactly as we do. Validation: our approach is architecturally sound.

---

## Stage 3: Contour Smoothing

### Why Gaussian smoothing has limits

Gaussian smoothing applies uniform smoothing to all boundary points. This:

- Rounds corners (bad — corners should be preserved)
- Doesn't distinguish between noise and real curvature
- Has no awareness of curvature magnitude

### Affine Shortening Flow (ASF) — the better approach

ASF evolves the curve according to: **∂C/∂t = κ^(1/3) · N**

Where:

- κ = discrete curvature at each point
- N = inward unit normal
- (1/3) exponent is the key — it's the affine-invariant exponent

**Why κ^(1/3) is better than Gaussian**:

- κ^(1/3) dampens high-curvature points LESS than κ (mean curvature flow). Sharp corners survive because their high κ value is cube-root-dampened.
- Affine invariance means the result is independent of parameterization
- Eliminates pixelation (oscillating curvature from pixel staircase) while preserving genuine corners

**Discrete implementation**:

```python
def affine_shortening_step(pts, dt=0.15):
    """One step of affine shortening flow on a closed polygon."""
    n = len(pts)
    new_pts = pts.copy()
    for i in range(n):
        p_prev = pts[(i-1) % n]
        p_curr = pts[i]
        p_next = pts[(i+1) % n]
        # Discrete curvature via cross product
        v1 = p_curr - p_prev
        v2 = p_next - p_curr
        cross = v1[0]*v2[1] - v1[1]*v2[0]  # scalar 2D cross product
        len1 = np.linalg.norm(v1) + 1e-8
        len2 = np.linalg.norm(v2) + 1e-8
        kappa = abs(cross) / (len1 * len2 * (len1 + len2) / 2)
        # Inward normal
        normal = np.array([-(p_next[1]-p_prev[1]), p_next[0]-p_prev[0]])
        normal = normal / (np.linalg.norm(normal) + 1e-8)
        # ASF update: kappa^(1/3) in normal direction
        new_pts[i] = p_curr + dt * (kappa ** (1/3)) * normal
    return new_pts
```

**Recommended parameters**: dt=0.1–0.2, 2–5 iterations, total T ≤ 1.0. More than 5 iterations starts rounding real corners.

**Where to insert**: Replace or augment `_smooth_contour` in `multilevel/__init__.py`. The current Gaussian sigma=0.5 operates on image space; ASF operates on polygon vertices.

**Caveat**: Line art contours should NOT use ASF (same reason they don't use Gaussian smoothing — binary threshold contours are already clean).

---

## Stage 4: Pre-Bézier Preprocessing

### Staircase removal (VTracer technique)

Pixel boundaries on diagonal or gently-curved edges produce characteristic **staircase artifacts**: alternating left-right or up-down steps.

**Detection**: The staircase signature in a polygon is a run of points where the **signed area** of consecutive triangles alternates sign in a regular pattern.

```python
def detect_staircase(pts):
    """Find runs of staircase steps using alternating signed area."""
    signs = []
    n = len(pts)
    for i in range(1, n-1):
        cross = np.cross(pts[i] - pts[i-1], pts[i+1] - pts[i])
        signs.append(np.sign(cross))
    # Staircase: alternating [1,-1,1,-1,...] or [-1,1,-1,1,...]
    # Find runs of alternating sign ≥ 3
    ...
```

**Resolution**: Replace the staircase run with a straight line or fewer points.

**Why this matters**: Staircase bumps cause the Bézier fitter to use control points compensating for noise rather than representing actual shape. Removing them pre-simplification reduces the "surface area" the fitter needs to cover, potentially reducing node count.

**When to apply**: BEFORE `cv2.approxPolyDP` (RDP), on the raw OpenCV contour points.

### Inflection-point splitting (VTracer — pre-simplification ONLY)

**Important**: We tried inflection-point splitting POST-simplification (June 2025) and it failed completely. See kb-what-failed.md.

VTracer detects inflection points on the **dense polygon** (before simplification) and splits there. After RDP, contours are too short for meaningful inflection detection.

**If trying this**: Apply it BEFORE `cv2.approxPolyDP`, on raw contour points from `cv2.findContours`. The inflection point is where the curvature sign changes — the curve transitions from convex to concave. Splitting here creates naturally-flowing sub-curves.

---

## LIVE: Progressive Closed-Path Encoding (CVPR 2022)

LIVE (Layered Image Vectorization Engine) represents images as a set of **closed parametric paths** defined by LIVE primitives (path + fill). It builds the SVG progressively:

1. Start with a solid background color
2. Iteratively add closed paths using differentiable rendering
3. Each new path minimizes the remaining pixel error

**Key insight**: The "artist's approach" — lay down broad fills first, then add detail on top. Our painter's algorithm follows the same principle.

**Why LIVE doesn't directly apply here**: LIVE uses GPU (differentiable renderer). Its quality on detailed images with many colors is lower than Potrace/VTracer. Best for stylized or logo-like content.

**What we can learn**: The progressive layering principle is sound. Our painter's algorithm (light → dark) implements this correctly.

---

## DiffVG: Differentiable SVG Rendering

DiffVG (Li et al. 2020) makes SVG rendering differentiable, enabling gradient-based optimization of path parameters directly against a raster input.

**Core capability**: Given an initial SVG approximation, DiffVG can refine control point positions, colors, and opacities by backpropagating the pixel loss.

**Why it's not viable for SVG-gen current pipeline**:

- Requires GPU (PyTorch CUDA)
- Slow per-image (minutes not seconds)
- Works best on simple shapes (logos, illustrations), not photorealistic content
- The gradient landscape for 100K+ control points is highly non-convex

**Potential future use**: As an optional post-processing step for single-image quality maximization, not for the batch pipeline. Could refine a completed SVG output further on GPU hardware.

**Status**: NEEDS GPU — tracked as H8 in research queue.

---

## Key Techniques Summary (actionable for SVG-gen)

| Technique                                     | Source         | Applicable?                                 | Status                   |
| --------------------------------------------- | -------------- | ------------------------------------------- | ------------------------ |
| Area region merging post-K-means              | He et al. 2024 | YES — directly addresses test4 123K nodes   | H-RM1, READY             |
| Pre-RDP staircase removal                     | VTracer        | YES — low risk, pre-simplification          | H-RM4, READY             |
| Affine shortening flow smoothing              | LIVE/geometry  | YES — can replace Gaussian smoothing        | H-RM2, READY             |
| Inflection-point splitting pre-simplification | VTracer        | POSSIBLE (different from June 2025 attempt) | See kb-what-failed §note |
| SLIC + RAG quantization                       | He et al. 2024 | YES — but higher effort/risk                | H7, READY                |
| Potrace optimal polygon DP                    | Potrace        | DONE — already in pipeline                  |                          |
| Painter's algorithm layering                  | VTracer/LIVE   | DONE — already in pipeline                  |                          |
| Differentiable optimization                   | DiffVG         | NOT viable (requires GPU)                   | H8, blocked              |

---

## Critical Pipeline Bottlenecks (diagnosis from research)

1. **test4 123K nodes** → K-means fragmentation. Area merging (H-RM1) is the directly confirmed fix. Evidence: 2418→151 fragments measured with identical algorithm family.
2. **test2 WdErr +25.19** → Soft field over-expansion on automotive paint reflection clusters. Not a fragmentation issue — a boundary-width issue. Affine shortening (H-RM2) might help by tightening the soft boundary representation.
3. **test3 Xtra% 14.4, WdErr +12.11** → Saturated cluster over-expansion. The cluster soft fields overlap more for high-saturation colors. Staircase removal (H-RM4) won't help here. Area merging (H-RM1) might if small satellite clusters are expanding.

---

## References

- He et al. (2024). "Vectorizing Raster Images with Perfect Precision." arXiv 2409.15940.
- Selinger, P. Potrace algorithm. DeepWiki documentation + original paper.
- Ma et al. (2022). "LIVE: Towards Layer-wise Image Vectorization." CVPR 2022.
- Li et al. (2020). "Differentiable Vector Graphics Rasterization for Editing and Learning." SIGGRAPH Asia 2020.
- VTracer source: github.com/visioncortex/vtracer
