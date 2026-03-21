# Raster-to-SVG Vectorization: Comprehensive Technical Research Report

**Goal**: Identify the best methods for converting raster images to SVG vector art that looks like a human artist created it in Adobe Illustrator — **not** pixel-perfect reproduction.

**Scope**: Algorithms and techniques implementable in Python with OpenCV, NumPy, SciPy, and scikit-image. Ranked by implementation impact on the current SVG-gen pipeline.

---

## Table of Contents

1. [Potrace and Classical Bitmap Tracing](#1-potrace-and-classical-bitmap-tracing)
2. [Adobe Image Trace — Reverse-Engineering the Gold Standard](#2-adobe-image-trace)
3. [Path Simplification: RDP, Visvalingam-Whyatt, and Beyond](#3-path-simplification)
4. [Schneider's Algorithm and Bézier Curve Fitting](#4-schneiders-algorithm-and-bézier-curve-fitting)
5. [Color Quantization: Mean-Shift, SLIC, and Perceptual Color Spaces](#5-color-quantization)
6. [Gradient Mesh, Linear Gradients, and Diffusion Curves](#6-gradient-and-smooth-shading)
7. [Variable-Width Strokes and Stroke Detection](#7-variable-width-strokes)
8. [Modern Optimization: DiffVG, LIVE, CLIPasso, VectorFusion, SVGDreamer](#8-modern-optimization-approaches)
9. [Post-Processing: Node Reduction, Path Merging, and Cleanup](#9-post-processing)
10. [The "Illustration Look": What Separates Professional Vector Art](#10-the-illustration-look)

---

## 1. Potrace and Classical Bitmap Tracing

### Algorithm Overview

Potrace (Peter Selinger, 2003) is the most widely-used open-source bitmap tracer, embedded in Inkscape, FontForge, and dozens of other tools. It operates on **binary (1-bit) images only** — all color handling must be done externally.

#### Potrace's 4-Step Pipeline

1. **Bitmap Decomposition**: Scan the binary image to find connected components of black pixels. Each connected region becomes a closed path. Nested white holes become inner contours. The `turnpolicy` parameter (-z) resolves ambiguities at checkerboard-pattern pixel corners (options: black, white, left, right, minority, majority, random).

2. **Optimal Polygon Finding**: For each closed path of pixel edges, Potrace finds the **optimal polygon** — the straightest possible polygon that follows the pixel grid. This uses dynamic programming to minimize a penalty function that balances segment count against deviation. This is the key innovation: instead of naively following pixel edges, it finds polygon vertices that produce the smoothest representation.

3. **Bézier Curve Fitting**: Each polygon edge is smoothed into a cubic Bézier curve or left as a straight line. The `alphamax` parameter (default 1.0, range 0–1.334) controls the corner threshold:
   - `alphamax = 0`: All corners preserved as sharp angles.
   - `alphamax = 1.334`: Maximum smoothing; only very sharp angles remain corners.
   - The magic number 1.334 ≈ 4(√2−1) represents the geometric limit where a right angle becomes indistinguishable from a curve.

4. **Curve Optimization**: An optional pass (`-O`, `opttolerance` default 0.2) merges adjacent Bézier segments when they can be represented by fewer segments within the tolerance. This dramatically reduces node count.

#### Key Parameters

| Parameter           | Default  | Purpose                                                          |
| ------------------- | -------- | ---------------------------------------------------------------- |
| `turdsize` (-t)     | 2        | Suppress speckles up to N pixels                                 |
| `alphamax` (-a)     | 1.0      | Corner detection threshold (0 = all corners, 1.334 = max smooth) |
| `opttolerance` (-O) | 0.2      | Curve optimization tolerance (higher = fewer nodes)              |
| `turnpolicy` (-z)   | minority | Ambiguity resolution at pixel intersections                      |

#### Strengths

- Produces extremely clean, minimal SVG output with very few nodes
- Polynomial-time optimal polygon step (not greedy)
- Well-tested on 20+ years of real-world usage
- No iterative optimization — deterministic and fast

#### Weaknesses

- **Binary only**: Requires external color decomposition. Each color layer must be thresholded separately and traced individually.
- **No anti-aliasing awareness**: Doesn't handle the soft boundaries our pipeline creates; expects hard binary masks.
- **No gradient support**: Each region is a single flat fill.
- **Single-pass**: No global optimization across all paths simultaneously.

#### Relevance to SVG-gen

Potrace is **not directly applicable** as a drop-in replacement because our pipeline works with soft membership fields, not binary masks. However, Potrace's **optimal polygon** step is a much stronger approach than our current marching-squares → RDP path, and its **curve optimization** pass (merging adjacent Bézier segments) is exactly what our output needs for node reduction.

**Actionable**: After thresholding our soft fields at iso=0.37, apply Potrace's polygon optimization logic before Bézier fitting. This would replace our "marching squares → raw contour → RDP simplify → Bézier fit" with "marching squares → optimal polygon → Potrace-style Bézier smoothing". Python binding: `pypotrace` (pip install pypotrace).

---

## 2. Adobe Image Trace

### Reverse-Engineering the Gold Standard

Adobe Illustrator's Image Trace (formerly Live Trace) is the industry standard for vectorization. While the source code is proprietary, the parameter panel reveals its architecture.

#### Image Trace Parameters (from Adobe's documentation)

**Mode**: Auto-Color, High Color, Low Color, Grayscale, Black and White, Outline

**Palette**:

- **Automatic**: Automatically switches between limited and full tone based on image
- **Limited**: Uses a small set of colors determined by the Colors slider (2–30)
- **Full Tone**: Uses the full spectrum of colors
- **Document Library**: Uses colors from a swatch group

**Advanced Panel**:
| Parameter | Range | Purpose |
|-----------|-------|---------|
| **Paths** | 0–100% | Tightness of path fit. Lower = looser/fewer nodes, Higher = tighter fit |
| **Corners** | 0–100% | Corner preservation. Higher = more corners, Lower = rounder |
| **Noise** | 1–100px | Area in pixels below which regions are ignored |
| **Method: Abutting** | — | Paths share edges; no overlap, no gaps |
| **Method: Overlapping** | — | Paths may overlap each other (simpler, fewer artifacts) |
| **Snap Curves To Lines** | on/off | Convert near-straight Bézier segments to true line segments |
| **Create: Fills** | on/off | Generate filled regions |
| **Create: Strokes** | on/off | Generate stroked paths (max width slider: 1–10px) |
| **Gradients** | on/off + smooth slider | Detect and generate SVG linear gradients |
| **Shapes** | on/off | Detect geometric primitives: circles, squares, rectangles |

#### Key Architectural Insights

1. **Abutting vs. Overlapping**: This is the most architecturally significant choice. Our pipeline uses overlapping (painter's algorithm — lighter first, darker last). Adobe offers both. **Abutting** produces cleaner, more professional results because:
   - No hidden overdraw → smaller file size
   - Each pixel belongs to exactly one path → no blending artifacts
   - Matches how a human artist works in Illustrator (select → divide → fill)
   - **Implementation**: After extracting all contour polygons, clip each polygon against its neighbors using computational geometry (Sutherland-Hodgman or Weiler-Atherton clipping).

2. **Gradient Detection**: Image Trace can detect linear gradients in regions that our pipeline posterizes into bands. The algorithm likely:
   - Identifies regions where K-means produces adjacent same-hue bands
   - Fits a linear color ramp across the merged region
   - Outputs `<linearGradient>` or `<radialGradient>` in the SVG

3. **Shape Detection**: Detecting circles, rectangles, and regular polygons transforms pixel blobs into clean geometric primitives. This is a huge part of the "illustration look."

4. **Stroke Detection**: Image Trace can output stroked paths with configurable width. This is critical for line art, text hairlines, and thin features.

5. **Snap Curves to Lines**: Converting near-straight Bézier segments to `L` commands. Our pipeline already does this with `line_tolerance=0.25`, but Image Trace likely has a more sophisticated detection that considers the visual difference.

#### What We Can Steal

**HIGH IMPACT** — Implement in priority order:

1. **Shape detection** (circles, rectangles) — replaces jagged pixel-traced shapes with perfect geometry
2. **Abutting path generation** — eliminates painter's algorithm artifacts
3. **Gradient detection** — merges banded regions into `<linearGradient>`
4. **Adaptive Noise slider** — our `min_contour_area=1` is too aggressive; need perceptual noise filtering

---

## 3. Path Simplification

### Ramer-Douglas-Peucker (RDP)

The Ramer-Douglas-Peucker algorithm recursively simplifies a polyline by removing points that deviate less than ε from the straight line between endpoints.

#### Algorithm

```
function RDP(points, ε):
    find point with maximum distance from line(first, last)
    if max_distance > ε:
        left = RDP(points[0..max_index], ε)
        right = RDP(points[max_index..end], ε)
        return left + right
    else:
        return [first, last]
```

#### Properties

- **Time**: O(n²) worst case, O(n log n) expected
- **Error**: The maximum deviation of any removed point from the simplified line is at most ε (guaranteed bound)
- **Behavior**: Preserves corners well (high-curvature points naturally have high deviation)
- **Problem**: Can create **self-intersecting** paths when ε is large, especially on narrow features
- **Problem**: Treats all points equally — doesn't consider perceptual importance
- **Our current usage**: `simplify_epsilon=0.05` in upscaled space (very conservative)

### Visvalingam-Whyatt

This algorithm iteratively removes the point that forms the smallest triangle area with its neighbors. More formally, for three consecutive points A, B, C, the effective area is the area of triangle ABC.

#### Algorithm

```
function Visvalingam(points, min_area):
    compute effective area for every internal point
    while smallest area < min_area:
        remove the point with smallest area
        recompute area for its surviving neighbors
    return surviving points
```

#### Properties

- **Time**: O(n log n) with a priority queue
- **Error**: No guaranteed maximum deviation bound (unlike RDP)
- **Behavior**: Produces visually smooth results because it prioritizes geometric significance
- **Problem**: Can erode fine detail — thin spikes and sharp corners may have small triangle areas but are visually significant
- **Advantage**: Never creates self-intersections (points are removed, not moved)

### Comparison and Recommendation

| Criterion              | RDP                                          | Visvalingam-Whyatt                 |
| ---------------------- | -------------------------------------------- | ---------------------------------- |
| Error bound            | Guaranteed ε                                 | No guaranteed bound                |
| Corner preservation    | Excellent                                    | Poor for sharp thin features       |
| Visual smoothness      | Can be jagged                                | Naturally smooth                   |
| Self-intersection risk | Yes, at high ε                               | No                                 |
| Implementation         | `cv2.approxPolyDP`                           | Custom or `simplification` package |
| Best for               | Contour simplification before Bézier fitting | Final SVG path optimization        |

**Recommendation for SVG-gen**: Use **RDP first** (at a very tight ε=0.02–0.05) to remove collinear noise points from marching squares output, then use **Visvalingam-Whyatt as a post-processing pass** on the final Bézier control polygon to remove visually insignificant nodes.

A hybrid approach: RDP for initial simplification (where guaranteed bounds matter), then Visvalingam-Whyatt as a perceptual polish pass.

---

## 4. Schneider's Algorithm and Bézier Curve Fitting

### The Fundamental Problem

Given a sequence of points (a digitized curve), find a sequence of cubic Bézier segments that approximates it within a given tolerance. This is the core of all vectorization tools.

### Schneider's Algorithm (Graphics Gems, 1990)

Philip Schneider's "An Algorithm for Automatically Fitting Digitized Curves" (Graphics Gems, 1990) is the most widely-implemented curve fitting algorithm. Nearly every vectorization tool (including Inkscape, Potrace's internals, and likely our own `fit_bezier`) derives from it.

#### Core Approach

1. **Estimate tangent directions** at the start and end of the point sequence using neighboring points.
2. **Parameterize** the points using chord-length parameterization: assign each point a $t$ value proportional to cumulative arc length.
3. **Least-squares fit** a single cubic Bézier to all points. The endpoints are fixed to the first and last data points. The two free parameters are the lengths of the control arms (distance from endpoint to control point along the tangent direction).
4. **Check error**: If the maximum deviation exceeds the tolerance, **split** the curve at the point of maximum error and recursively fit each half.
5. **Iterate**: Optionally re-parameterize (Newton-Raphson) to improve $t$ estimates and refit.

#### Mathematical Formulation

For a cubic Bézier $B(t) = (1-t)^3 P_0 + 3(1-t)^2 t P_1 + 3(1-t) t^2 P_2 + t^3 P_3$:

- $P_0$ and $P_3$ are fixed (first and last data points)
- $P_1 = P_0 + \alpha_1 \hat{t}_0$ (control point 1, along start tangent)
- $P_2 = P_3 - \alpha_2 \hat{t}_3$ (control point 2, along end tangent)
- Solve for $\alpha_1, \alpha_2$ via least-squares minimization of $\sum_i |B(t_i) - d_i|^2$

This reduces to solving a 2×2 linear system.

#### Known Problems

1. **Local minima**: For C-shaped curves, there are typically **three local minima** with very similar visual shape but different control arm lengths. Schneider's iterative refinement can get stuck in a suboptimal one.

2. **Split point sensitivity**: Splitting at the point of maximum error is a heuristic. The optimal split point may not be where the maximum error occurs.

3. **Tangent estimation**: Using finite differences on noisy data produces poor tangent estimates, leading to twisted curves. Better: use the direction of best-fit line through several neighboring points.

4. **Parameter sensitivity**: Chord-length parameterization is a good approximation but not optimal. The Newton-Raphson reparameterization step can diverge.

### Raph Levien's Optimal Fitting (2021)

Raph Levien (of Runebender/font design fame) published a fundamentally better approach to cubic Bézier fitting that addresses Schneider's weaknesses:

#### Key Insight: Area and Moment Matching

Instead of minimizing point-to-curve distance iteratively, match **signed area** and **x-moment** between the source curve and the Bézier. This:

- Reduces the problem to solving a **quartic polynomial** (closed-form, no iteration)
- **Finds all local minima** simultaneously (up to 4 solutions)
- Produces $O(n^6)$ error scaling (subdividing in half reduces error by 64×)
- Is deterministic and fast (no iterative convergence issues)

#### The Area-Preserving Property

For a cubic Bézier with chord-normalized coordinates:

$$\text{area} = \frac{3}{20}(2\delta_0 \sin\theta_0 + 2\delta_1 \sin\theta_1 - \delta_0 \delta_1 \sin(\theta_0 + \theta_1))$$

where $\delta_0, \delta_1$ are control arm lengths and $\theta_0, \theta_1$ are tangent angles.

This constrains the parameter space to a 1D curve. A second constraint (x-moment matching) then pins the solution to a discrete set of points.

#### Practical Significance

- **Font design**: Preserves ink area exactly, so stroke weight doesn't drift during simplification
- **For SVG-gen**: Bézier fitting that reliably finds the globally optimal curve (not just a local minimum), with guaranteed error bounds, and no iterative convergence problems
- **Implementation**: The algorithm is described in full detail at raphlinus.github.io and could be implemented in ~200 lines of Python using NumPy's polynomial root-finding

### Recommendation

**HIGH IMPACT**: Replace our current Schneider-derived recursive Bézier fitter with Levien's area-preserving optimal fitter. Benefits:

1. **Fewer nodes**: The optimal fit means each segment does more work → fewer subdivisions needed
2. **No convergence failures**: Closed-form solution, no iterative refinement
3. **Better corner detection**: The "near miss" detection handles ambiguous cases that Schneider's max-error splitting mishandles
4. **Deterministic**: Same input always produces same output (no random initialization sensitivity)

Estimated effort: 2–3 days to implement the quartic polynomial solver and integrate.

---

## 5. Color Quantization

### Current Approach: K-Means with PP-Center Initialization

Our pipeline uses K-means (OpenCV's implementation, 20 iterations, PP-centers) with auto-K estimation (LAB binning at 8 ΔE resolution, max K=48), followed by gradient-aware agglomerative merge.

**Strengths**: Fast, deterministic seed, handles well-separated color regions.
**Weaknesses**: Sensitive to K selection, doesn't respect spatial coherence, treats color space uniformly.

### Mean-Shift Clustering

Mean-shift is a non-parametric mode-finding algorithm that **automatically discovers K** by iterating kernel density estimation to find density peaks in feature space.

#### How It Works for Color Quantization

1. Operate in a **joint spatial-color domain**: each pixel is a 5D point $(x, y, L, a, b)$
2. Start a kernel window at each pixel's 5D position
3. Iteratively shift each window toward the local density maximum
4. Pixels that converge to the same mode belong to the same cluster

#### Key Parameters

- **Spatial bandwidth** $h_s$: radius in pixel space (typically 10–30). Controls spatial coherence — pixels within this radius influence each other.
- **Color bandwidth** $h_r$: radius in color space (typically 10–30 in LAB). Controls color precision — colors within this range are considered "same."

#### Advantages Over K-Means

- **No K required**: The number of clusters emerges from the data
- **Spatially aware**: Nearby pixels are more likely to cluster together, producing spatially coherent regions (fewer noisy "salt and pepper" artifacts)
- **Handles arbitrary cluster shapes**: Unlike K-means' spherical assumption, mean-shift finds arbitrarily-shaped density peaks
- **Preserves edges**: The spatial bandwidth naturally respects object boundaries

#### Disadvantages

- **Slow**: O(n² × iterations) for n pixels. A 1536×1024 image has 1.6M pixels — direct mean-shift is impractical.
- **Bandwidth selection**: Wrong $h_s$ or $h_r$ causes over/under-segmentation
- **Implementation**: `cv2.pyrMeanShiftFiltering(src, sp=20, sr=30)` does mean-shift _smoothing_ (not clustering), but it can be used as a preprocessing step before K-means.

#### Practical Recommendation

**Use mean-shift filtering as a preprocessing step**, not as the primary clustering:

```python
# Mean-shift prefilter: spatial=15, color=25
filtered = cv2.pyrMeanShiftFiltering(img_bgr, sp=15, sr=25, maxLevel=1)
# Then K-means on the filtered image
```

This preserves edges while flattening smooth regions — greatly reducing K-means sensitivity to noise and gradients. Adobe Image Trace almost certainly does something equivalent.

**Impact**: MEDIUM. Would reduce gradient posterization artifacts and improve cluster stability.

### SLIC Superpixels

SLIC (Simple Linear Iterative Clustering, Achanta et al., 2012) is a superpixel segmentation algorithm that partitions the image into roughly-equal-sized compact clusters.

#### Algorithm

1. Initialize cluster centers on a regular grid with spacing $S = \sqrt{N/k}$
2. For each center, search a $2S \times 2S$ neighborhood
3. Assign pixels to nearest cluster using a weighted distance:

$$D = \sqrt{\left(\frac{d_c}{m}\right)^2 + \left(\frac{d_s}{S}\right)^2}$$

where $d_c$ is CIELAB color distance, $d_s$ is spatial distance, and $m$ is the compactness factor (typically 10–40).

4. Recompute cluster centers. Iterate 5–10 times.

#### For Vectorization

SLIC superpixels aren't directly useful as final color regions (too many clusters, too uniform in size). But they're excellent as an **intermediate representation**:

1. Compute SLIC superpixels (N=1000–4000)
2. Build a **region adjacency graph (RAG)** where each superpixel is a node
3. Merge adjacent superpixels by color similarity (agglomerative clustering on the RAG)
4. Result: spatially coherent, edge-respecting color regions with adaptive sizes

**Implementation**: `skimage.segmentation.slic()` + `skimage.future.graph.rag_mean_color()` + `skimage.future.graph.merge_hierarchical()`

**Impact**: MEDIUM-HIGH. SLIC→RAG merging produces more perceptually meaningful regions than K-means, especially for images with gradients and textured areas. The spatial coherence eliminates scattered noise clusters.

### CIELAB Distance and Perceptual Uniformity

Our pipeline converts to LAB for auto-K estimation but does K-means in BGR space. This is suboptimal because:

- BGR Euclidean distance is **not perceptually uniform**: a distance of 30 in green looks very different from 30 in blue
- CIELAB was designed so that equal distances correspond to equal perceived color differences
- $\Delta E_{ab}$ (Euclidean distance in LAB) approximates perceptual difference. JND (just noticeable difference) ≈ 2.3 $\Delta E$
- **CIEDE2000** ($\Delta E_{00}$) is even more perceptually accurate, adding corrections for lightness, chroma, and hue weighting

**Recommendation**: Switch K-means and all merge thresholds to operate in **CIELAB space** using $\Delta E_{ab}$ distances. Change merge threshold from "BGR distance < 80" to "CIELAB ΔE < 15" (approximately equivalent but more perceptually consistent).

**Impact**: MEDIUM. More consistent behavior across different color palettes.

---

## 6. Gradient and Smooth Shading

### The Gradient Problem

Our current pipeline represents every region as a flat fill color. When the source image contains smooth gradients (sky fades, soft lighting, metallic reflections), K-means quantizes these into visible bands — the classic "posterization" artifact.

### SVG Gradient Primitives

SVG 1.1 natively supports:

1. **`<linearGradient>`**: Color interpolation along a line between two points
2. **`<radialGradient>`**: Color interpolation radiating from a center point
3. Both support multiple `<stop>` elements with positions and colors

These are universally supported by all SVG renderers and very compact to represent.

### Detecting Gradient Regions

**Algorithm to detect and replace banded regions with gradients:**

1. **Identify candidate regions**: After K-means, find sequences of adjacent clusters where:
   - Hue is within 15° in CIELAB
   - Lightness varies monotonically
   - Spatial extent forms a roughly convex region

2. **Fit a linear gradient**:
   - Compute the principal axis of the merged region (PCA of pixel positions)
   - Project all pixel colors onto this axis
   - Fit a linear color ramp by least-squares: `color(t) = start_color × (1-t) + end_color × t`
   - If residual error (max ΔE deviation from ramp) < 5.0, accept the gradient

3. **Generate SVG gradient**:

   ```xml
   <linearGradient id="g1" x1="10%" y1="0%" x2="90%" y2="100%">
     <stop offset="0%" stop-color="#aabbcc"/>
     <stop offset="100%" stop-color="#ddeeff"/>
   </linearGradient>
   <path d="..." fill="url(#g1)"/>
   ```

4. **Multi-stop gradients**: For complex fades, add intermediate `<stop>` elements at 25%, 50%, 75% positions.

**Implementation**: ~200 lines of Python. Requires: PCA (numpy), contour merging, and SVG gradient element generation.

**Impact**: HIGH for gradient-heavy images. Eliminates posterization entirely. No impact on flat-color images.

### Diffusion Curves

Diffusion curves (Orzan et al., 2008, INRIA/Adobe) are a vector primitive where you specify colors on **both sides** of a curve, and the colors diffuse outward from the boundaries. This is conceptually beautiful — instead of filling regions, you define the edges and let color propagate.

**SVG compatibility**: Not part of SVG 1.1 or SVG 2.0 spec. Would require a custom renderer. **Not recommended for our pipeline** — too far from the SVG standard.

### Gradient Mesh

Gradient mesh (used heavily in Adobe Illustrator) subdivides a region into a grid of patches, each with independently specified colors at grid points, interpolated smoothly across the patch.

**SVG compatibility**: SVG 2.0 proposed `<mesh>` element, but browser support is essentially zero. **Not recommended for our pipeline** unless targeting Illustrator-only output.

### Practical Recommendation

Focus on **`<linearGradient>` detection and generation** — it's universally supported, compact, and addresses the most visible artifact (banded skies, soft shadows). Radial gradients are a bonus for centered light sources.

---

## 7. Variable-Width Strokes

### The Problem

Our pipeline represents everything as filled paths. Thin features (pen strokes, text hairlines, wire outlines) end up as very narrow filled regions with two contours — an outside and inside edge. This produces:

- Double the nodes needed (two parallel paths instead of one centerline + width)
- Fragile geometry (thin fills can collapse or gap at rendering)
- No stroke caps or joins (filled paths have sharp ends, not round/butt)

### Stroke Detection Algorithm

1. **Compute the medial axis / skeleton**: For each cluster's binary mask, compute the morphological skeleton using `skimage.morphology.skeletonize()` or distance-transform-based thinning.

2. **Measure local width**: At each skeleton pixel, the distance transform value gives the half-width of the original feature. A stroke is a skeleton branch where width is roughly constant and below a threshold (e.g., < 5px in source resolution).

3. **Classify features**:
   - Width < 3px at source resolution → **stroke candidate**
   - Width > 5px → **filled region** (trace as filled path)
   - 3–5px → heuristic: if width variance < 20%, treat as stroke; otherwise filled

4. **Fit the centerline**: Apply Bézier curve fitting to the skeleton points (not the contour).

5. **Generate SVG stroke**:
   ```xml
   <path d="M... C..."
         stroke="#333"
         stroke-width="2.5"
         stroke-linecap="round"
         stroke-linejoin="round"
         fill="none"/>
   ```

#### Variable Width

SVG 1.1 doesn't support variable-width strokes. Options:

- **Constant-width approximation**: Use the median width along the stroke. Good enough for most cases.
- **SVG 2.0 `stroke-width` profile**: Proposed but not widely supported.
- **Fill path with tapering**: For calligraphic/brush strokes, keep as filled paths but with optimized node count.
- **Multiple overlapping constant-width segments**: Break a varying-width stroke into segments of roughly constant width. Practical for gradual tapers.

### Implementation

```python
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt

mask = (labels == cluster_id).astype(np.uint8)
skeleton = skeletonize(mask > 0)
dist = distance_transform_edt(mask)
widths = dist[skeleton]  # width at each skeleton pixel

# Branch extraction: connected components of skeleton
# For each branch: mean width, variance, length
# If mean_width < threshold and cv(width) < 0.2: → stroke
```

**Impact**: HIGH for line art, text, and technical drawings. MEDIUM for photographic content. Halves the node count for thin features and produces more natural-looking output.

---

## 8. Modern Optimization Approaches

### DiffVG — Differentiable Vector Graphics Rasterization

**Paper**: Li et al., SIGGRAPH Asia 2020 (MIT CSAIL + Adobe Research)
**Code**: github.com/BachiLi/diffvg

#### What It Does

DiffVG is a **differentiable SVG rasterizer**: given SVG parameters (Bézier control points, colors, stroke widths), it renders the image AND computes gradients of a loss function back through the rendering process to the SVG parameters. This enables:

- **Image vectorization by optimization**: Start with random Bézier curves, iteratively adjust parameters to minimize pixel-wise loss with the target image
- **Perceptual loss functions**: Use deep features (VGG loss, LPIPS) instead of pixel MSE
- **Painterly rendering**: Fit random brush-stroke Bézier curves to match an image aesthetically

#### Architecture

1. **Forward pass**: Rasterize SVG to pixels using analytical prefiltering (for speed) or multisampling anti-aliasing (for accuracy)
2. **Loss computation**: Compare rendered image to target using chosen loss function
3. **Backward pass**: Compute gradients ∂Loss/∂(control points, colors, opacity, stroke width) via automatic differentiation
4. **Update**: Adam optimizer steps on SVG parameters

#### Key Results

- 128 random Bézier paths → recognizable painterly rendering in ~200 optimization steps
- Can optimize 256 paths with 4 control points each to match a target image
- Enables editing SVG using raster operations (e.g., seam carving on SVG via differentiable rendering)

#### Relevance to SVG-gen

DiffVG is **complementary** to our pipeline, not a replacement:

1. **Post-optimization**: After our pipeline generates SVG, use DiffVG to fine-tune control point positions by optimizing pixel-wise or perceptual loss. This could fix subtle boundary misalignments.
2. **Color optimization**: Replace our iterative `optimize_svg_colors` (which is hacky: render → sample → replace) with DiffVG gradient-based color optimization — more principled and can optimize all colors simultaneously.
3. **Node budget enforcement**: Use DiffVG to re-optimize with a fixed number of paths/nodes, automatically distributing detail where it matters most.

**Implementation barrier**: DiffVG requires PyTorch and a C++/CUDA extension. Not trivial to integrate with our NumPy/OpenCV stack. Could be used as an optional post-processing step.

**Impact**: HIGH for quality, MEDIUM-HIGH effort. Best as a final polishing pass.

### LIVE — Layerwise Image Vectorization

**Paper**: Ma et al., CVPR 2022 Oral
**Core Idea**: Progressive, layer-by-layer path addition with differentiable rendering.

#### Algorithm

1. Start with an empty canvas
2. **Add one path** at a time:
   - Initialize the new path's position/shape/color using a component-wise strategy (e.g., place it where the current error is largest)
   - Optimize the new path's parameters (4 Bézier control points + RGBA color) via backpropagation through DiffVG
   - **Simultaneously re-optimize all existing paths** — this is crucial; it prevents early paths from "locking in" suboptimal positions
3. Repeat until error threshold is met or budget is exhausted

#### Key Insight

Adding paths one-at-a-time with global re-optimization produces dramatically more compact SVGs than DiffVG's "initialize all paths randomly" approach:

- LIVE needs only **~5 paths** for simple icons (DiffVG needs 256)
- Produces **semantically meaningful layers** (background, major shape, detail, highlight)
- Natural layer ordering mimics how an artist builds up an illustration

#### Relevance to SVG-gen

LIVE's progressive approach mirrors how our pipeline works (layers ordered lightest-to-darkest). The key difference is LIVE optimizes all layers globally after each addition. We don't re-optimize existing layers.

**Actionable**: Rather than adopting LIVE wholesale (it requires DiffVG + GPU), borrow the concept of **re-optimizing colors after each layer is added**, not just at the end. Our `optimize_svg_colors` runs 3 iterations at the end; instead, run 1 iteration of color optimization after each cluster's path is generated.

### CLIPasso — Semantically-Aware Object Sketching

**Paper**: Vinker et al., 2022 (SIGGRAPH)

#### What It Does

CLIPasso generates **sketches** of objects — minimalist Bézier curve drawings that capture the essential visual identity of a subject. It uses CLIP (Contrastive Language-Image Pretraining) as a perceptual loss function.

#### Algorithm

1. Define a sketch as a **set of Bézier curves** (4 control points each)
2. Rasterize via differentiable rendering (DiffVG)
3. Pass both the original image and the sketch rendering through CLIP's vision encoder
4. Minimize the CLIP embedding distance between original and sketch
5. Control abstraction level by varying the number of strokes (4 → 64)

#### Relevance to SVG-gen

CLIPasso is designed for **sketching** (line art abstraction), not photorealistic vectorization. However, two concepts are transferable:

1. **Perceptual loss via CLIP**: Instead of SSIM/MSE for evaluating our SVG output, CLIP-based losses better capture "does this vector art represent the same thing?" — useful for aggressive simplification
2. **Abstraction control**: The ability to dial the number of primitives and get a meaningful representation at each level is something our pipeline lacks. Currently, reducing K or increasing epsilon produces worse output, not a simpler artistic interpretation.

**Impact**: LOW for our current use case (we want faithful vectorization, not abstraction). Could be interesting for a future "simplify" mode.

### VectorFusion — Text-to-SVG via Score Distillation

**Paper**: Jain et al., CVPR 2023

Uses Score Distillation Sampling (SDS) from pre-trained diffusion models (Stable Diffusion) to generate SVGs from text prompts. The SVG is parameterized as Bézier paths and optimized via DiffVG so that, when rendered to pixels, the diffusion model considers it a high-probability sample for the text prompt.

**Relevance**: Minimal for raster-to-SVG conversion. VectorFusion is text-to-SVG, not image-to-SVG. However, it demonstrates that diffusion model guidance can improve SVG aesthetics — potentially usable as a "make it look more like professional vector art" post-processing loss.

### SVGDreamer — Text-Guided SVG with Semantic Decomposition

**Paper**: Xing et al., CVPR 2024

SVGDreamer extends VectorFusion with:

1. **SIVE (Semantic-Driven Image Vectorization)**: Decomposes the scene into foreground objects and background using attention masks, enabling independent editing of SVG layers
2. **VPSD (Vectorized Particle-based Score Distillation)**: Models SVGs as distributions of control points and colors, addressing shape over-smoothing and color over-saturation

**Relevance**: The SIVE decomposition concept — using attention/saliency maps to guide layer separation — is directly applicable. Our pipeline treats all regions equally, but a saliency-guided approach could allocate more detail (higher contour resolution, more Bézier segments) to the foreground subject and simplify the background more aggressively.

**Impact**: LOW for direct adoption. MEDIUM if we extract the saliency-guided detail allocation concept.

---

## 9. Post-Processing

### Node Reduction

Our SVGs currently contain 500K–1.5M nodes. Professional Illustrator files for comparable images typically have 5K–50K nodes. The gap is **10–100×** too many nodes.

#### Strategy 1: Merge Adjacent L Segments into Bézier Curves

Our output is dominated by line segments (L commands). Many sequences of consecutive L segments approximate curves that could be represented by a single cubic Bézier.

```
Current:  M 0,0 L 1,0.1 L 2,0.4 L 3,0.8 L 4,1.3 L 5,1.9 ...  (6 nodes)
Better:   M 0,0 C 1.5,0.1 3.5,0.6 5,1.9                        (2 nodes)
```

**Algorithm**: Sliding window over consecutive L segments. For each window of size N, attempt to fit a single cubic Bézier. If error < tolerance, replace the N segments with one C command. Slide forward and repeat.

**Expected impact**: 5–10× node reduction on typical output.

#### Strategy 2: Bézier Segment Merging (Potrace-style)

Adjacent Bézier segments that are nearly tangent-continuous can often be merged into a single segment. This is Potrace's `opttolerance` parameter.

**Algorithm**: For two adjacent cubic Béziers $C_1$ and $C_2$:

1. Check if they share a tangent at the join point (within angular tolerance)
2. If yes, fit a single cubic Bézier to the combined control polygon
3. If error < tolerance, replace both with the merged segment

#### Strategy 3: Visvalingam-Whyatt on Control Polygons

Apply Visvalingam-Whyatt (area-based removal) to the control polygon of the Bézier path. Remove control points that contribute minimal visual change (tiny triangle area).

#### Strategy 4: Optimize Circle/Arc Segments

Many contours contain near-circular arcs. Detect these and replace with SVG `A` (arc) commands, which represent the same geometry with fewer nodes.

**Detection**: For a sequence of Bézier segments, fit a circle. If residual < tolerance, replace with arc command.

### Path Merging

Multiple paths with the same fill that are adjacent can often be merged into a single compound path:

```xml
<!-- Before: two separate paths -->
<path d="M 0,0 ... Z" fill="#333"/>
<path d="M 50,50 ... Z" fill="#333"/>

<!-- After: one compound path -->
<path d="M 0,0 ... Z M 50,50 ... Z" fill="#333"/>
```

This doesn't reduce nodes but reduces SVG DOM elements and rendering overhead.

### Coordinate Precision

Current output uses high-precision coordinates (e.g., `123.456789`). Reducing to 1–2 decimal places saves ~30% file size with negligible visual impact at typical viewing scales.

### Path Data Optimization

- Use relative commands (m, l, c) instead of absolute (M, L, C) — often shorter
- Use shorthand Bézier commands (S for smooth cubic, Q for quadratic) where applicable
- Omit redundant whitespace and trailing zeros

**Tools**: SVGO (Node.js) or scour (Python) can apply these optimizations automatically.

### Priority Ranking

| Technique                      | Node Reduction         | Effort  | Impact      |
| ------------------------------ | ---------------------- | ------- | ----------- |
| L→C segment merging            | 5–10×                  | Medium  | **Highest** |
| Bézier segment merging         | 2–3×                   | Medium  | **High**    |
| Coordinate precision           | 0× (file size only)    | Trivial | Medium      |
| Circle/arc detection           | 1.5× on curved regions | Medium  | Medium      |
| Visvalingam on control polygon | 1.3–2×                 | Low     | Medium      |
| SVGO/scour cleanup             | 0× (file size only)    | Trivial | Low         |

---

## 10. The "Illustration Look"

### What Makes Professional Vector Art Look Professional

Having analyzed Adobe Illustrator files, Dribbble portfolio pieces, and professional iconography, the following properties distinguish hand-crafted vector art from algorithm output:

#### 1. **Decisive Edges**

Professional illustrations have edges that are either:

- **Sharp and clean** (geometric boundaries, text, hard shadows)
- **Intentionally soft** (gradients, atmospheric blur)

Algorithm output tends to produce edges that are **uniformly fuzzy** — everything has the same amount of smoothing regardless of artistic intent.

**Fix**: Edge classification. Detect high-contrast boundaries (sharp) vs. low-contrast transitions (soft) and apply different smoothing parameters. Our dual-sigma smoothing partially addresses this, but it uses a continuous interpolation rather than a binary sharp/soft decision.

#### 2. **Geometric Regularity**

Professionals use **perfect circles, exact rectangles, precise angles** (90°, 45°, 30°). Algorithmic output produces irregular approximations of these shapes.

**Fix**: Shape detection and snapping:

- Detect near-circular contours → replace with `<circle>` or arc commands
- Detect near-rectangular contours → replace with `<rect>` or right-angle polygon
- Snap near-horizontal/vertical line segments to exact horizontal/vertical
- Quantize common angles (0°, 30°, 45°, 60°, 90°) with a small tolerance

This is one of Adobe Image Trace's "Shapes" feature. Implementation:

```python
# Circle detection
(x, y), radius = cv2.minEnclosingCircle(contour)
circle_area = np.pi * radius**2
contour_area = cv2.contourArea(contour)
circularity = contour_area / circle_area
if circularity > 0.92:  # >92% circular
    # Replace contour with SVG circle
```

#### 3. **Limited, Harmonious Color Palette**

Professional illustrations use 5–15 colors that form a coherent palette. Our auto-K can produce 20–48 clusters with no harmony constraint.

**Fix**: After K-means, snap cluster colors to a pre-defined palette (extracted from the image via palette extraction algorithms, or from standard design palettes). Or: quantize the K-means centroids to the nearest perceptually distinct colors, enforcing a minimum ΔE between all pairs.

#### 4. **Consistent Stroke Weight**

In professional work, stroke weights are consistent (e.g., all outlines are 2px, all hairlines are 0.5px). Our pipeline produces randomly varying edge thicknesses based on contour geometry.

**Fix**: Quantize detected stroke widths to a small set of standard weights (0.5, 1, 1.5, 2, 3 px).

#### 5. **Clean Path Topology**

Professional vector art has:

- No overlapping paths (abutting edges)
- No stray points or micro-paths
- Compound paths for holes (letter O = outer circle + inner circle cutout)
- Logical path grouping (`<g>` elements for related paths)

**Fix**:

- Implement abutting path generation (see §2)
- Filter paths below perceptual threshold (< 0.5px at target resolution)
- Detect and generate compound paths with holes using fill-rule="evenodd"
- Group paths by spatial proximity or color similarity

#### 6. **Appropriate Level of Detail**

Professional illustrations **abstract away** fine detail. A photograph of a tree might become 5–10 shaped paths representing trunk, major branches, and leaf mass. Our pipeline tries to preserve every pixel, producing thousands of tiny paths for texture.

**Fix**: This is the hardest problem and requires semantic understanding. Pragmatic approaches:

- Use saliency detection to identify foreground regions → detailed vectorization
- Background/texture regions → aggressive simplification (larger min_contour_area, higher simplify_epsilon)
- Offer user control over a "detail" slider that adjusts these parameters per-region

---

## Implementation Priority Matrix

Ranked by **impact on illustration quality** × **feasibility in our Python stack**:

| Rank | Technique                             | Impact         | Effort   | Section |
| ---- | ------------------------------------- | -------------- | -------- | ------- |
| 1    | **L-segment → Bézier merging**        | 🔴 Critical    | 2–3 days | §9      |
| 2    | **Levien optimal Bézier fitting**     | 🔴 Critical    | 3–4 days | §4      |
| 3    | **Shape detection (circles, rects)**  | 🟠 High        | 2–3 days | §10     |
| 4    | **Linear gradient detection**         | 🟠 High        | 2–3 days | §6      |
| 5    | **Mean-shift prefiltering**           | 🟡 Medium-High | 0.5 days | §5      |
| 6    | **SLIC→RAG color quantization**       | 🟡 Medium-High | 2–3 days | §5      |
| 7    | **Abutting path generation**          | 🟠 High        | 3–5 days | §2      |
| 8    | **Stroke detection + centerline**     | 🟡 Medium-High | 3–4 days | §7      |
| 9    | **K-means in CIELAB space**           | 🟡 Medium      | 0.5 days | §5      |
| 10   | **Angle/line snapping**               | 🟡 Medium      | 1 day    | §10     |
| 11   | **Bézier segment merging**            | 🟡 Medium      | 2 days   | §9      |
| 12   | **SVGO-style cleanup**                | 🟢 Low         | 0.5 days | §9      |
| 13   | **DiffVG post-optimization**          | 🟠 High        | 5+ days  | §8      |
| 14   | **Saliency-guided detail allocation** | 🟡 Medium      | 3–4 days | §8      |
| 15   | **Perceptual palette harmonization**  | 🟢 Low         | 1 day    | §10     |

### Recommended Implementation Order

**Phase 1 — Node reduction (biggest immediate win)**

1. L-segment → Bézier merging (§9, Strategy 1)
2. Levien optimal fitter (§4) to replace Schneider fitter
3. Bézier segment merging (§9, Strategy 2)

Target: Reduce node count from 500K–1.5M to 20K–80K

**Phase 2 — Visual quality leap** 4. Shape detection: circles, rectangles, straight-line snapping (§10) 5. Linear gradient detection and generation (§6) 6. Mean-shift prefiltering before K-means (§5) 7. Switch K-means to CIELAB space (§5)

Target: Output that reads as "illustrated" rather than "traced"

**Phase 3 — Professional polish** 8. Abutting path generation (§2) 9. Stroke detection for thin features (§7) 10. Saliency-guided detail allocation (§8) 11. Coordinate precision and SVGO cleanup (§9)

Target: Output indistinguishable from human-created Illustrator files at normal viewing scale

---

## References

1. Selinger, P. (2003). _Potrace: a polygon-based tracing algorithm_. potrace.sourceforge.net
2. Ramer, U. (1972). An iterative procedure for the polygonal approximation of plane curves. _CGIP_ 1(3):244-256
3. Douglas, D. & Peucker, T. (1973). Algorithms for the Reduction of the Number of Points Required to Represent a Digitized Line. _Cartographica_ 10(2):112-122
4. Visvalingam, M. & Whyatt, J.D. (1993). Line generalisation by repeated elimination of points. _Cartographic Journal_ 30(1):46-51
5. Schneider, P. (1990). An Algorithm for Automatically Fitting Digitized Curves. _Graphics Gems_, Academic Press, pp. 612-626
6. Levien, R. (2021). Fitting cubic Bézier curves. raphlinus.github.io
7. Li, T.-M. et al. (2020). Differentiable Vector Graphics Rasterization for Editing and Learning. _ACM TOG_ 39(6), SIGGRAPH Asia 2020
8. Ma, X. et al. (2022). Towards Layer-wise Image Vectorization. _CVPR_ 2022 (Oral)
9. Vinker, Y. et al. (2022). CLIPasso: Semantically-Aware Object Sketching. _SIGGRAPH_ 2022
10. Jain, A. et al. (2023). VectorFusion: Text-to-SVG by Abstracting Pixel-Based Diffusion Models. _CVPR_ 2023
11. Xing, X. et al. (2024). SVGDreamer: Text Guided SVG Generation with Diffusion Model. _CVPR_ 2024
12. Orzan, A. et al. (2008). Diffusion Curves: A Vector Representation for Smooth-Shaded Images. _ACM TOG_ 27(3)
13. Achanta, R. et al. (2012). SLIC Superpixels Compared to State-of-the-Art Superpixel Methods. _IEEE TPAMI_ 34(11):2274-2281
14. Comaniciu, D. & Meer, P. (2002). Mean Shift: A Robust Approach Toward Feature Space Analysis. _IEEE TPAMI_ 24(5):603-619
15. de Boor, C. et al. (1987). High accuracy geometric Hermite interpolation. _CAGD_ 4(4):269-278
16. Penner, A. (2019). Fitting a Cubic Bézier to a Parametric Function. _College Mathematics Journal_ 50(4)
