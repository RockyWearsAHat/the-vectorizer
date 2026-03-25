---
name: "math-implementation-guide"
description: "Bridges research math to NumPy/OpenCV code for each high-priority algorithm. Read this before implementing any technique from the research report."
---

# Math-to-Code Implementation Guide

This skill file bridges the gap between VECTORIZATION_RESEARCH_REPORT.md (the _what_) and actual NumPy/OpenCV implementation (the _how_). Each section provides:

1. The mathematical formulation with variable definitions
2. The exact NumPy/OpenCV translation
3. How it wires into the existing pipeline
4. What to test and what to watch for

---

## 1. Visvalingam-Whyatt Node Reduction

### The Math

For three consecutive points $A_i, B_i, C_i$ on a polyline, the **effective area** is:

$$\text{area}_i = \frac{1}{2} |x_A(y_B - y_C) + x_B(y_C - y_A) + x_C(y_A - y_B)|$$

The algorithm maintains a min-heap of (area, index). Pop the smallest, remove that point, recompute its two neighbors' areas. Repeat until the smallest area exceeds a threshold.

### The Code

```python
import heapq

def visvalingam_whyatt(points: np.ndarray, min_area: float) -> np.ndarray:
    """Remove points with effective triangle area below min_area.

    points: (N, 2) array of x,y coordinates
    min_area: minimum triangle area to keep a point
    Returns: (M, 2) array with M <= N points
    """
    n = len(points)
    if n <= 3:
        return points

    # Linked list via prev/next arrays for O(1) removal
    prev_idx = np.arange(-1, n - 1)  # prev_idx[0] = -1 (sentinel)
    next_idx = np.arange(1, n + 1)   # next_idx[n-1] = n (sentinel)
    removed = np.zeros(n, dtype=bool)

    def tri_area(i):
        p, c, nx = prev_idx[i], i, next_idx[i]
        if p < 0 or nx >= n:
            return float('inf')  # endpoints can't be removed
        ax, ay = points[p]
        bx, by = points[c]
        cx, cy = points[nx]
        return 0.5 * abs(ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))

    # Build initial heap: (area, index)
    heap = []
    for i in range(1, n - 1):
        heapq.heappush(heap, (tri_area(i), i))

    while heap:
        area, i = heapq.heappop(heap)
        if removed[i]:
            continue
        if area >= min_area:
            break

        # Remove point i
        removed[i] = True
        p, nx = prev_idx[i], next_idx[i]
        if p >= 0:
            next_idx[p] = nx
        if nx < n:
            prev_idx[nx] = p

        # Recompute neighbors
        if p >= 0 and p > 0:
            heapq.heappush(heap, (tri_area(p), p))
        if nx < n and nx < n - 1:
            heapq.heappush(heap, (tri_area(nx), nx))

    return points[~removed]
```

### Pipeline Integration

**Where it goes:** In `curve_fitting/__init__.py`, call it inside `_fit_contour()` AFTER `merge_segments_artistic` and BEFORE SVG path string generation. Apply to the final control polygon of each fitted path.

**Threshold tuning:** Start with `min_area = (max_error * 0.5)²`. This removes points whose visual contribution is less than half the fitting tolerance.

**What to test:** Run `check_regression.py`. Node counts should drop 5-15%. Feature% should stay within 0.5pp.

**Watch for:** On test4 (123K nodes, dense texture), this could give the biggest wins. On Ref (line art), threshold must be very small to avoid losing thin strokes.

---

## 2. Potrace-Style Optimal Polygon (True DP)

### The Math

Given a closed contour of $N$ pixel-edge vertices $v_0, \ldots, v_{N-1}$, find the polygon with fewest sides that stays within distance $\varepsilon$ of the original contour.

**DP recurrence:**

$$\text{cost}[j] = \min_{0 \le i < j} \left( \text{cost}[i] + 1 \right) \quad \text{s.t.} \; \max_{k \in [i,j]} d(v_k, \overline{v_i v_j}) \le \varepsilon$$

where $d(v_k, \overline{v_i v_j})$ is the perpendicular distance from vertex $v_k$ to the line segment $v_i \to v_j$.

**Key difference from RDP:** RDP is greedy-recursive (split at max-error point). DP finds the globally optimal set of vertices that minimizes segment count subject to the error bound.

### The Code

```python
def dp_optimal_polygon(points: np.ndarray, epsilon: float) -> np.ndarray:
    """Find minimum-vertex polygon within epsilon of input polyline.

    Uses Potrace's dynamic programming approach.
    points: (N, 2) contour vertices
    epsilon: max perpendicular distance tolerance
    Returns: (M, 2) polygon vertices with M <= N
    """
    n = len(points)
    if n <= 4:
        return points

    INF = float('inf')
    cost = np.full(n, INF)
    parent = np.full(n, -1, dtype=int)
    cost[0] = 0

    for j in range(1, n):
        # Try all possible predecessors
        # In practice, limit lookback to avoid O(n²) on huge contours
        i_start = max(0, j - 800)  # bound window for large contours

        for i in range(j - 1, i_start - 1, -1):
            if cost[i] == INF:
                continue

            # Compute max deviation of points[i+1..j-1] from line points[i]->points[j]
            if j - i <= 1:
                max_dev = 0.0
            else:
                seg = points[j] - points[i]
                seg_len_sq = seg[0]**2 + seg[1]**2
                if seg_len_sq < 1e-10:
                    max_dev = np.max(np.linalg.norm(points[i+1:j] - points[i], axis=1))
                else:
                    # Vectorized perpendicular distance
                    rel = points[i+1:j] - points[i]
                    # Cross product gives signed area of parallelogram
                    cross = np.abs(rel[:, 0] * seg[1] - rel[:, 1] * seg[0])
                    max_dev = np.max(cross) / np.sqrt(seg_len_sq)

            if max_dev <= epsilon:
                new_cost = cost[i] + 1
                if new_cost < cost[j]:
                    cost[j] = new_cost
                    parent[j] = i
            else:
                # Once we find a predecessor that violates epsilon,
                # going further back will only get worse (for convex-ish contours)
                # But NOT guaranteed for non-convex — don't break
                pass

    # Backtrace
    if cost[n-1] == INF:
        return points  # fallback: DP couldn't find valid polygon

    indices = []
    j = n - 1
    while j >= 0:
        indices.append(j)
        j = parent[j]
    indices.reverse()

    return points[indices]
```

### Vectorized Max-Deviation (the bottleneck)

The inner loop's perpendicular distance computation is the bottleneck. Here's the vectorized version:

```python
# For segment from points[i] to points[j]:
seg = points[j] - points[i]                    # (2,)
seg_len = np.sqrt(seg[0]**2 + seg[1]**2)       # scalar
if seg_len < 1e-10:
    devs = np.linalg.norm(pts_between - points[i], axis=1)
else:
    rel = pts_between - points[i]               # (K, 2)
    cross = np.abs(rel[:, 0]*seg[1] - rel[:, 1]*seg[0])  # (K,)
    devs = cross / seg_len                      # (K,)
max_dev = np.max(devs) if len(devs) > 0 else 0.0
```

### Pipeline Integration

**Where it goes:** Replace `cv2.approxPolyDP()` calls in `_fit_contour()`. The DP polygon replaces RDP as the first simplification step.

**Important:** This is NOT the same as the existing DP in `merge_segments_artistic` (which operates on fitted Bézier segments, not raw contour points). This operates on raw contour vertex positions.

**Prior failure analysis:** `kb-what-failed.md` says "Direct DP on raw contours FAILED — raw contour points too dense, DP keeps MORE vertices than RDP even with high lambda." That was using **mean-SSD** as the penalty metric. The version above uses **max perpendicular deviation** (matching RDP's error metric), which should behave differently — it's the `max_dev <= epsilon` check, not a sum-of-squares.

**What to test:** Compare node counts AND Feature% vs RDP baseline. If max-deviation DP still produces more vertices than RDP at same epsilon, then look at whether the contour points from `cv2.findContours(CHAIN_APPROX_SIMPLE)` are sparse enough (CHAIN_APPROX_SIMPLE already removes collinear points).

---

## 3. Abutting Path Generation

### The Math

**Goal:** Partition the image plane into non-overlapping regions, each a polygon, such that adjacent regions share exact edges (no gaps, no overlaps).

**Current state:** Painter's algorithm — each cluster's contour may overlap others. Adjacent regions have visible gap slivers at boundaries.

**The approach:** For each pixel, assign it to the cluster with highest soft-field membership. Then extract boundary contours between adjacent cluster regions.

### The Critical Insight (why the naive attempt failed)

The prior naive attempt (March 2026, see kb-what-failed.md) tried to clip per-cluster binary masks. This failed because the painter's algorithm relies on overlap for seamless rendering.

The correct approach uses **topological boundary extraction**, not clipping:

### Algorithm: Shared Boundary Extraction

```
1. Create label map: label[y,x] = argmax_k(soft_field[k][y,x])
   Every pixel belongs to exactly one cluster.

2. Find boundary pixels: where label differs from any 4-neighbor
   boundary_mask[y,x] = (label[y,x] != label[y-1,x]) or
                         (label[y,x] != label[y,x-1])   etc.

3. For each pair of adjacent clusters (i,j):
   Extract the boundary curve between them as a polyline.
   This boundary is SHARED — it's the edge of both cluster i's polygon
   and cluster j's polygon.

4. Build each cluster's polygon from its shared boundary segments.
   Order segments to form a closed path.

5. The SVG uses these abutting polygons — NO overlap, NO gap.
```

### The Code (Step 1-2)

```python
def build_label_map(soft_fields: list[np.ndarray]) -> np.ndarray:
    """Assign each pixel to cluster with highest membership.

    soft_fields: list of K arrays, each (H, W) with values in [0,1]
    Returns: (H, W) int32 label map
    """
    # Stack into (K, H, W) then argmax
    stacked = np.stack(soft_fields, axis=0)  # (K, H, W)
    labels = np.argmax(stacked, axis=0).astype(np.int32)  # (H, W)
    return labels


def extract_boundary_segments(labels: np.ndarray) -> dict:
    """Extract boundary polylines between adjacent cluster pairs.

    Returns: {(i,j): list_of_polylines} where i < j
    """
    h, w = labels.shape
    boundaries = {}

    # Horizontal boundaries: compare label[y,x] vs label[y+1,x]
    diff_v = labels[:-1, :] != labels[1:, :]
    ys, xs = np.nonzero(diff_v)
    for y, x in zip(ys, xs):
        pair = (min(labels[y,x], labels[y+1,x]), max(labels[y,x], labels[y+1,x]))
        boundaries.setdefault(pair, []).append((x, y + 0.5))

    # Vertical boundaries: compare label[y,x] vs label[y,x+1]
    diff_h = labels[:, :-1] != labels[:, 1:]
    ys, xs = np.nonzero(diff_h)
    for y, x in zip(ys, xs):
        pair = (min(labels[y,x], labels[y,x+1]), max(labels[y,x], labels[y,x+1]))
        boundaries.setdefault(pair, []).append((x + 0.5, y))

    return boundaries
```

### Step 3-4: Boundary → Polygon Construction

This is the hardest part. The boundary pixels must be **ordered** into contiguous polylines, then each cluster's polygon is built from the ordered boundary segments.

```python
def order_boundary_points(points: list[tuple]) -> list[np.ndarray]:
    """Order scattered boundary points into connected polylines.

    Uses nearest-neighbor chaining with spatial index.
    """
    from scipy.spatial import cKDTree

    pts = np.array(points)
    tree = cKDTree(pts)
    used = np.zeros(len(pts), dtype=bool)
    polylines = []

    for start in range(len(pts)):
        if used[start]:
            continue
        # Chain forward from this point
        chain = [start]
        used[start] = True
        current = start
        while True:
            dists, idxs = tree.query(pts[current], k=6)
            found = False
            for d, idx in zip(dists[1:], idxs[1:]):
                if not used[idx] and d < 2.0:  # adjacent pixel boundary
                    chain.append(idx)
                    used[idx] = True
                    current = idx
                    found = True
                    break
            if not found:
                break
        if len(chain) >= 3:
            polylines.append(pts[chain])

    return polylines
```

### Pipeline Integration

**This is a major architecture change.** It replaces the painter's algorithm SVG construction in `multilevel/__init__.py` (the final SVG generation step).

**Phased approach:**

1. Implement `build_label_map()` + boundary extraction + polygon construction
2. Test on a single image (Ref.png, simplest case)
3. Compare SVG quality vs painter's algorithm output
4. If better, add a flag `--abutting` to switch modes

**Expected wins:** No gap slivers, smaller SVG (no hidden overdraw), better WdErr (no expansion from overlap).

**Expected risks:** Complex topology (multiple connected components per cluster, holes, islands). Need robust polygon construction.

---

## 4. Max-Deviation DP Penalty (Alternative to Mean-SSD)

### The Math

Current DP in `merge_segments_artistic` uses sum of squared point-to-curve distances:

$$\text{SSD} = \sum_{k=i}^{j-1} \sum_{p \in \text{seg}_k} |B(t_p) - p|^2$$

This favors MANY short segments (each with tiny SSD) over FEW long segments (each with moderate SSD). The penalty $\lambda$ must be tuned to balance.

**Alternative: max-deviation metric:**

$$\text{MaxDev} = \max_{k \in [i,j]} \max_{p \in \text{seg}_k} |B(t_p) - p|$$

This asks: "Can a single Bézier curve represent segments $i..j$ with no point deviating more than $\varepsilon$?"

### The Code Change

In `merge_segments_artistic()` in `curve_fitting/__init__.py`, the `_can_merge()` function currently computes max point-to-curve error. Change the DP cost function:

```python
# Current (segment-count minimization with max-error gating):
#   cost[j] = min(cost[i] + 1) for all valid i where _can_merge(i,j)

# Better (explicit max-deviation in cost):
#   cost[j] = min(cost[i] + max_dev(i,j)) where max_dev < tolerance
# This penalizes merges that use MORE of the tolerance budget,
# preferring merges that are well within tolerance.
```

This is a small code change (~10 lines in the DP loop) but changes the segment selection from "minimize count regardless of how close to tolerance each merge is" to "minimize count while preferring merges with low deviation."

### Pipeline Integration

**Where:** `merge_segments_artistic()` in `curve_fitting/__init__.py`
**Risk:** LOW — isolated to one function, easy to A/B test
**Test:** Compare node counts and Feature% before/after

---

## 5. Diagnostic Profiling: Where Quality Is Lost

### The Problem

Agents throw random parameter changes because nobody knows which pipeline step is destroying quality for each image. This section provides a method to MEASURE quality loss at each stage.

### The Method

For a given image, render the pipeline's intermediate results at each step and compare to the original:

```python
def profile_quality_loss(image_path: str):
    """Measure quality contribution of each pipeline stage."""
    import cv2
    from app.core.multilevel import multilevel_vectorize

    img = cv2.imread(image_path)

    # Run pipeline with debug hooks that capture intermediate states:
    # 1. After K-means: render label map as flat colors → measure SSIM vs original
    # 2. After merge: render merged label map → SSIM
    # 3. After soft field: render soft field as colors → SSIM
    # 4. After threshold: render binary masks as colors → SSIM
    # 5. After contour extraction: render contour fills → SSIM
    # 6. After Bézier fitting: render SVG → SSIM
    # 7. After node reduction: render SVG → SSIM

    # The DELTA between consecutive steps shows where quality is lost.
    # If SSIM drops sharply at step 4→5, the contour extraction is the bottleneck.
    # If it drops at step 1→2, the merge is too aggressive.
```

### Pipeline Integration

Create a `--profile` flag on `generate.py` that runs this analysis and prints a quality-loss waterfall:

```
Quality Loss Waterfall for test3.jpg:
  Original               → 1.000 SSIM
  After K-means (K=11)   → 0.972 SSIM  (Δ -0.028)
  After merge (K=8)      → 0.968 SSIM  (Δ -0.004)
  After soft field       → 0.965 SSIM  (Δ -0.003)
  After threshold (iso)  → 0.951 SSIM  (Δ -0.014) ← biggest drop
  After contours         → 0.948 SSIM  (Δ -0.003)
  After Bézier fit       → 0.945 SSIM  (Δ -0.003)
  After node reduce      → 0.943 SSIM  (Δ -0.002)
```

This tells agents: "focus on iso thresholding for test3, not on curve fitting."

---

## 6. Coordinate Precision Reduction

### The Math

SVG path data like `C 123.456789,234.567891 ...` uses too many decimal places. At a viewBox of ~4000 units wide, 1 decimal place = 0.1 units ≈ 0.025% of width. Two decimals is more than enough.

### The Code

This is a simple string-formatting change in the SVG generation:

```python
# Current (in multilevel/__init__.py SVG path generation):
f"C {x1},{y1} {x2},{y2} {x3},{y3}"

# Better (2 decimal places — negligible visual impact):
f"C {x1:.2f},{y1:.2f} {x2:.2f},{y2:.2f} {x3:.2f},{y3:.2f}"

# Even better — strip trailing zeros:
def fmt(v):
    s = f"{v:.2f}"
    if '.' in s:
        s = s.rstrip('0').rstrip('.')
    return s
```

### Impact

**SVG file size reduction: 20-35%** with zero visual impact. This is the single easiest win available. Takes 10 minutes to implement.

---

## 7. Applying Research Insights: Translation Rules

When reading a research paper or algorithm description, use these rules to translate math to code:

### Rule 1: Summation → np.sum / vectorized ops

$$\sum_{i=0}^{n} f(x_i) \rightarrow \texttt{np.sum(f(x))}$$

### Rule 2: Argmin/Argmax → np.argmin / np.argmax

$$\arg\min_i d_i \rightarrow \texttt{np.argmin(d)}$$

### Rule 3: Matrix multiply → @ operator or np.dot

$$\mathbf{A}\mathbf{x} = \mathbf{b} \rightarrow \texttt{x = np.linalg.solve(A, b)}$$

### Rule 4: Least squares → np.linalg.lstsq

$$\min_x \|\mathbf{A}\mathbf{x} - \mathbf{b}\|^2 \rightarrow \texttt{x, \_, \_, \_ = np.linalg.lstsq(A, b, rcond=None)}$$

### Rule 5: Distance transform → cv2.distanceTransform

$$d(p) = \min_q |p - q| \; \text{for } q \in \text{boundary} \rightarrow \texttt{cv2.distanceTransform(mask, cv2.DIST\_L2, 5)}$$

### Rule 6: PCA → np.linalg.eigh on covariance

$$\text{eigenvalues}, \text{eigenvectors} = \text{eig}(\text{Cov}(X))$$

```python
cov = np.cov(points.T)  # (2, 2) for 2D points
eigenvalues, eigenvectors = np.linalg.eigh(cov)
principal_axis = eigenvectors[:, -1]  # largest eigenvalue
```

### Rule 7: Convolution/smoothing → cv2.GaussianBlur or scipy.ndimage

$$G_\sigma * I \rightarrow \texttt{cv2.GaussianBlur(img, (0,0), sigma)}$$

### Rule 8: Cross product (2D, for signed area) → manual

$$\vec{u} \times \vec{v} = u_x v_y - u_y v_x$$

```python
cross = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]  # vectorized
```

### Rule 9: Bézier evaluation → De Casteljau or explicit

$$B(t) = (1-t)^3 P_0 + 3(1-t)^2 t P_1 + 3(1-t) t^2 P_2 + t^3 P_3$$

```python
def eval_bezier(t, p0, p1, p2, p3):
    u = 1 - t
    return u**3 * p0 + 3*u**2*t * p1 + 3*u*t**2 * p2 + t**3 * p3
# Vectorized for array of t values:
def eval_bezier_vec(ts, p0, p1, p2, p3):
    u = 1 - ts
    return (u**3)[:,None]*p0 + (3*u**2*ts)[:,None]*p1 + (3*u*ts**2)[:,None]*p2 + (ts**3)[:,None]*p3
```

### Rule 10: Newton-Raphson → iterate

$$x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}$$

```python
for _ in range(max_iters):
    fx = f(x)
    fpx = f_prime(x)
    if abs(fpx) < 1e-10:
        break
    x = x - fx / fpx
```

---

## Application Checklist

When implementing any technique from the research report:

1. **Read the math section here** for the NumPy translation
2. **Check kb-what-failed.md** — has this been tried before? If so, what exactly failed?
3. **Write a HYPOTHESIS** before coding (what metric will change and by how much?)
4. **Implement in isolation** — touch ONE function, add ONE call site
5. **Run `check_regression.py`** immediately after
6. **If regression:** revert. Update kb-what-failed.md with the specific evidence.
7. **If improvement:** update kb-what-works.md AND kb-baselines.md AND check_regression.py baselines
