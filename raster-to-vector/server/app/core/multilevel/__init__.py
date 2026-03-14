"""Archimedes convergence vectorization engine.

Like Archimedes bounding pi with inscribed and circumscribed polygons,
we converge on the true edge from both sides:

  1. K-means finds the real colours in the image.
  2. For each colour we build a *soft membership field*: how strongly
     each pixel belongs to that colour vs. its closest rival.
     - Pixels deep inside a region score ~1.0  (inscribed / inner bound)
     - Pixels deep outside score ~0.0          (circumscribed / outer bound)
     - Anti-aliased edge pixels score ≈0.5     (the true boundary)
  3. The soft-field gradient reveals *edge hardness* at every pixel.
     An adaptive per-pixel iso-threshold compresses the inner bound
     (iso=0.60) toward the outer bound (iso=0.30) based on local
     color strength — hard edges snap tight, soft blends stay generous.
  4. Marching squares extracts the contour where the soft field equals
     the adaptive threshold — the true visual edge the eye perceives.
  5. Bézier curves are fit to those sub-pixel contour points.

No fixed thresholding.  The boundary is found analytically between
the inner and outer bounds, weighted by color transition strength.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from scipy.ndimage import distance_transform_edt, gaussian_filter1d
from skimage.measure import find_contours
from ..curve_fitting import fit_closed_bezier, fit_bezier_path
from skimage.morphology import skeletonize as _skeletonize
from ..stroke_reconstruction import _prune_skeleton, _trace_skeleton_paths


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class VectorLayer:
    paths: list[str]        # SVG path `d` strings, one per iso-level
    opacities: list[float]  # opacity for each iso-level path
    color: str              # hex fill colour


@dataclass
class StrokeLayer:
    paths: list[str]        # SVG path `d` strings for stroke centre lines
    widths: list[float]     # stroke-width for each path
    color: str              # hex stroke colour


@dataclass
class MultilevelResult:
    layers: list[VectorLayer]
    stroke_layers: list[StrokeLayer]
    width: int
    height: int
    background_color: str
    path_count: int
    node_count: int


# ---------------------------------------------------------------------------
# Background detection
# ---------------------------------------------------------------------------

def detect_background(image_bgr: np.ndarray) -> tuple[np.ndarray, int]:
    """Return (BGR colour, gray value) of the dominant border colour."""
    h, w = image_bgr.shape[:2]
    border = np.concatenate([
        image_bgr[0, :],
        image_bgr[-1, :],
        image_bgr[1:-1, 0],
        image_bgr[1:-1, -1],
    ])
    color = np.median(border, axis=0).astype(np.uint8)
    gray = int(cv2.cvtColor(color.reshape(1, 1, 3), cv2.COLOR_BGR2GRAY)[0, 0])
    return color, gray


# ---------------------------------------------------------------------------
# Edge-density map for adaptive processing
# ---------------------------------------------------------------------------

def _compute_edge_weight(image_bgr: np.ndarray, blur_radius: int = 15) -> np.ndarray:
    """Build a [0,1] edge-density map.  1 = edge-rich, 0 = flat/background.

    Used to blend between crisp and smooth soft-field processing:
    edges get sharp contours, flat fills get noise-free contours.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    density = cv2.blur(mag, (blur_radius, blur_radius))
    mx = density.max()
    if mx > 0:
        density = density / mx
    return np.clip((density - 0.05) / 0.20, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Core: quantize → adaptive soft-membership → sub-pixel contour → fit
# ---------------------------------------------------------------------------

def multilevel_vectorize(
    image_bgr: np.ndarray,
    *,
    num_levels: int = 24,
    simplify_epsilon: float = 0.12,
    max_error: float = 0.30,
    line_tolerance: float = 0.25,
    corner_threshold: float = 60.0,
    min_contour_area: int = 1,
    contour_scale: int = 6,
    smooth_sigma: float = 0.7,
) -> MultilevelResult:
    h, w = image_bgr.shape[:2]

    if len(image_bgr.shape) == 2:
        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)

    bg_color, bg_gray = detect_background(image_bgr)
    bg_hex = _bgr_to_hex(bg_color)

    # --- Step 0: Edge-density map for adaptive processing ---
    edge_weight = _compute_edge_weight(image_bgr)

    # --- Step 0b: Dual-denoise strategy ---
    # SD images have subtle per-pixel static (3-8 BGR units of noise)
    # within flat regions.  Two bilateral passes with different roles:
    #   1. Strong spatial + medium color for K-means input → clean
    #      cluster centres that aren't split by noise.
    #   2. Mild spatial + very tight color gate (sc=5) for the distance
    #      map → kills SD static in flat fills while leaving real edges
    #      (20+ BGR difference) completely untouched.
    denoised_km = cv2.bilateralFilter(image_bgr, 15, 12, 30)
    denoised_dist = cv2.bilateralFilter(image_bgr, 7, 5, 20)

    # --- Step 1: K-means colour quantization ---
    pixels = denoised_km.reshape(-1, 3).astype(np.float32)
    K = max(2, min(num_levels, 64))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(
        pixels, K, None, criteria, 6, cv2.KMEANS_PP_CENTERS
    )

    # --- Step 1b: Merge close cluster centres ---
    centers, labels = _merge_close_clusters(
        centers, labels.flatten(), h, w, threshold=60.0,
    )
    K = len(centers)

    centers_u = centers.astype(np.uint8)
    centers_f = centers.astype(np.float32)

    # --- Step 1c: Identify background cluster ---
    # Only skip if the nearest cluster is genuinely close to the
    # detected border colour (distance < 40 in BGR space).
    bg_dists = np.array([
        np.linalg.norm(centers_f[k] - bg_color.astype(np.float32))
        for k in range(K)
    ])
    bg_cluster_idx = int(np.argmin(bg_dists))
    bg_cluster = bg_cluster_idx if bg_dists[bg_cluster_idx] < 40.0 else -1

    # --- Step 2: Distance from every pixel to every cluster centre ---
    # Use the mildly denoised image (tight color gate sc=5) so SD
    # static doesn't pollute soft-field boundaries, while real edges
    # remain razor-sharp.
    pixels_3d = denoised_dist.astype(np.float32)
    dist_map = np.empty((h, w, K), dtype=np.float32)
    for k in range(K):
        diff = pixels_3d - centers_f[k]
        dist_map[:, :, k] = np.sqrt(np.sum(diff * diff, axis=2))

    # --- Step 3: Sort lightest-first (painter's algorithm) ---
    grays = np.array([
        int(cv2.cvtColor(c.reshape(1, 1, 3), cv2.COLOR_BGR2GRAY)[0, 0])
        for c in centers_u
    ])
    order = np.argsort(-grays)

    # --- Step 3b: Classify clusters as FILL vs MEDIATOR ---
    # A mediator cluster is an AA interpolation artifact: its color
    # lies on the line segment between two neighbor colors in BGR space,
    # it's thin, and it has almost no interior.  Mediators get reduced
    # rendering (no halo, lower core opacity) because the fill cluster's
    # halo already provides smooth anti-aliasing.
    # This is color-space aware — works for any palette, not just gray.
    morph_kernel = np.ones((3, 3), np.uint8)
    cluster_mediator_score = np.zeros(K, dtype=np.float64)  # 0=fill, 1=pure mediator
    cluster_mean_thick = np.zeros(K, dtype=np.float64)
    cluster_interior_frac = np.zeros(K, dtype=np.float64)
    cluster_pix_count = np.zeros(K, dtype=np.int64)
    for k in range(K):
        if k == bg_cluster:
            continue
        mask_k = (labels == k).astype(np.uint8)
        pix_count = int(np.count_nonzero(mask_k))
        if pix_count == 0:
            continue

        # Thickness: distance transform
        dt = distance_transform_edt(mask_k)
        fg_dts = dt[mask_k > 0]
        mean_thick = float(fg_dts.mean())
        interior_frac = float(np.count_nonzero(fg_dts > 2.0)) / pix_count
        cluster_mean_thick[k] = mean_thick
        cluster_interior_frac[k] = interior_frac
        cluster_pix_count[k] = pix_count

        # Find neighbor cluster IDs
        dilated = cv2.dilate(mask_k, morph_kernel, iterations=1)
        border_zone = (dilated > 0) & (mask_k == 0)
        unique_neighbors = sorted(
            set(int(n) for n in np.unique(labels[border_zone])) - {k}
        )

        # Check if color is an interpolation between any neighbor pair
        min_interp = 999.0
        if len(unique_neighbors) >= 2:
            c = centers_f[k]
            for i in range(len(unique_neighbors)):
                for j in range(i + 1, len(unique_neighbors)):
                    d = _point_to_segment_dist(
                        c, centers_f[unique_neighbors[i]],
                        centers_f[unique_neighbors[j]],
                    )
                    if d < min_interp:
                        min_interp = d

        # Mediator score: 1.0 for perfect interpolation with no interior;
        # falls off for thicker clusters or less perfect interpolations.
        if mean_thick <= 2.0 and min_interp < 30.0 and interior_frac < 0.20:
            # Score: closer to 0 interp_dist and 0 interior → higher score
            interp_score = max(0.0, 1.0 - min_interp / 30.0)
            interior_penalty = max(0.0, 1.0 - interior_frac / 0.20)
            cluster_mediator_score[k] = interp_score * interior_penalty
        # else: stays 0 → full fill

    # --- Step 3c: Absorb mediator clusters into their nearest real neighbor ---
    # AA-interpolation clusters (gray pixels between black and white) should
    # not be rendered as separate layers.  Reassign their pixels to the
    # darkest neighboring cluster so the soft-field boundary naturally falls
    # at the visual edge.  This eliminates gray splotches entirely.
    mediator_ids = [k for k in range(K) if cluster_mediator_score[k] > 0.3 and k != bg_cluster]
    if mediator_ids:
        for k in mediator_ids:
            mask_k = (labels == k).astype(np.uint8)
            dilated = cv2.dilate(mask_k, morph_kernel, iterations=1)
            border_zone = (dilated > 0) & (mask_k == 0)
            neighbors = sorted(
                set(int(n) for n in np.unique(labels[border_zone]))
                - {k} - set(mediator_ids)
            )
            if not neighbors:
                # Fall back to any non-mediator non-bg neighbor
                neighbors = sorted(
                    set(int(n) for n in np.unique(labels[border_zone]))
                    - {k}
                )
            if neighbors:
                # Pick the darkest neighbor (lowest gray value) — AA pixels
                # visually belong to the darker side of the edge.
                target = min(neighbors, key=lambda n: grays[n])
                labels[labels == k] = target

        # Rebuild distance map and centers for surviving clusters
        alive = sorted(set(int(v) for v in np.unique(labels)))
        if bg_cluster not in alive:
            bg_cluster = -1
        # Remap labels to contiguous IDs
        remap = np.full(K, -1, dtype=np.int32)
        new_centers_list = []
        for new_id, old_id in enumerate(alive):
            remap[old_id] = new_id
        new_labels = remap[labels]
        new_centers = centers_f[alive]
        # Update everything
        labels = new_labels
        centers_f = new_centers.astype(np.float32)
        centers_u = new_centers.astype(np.uint8)
        K = len(centers_f)
        bg_cluster = int(remap[bg_cluster]) if bg_cluster >= 0 and remap[bg_cluster] >= 0 else -1
        # Rebuild distance map
        pixels_3d = denoised_dist.astype(np.float32)
        dist_map = np.empty((h, w, K), dtype=np.float32)
        for k_new in range(K):
            diff = pixels_3d - centers_f[k_new]
            dist_map[:, :, k_new] = np.sqrt(np.sum(diff * diff, axis=2))
        # Rebuild grays and order
        grays = np.array([
            int(cv2.cvtColor(c.reshape(1, 1, 3), cv2.COLOR_BGR2GRAY)[0, 0])
            for c in centers_u
        ])
        order = np.argsort(-grays)
        cluster_lightness = grays.astype(np.float64) / 255.0
        # Reset mediator scores — all survivors are real fills
        cluster_mediator_score = np.zeros(K, dtype=np.float64)

    # Compute perceptual lightness for each cluster (0=black, 1=white)
    cluster_lightness = grays.astype(np.float64) / 255.0

    layers: list[VectorLayer] = []
    stroke_layers: list[StrokeLayer] = []
    total_paths = 0
    total_nodes = 0

    # Pre-compute upscaled edge weight (shared across all clusters)
    S = contour_scale
    ew_up = cv2.resize(edge_weight, (w * S, h * S),
                       interpolation=cv2.INTER_LINEAR)

    for cluster_idx in order:
        # Skip background cluster — it's represented by the SVG canvas
        # or the background rect, so processing it wastes time.
        if cluster_idx == bg_cluster:
            continue

        color_hex = _bgr_to_hex(centers_u[cluster_idx])
        mediator = cluster_mediator_score[cluster_idx]
        lightness = cluster_lightness[cluster_idx]

        # --- Archimedes soft membership ---
        d_k = dist_map[:, :, cluster_idx]

        other_mask = np.ones(K, dtype=bool)
        other_mask[cluster_idx] = False
        d_other = np.min(dist_map[:, :, other_mask], axis=2)

        denom = d_k + d_other
        denom = np.where(denom < 1e-10, 1e-10, denom)
        soft_raw = d_other / denom

        # --- Superresolution contour extraction ---
        soft_up = cv2.resize(soft_raw, (w * S, h * S),
                             interpolation=cv2.INTER_CUBIC)
        # Clamp to [0,1] — cubic interpolation can overshoot.
        np.clip(soft_up, 0.0, 1.0, out=soft_up)

        # --- Dual-sigma smoothing ---
        # Edge-rich pixels get crisp sigma; flat fills get heavier
        # smoothing.  Mediator clusters use tighter smoothing so
        # their contours hug actual pixel locations.
        sigma_crisp = (0.35 - mediator * 0.15) * S
        sigma_smooth = (1.0 - mediator * 0.5) * S
        soft_crisp = cv2.GaussianBlur(soft_up, (0, 0), sigmaX=sigma_crisp)
        soft_smooth = cv2.GaussianBlur(soft_up, (0, 0), sigmaX=sigma_smooth)
        soft = ew_up * soft_crisp + (1.0 - ew_up) * soft_smooth

        # --- Adaptive iso boundary placement ---
        gx = cv2.Sobel(soft.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(soft.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(gx * gx + gy * gy)
        mx = grad.max()
        grad_norm = grad / mx if mx > 0 else grad

        inner_iso, outer_iso = 0.60, 0.30
        adaptive_iso = outer_iso + grad_norm * (inner_iso - outer_iso)
        adaptive_iso = cv2.GaussianBlur(adaptive_iso, (0, 0), sigmaX=1.0 * S)
        shifted = soft - adaptive_iso

        layer_paths: list[str] = []
        layer_opacities: list[float] = []

        # --- Halo: soft AA fringe ---
        halo_iso = 0.28
        base_halo_opacity = 0.40
        halo_opacity = base_halo_opacity * float((1.0 - lightness) ** 1.0)
        halo_contours = find_contours(soft, halo_iso) if halo_opacity > 0.03 else []
        halo_parts: list[str] = []
        for contour in halo_contours:
            if len(contour) < 4:
                continue
            xy = contour[:, ::-1].astype(np.float64)
            xy = _smooth_contour(xy, sigma=smooth_sigma * 1.5 * S)
            xy = xy / S
            area = abs(_polygon_area(xy))
            if area < min_contour_area:
                continue
            d = _fit_contour(xy, simplify_epsilon, max_error, corner_threshold, line_tolerance)
            if d:
                halo_parts.append(d)

        if halo_parts:
            combined = " ".join(halo_parts)
            layer_paths.append(combined)
            layer_opacities.append(halo_opacity)
            total_paths += 1
            total_nodes += combined.count("C") + combined.count("L") + combined.count("M")

        # --- Core contour at adaptive iso, FULL opacity ---
        # Fill tiny gaps in narrow features (e.g. thin letter legs)
        # Use minimal 3x3 kernel to avoid inflating junctions.
        if mediator < 0.3:
            _fill = (shifted > 0).astype(np.uint8)
            _kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            _closed = cv2.morphologyEx(_fill, cv2.MORPH_CLOSE, _kern)
            shifted[(_closed > 0) & (shifted <= 0)] = 0.01

        core_contours = find_contours(shifted, 0.0)
        core_parts: list[str] = []
        for contour in core_contours:
            if len(contour) < 4:
                continue
            xy = contour[:, ::-1].astype(np.float64)
            xy = _smooth_contour(xy, sigma=smooth_sigma * S)
            xy = xy / S
            area = abs(_polygon_area(xy))
            if area < min_contour_area:
                continue
            d = _fit_contour(xy, simplify_epsilon, max_error, corner_threshold, line_tolerance)
            if d:
                core_parts.append(d)

        if core_parts:
            combined = " ".join(core_parts)
            layer_paths.append(combined)
            layer_opacities.append(1.0)
            total_paths += 1
            total_nodes += combined.count("C") + combined.count("L") + combined.count("M")

        if layer_paths:
            layers.append(VectorLayer(
                paths=layer_paths,
                opacities=layer_opacities,
                color=color_hex,
            ))

    return MultilevelResult(
        layers=layers,
        stroke_layers=stroke_layers,
        width=w,
        height=h,
        background_color=bg_hex,
        path_count=total_paths,
        node_count=total_nodes,
    )


# ---------------------------------------------------------------------------
# SVG generation
# ---------------------------------------------------------------------------

def generate_svg(
    result: MultilevelResult,
    *,
    remove_background: bool = True,
) -> str:
    w, h = result.width, result.height
    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {w} {h}" width="{w}" height="{h}">',
    ]

    if not remove_background:
        parts.append(
            f'<rect width="{w}" height="{h}" fill="{result.background_color}"/>'
        )

    for layer in result.layers:
        # Archimedes squeeze: render iso-level contours from outermost
        # (faintest) to innermost (full opacity).  Each successive ring
        # overpaints the previous, creating a graduated transition that
        # reconstructs the anti-aliased gradient.
        for path_d, opacity in zip(layer.paths, layer.opacities):
            if not path_d:
                continue
            if opacity >= 1.0:
                parts.append(
                    f'<path d="{path_d}" fill="{layer.color}"'
                    f' fill-rule="evenodd"/>'
                )
            else:
                parts.append(
                    f'<path d="{path_d}" fill="{layer.color}"'
                    f' fill-rule="evenodd" opacity="{opacity:.2f}"/>'
                )

    # Render stroke paths on top of fills
    for sl in result.stroke_layers:
        for path_d, sw in zip(sl.paths, sl.widths):
            if not path_d:
                continue
            parts.append(
                f'<path d="{path_d}" fill="none" stroke="{sl.color}"'
                f' stroke-width="{sw:.2f}"'
                f' stroke-linecap="round" stroke-linejoin="round"/>'
            )

    parts.append("</svg>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _merge_close_clusters(
    centers: np.ndarray,
    labels_flat: np.ndarray,
    h: int,
    w: int,
    threshold: float = 35.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Agglomerative merge: keep merging the nearest pair until the
    closest pair is further than `threshold` apart, OR we reach a
    target count of ≥3 clusters.

    Uses pixel-weighted centre updates to maintain accurate colours.
    """
    centers_f = centers.astype(np.float64).copy()
    K = len(centers_f)
    pixel_counts = np.array([
        np.count_nonzero(labels_flat == i) for i in range(K)
    ], dtype=np.int64)
    alive = np.ones(K, dtype=bool)
    merge_into = np.arange(K, dtype=np.int32)

    while True:
        alive_ids = np.where(alive)[0]
        if len(alive_ids) <= 3:
            break

        # Find the closest pair among alive clusters
        best_dist = float("inf")
        best_i, best_j = -1, -1
        for idx_a in range(len(alive_ids)):
            for idx_b in range(idx_a + 1, len(alive_ids)):
                i, j = alive_ids[idx_a], alive_ids[idx_b]
                d = np.linalg.norm(centers_f[i] - centers_f[j])
                if d < best_dist:
                    best_dist = d
                    best_i, best_j = i, j

        if best_dist > threshold:
            break

        # Merge smaller into larger
        if pixel_counts[best_i] < pixel_counts[best_j]:
            src, dst = best_i, best_j
        else:
            src, dst = best_j, best_i

        ni, nj = pixel_counts[src], pixel_counts[dst]
        centers_f[dst] = (centers_f[dst] * nj + centers_f[src] * ni) / (ni + nj)
        pixel_counts[dst] += ni
        alive[src] = False
        merge_into[src] = dst

    # Resolve chains in merge_into
    for i in range(K):
        root = i
        while merge_into[root] != root:
            root = merge_into[root]
        merge_into[i] = root

    # Renumber
    alive_ids = np.where(alive)[0]
    id_map = {old: new for new, old in enumerate(alive_ids)}
    old_to_new = np.array([id_map[merge_into[i]] for i in range(K)], dtype=np.int32)

    new_centers = centers_f[alive_ids]
    new_labels = old_to_new[labels_flat].reshape(h, w)
    return new_centers.astype(np.float32), new_labels


def _polygon_area(pts: np.ndarray) -> float:
    """Signed area via shoelace formula."""
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _point_to_segment_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Distance from point p to line segment a-b in N-d color space."""
    ab = b - a
    ab_len_sq = float(np.dot(ab, ab))
    if ab_len_sq < 1e-10:
        return float(np.linalg.norm(p - a))
    t = np.clip(float(np.dot(p - a, ab)) / ab_len_sq, 0.0, 1.0)
    return float(np.linalg.norm(p - (a + t * ab)))


def _smooth_contour(pts: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    """Smooth a closed contour's coordinates with a 1D Gaussian filter.

    Removes pixel-grid staircase artifacts from marching-squares contours
    while preserving corners and overall shape.  The sigma is in units of
    contour-point spacing (≈1 upscaled pixel).

    Short contours get proportionally less smoothing so small shapes
    aren't distorted.
    """
    N = len(pts)
    if N < 6 or sigma < 0.1:
        return pts

    # Cap sigma to prevent over-smoothing short contours — never
    # average over more than 1/6 of the contour's total length.
    effective_sigma = min(sigma, N / 6.0)
    if effective_sigma < 0.5:
        return pts

    # 1D Gaussian along the contour (wrap for closed curve)
    sx = gaussian_filter1d(pts[:, 0], sigma=effective_sigma, mode='wrap')
    sy = gaussian_filter1d(pts[:, 1], sigma=effective_sigma, mode='wrap')
    smoothed = np.column_stack([sx, sy])

    # Discrete curvature: angle change at each vertex
    d1 = np.roll(pts, -1, axis=0) - pts        # forward edge
    d0 = pts - np.roll(pts, 1, axis=0)          # backward edge
    cross = d0[:, 0] * d1[:, 1] - d0[:, 1] * d1[:, 0]
    dot = d0[:, 0] * d1[:, 0] + d0[:, 1] * d1[:, 1]
    angle = np.abs(np.arctan2(cross, dot))      # 0=straight, pi=reversal

    # Blend: corners (angle > ~25°) keep original position
    corner_threshold = 0.45  # radians (~25°)
    alpha = np.clip((angle - corner_threshold) / (np.pi/2 - corner_threshold), 0, 1)
    alpha = alpha[:, np.newaxis]  # broadcast to (N,2)
    result = smoothed * (1.0 - alpha) + pts * alpha

    return result


def _merge_collinear(points: np.ndarray, tol: float) -> np.ndarray:
    """Collapse consecutive near-collinear points into straight runs.

    Walks the polygon and extends the current run as long as all
    intermediate points stay within *tol* of the chord from the run
    start to the candidate endpoint.  When a point would exceed *tol*,
    the previous endpoint is emitted and a new run begins.

    Only merges runs that span at least 3 points to avoid collapsing
    short curved sections that happen to be momentarily straight.
    """
    if len(points) < 4:
        return points
    keep = [0]
    run_start = 0
    for i in range(2, len(points)):
        seg = points[run_start:i + 1]
        p0, p1 = seg[0], seg[-1]
        d = p1 - p0
        seg_len = np.linalg.norm(d)
        if seg_len < 1e-10:
            continue
        n = np.array([-d[1], d[0]]) / seg_len
        perp = np.abs((seg[1:-1] - p0) @ n)
        if len(perp) > 0 and float(np.max(perp)) > tol:
            # Only merge if we skipped at least 2 intermediate points
            # (run of 4+ points).  Short runs (3 points) might be
            # part of a gentle curve and should keep their detail.
            if (i - 1) - run_start >= 3:
                keep.append(i - 1)
            else:
                # Keep all points in short non-collinear section
                for j in range(run_start + 1, i):
                    keep.append(j)
            run_start = i - 1
    keep.append(len(points) - 1)
    return points[sorted(set(keep))]


def _fit_contour(
    contour: np.ndarray,
    simplify_epsilon: float,
    max_error: float,
    corner_threshold: float,
    line_tolerance: float = 0.25,
) -> str:
    """Simplify contour and fit smooth closed Bézier curves; return SVG path d."""
    simplified = cv2.approxPolyDP(
        contour.reshape(-1, 1, 2).astype(np.float32),
        simplify_epsilon,
        closed=True,
    ).reshape(-1, 2)

    if len(simplified) < 3:
        return ""

    try:
        curve = fit_closed_bezier(
            simplified, max_error=max_error,
            corner_threshold=corner_threshold,
            line_tolerance=line_tolerance,
        )
    except Exception:
        return ""

    return _curve_to_d(curve)


def _curve_to_d(curve) -> str:
    """Convert a FittedCurve to an SVG path `d` string."""
    if not curve.segments:
        return ""

    p = curve.segments[0]
    parts = [f"M{p.p0[0]:.2f},{p.p0[1]:.2f}"]
    for seg in curve.segments:
        if seg.is_line:
            parts.append(f"L{seg.p3[0]:.2f},{seg.p3[1]:.2f}")
        else:
            parts.append(
                f"C{seg.p1[0]:.2f},{seg.p1[1]:.2f} "
                f"{seg.p2[0]:.2f},{seg.p2[1]:.2f} "
                f"{seg.p3[0]:.2f},{seg.p3[1]:.2f}"
            )
    if curve.is_closed:
        parts.append("Z")

    return "".join(parts)


def _process_stroke_cluster(
    labels: np.ndarray,
    cluster_idx: int,
    w: int,
    h: int,
    scale: int,
    simplify_epsilon: float,
    max_error: float,
    corner_threshold: float,
    line_tolerance: float,
) -> tuple[list[str], list[float]] | None:
    """Skeleton-based stroke reconstruction for thin line-art clusters.

    Returns (paths, widths) or None if no valid strokes found.
    """
    S = scale
    mask_k = (labels == cluster_idx).astype(np.uint8)

    # Upscale for sub-pixel skeleton quality
    mask_up = cv2.resize(mask_k * 255, (w * S, h * S),
                         interpolation=cv2.INTER_NEAREST)

    # Smooth blocky upscaled edges before skeletonization
    mask_smooth = cv2.GaussianBlur(
        mask_up.astype(np.float32), (0, 0), sigmaX=0.6 * S
    )
    mask_bin = (mask_smooth > 100).astype(np.uint8)

    # Distance transform for width estimation
    dt_up = distance_transform_edt(mask_bin)

    # Skeletonize
    skel = _skeletonize(mask_bin > 0).astype(np.uint8)

    # Prune short branches
    skel = _prune_skeleton(skel, min_branch_length=max(3, S))

    # Trace ordered paths through skeleton
    paths = _trace_skeleton_paths(skel)

    s_paths: list[str] = []
    s_widths: list[float] = []

    for pts in paths:
        if len(pts) < 3:
            continue

        # pts is Nx2 (x, y) in upscaled coordinates
        # Estimate width from distance transform at each point
        widths = []
        for px, py in pts:
            ix = int(np.clip(round(px), 0, mask_bin.shape[1] - 1))
            iy = int(np.clip(round(py), 0, mask_bin.shape[0] - 1))
            widths.append(dt_up[iy, ix] * 2.0)  # diameter
        avg_w = float(np.mean(widths)) / S
        if avg_w < 0.3:
            continue

        # Scale to original coordinates
        pts_orig = pts.astype(np.float64) / S

        # Simplify (RDP)
        pts_s = cv2.approxPolyDP(
            pts_orig.reshape(-1, 1, 2).astype(np.float32),
            simplify_epsilon, closed=False,
        ).reshape(-1, 2)

        if len(pts_s) < 2:
            continue

        is_closed = np.linalg.norm(pts_s[0] - pts_s[-1]) < 2.0
        try:
            curve = fit_bezier_path(
                pts_s,
                max_error=max_error,
                corner_threshold=corner_threshold,
                is_closed=is_closed,
                line_tolerance=line_tolerance,
            )
        except Exception:
            continue

        d = _curve_to_d(curve)
        if d:
            s_paths.append(d)
            s_widths.append(max(0.5, avg_w))

    if not s_paths:
        return None
    return s_paths, s_widths


def _bgr_to_hex(color) -> str:
    b = max(0, min(255, int(color[0])))
    g = max(0, min(255, int(color[1])))
    r = max(0, min(255, int(color[2])))
    return f"#{r:02x}{g:02x}{b:02x}"
