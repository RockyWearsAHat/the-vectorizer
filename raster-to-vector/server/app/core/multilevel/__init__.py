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

import math
import re
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from dataclasses import dataclass
from scipy.ndimage import distance_transform_edt, gaussian_filter1d
from skimage.measure import find_contours
from ..curve_fitting import (
    fit_closed_bezier, fit_bezier_path, reduce_nodes, FittedCurve,
    enforce_g1_continuity, merge_segments_artistic,
)
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
    shapes: list[str] | None = None  # SVG shape elements (circle, rect, ellipse)


@dataclass
class StrokeLayer:
    paths: list[str]        # SVG path `d` strings for stroke centre lines
    widths: list[float]     # stroke-width for each path
    color: str              # hex stroke colour


@dataclass
class GradientDef:
    id: str               # gradient ID for SVG defs
    x1: float
    y1: float
    x2: float
    y2: float
    color_start: str      # hex start color
    color_end: str        # hex end color
    color_mid: str | None = None  # optional hex mid-stop color


@dataclass
class MultilevelResult:
    layers: list[VectorLayer]
    stroke_layers: list[StrokeLayer]
    width: int
    height: int
    background_color: str
    path_count: int
    node_count: int
    gradient_defs: list[GradientDef] | None = None
    is_line_art: bool = False


# ---------------------------------------------------------------------------
# Background detection
# ---------------------------------------------------------------------------

def detect_background(image_bgr: np.ndarray) -> tuple[np.ndarray, int]:
    """Return (BGR colour, gray value) of the dominant border colour.

    Samples a 5-pixel-wide border strip and uses the mean of all
    non-ink pixels (gray > 128).  The mean captures subtle warm tones
    in the background that the median misses, while the ink filter
    prevents dark artwork on the border from pulling the result too dark.
    """
    h, w = image_bgr.shape[:2]
    bw = min(5, max(1, h // 8), max(1, w // 8))  # border width in px
    border = np.concatenate([
        image_bgr[:bw, :].reshape(-1, 3),
        image_bgr[-bw:, :].reshape(-1, 3),
        image_bgr[bw:-bw, :bw].reshape(-1, 3),
        image_bgr[bw:-bw, -bw:].reshape(-1, 3),
    ])
    # Filter out clearly dark pixels (ink/artwork touching the border)
    gray_vals = (0.299 * border[:, 2] + 0.587 * border[:, 1]
                 + 0.114 * border[:, 0])
    light_mask = gray_vals > 128
    if light_mask.sum() > 10:
        light_border = border[light_mask]
    else:
        light_border = border  # fallback: use all
    # Mean preserves warm tones better than median for near-white backgrounds
    color = light_border.mean(axis=0).astype(np.uint8)
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


def _compute_lab_gradient_magnitude(image_lab: np.ndarray) -> np.ndarray:
    """Build a normalized [0,1] Lab gradient magnitude map."""
    grad_sq = np.zeros(image_lab.shape[:2], dtype=np.float32)
    for channel in range(3):
        plane = image_lab[:, :, channel]
        gx = cv2.Sobel(plane, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(plane, cv2.CV_32F, 0, 1, ksize=3)
        grad_sq += gx * gx + gy * gy
    lab_grad_mag = np.sqrt(grad_sq)
    grad_scale = float(np.percentile(lab_grad_mag, 95.0))
    if grad_scale > 1e-6:
        lab_grad_mag = lab_grad_mag / grad_scale
    return np.clip(lab_grad_mag, 0.0, 1.0).astype(np.float32)


def _compute_hard_edge_confidence(
    image_lab: np.ndarray,
    soft_raw: np.ndarray,
    d_self: np.ndarray,
    d_other: np.ndarray,
    *,
    lab_grad_mag: np.ndarray | None = None,
) -> np.ndarray:
    """Estimate [0,1] confidence that the local 0.5 crossing is a hard edge."""
    if lab_grad_mag is None:
        lab_grad_mag = _compute_lab_gradient_magnitude(image_lab)

    boundary_focus = 1.0 - np.clip(np.abs(soft_raw - 0.5) / 0.18, 0.0, 1.0)
    boundary_focus = boundary_focus * boundary_focus

    # Near the decision boundary, sqrt(d_self) + sqrt(d_other) is a cheap
    # proxy for how far apart the competing Lab colours are.
    separation_proxy = np.sqrt(np.maximum(d_self, 0.0)) + np.sqrt(np.maximum(d_other, 0.0))
    separation_conf = np.clip((separation_proxy - 24.0) / 24.0, 0.0, 1.0)

    hard_edge = boundary_focus * lab_grad_mag * separation_conf
    return np.clip(hard_edge, 0.0, 1.0).astype(np.float32)


def _build_local_iso_map(
    base_iso: float,
    hard_edge_conf: np.ndarray,
    edge_weight: np.ndarray,
    *,
    mediator: float,
) -> np.ndarray:
    """Tighten the local extraction iso toward 0.5 where edges are confident.

    The base iso stays permissive for feature preservation. Hard-edge confidence
    only spends a bounded fraction of the remaining gap to the neutral 0.5
    decision boundary, so smooth/gradient regions keep the legacy behavior.
    """
    edge_locked_conf = hard_edge_conf * np.clip(edge_weight + 0.10, 0.0, 1.0)
    mediator_softening = 1.0 - min(max(mediator, 0.0), 1.0) * 0.35
    tighten_strength = 0.20 * mediator_softening
    return base_iso + (0.5 - base_iso) * (edge_locked_conf * tighten_strength)


# ---------------------------------------------------------------------------
# Core: quantize → adaptive soft-membership → sub-pixel contour → fit
# ---------------------------------------------------------------------------

def multilevel_vectorize(
    image_bgr: np.ndarray,
    *,
    num_levels: int = 0,
    simplify_epsilon: float = 1.5,
    max_error: float = 2.0,
    line_tolerance: float = 1.2,
    corner_threshold: float = 55.0,
    min_contour_area: int = 12,
    contour_scale: int = 4,
    smooth_sigma: float = 0.50,
    mediator_threshold: float = 0.3,   # backward compat (used in absorption)
) -> MultilevelResult:
    h, w = image_bgr.shape[:2]

    if len(image_bgr.shape) == 2:
        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)

    _t_start = time.time()
    bg_color, bg_gray = detect_background(image_bgr)
    bg_hex = _bgr_to_hex(bg_color)
    source_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    # --- Step 0: Edge-density map for adaptive processing ---
    edge_weight = _compute_edge_weight(image_bgr)
    source_lab_grad = _compute_lab_gradient_magnitude(source_lab)

    # --- Step 0b: Dual denoise ------------------------------------------------
    # Bilateral keeps edges while killing SD/AI per-pixel noise in flat regions.
    # For large images (>4MP), use smaller filter radii for speed.
    # Fast Gaussian blur instead of expensive bilateral filter.
    # Vectorization quantizes colors anyway, so edge-preserving denoising
    # is overkill. GaussianBlur is 10-100× faster.
    denoised_km = cv2.GaussianBlur(image_bgr, (7, 7), 0)
    denoised_dist = cv2.GaussianBlur(image_bgr, (5, 5), 0)
    _t_denoise = time.time()

    # --- Step 1: K-means colour quantization ---
    # K-means in LAB space for perceptually-uniform clustering.
    # This prevents dark chromatic colors (e.g. burgundy) from being
    # absorbed into black, since LAB separates luminance from chrominance.
    # For large images, subsample pixels for K-means center discovery
    # (centers converge with ~500K samples), then assign ALL pixels
    # using per-channel distance maps which are much faster than
    # pixel-by-pixel broadcasting.
    # Cap K lower for large images to limit total processing time.
    # Chromatic images need more clusters to preserve minority color regions
    _hsv_check = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    _mean_sat = float(_hsv_check[:, :, 1].mean())
    # Detect color using fraction of saturated pixels, not mean.
    # Mean misses minority chromatic regions (e.g. 2.5% red wheels on gray car).
    _sat_frac = float(np.count_nonzero(_hsv_check[:, :, 1] > 30)) / max(1, h * w)
    _has_color = _mean_sat > 15 or _sat_frac > 0.01
    # Dynamic max_k: match documented tiers that gave best quality.
    # Color distinction removed — extra clusters fragment features more than
    # they help color fidelity (test4 regressed 96.5% → 83% with K=10 vs K=7).
    if h * w > 16_000_000:
        _max_k = 8
    elif h * w > 8_000_000:
        _max_k = 12
    elif h * w > 4_000_000:
        _max_k = 12
    else:
        _max_k = 20
    K = _estimate_initial_k(image_bgr, max_k=_max_k) if num_levels <= 0 else max(2, min(num_levels, 64))
    # No fitting tolerance scaling — keep configured values to preserve width accuracy
    _km_iters = 10
    _km_attempts = 10
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, _km_iters, 0.5)
    # Convert to LAB for perceptually-uniform clustering
    _lab_for_km = cv2.cvtColor(denoised_km, cv2.COLOR_BGR2LAB)
    all_pixels = _lab_for_km.reshape(-1, 3).astype(np.float32)
    # Weight chrominance channels to prevent dark chromatic colors (e.g. burgundy)
    # from being absorbed into black. In LAB, L dominates (0-255) while a,b have
    # smaller useful range — dark red vs black differs by only ~16 in a,b.
    # Scaling a,b by 1.5 increases their effective ΔE from ~30 to ~45.
    _CHROMA_WEIGHT = 2.0
    all_pixels[:, 1] *= _CHROMA_WEIGHT
    all_pixels[:, 2] *= _CHROMA_WEIGHT
    n_pixels = len(all_pixels)
    _KM_SAMPLE = 500_000
    # Seed OpenCV's internal RNG so K-means++ is reproducible regardless of
    # what other images were processed before this one in the batch.
    cv2.setRNGSeed(42)
    if n_pixels > _KM_SAMPLE * 1.5:
        # Subsample for center discovery (in LAB space)
        rng = np.random.default_rng(42)
        idx = rng.choice(n_pixels, _KM_SAMPLE, replace=False)
        sample_pixels = all_pixels[idx]
        _, _, centers = cv2.kmeans(
            sample_pixels, K, None, criteria, _km_attempts, cv2.KMEANS_PP_CENTERS
        )
        # Fast label assignment using ||p-c||² = ||p||² - 2·p·cᵀ + ||c||²
        # Matrix multiply is much faster than broadcasting for large N.
        # Use LAB pixels for distance computation (perceptually uniform)
        _flat_pixels = all_pixels  # already LAB float32
        _centers_f32 = centers.astype(np.float32)
        _p_sq = np.sum(_flat_pixels ** 2, axis=1)  # (N,)
        _c_sq = np.sum(_centers_f32 ** 2, axis=1)  # (K,)
        # p·cᵀ is (N, K) — use matrix multiply in chunks to control memory
        _chunk_size = 2_000_000
        labels = np.empty(n_pixels, dtype=np.int32)
        for _cs in range(0, n_pixels, _chunk_size):
            _ce = min(_cs + _chunk_size, n_pixels)
            _pc = _flat_pixels[_cs:_ce] @ _centers_f32.T  # (chunk, K)
            _dists = _p_sq[_cs:_ce, None] - 2 * _pc + _c_sq[None, :]
            labels[_cs:_ce] = np.argmin(_dists, axis=1)
    else:
        _, labels, centers = cv2.kmeans(
            all_pixels, K, None, criteria, 6, cv2.KMEANS_PP_CENTERS
        )
        labels = labels.flatten()

    # Undo chrominance weighting before converting centers back to BGR
    centers[:, 1] /= _CHROMA_WEIGHT
    centers[:, 2] /= _CHROMA_WEIGHT
    centers_lab = centers.copy()
    centers_bgr = np.zeros_like(centers)
    for i in range(len(centers)):
        lab_pixel = centers[i].reshape(1, 1, 3).astype(np.uint8)
        bgr_pixel = cv2.cvtColor(lab_pixel, cv2.COLOR_LAB2BGR)
        centers_bgr[i] = bgr_pixel.reshape(3).astype(np.float32)
    centers = centers_bgr

    # LAB image for perceptual distance merging
    # Note: lab_image uses UNWEIGHTED LAB values from original conversion
    lab_image = _lab_for_km.astype(np.float32)

    # --- Step 1b: Merge close cluster centres (LAB-aware) ---
    centers, labels = _merge_close_clusters(
        centers, labels.flatten(), h, w, threshold=20.0,
        lab_image=lab_image, lab_threshold=6.0,
    )
    K = len(centers)
    _t_kmeans = time.time()

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

    # --- Step 1d: Gradient-aware merge ---
    # Collapse cluster pairs whose boundary has low color contrast
    # in the source image, preserving boundaries along real edges.
    labels, centers_f, bg_cluster = _gradient_aware_merge(
        labels, centers_f, denoised_dist, bg_cluster,
        boundary_contrast_thresh=22.0,
        max_color_dist=60.0,
    )
    K = len(centers_f)
    _t_merge = time.time()

    centers_u = centers_f.astype(np.uint8)

    # --- Step 1e: Merge dark chromatic clusters into dark achromatic ---
    # Human chromatic sensitivity drops to near-zero at very low luminance.
    # Dark reddish/brownish clusters (e.g. dark shadow reflections on car
    # panels) are perceptually indistinguishable from dark gray/black.
    # Leaving them as separate clusters creates false colored polygons
    # in areas that look neutral in the source image.
    _centers_lab_merge = np.array([
        cv2.cvtColor(c.reshape(1, 1, 3), cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
        for c in centers_u
    ])
    _dark_chroma_merged = []
    for k in range(K):
        if k == bg_cluster:
            continue
        _L_k = float(_centers_lab_merge[k, 0])
        if _L_k > 35:  # only very dark clusters
            continue
        _chroma_k = math.sqrt(
            (float(_centers_lab_merge[k, 1]) - 128.0) ** 2
            + (float(_centers_lab_merge[k, 2]) - 128.0) ** 2
        )
        if _chroma_k < 12:  # already achromatic
            continue
        # Find nearest dark achromatic cluster
        _best_target = -1
        _best_d = float('inf')
        for j in range(K):
            if j == k:
                continue
            _L_j = float(_centers_lab_merge[j, 0])
            _chroma_j = math.sqrt(
                (float(_centers_lab_merge[j, 1]) - 128.0) ** 2
                + (float(_centers_lab_merge[j, 2]) - 128.0) ** 2
            )
            if _chroma_j > 12:  # target must be achromatic
                continue
            if abs(_L_k - _L_j) > 30:  # must be similarly dark
                continue
            _d = float(np.linalg.norm(_centers_lab_merge[k] - _centers_lab_merge[j]))
            if _d < _best_d:
                _best_d = _d
                _best_target = j
        if _best_target >= 0 and _best_d < 40:
            _dark_chroma_merged.append((k, _best_target, _best_d))

    if _dark_chroma_merged:
        for src, dst, d in _dark_chroma_merged:
            labels[labels == src] = dst
        # Rebuild contiguous IDs
        alive = sorted(set(int(v) for v in np.unique(labels)))
        remap = np.full(K, -1, dtype=np.int32)
        for new_id, old_id in enumerate(alive):
            remap[old_id] = new_id
        labels = remap[labels]
        centers_f = centers_f[alive]
        centers_u = centers_f.astype(np.uint8)
        if bg_cluster >= 0 and remap[bg_cluster] >= 0:
            bg_cluster = int(remap[bg_cluster])
        else:
            bg_cluster = -1
        K = len(centers_f)

    # --- Step 2: Distance from every pixel to every cluster centre ---
    # Use the mildly denoised image (tight color gate sc=5) so SD
    # static doesn't pollute soft-field boundaries, while real edges
    # remain razor-sharp.
    # CIELAB soft field: compute distances in perceptual color space
    # so contour boundaries fall at perceptual midpoints, not BGR midpoints.
    lab_dist_img = cv2.cvtColor(denoised_dist, cv2.COLOR_BGR2LAB).astype(np.float32)
    centers_lab_f = np.array([
        cv2.cvtColor(c.reshape(1, 1, 3), cv2.COLOR_BGR2LAB).reshape(3).astype(np.float32)
        for c in centers_u
    ])
    # Apply same chrominance weighting as K-means so soft field distances
    # are consistent with cluster assignments. Without this, chromatic clusters
    # (e.g. yellow) lose soft field strength vs nearby achromatic clusters.
    lab_dist_img[:, :, 1] *= _CHROMA_WEIGHT
    lab_dist_img[:, :, 2] *= _CHROMA_WEIGHT
    centers_lab_f[:, 1] *= _CHROMA_WEIGHT
    centers_lab_f[:, 2] *= _CHROMA_WEIGHT

    # Compute squared-distance map (skip sqrt — much faster).
    # Use squared distances throughout: soft = d²_other / (d²_k + d²_other).
    # iso threshold is adjusted: iso_sq = iso² / (2*iso² - 2*iso + 1).
    _lab_flat = lab_dist_img.reshape(-1, 3)
    _lab_p_sq = np.sum(_lab_flat ** 2, axis=1)  # (N,)
    _lab_c_sq = np.sum(centers_lab_f ** 2, axis=1)  # (K,)
    _DIST_CHUNK = 4_000_000
    _n_dist = h * w
    dist_map_flat = np.empty((_n_dist, K), dtype=np.float32)
    for _ds in range(0, _n_dist, _DIST_CHUNK):
        _de = min(_ds + _DIST_CHUNK, _n_dist)
        _pc = _lab_flat[_ds:_de] @ centers_lab_f.T  # (chunk, K)
        _sq_dists = _lab_p_sq[_ds:_de, None] - 2 * _pc + _lab_c_sq[None, :]
        np.maximum(_sq_dists, 0, out=_sq_dists)
        np.sqrt(_sq_dists, out=_sq_dists)
        dist_map_flat[_ds:_de] = _sq_dists
    dist_map = dist_map_flat.reshape(h, w, K)

    # Pre-compute nearest-two clusters for fast soft-field computation.
    _nn_idx = np.argpartition(dist_map, min(2, K - 1), axis=2)[:, :, :2]
    _i_grid, _j_grid = np.mgrid[:h, :w]
    _nn1_idx = _nn_idx[:, :, 0]
    _nn2_idx = _nn_idx[:, :, 1]
    _nn1_dist = dist_map[_i_grid, _j_grid, _nn1_idx]
    _nn2_dist = dist_map[_i_grid, _j_grid, _nn2_idx]

    # --- Step 3: Lightest-first painter's algorithm ---
    # Lightest clusters paint first (background), darkest paint last (details on top).
    grays = np.array([
        int(cv2.cvtColor(c.reshape(1, 1, 3), cv2.COLOR_BGR2GRAY)[0, 0])
        for c in centers_u
    ])
    order = np.argsort(-grays)

    # --- Step 3b: Classify clusters as FILL vs MEDIATOR ---
    # For large images, skip mediator analysis entirely — it's mainly
    # useful for line art, and costs too much on high-res photos.
    _skip_mediator = False
    _med_scale = 2 if h * w > 1_000_000 else 1
    if _med_scale > 1:
        _med_labels = labels[::_med_scale, ::_med_scale]
        _med_h, _med_w = _med_labels.shape
    else:
        _med_labels = labels
        _med_h, _med_w = h, w
    morph_kernel = np.ones((3, 3), np.uint8)
    cluster_mediator_score = np.zeros(K, dtype=np.float64)
    cluster_mean_thick = np.zeros(K, dtype=np.float64)
    cluster_interior_frac = np.zeros(K, dtype=np.float64)
    cluster_pix_count = np.zeros(K, dtype=np.int64)
    total_pixels = labels.shape[0] * labels.shape[1]
    if not _skip_mediator:
      for k in range(K):
        if k == bg_cluster:
            continue
        mask_k = (_med_labels == k).astype(np.uint8)
        pix_count_med = int(np.count_nonzero(mask_k))
        pix_count = pix_count_med * (_med_scale * _med_scale)
        if pix_count == 0:
            continue

        # Thickness: distance transform on downsampled mask
        dt = cv2.distanceTransform(mask_k, cv2.DIST_L2, 5)
        fg_dts = dt[mask_k > 0]
        mean_thick = float(fg_dts.mean()) * _med_scale
        interior_thresh = 2.0 / _med_scale if _med_scale > 1 else 2.0
        interior_frac = float(np.count_nonzero(fg_dts > interior_thresh)) / pix_count_med
        cluster_mean_thick[k] = mean_thick
        cluster_interior_frac[k] = interior_frac
        cluster_pix_count[k] = pix_count

        # Find neighbor cluster IDs on downsampled labels
        dilated = cv2.dilate(mask_k, morph_kernel, iterations=1)
        border_zone = (dilated > 0) & (mask_k == 0)
        unique_neighbors = sorted(
            set(int(n) for n in np.unique(_med_labels[border_zone])) - {k}
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
        cluster_area_frac = pix_count / total_pixels
        if mean_thick <= 2.0 and min_interp < 30.0 and interior_frac < 0.20:
            # Score: closer to 0 interp_dist and 0 interior → higher score
            interp_score = max(0.0, 1.0 - min_interp / 30.0)
            interior_penalty = max(0.0, 1.0 - interior_frac / 0.20)
            cluster_mediator_score[k] = interp_score * interior_penalty
            # Area damping: larger clusters are less likely to be pure AA
            # artifacts.  Gradually reduce mediator score for clusters
            # covering >0.3% of image, full protection above 1%.
            if cluster_area_frac > 0.003:
                area_damping = max(0.0, 1.0 - (cluster_area_frac - 0.003) / 0.007)
                cluster_mediator_score[k] *= area_damping
            # Color-distance protection: clusters far from background
            # are real content, not AA artifacts
            if bg_cluster >= 0:
                _bg_lab = centers_lab_f[bg_cluster]
                _cl_lab = centers_lab_f[k]
                _cd_bg = float(np.linalg.norm(_bg_lab - _cl_lab))
                if _cd_bg > 20:
                    bg_damping = max(0.0, 1.0 - (_cd_bg - 20) / 20)
                    cluster_mediator_score[k] *= bg_damping
        # else: stays 0 → full fill

    # --- Step 3c: Absorb mediator clusters into their nearest real neighbor ---
    # AA-interpolation clusters (gray pixels between black and white) should
    # not be rendered as separate layers.  Reassign their pixels to the
    # darkest neighboring cluster so the soft-field boundary naturally falls
    # at the visual edge.  This eliminates gray splotches entirely.
    mediator_ids = [k for k in range(K) if cluster_mediator_score[k] > 0.3 and k != bg_cluster]
    if mediator_ids:
        for k in mediator_ids:
            # Use downsampled labels for neighbor finding (fast)
            mask_k = (_med_labels == k).astype(np.uint8)
            dilated = cv2.dilate(mask_k, morph_kernel, iterations=1)
            border_zone = (dilated > 0) & (mask_k == 0)
            neighbors = sorted(
                set(int(n) for n in np.unique(_med_labels[border_zone]))
                - {k} - set(mediator_ids)
            )
            if not neighbors:
                neighbors = sorted(
                    set(int(n) for n in np.unique(_med_labels[border_zone]))
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
        centers_lab_f = np.array([
            cv2.cvtColor(c.reshape(1, 1, 3), cv2.COLOR_BGR2LAB).reshape(3).astype(np.float32)
            for c in centers_u
        ])
        # Reindex dist_map columns and remap nearest-two indices
        dist_map = dist_map[:, :, alive].copy()
        _nn1_idx = remap[_nn1_idx]
        _nn2_idx = remap[_nn2_idx]
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

    # --- Step 3e: Classify clusters as thin-feature vs fill ---
    # Thin clusters (text, lines, serifs) need higher iso to avoid
    # thickening and tighter Bézier fitting.
    cluster_is_thin = np.zeros(K, dtype=bool)
    for k in range(K):
        if k == bg_cluster:
            continue
        if cluster_mean_thick[k] > 0 and cluster_mean_thick[k] < 2.5:
            if cluster_interior_frac[k] < 0.25:
                cluster_is_thin[k] = True

    # --- Step 3f: Detect gradient regions between adjacent clusters ---
    gradient_defs: list[GradientDef] = []
    gradient_fill_map: dict[int, str] = {}  # cluster_idx → gradient ID
    _detect_gradients(
        labels, centers_f, denoised_dist, bg_cluster,
        w, h, gradient_defs, gradient_fill_map,
    )

    layers: list[VectorLayer] = []
    stroke_layers: list[StrokeLayer] = []
    total_paths = 0
    total_nodes = 0

    # Adaptive superresolution: total pixel budget shared across all
    # rendered clusters.  Simple images (few K) get high S; complex
    # images (many K) trade S for color fidelity.
    K_render = max(1, K - 1 if bg_cluster >= 0 else K)
    _TOTAL_BUDGET = 2_000_000_000
    # Adaptive min S: higher S = smoother contour edges.
    _min_S = 2 if h * w > 8_000_000 else (2 if h * w > 4_000_000 else 3)
    _max_S = 4 if h * w > 4_000_000 else contour_scale
    S = max(_min_S, min(_max_S, int(math.sqrt(_TOTAL_BUDGET / max(h * w * K_render, 1)))))
    print(f"[DIAG] {w}x{h} K_initial={K} K_final={K_render} S={S}")

    _t_preprocess = time.time()
    _t_curves_total = 0.0

    # --- Line art fast path ---
    # For clean grayscale line art, bypass soft field entirely and use Otsu
    # threshold directly.  The soft field assigns AA fringe pixels to light
    # clusters, making lines faint.  Direct thresholding gives crisp strokes.
    _hsv_la = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    _mean_sat_la = float(_hsv_la[:, :, 1].mean())
    _is_line_art = False
    if _mean_sat_la < 20 and K <= 6:
        _gray_la = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        _otsu_val, _ = cv2.threshold(_gray_la, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _bg_frac = float((_gray_la > _otsu_val).sum()) / _gray_la.size
        if _bg_frac > 0.70:
            _is_line_art = True

    if _is_line_art:
        # Hysteresis thresholding: strict core + lenient fringe (connected only)
        # Use Otsu directly as lenient to capture full AA fringe of thin text
        _strict_thresh = min(int(_otsu_val * 0.82), 145)
        _lenient_thresh = min(int(_otsu_val), 185)

        # Strict mask: definitely line pixels
        _strict_mask = ((_gray_la <= _strict_thresh) * 255).astype(np.uint8)
        # Lenient mask: possible line pixels (includes AA fringe)
        _lenient_mask = ((_gray_la <= _lenient_thresh) * 255).astype(np.uint8)

        # Find connected components in lenient mask
        _num_cc, _cc_labels = cv2.connectedComponents(_lenient_mask)
        # Keep only components that overlap with strict mask
        _strict_label_ids = set(np.unique(_cc_labels[_strict_mask > 0]).tolist()) - {0}
        _keep = np.isin(_cc_labels, list(_strict_label_ids))
        _binary_la = (_keep.astype(np.uint8) * 255)

        # Morph close to bridge small gaps
        _k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        _binary_la = cv2.morphologyEx(_binary_la, cv2.MORPH_CLOSE, _k_close)

        # Upscale for better contour quality
        if S > 1:
            _h_la, _w_la = _binary_la.shape[:2]
            _binary_la = cv2.resize(_binary_la, (_w_la * S, _h_la * S),
                                    interpolation=cv2.INTER_LINEAR)
            _binary_la = (_binary_la > 127).astype(np.uint8) * 255

        # Extract contours
        _cv_la, _hier_la = cv2.findContours(
            _binary_la, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE,
        )

        # Determine line (darkest) and background (lightest) colors
        _gray_centers = [
            0.114 * c[0] + 0.587 * c[1] + 0.299 * c[2] for c in centers_f
        ]
        _darkest_ci = int(np.argmin(_gray_centers))
        _line_hex = _bgr_to_hex(centers_u[_darkest_ci])

        # Group contours by hierarchy (outer + holes)
        _la_paths: list[str] = []
        _la_nodes = 0
        if _hier_la is not None and len(_cv_la) > 0:
            _hier0 = _hier_la[0]
            _groups_la: list[dict] = []
            _outer_set: set[int] = set()
            for idx in range(len(_cv_la)):
                if _hier0[idx][3] == -1:  # outer contour
                    _outer_set.add(idx)
                    _groups_la.append({'outer': idx, 'holes': []})
            _outer_list = {g['outer']: g for g in _groups_la}
            for idx in range(len(_cv_la)):
                parent = _hier0[idx][3]
                if parent != -1 and parent in _outer_list:
                    _outer_list[parent]['holes'].append(idx)

            # Sort by area, limit groups
            _groups_la.sort(
                key=lambda g: cv2.contourArea(_cv_la[g['outer']]), reverse=True,
            )
            _groups_la = _groups_la[:200]

            # Filter tiny contours
            _min_area_la = 2 * S * S  # smaller threshold for line art to preserve thin features
            _groups_la = [
                g for g in _groups_la
                if cv2.contourArea(_cv_la[g['outer']]) >= _min_area_la
            ]

            for g in _groups_la:
                parts: list[str] = []
                # Outer contour
                _pts = _cv_la[g['outer']].squeeze(1).astype(np.float64)
                if len(_pts) < 3:
                    continue
                # Skip smoothing for line art — binary threshold contours are already clean
                _pts = _pts / S  # back to original image space
                _d = _fit_contour(_pts, simplify_epsilon * 0.2, max_error * 0.3,
                                  corner_threshold, line_tolerance * 0.6)
                if not _d:
                    continue
                parts.append(_d)
                # Holes
                for hi in g['holes']:
                    _hpts = _cv_la[hi].squeeze(1).astype(np.float64)
                    if len(_hpts) < 3:
                        continue
                    # Skip smoothing for line art holes too
                    _hpts = _hpts / S
                    _hd = _fit_contour(_hpts, simplify_epsilon * 0.2, max_error * 0.3,
                                       corner_threshold, line_tolerance * 0.6)
                    if _hd:
                        parts.append(_hd)
                combined = " ".join(parts)
                _la_paths.append(combined)
                _la_nodes += (combined.count("C") + combined.count("L")
                              + combined.count("M"))

        _t_la_end = time.time()

        _la_layer = VectorLayer(
            paths=_la_paths,
            opacities=[1.0] * len(_la_paths),
            color=_line_hex,
            shapes=None,
        )
        return MultilevelResult(
            layers=[_la_layer],
            stroke_layers=[],
            width=w,
            height=h,
            background_color=bg_hex,
            path_count=len(_la_paths),
            node_count=_la_nodes,
            gradient_defs=None,
            is_line_art=True,
        )

    def _process_cluster(cluster_idx):
        # Skip background cluster — it's represented by the SVG canvas
        # or the background rect, so processing it wastes time.
        if cluster_idx == bg_cluster:
            return None

        _local_paths = 0
        _local_nodes = 0
        _local_curves_time = 0.0
        color_hex = _bgr_to_hex(centers_u[cluster_idx])
        mediator = cluster_mediator_score[cluster_idx]
        lightness = cluster_lightness[cluster_idx]
        _t_cl = time.time()

        # --- Archimedes soft membership (using pre-computed nearest-two) ---
        d_k = dist_map[:, :, cluster_idx]
        d_other = np.where(_nn1_idx == cluster_idx, _nn2_dist, _nn1_dist)

        denom = d_k + d_other
        denom = np.where(denom < 1e-10, 1e-10, denom)
        soft_raw = d_other / denom
        _t_soft = time.time()

        hard_edge_conf = _compute_hard_edge_confidence(
            source_lab,
            soft_raw,
            d_k,
            d_other,
            lab_grad_mag=source_lab_grad,
        )

        # --- Superresolution contour extraction ---
        # Blur at native resolution first (much cheaper than blurring
        # the upscaled image) then upscale with cubic interpolation.
        sigma_crisp_nat = max(0.36 - mediator * 0.12, 0.18)
        sigma_smooth_nat = max(0.66 - mediator * 0.30, 0.30)
        crisp_nat = cv2.GaussianBlur(soft_raw, (0, 0), sigmaX=sigma_crisp_nat)
        smooth_nat = cv2.GaussianBlur(soft_raw, (0, 0), sigmaX=sigma_smooth_nat)
        edge_locked_conf = hard_edge_conf * np.clip(edge_weight + 0.15, 0.0, 1.0)
        smooth_protection = edge_locked_conf * 0.03
        protected_smooth = smooth_nat * (1.0 - smooth_protection) + soft_raw * smooth_protection
        soft_blended = edge_weight * crisp_nat + (1.0 - edge_weight) * protected_smooth
        soft_up = cv2.resize(soft_blended, (w * S, h * S),
                             interpolation=cv2.INTER_LINEAR)
        hard_edge_up = cv2.resize(hard_edge_conf, (w * S, h * S),
                      interpolation=cv2.INTER_LINEAR)
        edge_weight_up = cv2.resize(edge_weight, (w * S, h * S),
                        interpolation=cv2.INTER_LINEAR)
        # Clamp to [0,1] — cubic interpolation can overshoot.
        np.clip(soft_up, 0.0, 1.0, out=soft_up)
        np.clip(hard_edge_up, 0.0, 1.0, out=hard_edge_up)
        np.clip(edge_weight_up, 0.0, 1.0, out=edge_weight_up)
        soft = soft_up

        # Adaptive iso: thin features use higher iso to shrink
        # contours toward their centre, reducing stroke thickening.
        # Higher K → more, smaller clusters → need lower iso to preserve features.
        # Base iso 0.44, with K-adaptive reduction for high cluster counts.
        _k_iso_adj = max(0.0, (K - 6) * 0.005)  # +0.005 per cluster above 6
        # Chrominance-aware iso: saturated/chromatic clusters (warm yellows,
        # reds, cyans) tend to lose boundary pixels to adjacent achromatic
        # clusters in the soft field. Lower iso slightly to expand their contours.
        _cl_lab = cv2.cvtColor(centers_u[cluster_idx].reshape(1, 1, 3), cv2.COLOR_BGR2LAB)[0, 0]
        _cl_chroma = float(np.sqrt((float(_cl_lab[1]) - 128) ** 2 + (float(_cl_lab[2]) - 128) ** 2))
        _chroma_iso_adj = min(0.015, max(0.0, (_cl_chroma - 20) * 0.0005))
        if cluster_is_thin[cluster_idx]:
            iso_level = 0.44 - _k_iso_adj * 0.5
        else:
            iso_level = 0.40 - _k_iso_adj - _chroma_iso_adj
        iso_map = _build_local_iso_map(
            iso_level,
            hard_edge_up,
            edge_weight_up,
            mediator=mediator,
        )
        shifted = soft - iso_map

        # --- Core contour at adaptive iso, FULL opacity ---
        layer_paths: list[str] = []
        layer_opacities: list[float] = []
        layer_shapes: list[str] | None = None
        # Fill tiny gaps in narrow features (e.g. thin letter legs)
        # Use minimal 3x3 kernel to avoid inflating junctions.
        if mediator < 0.3:
            _fill = (shifted > 0).astype(np.uint8)
            _kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            _closed = cv2.morphologyEx(_fill, cv2.MORPH_CLOSE, _kern)
            shifted[(_closed > 0) & (shifted <= 0)] = 0.01

        # Remove isolated small spike artifacts via morph open (fast O(n)).
        if not cluster_is_thin[cluster_idx]:
            _fg_mask = (shifted > 0).astype(np.uint8)
            _open_kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            _opened = cv2.morphologyEx(_fg_mask, cv2.MORPH_OPEN, _open_kern)
            # Zero out pixels that were removed by the opening
            shifted[(_fg_mask > 0) & (_opened == 0)] = -0.01

        # Island removal: remove small disconnected blobs that survive morph ops.
        # E.g. dark specks scattered on white car body panels in test2.
        # Safe: runs on per-cluster binary mask at S× resolution (not full labels).
        # Island removal disabled — too aggressive, removes legitimate small features.
        # The morph open + contour area filtering handles noise adequately.

        # Use OpenCV findContours (C++, ~10× faster than skimage
        # marching squares) on the binary thresholded field.  At S×
        # upscaling, integer coords give 1/S px precision — well
        # within Bézier fitting tolerance.
        _binary = (shifted > 0).astype(np.uint8)

        _t_contour_start = time.time()
        _cv_contours, _hierarchy = cv2.findContours(
            _binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE,
        )

        # Build hierarchy-aware groups: each outer contour paired with
        # its child holes.  RETR_CCOMP gives a two-level hierarchy:
        #   hierarchy[0][i] = [next, prev, first_child, parent]
        # parent == -1 → outer contour; parent != -1 → hole.
        # Group them so each outer + its holes become one evenodd path,
        # preventing unrelated sub-paths from interfering with hole cuts.
        _contour_groups: list[list[int]] = []  # [[outer_idx, hole_idx, ...], ...]
        if _hierarchy is not None and len(_cv_contours) > 0:
            hier = _hierarchy[0]
            for ci in range(len(_cv_contours)):
                if hier[ci][3] == -1:  # outer contour (no parent)
                    group = [ci]
                    # Walk child chain
                    child = hier[ci][2]  # first_child
                    while child != -1:
                        group.append(child)
                        child = hier[child][0]  # next sibling
                    _contour_groups.append(group)

        # Cap contour groups BEFORE fitting to avoid wasting compute.
        # Dynamic budget: aim for total fitting time < 7s across all clusters.
        # More clusters → fewer groups per cluster to stay in budget.
        pixel_count = h * w
        _fitting_budget_groups = max(300, 1000 // max(K_render, 1))
        if pixel_count < 4_000_000:
            MAX_GROUPS = min(1000, _fitting_budget_groups)
        else:
            MAX_GROUPS = _fitting_budget_groups
        if len(_contour_groups) > MAX_GROUPS:
            _raw_areas = [abs(cv2.contourArea(_cv_contours[g[0]])) / (S * S)
                          for g in _contour_groups]
            _keep_idx = np.argsort(_raw_areas)[::-1][:MAX_GROUPS]
            _contour_groups = [_contour_groups[i] for i in _keep_idx]

        # Convert contour groups to xy arrays, fit Béziers, build paths
        core_parts_per_group: list[list[str]] = []
        total_group_area: list[float] = []
        _t_cluster_fit_start = time.time()
        _fit_budget_exceeded = False
        _FIT_BUDGET = min(5.0, 20.0 / max(K_render, 1))  # dynamic: K=4→5s, K=8→2.5s
        for group in _contour_groups:
            # Check time budget at group level too
            if time.time() - _t_cluster_fit_start > _FIT_BUDGET:
                _fit_budget_exceeded = True
            group_parts: list[str] = []
            group_area = 0.0
            for ci in group:
                _c = _cv_contours[ci]
                pts = _c.squeeze(1).astype(np.float64)
                if len(pts) < 10:
                    continue
                area_raw = abs(cv2.contourArea(_c)) / (S * S)

                # Skip micro-fragments: tiny noisy shards that create
                # visual clutter at any zoom level.
                # Protect elongated thin shapes (lines), filter compact fragments.
                perim_raw = cv2.arcLength(_c, True) / S
                elongation = (perim_raw * perim_raw) / (area_raw + 1)
                # Uniform area multiplier — preserve detail at all sizes
                _area_mult = 1.0
                if elongation > 50 and perim_raw > 8:
                    # Elongated thin shape — likely a line, use lower threshold
                    min_frag_area = min_contour_area * 0.5 * _area_mult
                elif cluster_is_thin[cluster_idx]:
                    # Thin cluster — preserve more fragments
                    min_frag_area = min_contour_area * _area_mult
                else:
                    min_frag_area = min_contour_area * 1.5 * _area_mult
                if area_raw < min_frag_area:
                    continue

                xy = pts.copy()
                perim = float(np.sum(np.sqrt(np.sum(np.diff(xy, axis=0)**2, axis=1))))
                raw_area = abs(_polygon_area(xy))

                # Compactness filter: reject spindly shard fragments
                # (area/perim² < threshold) that look like noise.
                # A circle has compactness ~0.08; a thin sliver < 0.01.
                if perim > 1e-6:
                    compactness = raw_area / (perim * perim)
                    # Skip compact spike-like shapes, but protect elongated lines
                    perim_c = perim / S
                    elong_c = (perim_c * perim_c) / (raw_area / (S * S) + 1) if raw_area > 0 else 0
                    _area_real = raw_area / (S * S)
                    # Standard compactness filter: reject tiny spindly shards
                    if compactness < 0.005 and _area_real < 80 and elong_c < 50:
                        continue
                    # Stronger filter: small blobs (< 200 real px) that are
                    # irregularly shaped (compactness < 0.15 on 4*pi*A/P² scale)
                    # are almost always noise artifacts / texture fragments.
                    _compact_4pi = (4.0 * 3.14159265 * _area_real) / (perim_c * perim_c + 1e-9)
                    if _area_real < 100 and _compact_4pi < 0.10 and elong_c < 50:
                        continue

                width_est = (raw_area / perim) if perim > 1e-6 else 0.0
                width_est = (raw_area / perim) if perim > 1e-6 else 0.0
                t = min(max((width_est - 1.5 * S) / (1.5 * S), 0.0), 1.0)
                # Stronger smoothing: thin features get lighter sigma to
                # preserve detail; wide regions get heavy sigma to eliminate
                # polygon faceting.
                if cluster_is_thin[cluster_idx]:
                    contour_sigma = (0.3 + t * 0.3) * S
                else:
                    contour_sigma = (0.4 + t * 0.5) * S
                contour_sigma = max(contour_sigma, 0.8)

                xy = _smooth_contour(xy, sigma=contour_sigma)
                xy = xy / S
                area = abs(_polygon_area(xy))
                # Use lower area threshold for elongated shapes (thin lines)
                if len(xy) > 2:
                    _closed_xy = np.vstack([xy, xy[:1]])
                    perim_s = float(np.sum(np.sqrt(np.sum(np.diff(_closed_xy, axis=0)**2, axis=1))))
                else:
                    perim_s = 0
                elong_s = (perim_s * perim_s) / (area + 1) if area > 0 else 0
                area_thresh = min_contour_area * 0.3 if (elong_s > 50 and perim_s > 6) else min_contour_area
                if area < area_thresh:
                    continue

                # Geometric shape detection (only for outer contours
                # in non-thin clusters that have no child holes)
                is_hole = (_hierarchy is not None and _hierarchy[0][ci][3] != -1)
                has_holes = len(group) > 1
                if not is_hole:
                    if not cluster_is_thin[cluster_idx] and not has_holes:
                        shape_svg = _detect_shape(xy, min_area=min_contour_area * 4)
                        if shape_svg is not None:
                            if layer_shapes is None:
                                layer_shapes = []
                            layer_shapes.append(shape_svg)
                            _local_nodes += 1
                            continue
                    group_area = area

                # Time budget: use simplified polygon for remaining contours
                if _fit_budget_exceeded or time.time() - _t_cluster_fit_start > _FIT_BUDGET:
                    # Use simplified polygon path for remaining contours
                    _poly_pts = cv2.approxPolyDP(
                        (xy * 1.0).reshape(-1, 1, 2).astype(np.float32),
                        simplify_epsilon * 2, True,
                    ).squeeze()
                    if _poly_pts is not None and len(_poly_pts) >= 3:
                        _d = f"M {_poly_pts[0][0]:.2f},{_poly_pts[0][1]:.2f}"
                        for _pp in _poly_pts[1:]:
                            _d += f" L {_pp[0]:.2f},{_pp[1]:.2f}"
                        _d += " Z"
                        group_parts.append(_d)
                    continue

                _me = max_error * 0.5 if cluster_is_thin[cluster_idx] else max_error
                _lt = line_tolerance * 0.6 if cluster_is_thin[cluster_idx] else line_tolerance
                _t_fc = time.time()
                d_str = _fit_contour(xy, simplify_epsilon, _me, corner_threshold, _lt)
                _local_curves_time += time.time() - _t_fc
                if d_str:
                    group_parts.append(d_str)

            if group_parts:
                core_parts_per_group.append(group_parts)
                total_group_area.append(group_area)

        # Each group becomes one SVG path (outer + holes = evenodd cutouts)
        for group_parts in core_parts_per_group:
            combined = " ".join(group_parts)
            layer_paths.append(combined)
            layer_opacities.append(1.0)
            _local_paths += 1
            _local_nodes += combined.count("C") + combined.count("L") + combined.count("M")

        # Stroke-mode rendering is available for very thin clusters but
        # currently disabled — fill + adaptive iso handles thin features
        # well enough without the complexity of stroke reconstruction.
        used_stroke = False
        _t_cl_end = time.time()
        _n_groups = len(_contour_groups)
        _total_contour_area = sum(total_group_area)


        # Build fill layer
        if (layer_paths or layer_shapes) and not used_stroke:
            # Use gradient fill if this cluster was detected as a gradient zone
            fill_ref = gradient_fill_map.get(cluster_idx, color_hex)
            return (VectorLayer(
                paths=layer_paths,
                opacities=layer_opacities,
                color=fill_ref,
                shapes=layer_shapes,
            ), _local_paths, _local_nodes, _local_curves_time)
        return None

    # --- Parallel cluster processing ---
    _cluster_order = [ci for ci in order if ci != bg_cluster]

    _n_workers = min(len(_cluster_order), 8) if len(_cluster_order) > 1 else 1
    if _n_workers > 1:
        with ThreadPoolExecutor(max_workers=_n_workers) as _executor:
            _cluster_results = list(_executor.map(_process_cluster, _cluster_order))
    else:
        _cluster_results = [_process_cluster(ci) for ci in _cluster_order]

    # Pair each result with its cluster index
    _ci_results = list(zip(_cluster_order, _cluster_results))

    for _ci, _cr in _ci_results:
        if _cr is not None:
            _layer, _pc, _nc, _ct = _cr
            layers.append(_layer)
            total_paths += _pc
            total_nodes += _nc
            _t_curves_total += _ct

    _t_end = time.time()

    return MultilevelResult(
        layers=layers,
        stroke_layers=stroke_layers,
        width=w,
        height=h,
        background_color=bg_hex,
        path_count=total_paths,
        node_count=total_nodes,
        gradient_defs=gradient_defs if gradient_defs else None,
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

    # Emit gradient definitions if present
    if result.gradient_defs:
        parts.append('<defs>')
        for gd in result.gradient_defs:
            stops = f'<stop offset="0%" stop-color="{gd.color_start}"/>'
            if gd.color_mid:
                stops += f'<stop offset="50%" stop-color="{gd.color_mid}"/>'
            stops += f'<stop offset="100%" stop-color="{gd.color_end}"/>'
            parts.append(
                f'<linearGradient id="{gd.id}" '
                f'x1="{gd.x1:.1f}" y1="{gd.y1:.1f}" '
                f'x2="{gd.x2:.1f}" y2="{gd.y2:.1f}" '
                f'gradientUnits="userSpaceOnUse">'
                f'{stops}'
                f'</linearGradient>'
            )
        parts.append('</defs>')

    if not remove_background:
        parts.append(
            f'<rect width="{w}" height="{h}" fill="{result.background_color}"/>'
        )

    _shape_render = "crispEdges" if result.is_line_art else "geometricPrecision"

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
                    f' fill-rule="evenodd" shape-rendering="{_shape_render}"/>'
                )
            else:
                parts.append(
                    f'<path d="{path_d}" fill="{layer.color}"'
                    f' fill-rule="evenodd" opacity="{opacity:.2f}"'
                    f' shape-rendering="{_shape_render}"/>'
                )
        # Emit detected shapes (circles, ellipses, rectangles)
        if layer.shapes:
            for shape_tag in layer.shapes:
                parts.append(
                    f'{shape_tag} fill="{layer.color}" shape-rendering="geometricPrecision"/>'
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


def _detect_gradients(
    labels: np.ndarray,
    centers_f: np.ndarray,
    source_bgr: np.ndarray,
    bg_cluster: int,
    w: int,
    h: int,
    gradient_defs: list,
    gradient_fill_map: dict,
    min_region_pct: float = 2.0,
    color_dist_range: tuple[float, float] = (25.0, 150.0),
) -> None:
    """Detect gradient regions between adjacent clusters via PCA on pixel colors.

    For each cluster covering >min_region_pct of the image, checks if the
    source pixels within that cluster span a significant color range along
    a principal axis.  If so, creates a linearGradient definition.

    Modifies gradient_defs and gradient_fill_map in-place.
    """
    K = len(centers_f)
    total_px = h * w
    src_f = source_bgr.astype(np.float64)
    grad_id = 0

    for k in range(K):
        if k == bg_cluster:
            continue
        mask = labels == k
        pix_count = int(np.count_nonzero(mask))
        if pix_count < total_px * min_region_pct / 100.0:
            continue

        # Sample source pixels in this cluster
        ys, xs = np.where(mask)
        if len(ys) > 8000:
            rng = np.random.default_rng(k)
            idx = rng.choice(len(ys), 8000, replace=False)
            ys_s, xs_s = ys[idx], xs[idx]
        else:
            ys_s, xs_s = ys, xs
        colors = src_f[ys_s, xs_s]  # Nx3

        # PCA on the colors to find dominant color axis
        color_mean = colors.mean(axis=0)
        centered = colors - color_mean
        try:
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            continue

        # Largest eigenvalue = dominant color variation
        principal_idx = np.argmax(eigenvalues)
        spread = float(np.sqrt(eigenvalues[principal_idx]))

        # Only create gradient if color spread is significant
        if spread < color_dist_range[0] * 0.3 or spread > color_dist_range[1]:
            continue

        principal_axis = eigenvectors[:, principal_idx]
        projections = centered @ principal_axis  # scalar projection per pixel

        # Also project spatial coordinates to find gradient direction
        coords = np.column_stack([xs_s.astype(np.float64), ys_s.astype(np.float64)])
        coord_mean = coords.mean(axis=0)
        coords_c = coords - coord_mean

        # Correlate spatial position with color projection
        try:
            spatial_cov = np.cov(coords_c.T, projections)
        except Exception:
            continue
        if spatial_cov.shape != (3, 3):
            continue
        # spatial_cov[0:2, 2] = covariance of (x, y) with color projection
        spatial_color_cov = spatial_cov[0:2, 2]
        grad_dir = spatial_color_cov / (np.linalg.norm(spatial_color_cov) + 1e-10)

        # Gradient endpoints: project cluster bounding box onto gradient direction
        proj_spatial = coords_c @ grad_dir
        p_min, p_max = float(proj_spatial.min()), float(proj_spatial.max())
        if p_max - p_min < 10:  # too short spatially
            continue

        # Compute colors at the gradient endpoints
        low_mask = proj_spatial < np.percentile(proj_spatial, 15)
        high_mask = proj_spatial > np.percentile(proj_spatial, 85)
        if low_mask.sum() < 5 or high_mask.sum() < 5:
            continue
        color_start = colors[low_mask].mean(axis=0)
        color_end = colors[high_mask].mean(axis=0)

        # Verify the endpoint colors are meaningfully different
        endpoint_dist = float(np.linalg.norm(color_end - color_start))
        if endpoint_dist < 10:
            continue

        # SVG coordinates for gradient line
        x1 = float(coord_mean[0] + grad_dir[0] * p_min)
        y1 = float(coord_mean[1] + grad_dir[1] * p_min)
        x2 = float(coord_mean[0] + grad_dir[0] * p_max)
        y2 = float(coord_mean[1] + grad_dir[1] * p_max)

        # Compute mid-stop color for smoother 3-stop gradient
        mid_lo = np.percentile(proj_spatial, 40)
        mid_hi = np.percentile(proj_spatial, 60)
        mid_mask = (proj_spatial >= mid_lo) & (proj_spatial <= mid_hi)
        color_mid_hex = None
        if mid_mask.sum() >= 5:
            color_mid = colors[mid_mask].mean(axis=0)
            color_mid_hex = _bgr_to_hex(np.clip(color_mid, 0, 255).astype(np.uint8))

        gid = f"g{grad_id}"
        grad_id += 1
        gradient_defs.append(GradientDef(
            id=gid,
            x1=np.clip(x1, 0, w), y1=np.clip(y1, 0, h),
            x2=np.clip(x2, 0, w), y2=np.clip(y2, 0, h),
            color_start=_bgr_to_hex(np.clip(color_start, 0, 255).astype(np.uint8)),
            color_end=_bgr_to_hex(np.clip(color_end, 0, 255).astype(np.uint8)),
            color_mid=color_mid_hex,
        ))
        gradient_fill_map[k] = f"url(#{gid})"


def _gradient_aware_merge(
    labels: np.ndarray,
    centers_f: np.ndarray,
    source_bgr: np.ndarray,
    bg_cluster: int = -1,
    max_color_dist: float = 80.0,
    boundary_contrast_thresh: float = 30.0,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Merge adjacent clusters whose boundary has low color contrast.

    Instead of relying on a blurred edge-density map, computes the
    actual per-pixel colour difference at each cluster boundary.  If
    adjacent pixels across the boundary are similar (smooth gradient),
    the clusters are merged.  Hard edges (high colour contrast) are
    preserved.
    """
    h, w = labels.shape
    K = len(centers_f)
    centers = centers_f.astype(np.float64).copy()
    src_f = source_bgr.astype(np.float64)
    pixel_counts = np.array(
        [np.count_nonzero(labels == k) for k in range(K)], dtype=np.int64,
    )
    alive = np.ones(K, dtype=bool)
    _merge_iters = 0

    while _merge_iters < 3:
        _merge_iters += 1
        if int(alive.sum()) <= 3:
            break

        # Horizontal boundaries
        l_left = labels[:, :-1].ravel()
        l_right = labels[:, 1:].ravel()
        h_mask = l_left != l_right
        c_left = src_f[:, :-1].reshape(-1, 3)[h_mask]
        c_right = src_f[:, 1:].reshape(-1, 3)[h_mask]
        h_diff = np.sqrt(np.sum((c_left - c_right) ** 2, axis=1))

        # Vertical boundaries
        l_top = labels[:-1, :].ravel()
        l_bot = labels[1:, :].ravel()
        v_mask = l_top != l_bot
        c_top = src_f[:-1, :].reshape(-1, 3)[v_mask]
        c_bot = src_f[1:, :].reshape(-1, 3)[v_mask]
        v_diff = np.sqrt(np.sum((c_top - c_bot) ** 2, axis=1))

        all_k1 = np.concatenate([
            np.minimum(l_left[h_mask], l_right[h_mask]),
            np.minimum(l_top[v_mask], l_bot[v_mask]),
        ])
        all_k2 = np.concatenate([
            np.maximum(l_left[h_mask], l_right[h_mask]),
            np.maximum(l_top[v_mask], l_bot[v_mask]),
        ])
        all_diffs = np.concatenate([h_diff, v_diff])

        if len(all_k1) == 0:
            break

        # Aggregate median boundary contrast per pair using bincount
        pair_keys = all_k1.astype(np.int64) * K + all_k2
        max_key = int(pair_keys.max()) + 1
        diff_sums = np.bincount(pair_keys, weights=all_diffs, minlength=max_key)
        diff_counts = np.bincount(pair_keys, minlength=max_key)

        nonzero = diff_counts > 0
        mean_diffs = np.zeros(max_key, dtype=np.float64)
        mean_diffs[nonzero] = diff_sums[nonzero] / diff_counts[nonzero]

        # Find candidate: low boundary contrast + moderate cluster distance
        # + boundary must be substantial (>0.5% of image) to avoid merging
        # small distinct color regions that happen to have smooth transitions.
        min_boundary = int(h * w * 0.005)
        cand_mask = ((diff_counts >= min_boundary)
                     & (mean_diffs < boundary_contrast_thresh))
        best = None
        best_contrast = boundary_contrast_thresh
        for key in np.where(cand_mask)[0]:
            k1, k2 = divmod(int(key), K)
            if not alive[k1] or not alive[k2]:
                continue
            if k1 == bg_cluster or k2 == bg_cluster:
                continue
            color_dist = float(np.linalg.norm(centers[k1] - centers[k2]))
            if color_dist >= max_color_dist:
                continue
            if mean_diffs[key] < best_contrast:
                best_contrast = mean_diffs[key]
                best = (k1, k2)

        if best is None:
            break

        k1, k2 = best
        if pixel_counts[k1] >= pixel_counts[k2]:
            src, dst = k2, k1
        else:
            src, dst = k1, k2

        w_src, w_dst = float(pixel_counts[src]), float(pixel_counts[dst])
        centers[dst] = (centers[dst] * w_dst + centers[src] * w_src) / (w_src + w_dst)
        pixel_counts[dst] += pixel_counts[src]
        pixel_counts[src] = 0
        alive[src] = False
        labels[labels == src] = dst

    # Renumber contiguously
    alive_ids = np.where(alive)[0]
    remap = np.full(K, -1, dtype=np.int32)
    for new_id, old_id in enumerate(alive_ids):
        remap[old_id] = new_id
    new_labels = remap[labels]
    new_centers = centers[alive_ids].astype(np.float32)
    new_bg = int(remap[bg_cluster]) if bg_cluster >= 0 and remap[bg_cluster] >= 0 else -1
    return new_labels, new_centers, new_bg


def _merge_close_clusters(
    centers: np.ndarray,
    labels_flat: np.ndarray,
    h: int,
    w: int,
    threshold: float = 35.0,
    lab_image: np.ndarray | None = None,
    lab_threshold: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Agglomerative merge: keep merging the nearest pair until the
    closest pair is further than `threshold` apart, OR we reach a
    target count of ≥3 clusters.

    Uses pixel-weighted centre updates to maintain accurate colours.
    When lab_image is provided, computes LAB-space distances for
    perceptually accurate merging.
    """
    centers_f = centers.astype(np.float64).copy()
    K = len(centers_f)
    pixel_counts = np.array([
        np.count_nonzero(labels_flat == i) for i in range(K)
    ], dtype=np.int64)
    alive = np.ones(K, dtype=bool)
    merge_into = np.arange(K, dtype=np.int32)

    # Compute LAB centres if LAB image provided
    if lab_image is not None:
        lab_flat = lab_image.reshape(-1, 3).astype(np.float64)
        centers_lab = np.zeros((K, 3), dtype=np.float64)
        for i in range(K):
            mask_i = labels_flat == i
            if mask_i.any():
                centers_lab[i] = lab_flat[mask_i].mean(axis=0)
    else:
        centers_lab = None

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
                if centers_lab is not None:
                    d = np.linalg.norm(centers_lab[i] - centers_lab[j])
                else:
                    d = np.linalg.norm(centers_f[i] - centers_f[j])
                if d < best_dist:
                    best_dist = d
                    best_i, best_j = i, j

        # LAB threshold: configurable ΔE — higher for grayscale to merge close shades
        merge_thresh = lab_threshold if centers_lab is not None else threshold
        if best_dist > merge_thresh:
            break

        # Merge smaller into larger
        if pixel_counts[best_i] < pixel_counts[best_j]:
            src, dst = best_i, best_j
        else:
            src, dst = best_j, best_i

        ni, nj = pixel_counts[src], pixel_counts[dst]
        centers_f[dst] = (centers_f[dst] * nj + centers_f[src] * ni) / (ni + nj)
        if centers_lab is not None:
            centers_lab[dst] = (centers_lab[dst] * nj + centers_lab[src] * ni) / (ni + nj)
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


def _smooth_contour(pts: np.ndarray, sigma: float = 3.0, light_mode: bool = False) -> np.ndarray:
    """Curvature-adaptive contour smoothing with multi-pass Gaussian filter.

    Applies heavy smoothing to gently-curving boundary sections (eliminating
    polygon faceting on smooth surfaces) while preserving sharp corners.
    This mimics how an artist traces: long sweeping curves for smooth
    regions, crisp angles where edges meet.

    Pass 1: Strong global smooth (1.5× base sigma) to eliminate staircase.
    Pass 2: Per-vertex curvature-adaptive blend — low-curvature regions
             stay at the heavy-smooth result; high-curvature/corners blend
             back toward the original positions.

    When light_mode=True (S<=2), skip the curvature-blend refinement pass
    and return the heavy smooth directly for speed.
    """
    N = len(pts)
    if N < 6 or sigma < 0.1:
        return pts

    # Cap sigma to prevent over-smoothing short contours
    max_sigma = N / 5.0
    base_sigma = min(sigma, max_sigma)
    if base_sigma < 0.5:
        return pts

    # Pass 1: Heavy global smooth (1.5× base) for staircase elimination
    heavy_sigma = min(base_sigma * 1.5, max_sigma)
    sx_h = gaussian_filter1d(pts[:, 0], sigma=heavy_sigma, mode='wrap')
    sy_h = gaussian_filter1d(pts[:, 1], sigma=heavy_sigma, mode='wrap')
    heavy_smooth = np.column_stack([sx_h, sy_h])

    # Pass 2: Light smooth for corner regions
    light_sigma = min(base_sigma * 0.5, max_sigma)
    sx_l = gaussian_filter1d(pts[:, 0], sigma=light_sigma, mode='wrap')
    sy_l = gaussian_filter1d(pts[:, 1], sigma=light_sigma, mode='wrap')
    light_smooth = np.column_stack([sx_l, sy_l])

    # Discrete curvature: angle change at each vertex
    d1 = np.roll(pts, -1, axis=0) - pts        # forward edge
    d0 = pts - np.roll(pts, 1, axis=0)          # backward edge
    cross = d0[:, 0] * d1[:, 1] - d0[:, 1] * d1[:, 0]
    dot_prod = d0[:, 0] * d1[:, 0] + d0[:, 1] * d1[:, 1]
    angle = np.abs(np.arctan2(cross, dot_prod))  # 0=straight, pi=reversal

    # Smooth the curvature signal to avoid noisy vertex-level switching
    angle_smooth = gaussian_filter1d(angle, sigma=max(2.0, base_sigma * 0.3), mode='wrap')

    # Adaptive blend based on curvature:
    #   low curvature (< 0.15 rad ≈ 8°) → use heavy smooth (alpha=0)
    #   medium curvature → blend
    #   high curvature (> 0.5 rad ≈ 29°) → use light smooth
    #   very high (> 1.0 rad ≈ 57°) → use original points
    corner_lo = 0.15   # below this = smooth region
    corner_hi = 0.50   # above this = corner region
    corner_sharp = 1.0 # above this = true corner, keep original

    # alpha: 0 = heavy smooth, 1 = light smooth
    alpha = np.clip((angle_smooth - corner_lo) / (corner_hi - corner_lo), 0, 1)
    # beta: 0 = smoothed, 1 = original
    beta = np.clip((angle_smooth - corner_hi) / (corner_sharp - corner_hi), 0, 1)

    alpha = alpha[:, np.newaxis]
    beta = beta[:, np.newaxis]

    # Blend: heavy → light based on curvature, then light → original at corners
    blended = heavy_smooth * (1.0 - alpha) + light_smooth * alpha
    result = blended * (1.0 - beta) + pts * beta

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


def _detect_shape(contour: np.ndarray, min_area: float = 50.0) -> str | None:
    """Try to fit a geometric primitive to a contour.

    Returns an SVG shape string (circle, ellipse, rect) without fill attrs,
    or None if no shape fits.  Tests from most constrained to least:
    circle → ellipse → rectangle → None.
    """
    if len(contour) < 5:
        return None
    area = abs(cv2.contourArea(contour.reshape(-1, 1, 2).astype(np.float32)))
    if area < min_area:
        return None

    c32 = contour.reshape(-1, 1, 2).astype(np.float32)

    # --- Circle test ---
    (cx, cy), radius = cv2.minEnclosingCircle(c32)
    if radius > 2.0:
        circle_area = math.pi * radius * radius
        circularity = area / circle_area if circle_area > 0 else 0
        if circularity > 0.82:
            return f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radius:.1f}"'

    # --- Ellipse test ---
    if len(contour) >= 5:
        try:
            ((ecx, ecy), (major, minor), angle) = cv2.fitEllipse(c32)
            rx, ry = major / 2, minor / 2
            if rx > 1.5 and ry > 1.5:
                ellipse_area = math.pi * rx * ry
                ellipticity = area / ellipse_area if ellipse_area > 0 else 0
                if ellipticity > 0.80:
                    if abs(angle) < 2 or abs(angle - 180) < 2:
                        return (f'<ellipse cx="{ecx:.1f}" cy="{ecy:.1f}" '
                                f'rx="{rx:.1f}" ry="{ry:.1f}"')
                    else:
                        return (f'<ellipse cx="{ecx:.1f}" cy="{ecy:.1f}" '
                                f'rx="{rx:.1f}" ry="{ry:.1f}" '
                                f'transform="rotate({angle:.1f},{ecx:.1f},{ecy:.1f})"')
        except cv2.error:
            pass

    # --- Rectangle test ---
    rect = cv2.minAreaRect(c32)
    (rcx, rcy), (rw, rh), rangle = rect
    if rw > 2 and rh > 2:
        rect_area = rw * rh
        rectangularity = area / rect_area if rect_area > 0 else 0
        if rectangularity > 0.90:
            # Check if nearly axis-aligned
            if abs(rangle) < 3 or abs(rangle - 90) < 3 or abs(rangle + 90) < 3:
                x = rcx - rw / 2
                y = rcy - rh / 2
                return f'<rect x="{x:.1f}" y="{y:.1f}" width="{rw:.1f}" height="{rh:.1f}"'
            else:
                x = rcx - rw / 2
                y = rcy - rh / 2
                return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{rw:.1f}" height="{rh:.1f}" '
                        f'transform="rotate({rangle:.1f},{rcx:.1f},{rcy:.1f})"')

    return None


def _fit_contour(
    contour: np.ndarray,
    simplify_epsilon: float,
    max_error: float,
    corner_threshold: float,
    line_tolerance: float = 0.15,
) -> str:
    """Simplify contour and fit smooth closed Bezier curves; return SVG path d.

    Uses an artistic pipeline: RDP simplification, Bezier fitting,
    line-to-curve promotion, G1 continuity enforcement, segment
    merging.  The result has fewer, smoother curves like hand-crafted
    vector art rather than pixel-tracing.
    """
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
        # Step 1: Promote line segments to curves where possible
        curve = reduce_nodes(curve, max_error=max_error * 2.5)
        # Step 2: Merge short segments into longer artistic curves
        curve = _merge_short_curves(curve, max_error=max_error)
        # Step 3: DP-optimal segment merging (Potrace-style)
        curve = merge_segments_artistic(curve, tolerance=max_error * 2.0)
        # Step 4: Enforce G1 tangent continuity at all joins
        curve = enforce_g1_continuity(curve)
        # Step 5: Second merge pass — skip for complex contours (>200 segs)
        # where the O(n²) DP becomes a bottleneck
        if len(curve.segments) <= 200:
            curve = merge_segments_artistic(curve, tolerance=max_error * 2.0)
            # Step 6: Final G1 enforcement after second merge
            curve = enforce_g1_continuity(curve)
        # Step 7: Safety valve — aggressively merge dense contours
        if len(curve.segments) > 80:
            curve = merge_segments_artistic(curve, tolerance=max_error * 3.0)
            curve = enforce_g1_continuity(curve)
    except Exception:
        return ""

    return _curve_to_d(curve)


def _curve_to_d(curve) -> str:
    """Convert a FittedCurve to an SVG path `d` string."""
    if not curve.segments:
        return ""

    p = curve.segments[0]
    parts = [f"M{p.p0[0]:.1f},{p.p0[1]:.1f}"]
    for seg in curve.segments:
        if seg.is_line:
            parts.append(f"L{seg.p3[0]:.1f},{seg.p3[1]:.1f}")
        else:
            parts.append(
                f"C{seg.p1[0]:.1f},{seg.p1[1]:.1f} "
                f"{seg.p2[0]:.1f},{seg.p2[1]:.1f} "
                f"{seg.p3[0]:.1f},{seg.p3[1]:.1f}"
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


def _estimate_initial_k(image_bgr: np.ndarray, max_k: int = 12) -> int:
    """Estimate initial K for K-means from image color complexity.

    Samples the image in CIELAB space and counts perceptually distinct
    color bins.  Returns K = n_distinct (merge step handles AA
    intermediates, so no headroom multiplier is needed).
    """
    h, w = image_bgr.shape[:2]
    scale = max(1, min(h, w) // 64)
    small = cv2.resize(image_bgr, (w // scale, h // scale),
                       interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    # Bin at ~8 ΔE resolution
    binned = (lab / 8.0).astype(np.int32)
    n_distinct = len(np.unique(binned, axis=0))
    # Cap K for grayscale/low-saturation images — fewer clusters = larger contiguous regions
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mean_saturation = hsv[:, :, 1].mean()
    # Check for chromatic content that needs dedicated clusters
    sat_mask = hsv[:, :, 1] > 40  # pixels with meaningful saturation
    sat_fraction = sat_mask.sum() / max(1, hsv.shape[0] * hsv.shape[1])
    # Only cap K for truly achromatic images — if >1% of pixels are
    # chromatic, they need their own clusters even if the mean is low
    if mean_saturation < 30 and sat_fraction < 0.01:
        max_k = min(max_k, 6)
    if sat_fraction > 0.005:  # at least 0.5% of pixels are chromatic
        hue_vals = hsv[:, :, 0][sat_mask]
        hue_sectors = len(np.unique(hue_vals // 30))  # 30° sectors = 6 sectors max
        min_chromatic_k = max(4, 3 + hue_sectors)  # 3 for luminance spread + 1 per hue sector
        n_distinct = max(n_distinct, min_chromatic_k)
    return max(4, min(max_k, n_distinct))


def _merge_short_curves(
    curve: "FittedCurve",
    max_error: float = 1.5,
) -> "FittedCurve":
    """Merge consecutive short Bézier segments into longer, smoother curves.

    Collects runs of short segments (< 8px chord) and re-fits them as
    fewer curves.  This eliminates the micro-segment stutter that makes
    vector art look like a pixel trace.
    """
    if not curve.segments or len(curve.segments) <= 3:
        return curve

    new_segs: list = []
    run: list = []

    def _flush():
        if not run:
            return
        if len(run) <= 2:
            new_segs.extend(run)
            return
        # Collect all endpoints
        pts = [run[0].p0]
        for s in run:
            pts.append(s.p3)
        pts_arr = np.array(pts, dtype=np.float64)
        # Re-fit with generous tolerance for smooth artistic curves
        merged_curve = fit_bezier_path(
            pts_arr, max_error=max_error * 1.5,
            corner_threshold=55.0, line_tolerance=max_error,
        )
        new_segs.extend(merged_curve.segments)

    SHORT_THRESHOLD = 12.0  # pixels — merge fragments shorter than this
    for seg in curve.segments:
        chord = float(np.linalg.norm(seg.p3 - seg.p0))
        if chord < SHORT_THRESHOLD and not seg.is_line:
            run.append(seg)
        else:
            _flush()
            run = []
            new_segs.append(seg)
    _flush()

    return FittedCurve(segments=new_segs, is_closed=curve.is_closed)


def optimize_svg_colors(
    svg_string: str,
    source_bgr: np.ndarray,
    *,
    iterations: int = 3,
) -> str:
    """Adjust SVG fill colours to match actual pixel means in the source.

    Renders the SVG, identifies which pixels each fill colour covers,
    then replaces the colour with the mean of the source pixels in
    that region.  Iterates until colours stabilise.
    """
    import cairosvg
    h, w = source_bgr.shape[:2]
    svg = svg_string
    _fringe_colors: set[str] = set()
    for m in re.finditer(r'fill="(#[0-9a-fA-F]{6})"[^/]*opacity="', svg):
        _fringe_colors.add(m.group(1))
    for _ in range(iterations):
        png = cairosvg.svg2png(
            bytestring=svg.encode(), output_width=w, output_height=h,
        )
        rendered = cv2.imdecode(
            np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR,
        )
        fills = re.findall(r'fill="(#[0-9a-fA-F]{6})"', svg)
        unique_fills = list(dict.fromkeys(fills))
        color_map: dict[str, str] = {}
        for hex_color in unique_fills:
            if hex_color in _fringe_colors:
                continue
            r_val = int(hex_color[1:3], 16)
            g_val = int(hex_color[3:5], 16)
            b_val = int(hex_color[5:7], 16)
            bgr = np.array([b_val, g_val, r_val], dtype=np.float64)
            diff = np.abs(rendered.astype(np.float64) - bgr)
            mask = np.all(diff < 2, axis=2)
            if np.count_nonzero(mask) < 50:
                continue
            _kern_c = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            _eroded = cv2.erode(mask.astype(np.uint8), _kern_c).astype(bool)
            if np.count_nonzero(_eroded) > 200:
                sample_mask = _eroded
            else:
                sample_mask = mask
            # Weight source pixels by proximity to current fill color.
            # This prevents over-expanded contours (e.g. line art that
            # bleeds into background) from dragging the color away.
            src_samples = source_bgr[sample_mask].astype(np.float64)
            src_diff = np.sqrt(np.sum((src_samples - bgr) ** 2, axis=1))
            # Accept pixels within 80 BGR distance of current fill
            close_mask = src_diff < 80.0
            if np.count_nonzero(close_mask) >= 20:
                mean_orig = np.median(src_samples[close_mask], axis=0)
            else:
                mean_orig = np.median(src_samples, axis=0)
            nb, ng, nr = (int(round(np.clip(v, 0, 255))) for v in mean_orig)
            new_hex = f"#{nr:02x}{ng:02x}{nb:02x}"
            if new_hex != hex_color:
                color_map[hex_color] = new_hex
        if not color_map:
            break
        for old_c, new_c in color_map.items():
            svg = svg.replace(f'fill="{old_c}"', f'fill="{new_c}"')
    return svg
