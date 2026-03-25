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
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from dataclasses import dataclass
from scipy.ndimage import distance_transform_edt, gaussian_filter1d
from ..curve_fitting import (
    fit_closed_bezier, fit_bezier_path, reduce_nodes, FittedCurve,
    enforce_g1_continuity, merge_segments_artistic,
)
from skimage.morphology import skeletonize as _skeletonize
from ..stroke_reconstruction import _prune_skeleton, _trace_skeleton_paths

cv2.setNumThreads(1)
cv2.ocl.setUseOpenCL(False)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class VectorLayer:
    paths: list[str]        # SVG path `d` strings, one per iso-level
    opacities: list[float]  # opacity for each iso-level path
    color: str              # hex fill colour
    shapes: list[str] | None = None  # SVG shape elements (circle, rect, ellipse)
    path_fills: list[str] | None = None  # optional per-path fill refs


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
    kind: str = "linear"
    cx: float | None = None
    cy: float | None = None
    r: float | None = None
    fx: float | None = None
    fy: float | None = None


@dataclass
class GradientRegionAssignment:
    fill_ref: str
    bbox: tuple[float, float, float, float]
    mask: np.ndarray


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


def _render_color_from_samples(samples: np.ndarray, base_color: np.ndarray) -> np.ndarray:
    lo = np.percentile(samples, 10.0, axis=0)
    hi = np.percentile(samples, 90.0, axis=0)
    clipped = np.clip(samples, lo, hi)
    raw_mean = samples.mean(axis=0)
    clipped_mean = clipped.mean(axis=0)
    base_hsv = cv2.cvtColor(base_color.reshape(1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2HSV)[0, 0]
    hue = int(base_hsv[0])
    is_warm_family = hue <= 45 or hue >= 170
    if float(base_hsv[1]) >= 80 and is_warm_family:
        sat_blend = min(0.35, float(base_hsv[1]) / 255.0 * 0.35)
        return (
            clipped_mean * (1.0 - sat_blend)
            + raw_mean * 0.5 * sat_blend
            + base_color * 0.5 * sat_blend
        )
    return clipped_mean


def _compute_render_centers(
    labels: np.ndarray,
    source_bgr: np.ndarray,
    base_centers_u: np.ndarray,
) -> np.ndarray:
    """Re-estimate display colors from original pixels without affecting geometry."""
    K = len(base_centers_u)
    render_centers = base_centers_u.astype(np.float32).copy()
    rng = np.random.default_rng(0)

    for k in range(K):
        ys, xs = np.where(labels == k)
        if len(ys) == 0:
            continue
        if len(ys) > 6000:
            idx = rng.choice(len(ys), 6000, replace=False)
            ys = ys[idx]
            xs = xs[idx]
        samples = source_bgr[ys, xs].astype(np.float32)
        base_color = base_centers_u[k].astype(np.float32)
        render_centers[k] = _render_color_from_samples(samples, base_color)

    return np.clip(np.round(render_centers), 0, 255).astype(np.uint8)


def _precompute_nearest_clusters(
    dist_map: np.ndarray,
    k_neighbors: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Return up to k nearest cluster indices and distances per pixel."""
    k_use = max(1, min(k_neighbors, dist_map.shape[2]))
    nn_idx = np.argpartition(dist_map, k_use - 1, axis=2)[:, :, :k_use]
    nn_dist = np.take_along_axis(dist_map, nn_idx, axis=2)
    order = np.argsort(nn_dist, axis=2)
    nn_idx = np.take_along_axis(nn_idx, order, axis=2)
    nn_dist = np.take_along_axis(nn_dist, order, axis=2)
    return nn_idx, nn_dist


def _soft_competing_distance(
    cluster_idx: int,
    nn_idx: np.ndarray,
    nn_dist: np.ndarray,
    max_competitors: int = 3,
) -> np.ndarray:
    """Ambiguity-gated softmin competitor distance from nearest non-self clusters."""
    candidate_dists = np.where(nn_idx == cluster_idx, np.inf, nn_dist)
    candidate_dists = np.sort(candidate_dists, axis=2)
    competitor_count = min(max_competitors, candidate_dists.shape[2])
    competitors = candidate_dists[:, :, :competitor_count]
    finite = np.isfinite(competitors)
    nearest = np.where(finite[:, :, 0], competitors[:, :, 0], nn_dist[:, :, 0])

    if os.environ.get("SVG_SOFT_COMP_MODE") == "nearest":
        return nearest.astype(np.float32)

    tau = 6.0
    shifted = np.where(finite, np.exp(-(competitors - nearest[:, :, None]) / tau), 0.0)
    softmin = nearest - tau * np.log(np.maximum(np.sum(shifted, axis=2), 1e-6))
    softmin = np.clip(softmin, 0.55 * nearest, nearest)

    if competitor_count > 1:
        second = np.where(finite[:, :, 1], competitors[:, :, 1], np.inf)
        ambiguity = 1.0 - np.clip((second - nearest) / np.maximum(8.0, nearest * 0.5), 0.0, 1.0)
    else:
        ambiguity = np.zeros_like(nearest)

    blend = 0.18 * ambiguity
    soft_other = nearest * (1.0 - blend) + softmin * blend
    return soft_other.astype(np.float32)


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
    simplify_epsilon: float = 1.0,
    max_error: float = 1.5,
    line_tolerance: float = 0.5,
    corner_threshold: float = 55.0,
    min_contour_area: int = 12,
    contour_scale: int = 4,
    smooth_sigma: float = 0.50,
    mediator_threshold: float = 0.3,   # backward compat (used in absorption)
) -> MultilevelResult:
    h, w = image_bgr.shape[:2]
    _warm_debug = os.environ.get("SVG_WARM_DEBUG") == "1"
    _cluster_debug = os.environ.get("SVG_CLUSTER_DEBUG") == "1"

    if len(image_bgr.shape) == 2:
        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)

    _t_start = time.time()
    bg_color, bg_gray = detect_background(image_bgr)
    bg_hex = _bgr_to_hex(bg_color)
    source_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    source_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # --- Step 0: Edge-density map for adaptive processing ---
    edge_weight = _compute_edge_weight(image_bgr)
    source_lab_grad = _compute_lab_gradient_magnitude(source_lab)

    # --- Step 0b: Dual denoise ------------------------------------------------
    # Bilateral keeps edges while killing SD/AI per-pixel noise in flat regions.
    # For large images (>4MP), use smaller filter radii for speed.
    # Fast Gaussian blur instead of expensive bilateral filter.
    # Vectorization quantizes colors anyway, so edge-preserving denoising
    # is overkill. GaussianBlur is 10-100× faster.
    denoised_dist = cv2.GaussianBlur(image_bgr, (5, 5), 0)

    # Mean-shift prefilter for K-means: flattens uniform regions while
    # preserving edges — produces more coherent color clusters than
    # isotropic Gaussian blur alone. Skip for very large images (>12MP).
    if h * w <= 12_000_000:
        _ms_level = 2 if h * w > 6_000_000 else 1
        denoised_km = cv2.pyrMeanShiftFiltering(image_bgr, sp=8, sr=16,
                                                 maxLevel=_ms_level)
    else:
        denoised_km = cv2.GaussianBlur(image_bgr, (7, 7), 0)
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
    _sat_mask = _hsv_check[:, :, 1] > 35
    _sat_count = int(np.count_nonzero(_sat_mask))
    if _sat_count > 0:
        _hue_sat = _hsv_check[:, :, 0]
        _warm_yellow_frac = float(np.count_nonzero(_sat_mask & (_hue_sat >= 12) & (_hue_sat <= 45))) / _sat_count
        _red_frac = float(np.count_nonzero(_sat_mask & ((_hue_sat <= 10) | (_hue_sat >= 170)))) / _sat_count
    else:
        _warm_yellow_frac = 0.0
        _red_frac = 0.0
    _enable_warm_fill_relax = _sat_frac > 0.25 and _warm_yellow_frac > 0.72 and _red_frac < 0.12
    _has_color = _mean_sat > 15 or _sat_frac > 0.01
    # Dynamic max_k: match documented tiers that gave best quality.
    # Color distinction removed — extra clusters fragment features more than
    # they help color fidelity (test4 regressed 96.5% → 83% with K=10 vs K=7).
    if h * w > 16_000_000:
        _max_k = 10
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
    _flat_pixels = all_pixels
    _p_sq = np.sum(_flat_pixels ** 2, axis=1)
    _chunk_size = 2_000_000
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
        _centers_f32 = centers.astype(np.float32)
        _c_sq = np.sum(_centers_f32 ** 2, axis=1)  # (K,)
        # p·cᵀ is (N, K) — use matrix multiply in chunks to control memory
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
    if _cluster_debug:
        _premerge_hsv = cv2.cvtColor(centers_u.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
        _parts = []
        for k in range(K):
            _parts.append(
                f"k{k}:bgr=({int(centers_u[k,0])},{int(centers_u[k,1])},{int(centers_u[k,2])})"
                f" hsv=({int(_premerge_hsv[k,0])},{int(_premerge_hsv[k,1])},{int(_premerge_hsv[k,2])})"
                f" px={int(np.count_nonzero(labels == k))}"
            )
        print(f"[CLUSTER] bg_color=({int(bg_color[0])},{int(bg_color[1])},{int(bg_color[2])}) bg_cluster={bg_cluster}")
        print("[CLUSTER] centers_pre " + " | ".join(_parts[:16]))

    # --- Step 1d: Gradient-aware merge ---
    # Collapse cluster pairs whose boundary has low color contrast
    # in the source image, preserving boundaries along real edges.
    labels, centers_f, bg_cluster = _gradient_aware_merge(
        labels, centers_f, denoised_dist, bg_cluster,
        boundary_contrast_thresh=22.0,
        max_color_dist=60.0,
    )
    K = len(centers_f)
    if _cluster_debug:
        _centers_post_u = centers_f.astype(np.uint8)
        _post_hsv = cv2.cvtColor(_centers_post_u.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
        _parts = []
        for k in range(K):
            _parts.append(
                f"k{k}:bgr=({int(_centers_post_u[k,0])},{int(_centers_post_u[k,1])},{int(_centers_post_u[k,2])})"
                f" hsv=({int(_post_hsv[k,0])},{int(_post_hsv[k,1])},{int(_post_hsv[k,2])})"
                f" px={int(np.count_nonzero(labels == k))}"
            )
        print(f"[CLUSTER] post_merge bg_cluster={bg_cluster}")
        print("[CLUSTER] centers_post " + " | ".join(_parts[:16]))
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
    _centers_hsv_merge = cv2.cvtColor(centers_u.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    _cluster_sizes_premerge = np.bincount(labels.ravel(), minlength=K)
    _dark_chroma_merged = []
    for k in range(K):
        if k == bg_cluster:
            continue
        _L_k = float(_centers_lab_merge[k, 0])
        if _L_k > 35:  # only very dark clusters
            continue
        if int(_centers_hsv_merge[k, 1]) >= 90 and int(_cluster_sizes_premerge[k]) < int(h * w * 0.02):
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

    # Pre-compute nearest nearby clusters for fast soft-field competition.
    _nn_idx, _nn_dist = _precompute_nearest_clusters(dist_map, k_neighbors=min(4, K))

    warm_debug_primary_competitor: dict[int, int] = {}

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
        # Reindex dist_map columns and rebuild nearest competitors
        dist_map = dist_map[:, :, alive].copy()
        _nn_idx, _nn_dist = _precompute_nearest_clusters(dist_map, k_neighbors=min(4, K))
        # Rebuild grays and order
        grays = np.array([
            int(cv2.cvtColor(c.reshape(1, 1, 3), cv2.COLOR_BGR2GRAY)[0, 0])
            for c in centers_u
        ])
        order = np.argsort(-grays)
        # Reset mediator scores — all survivors are real fills
        cluster_mediator_score = np.zeros(K, dtype=np.float64)

    centers_hsv_render = cv2.cvtColor(centers_u.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    warm_debug_clusters = [
        k for k in range(K)
        if int(centers_hsv_render[k, 1]) >= 35 and 12 <= int(centers_hsv_render[k, 0]) <= 45
    ]
    if _warm_debug and warm_debug_clusters:
        debug_parts = []
        for k in warm_debug_clusters:
            debug_parts.append(
                f"k{k}:hsv=({int(centers_hsv_render[k,0])},{int(centers_hsv_render[k,1])},{int(centers_hsv_render[k,2])}) "
                f"bgr=({int(centers_u[k,0])},{int(centers_u[k,1])},{int(centers_u[k,2])}) px={int(np.count_nonzero(labels == k))}"
            )
        print("[WARM] clusters " + " | ".join(debug_parts[:12]))
        for k in warm_debug_clusters:
            mask_k = labels == k
            if not np.any(mask_k):
                continue
            competitor_samples = []
            for rank in range(_nn_idx.shape[2]):
                candidates = _nn_idx[:, :, rank][mask_k]
                candidates = candidates[candidates != k]
                if candidates.size:
                    competitor_samples.append(candidates)
            if not competitor_samples:
                continue
            competitor_all = np.concatenate(competitor_samples)
            competitor_hist = np.bincount(competitor_all.astype(np.int32), minlength=K)
            competitor_hist[k] = 0
            warm_debug_primary_competitor[k] = int(np.argmax(competitor_hist))
            top = np.argsort(competitor_hist)[::-1][:3]
            top_desc = ", ".join(
                (
                    f"k{int(idx)}:{int(competitor_hist[idx])}"
                    f"(hsv={int(centers_hsv_render[idx,0])},{int(centers_hsv_render[idx,1])},{int(centers_hsv_render[idx,2])})"
                )
                for idx in top if competitor_hist[idx] > 0
            )
            if top_desc:
                print(f"[WARM] cluster k{k} nearest competitors {top_desc}")

    # --- Step 3e: Classify clusters as thin-feature vs fill ---
    # Thin clusters (text, lines, serifs) need higher iso to avoid
    # thickening and tighter Bézier fitting.
    render_centers_u = _compute_render_centers(labels, image_bgr, centers_u)
    if _warm_debug and warm_debug_clusters:
        render_hsv = cv2.cvtColor(render_centers_u.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
        for k in warm_debug_clusters:
            mask_k = labels == k
            if not np.any(mask_k):
                continue
            src_hsv_mean = source_hsv[mask_k].reshape(-1, 3).mean(axis=0)
            print(
                f"[WARM] k{k} source_hsv_mean=({src_hsv_mean[0]:.1f},{src_hsv_mean[1]:.1f},{src_hsv_mean[2]:.1f}) "
                f"center_hsv=({int(centers_hsv_render[k,0])},{int(centers_hsv_render[k,1])},{int(centers_hsv_render[k,2])}) "
                f"render_hsv=({int(render_hsv[k,0])},{int(render_hsv[k,1])},{int(render_hsv[k,2])})"
            )
    cluster_is_thin = np.zeros(K, dtype=bool)
    cluster_large_chromatic_fill = np.zeros(K, dtype=bool)
    for k in range(K):
        if k == bg_cluster:
            continue
        cluster_area_frac = float(np.count_nonzero(labels == k)) / max(1, total_pixels)
        cluster_large_chromatic_fill[k] = (
            _enable_warm_fill_relax
            and cluster_area_frac > 0.012
            and int(centers_hsv_render[k, 1]) >= 35
        )
        if cluster_mean_thick[k] > 0 and cluster_mean_thick[k] < 2.5:
            if cluster_interior_frac[k] < 0.25 and not cluster_large_chromatic_fill[k]:
                cluster_is_thin[k] = True

    # --- Step 3f: Detect gradient regions between adjacent clusters ---
    gradient_defs: list[GradientDef] = []
    gradient_fill_map: dict[int, str] = {}  # cluster_idx → gradient ID
    single_gradient_regions: dict[int, list[GradientRegionAssignment]] = {}
    _detect_gradients(
        labels, centers_f, image_bgr, bg_cluster,
        w, h, gradient_defs, gradient_fill_map, single_gradient_regions,
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
    _s_target = math.sqrt(_TOTAL_BUDGET / max(h * w * K_render, 1))
    if h * w > 16_000_000:
        _s_choice = int(round(_s_target))
    else:
        _s_choice = int(_s_target)
    S = max(_min_S, min(_max_S, _s_choice))
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
        # Hysteresis thresholding: strict core + selective AA fringe.
        # Build a solid core plus a lighter fringe layer so anti-aliased
        # edge pixels don't become full-strength black ink.
        _strict_thresh = min(int(_otsu_val * 0.90), 155)
        _core_thresh = min(int(_otsu_val), 165)

        # Strict mask: definitely line pixels
        _strict_mask = ((_gray_la <= _strict_thresh) * 255).astype(np.uint8)
        # Core mask: strong line pixels connected to strict cores.
        _core_mask = ((_gray_la <= _core_thresh) * 255).astype(np.uint8)

        # Keep only connected components that overlap the strict mask.
        _num_core_cc, _core_labels = cv2.connectedComponents(_core_mask)
        _strict_core_ids = set(np.unique(_core_labels[_strict_mask > 0]).tolist()) - {0}
        _core_keep = np.isin(_core_labels, list(_strict_core_ids))
        _binary_la = (_core_keep.astype(np.uint8) * 255)

        # Morph close to bridge small gaps
        _k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        _binary_la = cv2.morphologyEx(_binary_la, cv2.MORPH_CLOSE, _k_close)

        # Restore the erosion split from the known-good line-art path:
        # keep a thin core fill, route hairlines to strokes, and use the
        # outer ring as a low-opacity fringe instead of solid ink.
        _num_line_cc, _line_labels = cv2.connectedComponents((_binary_la > 0).astype(np.uint8))
        _line_hair_stroke_mask = np.zeros_like(_binary_la, dtype=np.uint8)
        _line_fill_mask = np.zeros_like(_binary_la, dtype=np.uint8)
        for _fid in range(1, _num_line_cc):
            _component = (_line_labels == _fid).astype(np.uint8)
            if not np.any(_component):
                continue
            _dt_comp = cv2.distanceTransform(_component * 255, cv2.DIST_L2, 3)
            _max_dt = float(_dt_comp.max())
            if _max_dt <= 1.5:
                _line_hair_stroke_mask = cv2.bitwise_or(_line_hair_stroke_mask, _component * 255)
            else:
                _line_fill_mask = cv2.bitwise_or(_line_fill_mask, _component * 255)

        # Split fill mask into strict core (truly dark) and hysteresis fringe
        # (AA pixels captured by lenient threshold).  The strict mask IS the
        # artist's line; the fringe IS the anti-aliasing.
        _strict_fill = cv2.bitwise_and(_strict_mask, _line_fill_mask)
        _hyst_fringe = cv2.bitwise_and(
            _line_fill_mask,
            cv2.bitwise_and(
                cv2.bitwise_not(_strict_mask),
                ((_gray_la > _strict_thresh) & (_gray_la <= _core_thresh)).astype(np.uint8) * 255,
            ),
        )
        _all_line_strokes = _line_hair_stroke_mask

        # Upscale for better contour quality
        if S > 1:
            _h_la, _w_la = _line_fill_mask.shape[:2]
            _binary_la = cv2.resize(_strict_fill, (_w_la * S, _h_la * S),
                                    interpolation=cv2.INTER_LINEAR)
            _binary_la = (_binary_la > 127).astype(np.uint8) * 255
            _binary_la_fringe = cv2.resize(_hyst_fringe, (_w_la * S, _h_la * S),
                                           interpolation=cv2.INTER_LINEAR)
            _binary_la_fringe = (_binary_la_fringe > 127).astype(np.uint8) * 255
        else:
            _binary_la = _strict_fill.copy()
            _binary_la_fringe = _hyst_fringe.copy()

        # Extract contours
        _cv_la, _hier_la = cv2.findContours(
            _binary_la, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE,
        )
        _cv_la_fringe, _hier_la_fringe = cv2.findContours(
            _binary_la_fringe, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE,
        )

        # Determine line (darkest) and background (lightest) colors
        _gray_centers = [
            0.114 * c[0] + 0.587 * c[1] + 0.299 * c[2] for c in centers_f
        ]
        _darkest_ci = int(np.argmin(_gray_centers))
        _line_hex = _bgr_to_hex(centers_u[_darkest_ci])

        def _fit_line_art_paths(_cv_contours, _hierarchy):
            _paths: list[str] = []
            _nodes = 0
            if _hierarchy is None or len(_cv_contours) == 0:
                return _paths, _nodes
            _hier0 = _hierarchy[0]
            _groups_la: list[dict] = []
            for idx in range(len(_cv_contours)):
                if _hier0[idx][3] == -1:
                    _groups_la.append({'outer': idx, 'holes': []})
            _outer_list = {g['outer']: g for g in _groups_la}
            for idx in range(len(_cv_contours)):
                parent = _hier0[idx][3]
                if parent != -1 and parent in _outer_list:
                    _outer_list[parent]['holes'].append(idx)

            _groups_la.sort(
                key=lambda g: cv2.contourArea(_cv_contours[g['outer']]), reverse=True,
            )
            _groups_la = _groups_la[:200]

            _min_area_la = max(S * S, 8)
            _groups_la = [
                g for g in _groups_la
                if cv2.contourArea(_cv_contours[g['outer']]) >= _min_area_la
            ]

            for g in _groups_la:
                parts: list[str] = []
                _pts = _cv_contours[g['outer']].squeeze(1).astype(np.float64)
                if len(_pts) < 3:
                    continue
                _pts = _pts / S
                _d = _fit_contour(_pts, simplify_epsilon * 0.14, max_error * 0.22,
                                  corner_threshold, line_tolerance * 0.6)
                if not _d:
                    continue
                parts.append(_d)
                for hi in g['holes']:
                    _hpts = _cv_contours[hi].squeeze(1).astype(np.float64)
                    if len(_hpts) < 3:
                        continue
                    _hpts = _hpts / S
                    _hd = _fit_contour(_hpts, simplify_epsilon * 0.14, max_error * 0.22,
                                       corner_threshold, line_tolerance * 0.6)
                    if _hd:
                        parts.append(_hd)
                combined = " ".join(parts)
                _paths.append(combined)
                _nodes += (combined.count("C") + combined.count("L")
                           + combined.count("M"))
            return _paths, _nodes

        # Group contours by hierarchy (outer + holes)
        _la_paths: list[str] = []
        _la_nodes = 0
        _la_paths, _la_nodes = _fit_line_art_paths(_cv_la, _hier_la)
        _la_fringe_paths, _la_fringe_nodes = _fit_line_art_paths(_cv_la_fringe, _hier_la_fringe)
        _la_hair_stroke = _process_stroke_mask(
            (_line_hair_stroke_mask > 0).astype(np.uint8),
            scale=S,
            simplify_epsilon=simplify_epsilon * 0.15,
            max_error=max_error * 0.25,
            corner_threshold=corner_threshold,
            line_tolerance=line_tolerance * 0.5,
            min_branch_length=1,
        )
        if _la_hair_stroke is None:
            _la_hair_stroke_paths, _la_hair_stroke_widths = [], []
        else:
            _la_hair_stroke_paths, _la_hair_stroke_widths = _la_hair_stroke
        _la_stroke_paths = _la_hair_stroke_paths
        _la_stroke_widths = _la_hair_stroke_widths
        _la_stroke_nodes = sum(
            path_d.count("C") + path_d.count("L") + path_d.count("M")
            for path_d in _la_stroke_paths
        )

        _t_la_end = time.time()

        _la_layer = VectorLayer(
            paths=_la_paths,
            opacities=[1.0] * len(_la_paths),
            color=_line_hex,
            shapes=None,
        )
        _layers = []
        if _la_fringe_paths:
            _layers.append(VectorLayer(
                paths=_la_fringe_paths,
                opacities=[0.55] * len(_la_fringe_paths),
                color=_line_hex,
                shapes=None,
            ))
        _layers.append(_la_layer)
        return MultilevelResult(
            layers=_layers,
            stroke_layers=[StrokeLayer(
                paths=_la_stroke_paths,
                widths=_la_stroke_widths,
                color=_line_hex,
            )] if _la_stroke_paths else [],
            width=w,
            height=h,
            background_color=bg_hex,
            path_count=len(_la_paths) + len(_la_fringe_paths) + len(_la_stroke_paths),
            node_count=_la_nodes + _la_fringe_nodes + _la_stroke_nodes,
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
        color_hex = _bgr_to_hex(render_centers_u[cluster_idx])
        mediator = cluster_mediator_score[cluster_idx]
        _t_cl = time.time()

        # --- Archimedes soft membership (using nearby competing clusters) ---
        d_k = dist_map[:, :, cluster_idx]
        d_other = _soft_competing_distance(cluster_idx, _nn_idx, _nn_dist)

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
        _k_iso_adj = min(0.02, max(0.0, (K - 8) * 0.0025))
        # Chrominance-aware iso: saturated/chromatic clusters (warm yellows,
        # reds, cyans) tend to lose boundary pixels to adjacent achromatic
        # clusters in the soft field. Lower iso slightly to expand their contours.
        _cl_lab = cv2.cvtColor(centers_u[cluster_idx].reshape(1, 1, 3), cv2.COLOR_BGR2LAB)[0, 0]
        _cl_chroma = float(np.sqrt((float(_cl_lab[1]) - 128) ** 2 + (float(_cl_lab[2]) - 128) ** 2))
        _chroma_iso_adj = min(0.015, max(0.0, (_cl_chroma - 20) * 0.0005))
        if cluster_is_thin[cluster_idx]:
            iso_level = 0.45 - _k_iso_adj * 0.5
        else:
            iso_level = 0.42 - _k_iso_adj - _chroma_iso_adj
            if _enable_warm_fill_relax and cluster_large_chromatic_fill[cluster_idx]:
                iso_level -= 0.008
            elif _sat_frac > 0.50 and K_render >= 10 and not _enable_warm_fill_relax and int(centers_hsv_render[cluster_idx, 1]) >= 70:
                iso_level += 0.008
            # Large prominent clusters: relax iso slightly to preserve major features
            _cluster_area_frac_iso = float(cluster_pix_count[cluster_idx]) / max(1, total_pixels)
            if _cluster_area_frac_iso >= 0.05 and not cluster_is_thin[cluster_idx]:
                iso_level -= 0.005

        if _warm_debug and cluster_idx in warm_debug_clusters:
            label_mask = labels == cluster_idx
            if np.any(label_mask):
                src_h = source_hsv[:, :, 0]
                src_s = source_hsv[:, :, 1]
                src_v = source_hsv[:, :, 2]
                dark_warm_mask = label_mask & (src_s >= 35) & (src_h >= 12) & (src_h <= 45) & (src_v <= 170)
                own_keep = int(np.count_nonzero(label_mask & (soft_raw > iso_level)))
                own_total = int(np.count_nonzero(label_mask))
                dark_keep = int(np.count_nonzero(dark_warm_mask & (soft_raw > iso_level)))
                dark_total = int(np.count_nonzero(dark_warm_mask))
                competitor_idx = warm_debug_primary_competitor.get(cluster_idx, -1)
                competitor_desc = f"k{competitor_idx}" if competitor_idx >= 0 else "none"
                if competitor_idx >= 0:
                    comp_dist = dist_map[:, :, competitor_idx][dark_warm_mask] if dark_total else np.empty(0, dtype=np.float32)
                else:
                    comp_dist = np.empty(0, dtype=np.float32)
                own_dist = d_k[dark_warm_mask] if dark_total else np.empty(0, dtype=np.float32)
                mean_own = float(own_dist.mean()) if own_dist.size else 0.0
                mean_comp = float(comp_dist.mean()) if comp_dist.size else 0.0
                print(
                    f"[WARM] k{cluster_idx} iso={iso_level:.3f} keep={own_keep}/{own_total} "
                    f"dark_keep={dark_keep}/{dark_total} competitor={competitor_desc} "
                    f"mean_d_self={mean_own:.2f} mean_d_comp={mean_comp:.2f} "
                    f"thin={int(cluster_is_thin[cluster_idx])} thick={cluster_mean_thick[cluster_idx]:.2f} "
                    f"interior={cluster_interior_frac[cluster_idx]:.3f} mediator={mediator:.3f}"
                )
        elif _cluster_debug:
            label_mask = labels == cluster_idx
            if np.any(label_mask):
                own_keep = int(np.count_nonzero(label_mask & (soft_raw > iso_level)))
                own_total = int(np.count_nonzero(label_mask))
                keep_ratio = own_keep / max(1, own_total)
                if own_total >= max(10000, int(total_pixels * 0.002)):
                    center_hsv = centers_hsv_render[cluster_idx]
                    print(
                        f"[CLUSTER] k{cluster_idx} keep={own_keep}/{own_total} ratio={keep_ratio:.3f} "
                        f"iso={iso_level:.3f} thin={int(cluster_is_thin[cluster_idx])} "
                        f"thick={cluster_mean_thick[cluster_idx]:.2f} interior={cluster_interior_frac[cluster_idx]:.3f} "
                        f"mediator={mediator:.3f} hsv=({int(center_hsv[0])},{int(center_hsv[1])},{int(center_hsv[2])})"
                    )
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
        layer_path_fills: list[str] = []
        cluster_gradient_regions = single_gradient_regions.get(cluster_idx)
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
        _cluster_area_frac = float(cluster_pix_count[cluster_idx]) / max(1, total_pixels)
        _cluster_sat = int(centers_hsv_render[cluster_idx, 1])
        _texture_rich_cluster = _cluster_sat >= 55 and _cluster_area_frac >= 0.006
        _fitting_budget_groups = max(400, 1600 // max(K_render, 1))
        if pixel_count < 4_000_000:
            MAX_GROUPS = min(1400, _fitting_budget_groups)
        else:
            MAX_GROUPS = _fitting_budget_groups
        if _cluster_area_frac >= 0.05:
            MAX_GROUPS += 900
        elif _cluster_area_frac >= 0.02:
            MAX_GROUPS += 450
        elif _cluster_area_frac >= 0.01:
            MAX_GROUPS += 200
        if _texture_rich_cluster:
            MAX_GROUPS += 250
        if cluster_is_thin[cluster_idx] and _cluster_area_frac >= 0.003:
            MAX_GROUPS += 150
        MAX_GROUPS = min(2200, MAX_GROUPS)
        if len(_contour_groups) > MAX_GROUPS:
            _raw_areas = [abs(cv2.contourArea(_cv_contours[g[0]])) / (S * S)
                          for g in _contour_groups]
            _keep_idx = np.argsort(_raw_areas)[::-1][:MAX_GROUPS]
            _contour_groups = [_contour_groups[i] for i in _keep_idx]

        # Convert contour groups to xy arrays, fit Béziers, build paths
        core_parts_per_group: list[list[str]] = []
        group_bboxes: list[tuple[float, float, float, float]] = []
        group_probe_points: list[np.ndarray] = []
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
                if _texture_rich_cluster:
                    _area_mult *= 0.75
                elif _cluster_area_frac >= 0.02:
                    _area_mult *= 0.85
                # Dense-texture clusters: raise area floor to cull angular
                # micro-fragments (e.g. forest canopy shards)
                if len(_contour_groups) > 300:
                    _area_mult *= 1.5
                if elongation > 50 and perim_raw > 8:
                    # Elongated thin shape — likely a line, use lower threshold
                    min_frag_area = min_contour_area * 0.5 * _area_mult
                elif cluster_is_thin[cluster_idx]:
                    # Thin cluster — preserve more fragments
                    min_frag_area = min_contour_area * 0.8 * _area_mult
                else:
                    # Non-thin clusters: slightly tighter filtering to reduce noise
                    min_frag_area = min_contour_area * 1.3 * _area_mult
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
                    contour_sigma = (0.25 + t * 0.25) * S
                else:
                    contour_sigma = (0.38 + t * 0.46) * S
                    # Extra smoothing for dense-texture clusters (many groups
                    # = angular fragment-rich areas like forest canopy)
                    if len(_contour_groups) > 300:
                        contour_sigma *= 1.4
                contour_sigma = max(contour_sigma, 0.7)

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
                    if cluster_gradient_regions is None and not cluster_is_thin[cluster_idx] and not has_holes:
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
                _outer = _cv_contours[group[0]].reshape(-1, 2).astype(np.float64) / S
                group_bboxes.append((
                    float(np.min(_outer[:, 0])),
                    float(np.min(_outer[:, 1])),
                    float(np.max(_outer[:, 0])),
                    float(np.max(_outer[:, 1])),
                ))
                if len(_outer) > 12:
                    _probe_idx = np.linspace(0, len(_outer) - 1, 12, dtype=int)
                    _probes = _outer[_probe_idx]
                else:
                    _probes = _outer
                group_probe_points.append(_probes)
                total_group_area.append(group_area)

        def _score_gradient_region_match(
            region: GradientRegionAssignment,
            group_bbox: tuple[float, float, float, float],
            probe_pts: np.ndarray,
        ) -> tuple[float, str]:
            gx0, gy0, gx1, gy1 = group_bbox
            rx0, ry0, rx1, ry1 = region.bbox
            overlap_x0 = max(gx0, rx0)
            overlap_y0 = max(gy0, ry0)
            overlap_x1 = min(gx1, rx1)
            overlap_y1 = min(gy1, ry1)
            if overlap_x1 <= overlap_x0 or overlap_y1 <= overlap_y0:
                return -1.0, region.fill_ref

            group_w = max(gx1 - gx0, 1e-6)
            group_h = max(gy1 - gy0, 1e-6)
            group_area = group_w * group_h
            overlap_area = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)

            sample_points = [
                (0.5 * (gx0 + gx1), 0.5 * (gy0 + gy1)),
                (gx0 + group_w * 0.35, gy0 + group_h * 0.35),
                (gx0 + group_w * 0.65, gy0 + group_h * 0.35),
                (gx0 + group_w * 0.35, gy0 + group_h * 0.65),
                (gx0 + group_w * 0.65, gy0 + group_h * 0.65),
            ]

            interior_hits = 0
            center_in_mask = False
            for sample_idx, (px, py) in enumerate(sample_points):
                ix = int(np.clip(round(px), 0, w - 1))
                iy = int(np.clip(round(py), 0, h - 1))
                if region.mask[iy, ix]:
                    interior_hits += 1
                    if sample_idx == 0:
                        center_in_mask = True

            boundary_hits = 0
            for px, py in probe_pts:
                ix = int(np.clip(round(px), 0, w - 1))
                iy = int(np.clip(round(py), 0, h - 1))
                if region.mask[iy, ix]:
                    boundary_hits += 1

            score = 0.0
            if center_in_mask:
                score += 10.0
            score += interior_hits * 3.0
            score += boundary_hits * 0.75
            score += min(3.0, 3.0 * overlap_area / group_area)
            return score, region.fill_ref

        def _compute_group_fill_hex(group: list[int], group_area: float) -> str:
            if _sat_frac <= 0.25 or cluster_is_thin[cluster_idx] or group_area < total_pixels * 0.001:
                return color_hex

            native_contours: list[np.ndarray] = []
            x0 = w
            y0 = h
            x1 = 0
            y1 = 0
            for ci in group:
                pts = _cv_contours[ci].reshape(-1, 2).astype(np.float32) / S
                pts = np.round(pts).astype(np.int32)
                if len(pts) < 3:
                    continue
                native_contours.append(pts)
                x0 = min(x0, int(pts[:, 0].min()))
                y0 = min(y0, int(pts[:, 1].min()))
                x1 = max(x1, int(pts[:, 0].max()))
                y1 = max(y1, int(pts[:, 1].max()))

            if not native_contours or x1 <= x0 or y1 <= y0:
                return color_hex

            x0 = max(0, x0)
            y0 = max(0, y0)
            x1 = min(w - 1, x1)
            y1 = min(h - 1, y1)
            if x1 <= x0 or y1 <= y0:
                return color_hex

            roi_mask = np.zeros((y1 - y0 + 1, x1 - x0 + 1), dtype=np.uint8)
            outer = (native_contours[0] - np.array([x0, y0], dtype=np.int32)).reshape(-1, 1, 2)
            cv2.fillPoly(roi_mask, [outer], 255)
            for hole in native_contours[1:]:
                hole_pts = (hole - np.array([x0, y0], dtype=np.int32)).reshape(-1, 1, 2)
                cv2.fillPoly(roi_mask, [hole_pts], 0)

            roi_cluster = labels[y0:y1 + 1, x0:x1 + 1] == cluster_idx
            sample_mask = (roi_mask > 0) & roi_cluster
            if int(np.count_nonzero(sample_mask)) < 64:
                return color_hex

            samples = image_bgr[y0:y1 + 1, x0:x1 + 1][sample_mask].astype(np.float32)
            group_color = _render_color_from_samples(samples, render_centers_u[cluster_idx].astype(np.float32))
            return _bgr_to_hex(np.clip(np.round(group_color), 0, 255).astype(np.uint8))

        def _try_group_gradient_fill(group: list[int], group_area: float) -> str | None:
            if _sat_frac <= 0.25 or cluster_is_thin[cluster_idx] or group_area < total_pixels * 0.004:
                return None

            native_contours: list[np.ndarray] = []
            x0 = w
            y0 = h
            x1 = 0
            y1 = 0
            for ci in group:
                pts = _cv_contours[ci].reshape(-1, 2).astype(np.float32) / S
                pts = np.round(pts).astype(np.int32)
                if len(pts) < 3:
                    continue
                native_contours.append(pts)
                x0 = min(x0, int(pts[:, 0].min()))
                y0 = min(y0, int(pts[:, 1].min()))
                x1 = max(x1, int(pts[:, 0].max()))
                y1 = max(y1, int(pts[:, 1].max()))

            if not native_contours or x1 <= x0 or y1 <= y0:
                return None

            x0 = max(0, x0)
            y0 = max(0, y0)
            x1 = min(w - 1, x1)
            y1 = min(h - 1, y1)
            if x1 <= x0 or y1 <= y0:
                return None

            roi_mask = np.zeros((y1 - y0 + 1, x1 - x0 + 1), dtype=np.uint8)
            outer = (native_contours[0] - np.array([x0, y0], dtype=np.int32)).reshape(-1, 1, 2)
            cv2.fillPoly(roi_mask, [outer], 255)
            for hole in native_contours[1:]:
                hole_pts = (hole - np.array([x0, y0], dtype=np.int32)).reshape(-1, 1, 2)
                cv2.fillPoly(roi_mask, [hole_pts], 0)

            roi_cluster = labels[y0:y1 + 1, x0:x1 + 1] == cluster_idx
            sample_mask = (roi_mask > 0) & roi_cluster
            if int(np.count_nonzero(sample_mask)) < 256:
                return None

            ys_local, xs_local = np.where(sample_mask)
            coords = np.column_stack([
                xs_local.astype(np.float64) + x0,
                ys_local.astype(np.float64) + y0,
            ])
            colors = image_bgr[y0:y1 + 1, x0:x1 + 1][sample_mask].astype(np.float64)
            color_mean = colors.mean(axis=0)
            centered = colors - color_mean
            try:
                cov = np.cov(centered.T)
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
            except np.linalg.LinAlgError:
                return None

            principal_idx = int(np.argmax(eigenvalues))
            spread = float(np.sqrt(eigenvalues[principal_idx]))
            if spread < 10.0 or spread > 120.0:
                return None

            projections = centered @ eigenvectors[:, principal_idx]
            coord_mean = coords.mean(axis=0)
            coords_c = coords - coord_mean
            try:
                spatial_cov = np.cov(coords_c.T, projections)
            except Exception:
                return None
            if spatial_cov.shape != (3, 3):
                return None
            spatial_color_cov = spatial_cov[0:2, 2]
            grad_dir = spatial_color_cov / (np.linalg.norm(spatial_color_cov) + 1e-10)
            proj_spatial = coords_c @ grad_dir
            p_min = float(proj_spatial.min())
            p_max = float(proj_spatial.max())
            if p_max - p_min < 40.0:
                return None

            try:
                spatial_corr = float(np.corrcoef(proj_spatial, projections)[0, 1])
            except Exception:
                return None
            if not np.isfinite(spatial_corr) or abs(spatial_corr) < 0.35:
                return None

            low_mask = proj_spatial < np.percentile(proj_spatial, 10)
            high_mask = proj_spatial > np.percentile(proj_spatial, 90)
            if low_mask.sum() < 12 or high_mask.sum() < 12:
                return None

            color_start = colors[low_mask].mean(axis=0)
            color_end = colors[high_mask].mean(axis=0)
            endpoint_dist = float(np.linalg.norm(color_end - color_start))
            if endpoint_dist < 12.0:
                return None

            span = max(p_max - p_min, 1e-6)
            t_vals = np.clip((proj_spatial - p_min) / span, 0.0, 1.0)
            color_line = color_start[None, :] + (color_end - color_start)[None, :] * t_vals[:, None]
            fit_residual = float(np.mean(np.linalg.norm(colors - color_line, axis=1)))
            if fit_residual > max(18.0, endpoint_dist * 0.42):
                return None

            mid_mask = (t_vals >= 0.35) & (t_vals <= 0.65)
            color_mid_hex = None
            if mid_mask.sum() >= 12:
                color_mid_hex = _bgr_to_hex(np.clip(np.round(colors[mid_mask].mean(axis=0)), 0, 255).astype(np.uint8))

            gid = f"g{len(gradient_defs)}"
            gradient_defs.append(GradientDef(
                id=gid,
                x1=float(np.clip(coord_mean[0] + grad_dir[0] * p_min, 0, w)),
                y1=float(np.clip(coord_mean[1] + grad_dir[1] * p_min, 0, h)),
                x2=float(np.clip(coord_mean[0] + grad_dir[0] * p_max, 0, w)),
                y2=float(np.clip(coord_mean[1] + grad_dir[1] * p_max, 0, h)),
                color_start=_bgr_to_hex(np.clip(np.round(color_start), 0, 255).astype(np.uint8)),
                color_end=_bgr_to_hex(np.clip(np.round(color_end), 0, 255).astype(np.uint8)),
                color_mid=color_mid_hex,
            ))
            return f"url(#{gid})"

        def _extract_group_detail_overlays(group: list[int], group_area: float) -> list[tuple[str, str]]:
            if _sat_frac <= 0.25 or cluster_is_thin[cluster_idx] or group_area < total_pixels * 0.012:
                return []

            _cluster_sat = int(centers_hsv_render[cluster_idx, 1])
            _is_texture_rich_sat = _cluster_sat >= 70 and _sat_frac > 0.55

            native_contours: list[np.ndarray] = []
            x0 = w
            y0 = h
            x1 = 0
            y1 = 0
            for ci in group:
                pts = _cv_contours[ci].reshape(-1, 2).astype(np.float32) / S
                pts = np.round(pts).astype(np.int32)
                if len(pts) < 3:
                    continue
                native_contours.append(pts)
                x0 = min(x0, int(pts[:, 0].min()))
                y0 = min(y0, int(pts[:, 1].min()))
                x1 = max(x1, int(pts[:, 0].max()))
                y1 = max(y1, int(pts[:, 1].max()))

            if not native_contours or x1 <= x0 or y1 <= y0:
                return []

            x0 = max(0, x0)
            y0 = max(0, y0)
            x1 = min(w - 1, x1)
            y1 = min(h - 1, y1)
            if x1 <= x0 or y1 <= y0:
                return []

            roi_mask = np.zeros((y1 - y0 + 1, x1 - x0 + 1), dtype=np.uint8)
            outer = (native_contours[0] - np.array([x0, y0], dtype=np.int32)).reshape(-1, 1, 2)
            cv2.fillPoly(roi_mask, [outer], 255)
            for hole in native_contours[1:]:
                hole_pts = (hole - np.array([x0, y0], dtype=np.int32)).reshape(-1, 1, 2)
                cv2.fillPoly(roi_mask, [hole_pts], 0)

            roi_cluster = labels[y0:y1 + 1, x0:x1 + 1] == cluster_idx
            sample_mask = (roi_mask > 0) & roi_cluster
            sample_count = int(np.count_nonzero(sample_mask))
            if sample_count < (250 if _is_texture_rich_sat else 400):
                return []

            roi_gray = cv2.cvtColor(image_bgr[y0:y1 + 1, x0:x1 + 1], cv2.COLOR_BGR2GRAY)
            gray_samples = roi_gray[sample_mask].astype(np.float32)
            gray_span = float(np.percentile(gray_samples, 90) - np.percentile(gray_samples, 10))
            if gray_span < (15.0 if _is_texture_rich_sat else 24.0):
                return []

            overlays: list[tuple[str, str]] = []
            _max_overlays = 4 if _is_texture_rich_sat else 2
            _components_per_polarity = 2 if _is_texture_rich_sat else 1
            for detail_mask_raw in (
                roi_gray <= np.percentile(gray_samples, 34),
                roi_gray >= np.percentile(gray_samples, 66),
            ):
                detail_mask = (detail_mask_raw & sample_mask).astype(np.uint8)
                if int(np.count_nonzero(detail_mask)) < sample_count * 0.08:
                    continue
                cc_count, cc_labels = cv2.connectedComponents(detail_mask)
                if cc_count <= 1:
                    continue
                cc_sizes = np.bincount(cc_labels.ravel())[1:]
                if len(cc_sizes) == 0:
                    continue
                component_order = np.argsort(cc_sizes)[::-1]
                for component_rank in component_order[:_components_per_polarity]:
                    component_idx = int(component_rank) + 1
                    component_size = int(cc_sizes[component_rank])
                    if component_size < sample_count * 0.08 or component_size > sample_count * 0.72:
                        continue
                    largest_mask = (cc_labels == component_idx).astype(np.uint8)
                    largest_mask = cv2.morphologyEx(
                        largest_mask,
                        cv2.MORPH_OPEN,
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                    )
                    if int(np.count_nonzero(largest_mask)) < sample_count * 0.06:
                        continue

                    detail_contours, detail_hierarchy = cv2.findContours(
                        (largest_mask * 255).astype(np.uint8),
                        cv2.RETR_CCOMP,
                        cv2.CHAIN_APPROX_SIMPLE,
                    )
                    if detail_hierarchy is None or len(detail_contours) == 0:
                        continue

                    for detail_idx in range(len(detail_contours)):
                        if detail_hierarchy[0][detail_idx][3] != -1:
                            continue
                        contour = detail_contours[detail_idx].squeeze(1).astype(np.float64)
                        if len(contour) < 10:
                            continue
                        area_real = abs(cv2.contourArea(detail_contours[detail_idx]))
                        if area_real < min_contour_area * 8:
                            continue
                        contour[:, 0] += x0
                        contour[:, 1] += y0
                        contour = _smooth_contour(contour, sigma=max(0.8, 0.28 * S))
                        d_detail = _fit_contour(
                            contour,
                            simplify_epsilon * 0.8,
                            max_error * 0.8,
                            corner_threshold,
                            line_tolerance,
                        )
                        if not d_detail:
                            continue
                        detail_pixels = image_bgr[y0:y1 + 1, x0:x1 + 1][largest_mask > 0].astype(np.float32)
                        if len(detail_pixels) < 32:
                            continue
                        detail_color = _render_color_from_samples(
                            detail_pixels,
                            render_centers_u[cluster_idx].astype(np.float32),
                        )
                        detail_hex = _bgr_to_hex(np.clip(np.round(detail_color), 0, 255).astype(np.uint8))
                        overlays.append((d_detail, detail_hex))
                        break
                    if len(overlays) >= _max_overlays:
                        break
            return overlays[:_max_overlays]

        # Each group becomes one SVG path (outer + holes = evenodd cutouts)
        cluster_fill_ref = gradient_fill_map.get(cluster_idx)
        group_gradient_scores: list[tuple[float, str]] = []
        for group, group_parts, group_bbox, probe_pts, group_area in zip(
            _contour_groups,
            core_parts_per_group,
            group_bboxes,
            group_probe_points,
            total_group_area,
        ):
            combined = " ".join(group_parts)
            layer_paths.append(combined)
            layer_opacities.append(1.0)
            solid_fill_hex = _compute_group_fill_hex(group, group_area)
            local_gradient_fill = _try_group_gradient_fill(group, group_area)
            if cluster_gradient_regions:
                fill_ref = local_gradient_fill or solid_fill_hex
                best_score = -1.0
                for region in cluster_gradient_regions:
                    region_score, region_fill_ref = _score_gradient_region_match(region, group_bbox, probe_pts)
                    if region_score > best_score:
                        best_score = region_score
                        fill_ref = region_fill_ref
                if best_score < 6.0:
                    fill_ref = local_gradient_fill or solid_fill_hex
                group_gradient_scores.append((best_score, fill_ref))
                layer_path_fills.append(fill_ref)
            elif cluster_fill_ref is not None:
                layer_path_fills.append(cluster_fill_ref)
            else:
                layer_path_fills.append(local_gradient_fill or solid_fill_hex)
            _local_paths += 1
            _local_nodes += combined.count("C") + combined.count("L") + combined.count("M")

            for detail_path, detail_fill in _extract_group_detail_overlays(group, group_area):
                layer_paths.append(detail_path)
                layer_opacities.append(1.0)
                layer_path_fills.append(detail_fill)
                _local_paths += 1
                _local_nodes += detail_path.count("C") + detail_path.count("L") + detail_path.count("M")

        if cluster_gradient_regions and layer_path_fills and not any(fill.startswith("url(#") for fill in layer_path_fills):
            _best_group_idx = -1
            _best_group_score = 0.0
            _best_group_fill = ""
            for _group_idx, (_score, _fill_ref) in enumerate(group_gradient_scores):
                if _score > _best_group_score:
                    _best_group_idx = _group_idx
                    _best_group_score = _score
                    _best_group_fill = _fill_ref
            if _best_group_idx >= 0 and _best_group_score >= 4.0:
                layer_path_fills[_best_group_idx] = _best_group_fill

        # Stroke-mode rendering is available for very thin clusters but
        # currently disabled — fill + adaptive iso handles thin features
        # well enough without the complexity of stroke reconstruction.
        used_stroke = False
        _t_cl_end = time.time()
        _n_groups = len(_contour_groups)
        _total_contour_area = sum(total_group_area)


        # Build fill layer
        if (layer_paths or layer_shapes) and not used_stroke:
            return (VectorLayer(
                paths=layer_paths,
                opacities=layer_opacities,
                color=color_hex,
                shapes=layer_shapes,
                path_fills=layer_path_fills if layer_path_fills else None,
            ), _local_paths, _local_nodes, _local_curves_time)
        return None

    # --- Parallel cluster processing ---
    _cluster_order = [ci for ci in order if ci != bg_cluster]

    _n_workers = 1
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

    used_gradient_ids: set[str] = set()
    for layer in result.layers:
        if layer.color.startswith("url(#") and layer.color.endswith(")"):
            used_gradient_ids.add(layer.color[5:-1])
        if layer.path_fills:
            for fill_ref in layer.path_fills:
                if fill_ref.startswith("url(#") and fill_ref.endswith(")"):
                    used_gradient_ids.add(fill_ref[5:-1])

    active_gradient_defs = [
        gd for gd in (result.gradient_defs or [])
        if gd.id in used_gradient_ids
    ]

    # Emit gradient definitions if present
    if active_gradient_defs:
        parts.append('<defs>')
        for gd in active_gradient_defs:
            stops = f'<stop offset="0%" stop-color="{gd.color_start}"/>'
            if gd.color_mid:
                stops += f'<stop offset="50%" stop-color="{gd.color_mid}"/>'
            stops += f'<stop offset="100%" stop-color="{gd.color_end}"/>'
            if gd.kind == "radial" and gd.cx is not None and gd.cy is not None and gd.r is not None:
                _fx = gd.fx if gd.fx is not None else gd.cx
                _fy = gd.fy if gd.fy is not None else gd.cy
                parts.append(
                    f'<radialGradient id="{gd.id}" '
                    f'cx="{gd.cx:.1f}" cy="{gd.cy:.1f}" '
                    f'r="{gd.r:.1f}" '
                    f'fx="{_fx:.1f}" fy="{_fy:.1f}" '
                    f'gradientUnits="userSpaceOnUse">'
                    f'{stops}'
                    f'</radialGradient>'
                )
            else:
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
        for idx, (path_d, opacity) in enumerate(zip(layer.paths, layer.opacities)):
            if not path_d:
                continue
            fill_ref = layer.path_fills[idx] if layer.path_fills and idx < len(layer.path_fills) else layer.color
            if opacity >= 1.0:
                parts.append(
                    f'<path d="{path_d}" fill="{fill_ref}"'
                    f' fill-rule="evenodd" shape-rendering="{_shape_render}"/>'
                )
            else:
                parts.append(
                    f'<path d="{path_d}" fill="{fill_ref}"'
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
    single_gradient_regions: dict,
    min_region_pct: float = 2.0,
    color_dist_range: tuple[float, float] = (25.0, 150.0),
) -> None:
    """Detect gradient regions between adjacent clusters via PCA on pixel colors.


    Modifies gradient_defs and gradient_fill_map in-place.
    """
    K = len(centers_f)
    total_px = h * w
    src_f = source_bgr.astype(np.float64)
    src_gray = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    src_gray_blur = cv2.GaussianBlur(src_gray, (0, 0), sigmaX=3.0)
    grad_id = 0
    _diag = os.environ.get("SVG_GRADIENT_DEBUG") == "1"
    _diag_counts: dict[str, int] = {
        "pair_candidates": 0,
        "pair_accept": 0,
        "single_candidates": 0,
        "single_accept": 0,
        "reject_small": 0,
        "reject_cov": 0,
        "reject_spread": 0,
        "reject_short": 0,
        "reject_corr": 0,
        "reject_endpoints": 0,
        "reject_endpoint_dist": 0,
        "reject_texture": 0,
        "reject_pair_hue": 0,
        "reject_pair_cc": 0,
        "reject_single_cc": 0,
    }
    _diag_corr_samples: list[tuple[int, int, float]] = []
    _diag_accept_regions: list[str] = []
    _diag_component_candidates: list[str] = []
    _diag_reject_regions: list[str] = []
    _diag_radial_regions: list[str] = []

    def _diag_reject(diag_label: str | None, reason: str) -> None:
        if _diag and diag_label and len(_diag_reject_regions) < 24:
            _diag_reject_regions.append(f"{diag_label}:{reason}")

    def _fit_gradient_region(
        mask: np.ndarray,
        seed: int,
        min_region_pct_override: float | None = None,
        diag_label: str | None = None,
    ) -> GradientDef | None:
        pix_count = int(np.count_nonzero(mask))
        _min_region_pct = min_region_pct if min_region_pct_override is None else min_region_pct_override
        if pix_count < total_px * _min_region_pct / 100.0:
            _diag_counts["reject_small"] += 1
            _diag_reject(diag_label, "small")
            return None

        ys, xs = np.where(mask)
        if len(ys) > 8000:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(ys), 8000, replace=False)
            ys_s, xs_s = ys[idx], xs[idx]
        else:
            ys_s, xs_s = ys, xs
        colors = src_f[ys_s, xs_s]

        color_mean = colors.mean(axis=0)
        mean_hsv = cv2.cvtColor(
            np.clip(color_mean, 0, 255).astype(np.uint8).reshape(1, 1, 3),
            cv2.COLOR_BGR2HSV,
        )[0, 0]
        _warm_mid_sat = 5 <= int(mean_hsv[0]) <= 25 and 25 <= int(mean_hsv[1]) <= 140
        _small_warm_region = _warm_mid_sat and pix_count < total_px * 0.02
        coords = np.column_stack([xs_s.astype(np.float64), ys_s.astype(np.float64)])
        gray_vals = src_gray_blur[ys_s, xs_s]

        def _try_small_warm_radial_gradient() -> GradientDef | None:
            if not _small_warm_region:
                return None

            bright_mask = gray_vals >= np.percentile(gray_vals, 82)
            dark_mask = gray_vals <= np.percentile(gray_vals, 18)
            if bright_mask.sum() < 5 or dark_mask.sum() < 5:
                if _diag and diag_label and len(_diag_radial_regions) < 24:
                    _diag_radial_regions.append(f"{diag_label}:radial_endpoints")
                return None

            bright_center = coords[bright_mask].mean(axis=0)
            radial_dist = np.linalg.norm(coords - bright_center[None, :], axis=1)
            radial_span = float(np.percentile(radial_dist, 95))
            if radial_span < 8.0:
                if _diag and diag_label and len(_diag_radial_regions) < 24:
                    _diag_radial_regions.append(f"{diag_label}:radial_span<{radial_span:.1f}")
                return None

            inner_cut = np.percentile(radial_dist, 24)
            outer_cut = np.percentile(radial_dist, 76)
            inner_mask = radial_dist <= inner_cut
            outer_mask = radial_dist >= outer_cut
            if inner_mask.sum() < 5 or outer_mask.sum() < 5:
                if _diag and diag_label and len(_diag_radial_regions) < 24:
                    _diag_radial_regions.append(f"{diag_label}:radial_masks")
                return None

            color_inner = colors[inner_mask].mean(axis=0)
            color_outer = colors[outer_mask].mean(axis=0)
            if dark_mask.sum() >= 5:
                color_outer_dark = colors[dark_mask].mean(axis=0)
                if np.linalg.norm(color_outer_dark - color_inner) > np.linalg.norm(color_outer - color_inner):
                    color_outer = color_outer_dark
            endpoint_dist = float(np.linalg.norm(color_outer - color_inner))
            if endpoint_dist < 10.0:
                if _diag and diag_label and len(_diag_radial_regions) < 24:
                    _diag_radial_regions.append(f"{diag_label}:radial_endpoint<{endpoint_dist:.1f}")
                return None

            t_vals_radial = np.clip(radial_dist / max(radial_span, 1e-6), 0.0, 1.0)
            color_radial = color_inner[None, :] + (color_outer - color_inner)[None, :] * t_vals_radial[:, None]
            radial_fit = float(np.mean(np.linalg.norm(colors - color_radial, axis=1)))
            radial_fit_thresh = max(14.0, endpoint_dist * 0.34) * 1.90
            if radial_fit > radial_fit_thresh:
                if _diag and diag_label and len(_diag_radial_regions) < 24:
                    _diag_radial_regions.append(f"{diag_label}:radial_fit:{radial_fit:.1f}>{radial_fit_thresh:.1f}")
                return None

            mid_mask = (t_vals_radial >= 0.35) & (t_vals_radial <= 0.65)
            color_mid_hex = None
            if mid_mask.sum() >= 5:
                color_mid = colors[mid_mask].mean(axis=0)
                color_mid_hex = _bgr_to_hex(np.clip(color_mid, 0, 255).astype(np.uint8))

            nonlocal grad_id
            gid = f"g{grad_id}"
            grad_id += 1
            region_center = coords.mean(axis=0)
            radius = float(np.percentile(radial_dist, 92))
            if _diag and diag_label and len(_diag_radial_regions) < 24:
                _diag_radial_regions.append(f"{diag_label}:radial_ok:r={radius:.1f}")
            return GradientDef(
                id=gid,
                x1=float(bright_center[0]), y1=float(bright_center[1]),
                x2=float(region_center[0]), y2=float(region_center[1]),
                kind="radial",
                cx=float(np.clip(region_center[0], 0, w)),
                cy=float(np.clip(region_center[1], 0, h)),
                r=float(np.clip(radius, 1.0, max(w, h))),
                fx=float(np.clip(bright_center[0], 0, w)),
                fy=float(np.clip(bright_center[1], 0, h)),
                color_start=_bgr_to_hex(np.clip(color_inner, 0, 255).astype(np.uint8)),
                color_end=_bgr_to_hex(np.clip(color_outer, 0, 255).astype(np.uint8)),
                color_mid=color_mid_hex,
            )

        centered = colors - color_mean
        try:
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            _diag_counts["reject_cov"] += 1
            _diag_reject(diag_label, "cov")
            return None

        principal_idx = int(np.argmax(eigenvalues))
        spread = float(np.sqrt(eigenvalues[principal_idx]))
        if spread < color_dist_range[0] * 0.3 or spread > color_dist_range[1]:
            _diag_counts["reject_spread"] += 1
            _diag_reject(diag_label, "spread")
            return None

        principal_axis = eigenvectors[:, principal_idx]
        projections = centered @ principal_axis

        coord_mean = coords.mean(axis=0)
        coords_c = coords - coord_mean

        try:
            spatial_cov = np.cov(coords_c.T, projections)
        except Exception:
            _diag_counts["reject_cov"] += 1
            _diag_reject(diag_label, "cov")
            return None
        if spatial_cov.shape != (3, 3):
            _diag_counts["reject_cov"] += 1
            _diag_reject(diag_label, "cov")
            return None
        spatial_color_cov = spatial_cov[0:2, 2]
        grad_dir = spatial_color_cov / (np.linalg.norm(spatial_color_cov) + 1e-10)

        proj_spatial = coords_c @ grad_dir
        p_min, p_max = float(proj_spatial.min()), float(proj_spatial.max())
        if p_max - p_min < 10:
            _diag_counts["reject_short"] += 1
            _diag_reject(diag_label, "short")
            return None
        try:
            spatial_corr = float(np.corrcoef(proj_spatial, projections)[0, 1])
        except Exception:
            _diag_counts["reject_corr"] += 1
            _diag_reject(diag_label, "corr_exc")
            return None
        if _diag and len(_diag_corr_samples) < 12:
            _diag_corr_samples.append((seed, pix_count, spatial_corr))
        if pix_count >= total_px * 0.03:
            _min_spatial_corr = 0.30
        elif _small_warm_region:
            _min_spatial_corr = 0.28
        else:
            _min_spatial_corr = 0.65
        if not np.isfinite(spatial_corr) or abs(spatial_corr) < _min_spatial_corr:
            _used_alt_axis = False
            if _small_warm_region:
                _bright_mask = gray_vals >= np.percentile(gray_vals, 82)
                _dark_mask = gray_vals <= np.percentile(gray_vals, 18)
                if _bright_mask.sum() >= 5 and _dark_mask.sum() >= 5:
                    _alt_vec = coords[_bright_mask].mean(axis=0) - coords[_dark_mask].mean(axis=0)
                    _alt_norm = float(np.linalg.norm(_alt_vec))
                    if _alt_norm > 1e-6:
                        grad_dir = _alt_vec / _alt_norm
                        proj_spatial = coords_c @ grad_dir
                        p_min, p_max = float(proj_spatial.min()), float(proj_spatial.max())
                        if p_max - p_min >= 10:
                            try:
                                _alt_corr = float(np.corrcoef(proj_spatial, gray_vals)[0, 1])
                            except Exception:
                                _alt_corr = 0.0
                            if np.isfinite(_alt_corr) and abs(_alt_corr) >= 0.28:
                                spatial_corr = _alt_corr
                                _used_alt_axis = True
            if not _used_alt_axis:
                radial_gd = _try_small_warm_radial_gradient()
                if radial_gd is not None:
                    return radial_gd
                _diag_counts["reject_corr"] += 1
                _diag_reject(diag_label, f"corr<{_min_spatial_corr:.2f}:{spatial_corr:.2f}")
                return None

        low_mask = proj_spatial < np.percentile(proj_spatial, 8)
        high_mask = proj_spatial > np.percentile(proj_spatial, 92)
        if low_mask.sum() < 5 or high_mask.sum() < 5:
            _diag_counts["reject_endpoints"] += 1
            _diag_reject(diag_label, "endpoints")
            return None
        color_start = colors[low_mask].mean(axis=0)
        color_end = colors[high_mask].mean(axis=0)
        endpoint_dist = float(np.linalg.norm(color_end - color_start))
        if endpoint_dist < 10:
            _diag_counts["reject_endpoint_dist"] += 1
            _diag_reject(diag_label, f"endpoint:{endpoint_dist:.1f}")
            return None

        texture_residual = float(np.std(src_gray[ys_s, xs_s] - src_gray_blur[ys_s, xs_s]))
        _warm_scale = 1.55 if _small_warm_region else (1.35 if _warm_mid_sat else 1.0)
        _texture_thresh = max(6.5, endpoint_dist * 0.42) * _warm_scale
        if texture_residual > _texture_thresh:
            _diag_counts["reject_texture"] += 1
            _diag_reject(diag_label, f"texture:{texture_residual:.1f}>{_texture_thresh:.1f}")
            return None

        spatial_span = max(p_max - p_min, 1e-6)
        t_vals = np.clip((proj_spatial - p_min) / spatial_span, 0.0, 1.0)
        color_line = color_start[None, :] + (color_end - color_start)[None, :] * t_vals[:, None]
        fit_residual = float(np.mean(np.linalg.norm(colors - color_line, axis=1)))
        _fit_thresh = max(12.0, endpoint_dist * 0.30) * _warm_scale
        if fit_residual > _fit_thresh:
            radial_gd = _try_small_warm_radial_gradient()
            if radial_gd is not None:
                return radial_gd
            _diag_counts["reject_texture"] += 1
            _diag_reject(diag_label, f"fit:{fit_residual:.1f}>{_fit_thresh:.1f}")
            return None

        x1 = float(coord_mean[0] + grad_dir[0] * p_min)
        y1 = float(coord_mean[1] + grad_dir[1] * p_min)
        x2 = float(coord_mean[0] + grad_dir[0] * p_max)
        y2 = float(coord_mean[1] + grad_dir[1] * p_max)

        mid_lo = np.percentile(proj_spatial, 35)
        mid_hi = np.percentile(proj_spatial, 65)
        mid_mask = (proj_spatial >= mid_lo) & (proj_spatial <= mid_hi)
        color_mid_hex = None
        if mid_mask.sum() >= 5:
            color_mid = colors[mid_mask].mean(axis=0)
            color_mid_hex = _bgr_to_hex(np.clip(color_mid, 0, 255).astype(np.uint8))

        nonlocal grad_id
        gid = f"g{grad_id}"
        grad_id += 1
        return GradientDef(
            id=gid,
            x1=np.clip(x1, 0, w), y1=np.clip(y1, 0, h),
            x2=np.clip(x2, 0, w), y2=np.clip(y2, 0, h),
            color_start=_bgr_to_hex(np.clip(color_start, 0, 255).astype(np.uint8)),
            color_end=_bgr_to_hex(np.clip(color_end, 0, 255).astype(np.uint8)),
            color_mid=color_mid_hex,
        )

    # First pass: fit gradients across adjacent low-contrast cluster pairs.
    centers_u8 = np.clip(centers_f, 0, 255).astype(np.uint8).reshape(-1, 1, 3)
    centers_hsv = cv2.cvtColor(centers_u8, cv2.COLOR_BGR2HSV).reshape(-1, 3)

    l_left = labels[:, :-1].ravel()
    l_right = labels[:, 1:].ravel()
    h_mask = l_left != l_right
    c_left = src_f[:, :-1].reshape(-1, 3)[h_mask]
    c_right = src_f[:, 1:].reshape(-1, 3)[h_mask]
    h_diff = np.sqrt(np.sum((c_left - c_right) ** 2, axis=1))

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
    if len(all_k1) > 0:
        pair_keys = all_k1.astype(np.int64) * K + all_k2
        max_key = int(pair_keys.max()) + 1
        diff_sums = np.bincount(pair_keys, weights=all_diffs, minlength=max_key)
        diff_counts = np.bincount(pair_keys, minlength=max_key)
        nonzero = diff_counts > 0
        mean_diffs = np.zeros(max_key, dtype=np.float64)
        mean_diffs[nonzero] = diff_sums[nonzero] / diff_counts[nonzero]

        min_boundary = int(h * w * 0.0025)
        candidate_keys = np.where((diff_counts >= min_boundary) & (mean_diffs < 18.0))[0]
        used_clusters: set[int] = set()
        for key in candidate_keys[np.argsort(mean_diffs[candidate_keys])]:
            _diag_counts["pair_candidates"] += 1
            k1, k2 = divmod(int(key), K)
            if k1 == bg_cluster or k2 == bg_cluster:
                continue
            if k1 in used_clusters or k2 in used_clusters:
                continue
            sat1, sat2 = int(centers_hsv[k1, 1]), int(centers_hsv[k2, 1])
            hue1, hue2 = int(centers_hsv[k1, 0]), int(centers_hsv[k2, 0])
            if sat1 > 20 and sat2 > 20:
                hue_diff = abs(hue1 - hue2)
                hue_diff = min(hue_diff, 180 - hue_diff)
                if hue_diff > 15:
                    _diag_counts["reject_pair_hue"] += 1
                    continue
            pair_mask = (labels == k1) | (labels == k2)
            fit_mask = pair_mask
            _cc_count, _cc_labels = cv2.connectedComponents(pair_mask.astype(np.uint8))
            if _cc_count > 2:
                _cc_sizes = np.bincount(_cc_labels.ravel())[1:]
                if len(_cc_sizes) == 0 or int(_cc_sizes.max()) < int(np.count_nonzero(pair_mask) * 0.85):
                    _diag_counts["reject_pair_cc"] += 1
                    continue
                _largest_idx = int(np.argmax(_cc_sizes)) + 1
                fit_mask = _cc_labels == _largest_idx
            gd = _fit_gradient_region(fit_mask, seed=k1 * K + k2, diag_label=f"pair:{k1},{k2}")
            if gd is None:
                continue
            gradient_defs.append(gd)
            ys_fit, xs_fit = np.where(fit_mask)
            if len(xs_fit) == 0:
                gradient_defs.pop()
                _diag_counts["reject_pair_cc"] += 1
                continue
            _fill_ref = f"url(#{gd.id})"
            _bbox = (
                float(xs_fit.min()),
                float(ys_fit.min()),
                float(xs_fit.max()),
                float(ys_fit.max()),
            )
            single_gradient_regions.setdefault(k1, []).append(
                GradientRegionAssignment(fill_ref=_fill_ref, bbox=_bbox, mask=fit_mask.copy())
            )
            single_gradient_regions.setdefault(k2, []).append(
                GradientRegionAssignment(fill_ref=_fill_ref, bbox=_bbox, mask=fit_mask.copy())
            )
            gradient_fill_map[k1] = _fill_ref
            gradient_fill_map[k2] = _fill_ref
            _diag_counts["pair_accept"] += 1
            _diag_accept_regions.append(
                f"pair:{k1},{k2}:{gd.kind}:bbox=({_bbox[0]:.0f},{_bbox[1]:.0f})-({_bbox[2]:.0f},{_bbox[3]:.0f})"
            )
            used_clusters.add(k1)
            used_clusters.add(k2)

    for k in range(K):
        if k == bg_cluster:
            continue
        if k in gradient_fill_map:
            continue
        _diag_counts["single_candidates"] += 1
        cluster_mask = labels == k
        fit_masks: list[np.ndarray] = [cluster_mask]
        _cc_count, _cc_labels = cv2.connectedComponents(cluster_mask.astype(np.uint8))
        _hue_k = int(centers_hsv[k, 0])
        _sat_k = int(centers_hsv[k, 1])
        _warm_mid_cluster = 5 <= _hue_k <= 25 and 25 <= _sat_k <= 140
        if _cc_count > 2:
            _cc_sizes = np.bincount(_cc_labels.ravel())[1:]
            if len(_cc_sizes) == 0:
                _diag_counts["reject_single_cc"] += 1
                continue
            _cluster_px = int(np.count_nonzero(cluster_mask))
            _largest_idx = int(np.argmax(_cc_sizes)) + 1
            _largest_size = int(_cc_sizes[_largest_idx - 1])
            if _largest_size < int(_cluster_px * 0.12):
                _diag_counts["reject_single_cc"] += 1
                continue
            fit_masks = [_cc_labels == _largest_idx]
            if _warm_mid_cluster:
                _candidate_components: list[tuple[float, int]] = []
                for _rank_idx in np.argsort(_cc_sizes)[::-1][1:4]:
                    _component_size = int(_cc_sizes[_rank_idx])
                    if _component_size < int(_cluster_px * 0.03):
                        continue
                    _component_mask = _cc_labels == (_rank_idx + 1)
                    _component_vals = src_gray_blur[_component_mask]
                    if _component_vals.size == 0:
                        continue
                    _ys_c, _xs_c = np.where(_component_mask)
                    _luma_span = float(np.percentile(_component_vals, 95) - np.percentile(_component_vals, 5))
                    _candidate_score = _luma_span + 30.0 * (_component_size / max(_cluster_px, 1))
                    _candidate_components.append((_candidate_score, _rank_idx + 1))
                    if _diag and len(_diag_component_candidates) < 12:
                        _diag_component_candidates.append(
                            f"k={k}:cc={_rank_idx + 1}:score={_candidate_score:.1f}:px={_component_size}:bbox=({int(_xs_c.min())},{int(_ys_c.min())})-({int(_xs_c.max())},{int(_ys_c.max())})"
                        )
                for _, _component_label in sorted(_candidate_components, reverse=True)[:2]:
                    fit_masks.append(_cc_labels == _component_label)
        _accepted_for_cluster = 0
        for _component_idx, fit_mask in enumerate(fit_masks):
            _min_region_override = 0.6 if (_warm_mid_cluster and _component_idx > 0) else None
            gd = _fit_gradient_region(
                fit_mask,
                seed=k * 10 + _component_idx,
                min_region_pct_override=_min_region_override,
                diag_label=f"single:{k}:comp={_component_idx}",
            )
            if gd is None:
                continue
            gradient_defs.append(gd)
            ys_fit, xs_fit = np.where(fit_mask)
            if len(xs_fit) == 0:
                _diag_counts["reject_single_cc"] += 1
                gradient_defs.pop()
                continue
            single_gradient_regions.setdefault(k, []).append(GradientRegionAssignment(
                fill_ref=f"url(#{gd.id})",
                bbox=(
                    float(xs_fit.min()),
                    float(ys_fit.min()),
                    float(xs_fit.max()),
                    float(ys_fit.max()),
                ),
                mask=fit_mask.copy(),
            ))
            _diag_counts["single_accept"] += 1
            _diag_accept_regions.append(
                f"single:{k}:comp={_component_idx}:{gd.kind}:bbox=({float(xs_fit.min()):.0f},{float(ys_fit.min()):.0f})-({float(xs_fit.max()):.0f},{float(ys_fit.max()):.0f}) hsv=({_hue_k},{_sat_k},{int(centers_hsv[k, 2])})"
            )
            _accepted_for_cluster += 1
            if _accepted_for_cluster >= 2:
                break

    if _diag:
        print(f"[GRAD] candidates pair={_diag_counts['pair_candidates']} single={_diag_counts['single_candidates']} accept pair={_diag_counts['pair_accept']} single={_diag_counts['single_accept']}")
        print(
            "[GRAD] rejects "
            f"small={_diag_counts['reject_small']} spread={_diag_counts['reject_spread']} "
            f"corr={_diag_counts['reject_corr']} short={_diag_counts['reject_short']} "
            f"endpoints={_diag_counts['reject_endpoints']} endpoint_dist={_diag_counts['reject_endpoint_dist']} "
            f"texture={_diag_counts['reject_texture']} "
            f"pair_hue={_diag_counts['reject_pair_hue']} pair_cc={_diag_counts['reject_pair_cc']} "
            f"single_cc={_diag_counts['reject_single_cc']} cov={_diag_counts['reject_cov']}"
        )
        if _diag_corr_samples:
            _diag_corr_samples.sort(key=lambda item: abs(item[2]), reverse=True)
            print("[GRAD] corr samples " + ", ".join(
                f"seed={seed}:px={pix}:corr={corr:.3f}"
                for seed, pix, corr in _diag_corr_samples[:8]
            ))
        if _diag_accept_regions:
            print("[GRAD] accepted " + " | ".join(_diag_accept_regions))
        if _diag_component_candidates:
            print("[GRAD] warm components " + " | ".join(_diag_component_candidates))
        if _diag_reject_regions:
            print("[GRAD] rejected regions " + " | ".join(_diag_reject_regions))
        if _diag_radial_regions:
            print("[GRAD] radial regions " + " | ".join(_diag_radial_regions))


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

        # Find the closest mergeable pair (adaptive threshold for dark clusters).
        # Weber's law: JND is larger at low luminance, so very dark clusters
        # that look perceptually identical can have larger ΔE in LAB.
        best_dist = float("inf")
        best_i, best_j = -1, -1
        for idx_a in range(len(alive_ids)):
            for idx_b in range(idx_a + 1, len(alive_ids)):
                i, j = alive_ids[idx_a], alive_ids[idx_b]
                if centers_lab is not None:
                    d = float(np.linalg.norm(centers_lab[i] - centers_lab[j]))
                    avg_L = (centers_lab[i][0] + centers_lab[j][0]) / 2.0
                    pair_thresh = lab_threshold * max(1.0, 2.0 - avg_L / 25.0)
                else:
                    d = float(np.linalg.norm(centers_f[i] - centers_f[j]))
                    pair_thresh = threshold
                if d < pair_thresh and d < best_dist:
                    best_dist = d
                    best_i, best_j = i, j

        if best_i < 0:
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


def _subdivide_4point_closed(pts: np.ndarray, corner_threshold_deg: float = 55.0) -> np.ndarray:
    """Corner-preserving 4-point interpolatory subdivision for closed polygons.

    Inserts new midpoints between non-corner vertices using the 4-point
    scheme (Dyn, Levin, Gregory 1987).  Original vertices are preserved
    (interpolatory).  Corners and their immediate neighbours are skipped.

    Provides the Bézier fitter with denser, geometrically optimal input
    in smooth sections, improving curve quality without moving existing
    vertices.
    """
    n = len(pts)
    if n < 5:
        return pts

    # Vectorised corner detection for closed polygon
    incoming = pts - np.roll(pts, 1, axis=0)
    outgoing = np.roll(pts, -1, axis=0) - pts
    dot_ = np.sum(incoming * outgoing, axis=1)
    len_in = np.linalg.norm(incoming, axis=1)
    len_out = np.linalg.norm(outgoing, axis=1)
    valid = (len_in > 1e-10) & (len_out > 1e-10)
    cos_a = np.ones(n)
    cos_a[valid] = np.clip(dot_[valid] / (len_in[valid] * len_out[valid]), -1.0, 1.0)
    angles = np.arccos(cos_a)
    corners = set(np.where(valid & (angles > np.radians(corner_threshold_deg)))[0])

    # Build new polygon with inserted midpoints
    new_pts = []
    for i in range(n):
        new_pts.append(pts[i])

        j = (i + 1) % n
        # Skip near corners — preserve sharp features
        if i in corners or j in corners:
            continue
        im1 = (i - 1) % n
        jp1 = (j + 1) % n
        if im1 in corners or jp1 in corners:
            continue

        # 4-point interpolatory rule: ω = 1/16
        mid = (-pts[im1] + 9.0 * pts[i] + 9.0 * pts[j] - pts[jp1]) / 16.0
        new_pts.append(mid)

    return np.array(new_pts)


def _fit_contour(
    contour: np.ndarray,
    simplify_epsilon: float,
    max_error: float,
    corner_threshold: float,
    line_tolerance: float = 0.15,
) -> str:
    """Simplify contour and fit smooth closed Bézier curves; return SVG path d.

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

    # Corner-preserving 4-point subdivision: insert geometrically optimal
    # midpoints in smooth sections for better Bézier fitting
    if len(simplified) >= 5:
        simplified = _subdivide_4point_closed(simplified, corner_threshold)

    try:
        curve = fit_closed_bezier(
            simplified, max_error=max_error,
            corner_threshold=corner_threshold,
            line_tolerance=line_tolerance,
        )
        # Promote line-segment runs to curves where beneficial
        curve = reduce_nodes(curve, max_error=max_error * 2.0)
        # Artistic merge for dense contours
        if len(curve.segments) > 100:
            curve = merge_segments_artistic(curve, tolerance=max_error * 1.5)
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


def _process_stroke_mask(
    mask_k: np.ndarray,
    scale: int,
    simplify_epsilon: float,
    max_error: float,
    corner_threshold: float,
    line_tolerance: float,
    min_branch_length: int | None = None,
) -> tuple[list[str], list[float]] | None:
    """Skeleton-based stroke reconstruction from a binary mask.

    Returns (paths, widths) or None if no valid strokes found.
    """
    S = scale
    if mask_k.dtype != np.uint8:
        mask_k = mask_k.astype(np.uint8)
    if mask_k.ndim != 2 or np.count_nonzero(mask_k) == 0:
        return None
    h, w = mask_k.shape[:2]

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
    _prune_len = max(3, S) if min_branch_length is None else max(1, int(min_branch_length))
    skel = _prune_skeleton(skel, min_branch_length=_prune_len)

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
    mask_k = (labels == cluster_idx).astype(np.uint8)
    return _process_stroke_mask(
        mask_k,
        scale=scale,
        simplify_epsilon=simplify_epsilon,
        max_error=max_error,
        corner_threshold=corner_threshold,
        line_tolerance=line_tolerance,
    )


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

    # This post-pass is helpful on low-saturation artwork, but it washes out
    # high-saturation photos by dragging clustered fills toward broad median
    # samples. Skip it for strongly chromatic images and keep the vectorizer's
    # own render-center colors instead.
    hsv = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2HSV)
    sat_frac = float(np.count_nonzero(hsv[:, :, 1] > 35)) / max(1, h * w)
    if sat_frac > 0.25:
        return svg_string

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
