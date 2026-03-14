"""Test: Inner/Outer boundary compression approach.

IDEA (from user):
  1. Trace the INNER boundary (where we're 100% sure this is our cluster)
  2. Trace the OUTER boundary (where any presence of our cluster ends)
  3. At each point along the boundary, compress them together based on
     how strong the actual pixel colors match the cluster — finding the
     "true visual edge" that the eye would see.

Instead of picking a single iso-level (like 0.47), we let pixel color
strength determine WHERE between inner and outer each boundary point
should land.  This is Archimedes convergence done properly:
  - Where colors are strong/clear: boundary moves close to inner (tight)
  - Where colors blend gradually: boundary sits between inner and outer
  - Where colors barely differ: boundary moves close to outer (generous)

The key insight: the current fixed iso-contour treats every boundary
point identically.  But a sharp dark stroke against white paper has
a totally different transition profile than two similar gray tones
meeting.  The boundary "hardness" should vary point-by-point.
"""

import sys, os, time
import cv2
import numpy as np
from pathlib import Path
from skimage.measure import find_contours
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import map_coordinates

sys.path.insert(0, str(Path(__file__).parent))
from app.core.multilevel import (
    detect_background, _compute_edge_weight, _merge_close_clusters,
    _bgr_to_hex, _polygon_area, _fit_contour,
)


def render_svg(svg_str, w, h):
    import cairosvg
    from io import BytesIO
    from PIL import Image
    png = cairosvg.svg2png(bytestring=svg_str.encode(), output_width=w, output_height=h)
    return cv2.cvtColor(np.array(Image.open(BytesIO(png)).convert("RGB")), cv2.COLOR_RGB2BGR)


def measure(src, rend, blur_sigma=1.5):
    g1, g2 = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY), cv2.cvtColor(rend, cv2.COLOR_BGR2GRAY)
    if blur_sigma > 0:
        k = int(blur_sigma * 6) | 1
        g1 = cv2.GaussianBlur(g1, (k, k), blur_sigma)
        g2 = cv2.GaussianBlur(g2, (k, k), blur_sigma)
    return ssim(g1, g2)


def measure_mae(src, rend):
    return np.mean(np.abs(src.astype(np.float32) - rend.astype(np.float32)))


# -------------------------------------------------------------------------
# Shared pipeline stages (same as production)
# -------------------------------------------------------------------------

def precompute(image_bgr):
    """Run K-means, merge, compute distance maps, soft fields, edge weight."""
    h, w = image_bgr.shape[:2]
    bg_color, _ = detect_background(image_bgr)
    bg_hex = _bgr_to_hex(bg_color)

    denoised_km = cv2.bilateralFilter(image_bgr, 15, 12, 30)
    denoised_dist = cv2.bilateralFilter(image_bgr, 7, 5, 20)

    pixels = denoised_km.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(pixels, 24, None, criteria, 6, cv2.KMEANS_PP_CENTERS)
    centers, labels = _merge_close_clusters(centers, labels.flatten(), h, w, threshold=60.0)

    K = len(centers)
    centers_u = centers.astype(np.uint8)
    centers_f = centers.astype(np.float32)

    bg_dists = np.array([np.linalg.norm(centers_f[k] - bg_color.astype(np.float32)) for k in range(K)])
    bg_idx = int(np.argmin(bg_dists))
    bg_cluster = bg_idx if bg_dists[bg_idx] < 40.0 else -1

    pixels_3d = denoised_dist.astype(np.float32)
    dist_map = np.empty((h, w, K), dtype=np.float32)
    for k in range(K):
        diff = pixels_3d - centers_f[k]
        dist_map[:, :, k] = np.sqrt(np.sum(diff * diff, axis=2))

    grays = np.array([int(cv2.cvtColor(c.reshape(1, 1, 3), cv2.COLOR_BGR2GRAY)[0, 0]) for c in centers_u])
    order = np.argsort(-grays)

    edge_weight = _compute_edge_weight(image_bgr)

    soft_fields = {}
    for ci in range(K):
        d_k = dist_map[:, :, ci]
        omask = np.ones(K, dtype=bool); omask[ci] = False
        d_other = np.min(dist_map[:, :, omask], axis=2)
        denom = np.maximum(d_k + d_other, 1e-10)
        soft_fields[ci] = d_other / denom

    return dict(
        h=h, w=w, K=K, bg_hex=bg_hex, bg_cluster=bg_cluster,
        centers_u=centers_u, centers_f=centers_f, order=order,
        edge_weight=edge_weight, soft_fields=soft_fields, dist_map=dist_map,
        denoised_dist=denoised_dist,
    )


# -------------------------------------------------------------------------
# APPROACH A: Current production (dual iso-contour with fixed halo)
# -------------------------------------------------------------------------

def make_svg_production(pre, scale=4):
    """Exact replica of production pipeline."""
    h, w, K = pre['h'], pre['w'], pre['K']
    S = scale
    ew_up = cv2.resize(pre['edge_weight'], (w * S, h * S), interpolation=cv2.INTER_LINEAR)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {w} {h}" width="{w}" height="{h}">',
        f'<rect width="{w}" height="{h}" fill="{pre["bg_hex"]}"/>',
    ]
    total_nodes = 0

    for ci in pre['order']:
        if ci == pre['bg_cluster']:
            continue
        color = _bgr_to_hex(pre['centers_u'][ci])
        soft_up = cv2.resize(pre['soft_fields'][ci], (w * S, h * S),
                             interpolation=cv2.INTER_LINEAR)
        sc = cv2.GaussianBlur(soft_up, (0, 0), sigmaX=0.6 * S)
        ss = cv2.GaussianBlur(soft_up, (0, 0), sigmaX=1.5 * S)
        soft = ew_up * sc + (1 - ew_up) * ss

        for iso, op in [(0.20, 0.50), (0.47, 1.00)]:
            contours = find_contours(soft, iso)
            path_parts = []
            for c in contours:
                if len(c) < 4: continue
                xy = c[:, ::-1].astype(np.float64) / S
                if abs(_polygon_area(xy)) < 15: continue
                d = _fit_contour(xy, 0.08, 0.15, 60.0)
                if d:
                    path_parts.append(d)
                    total_nodes += d.count("C") + d.count("L") + d.count("M")
            if path_parts:
                d = " ".join(path_parts)
                if op < 1.0:
                    parts.append(f'<path d="{d}" fill="{color}" fill-rule="evenodd" opacity="{op:.2f}"/>')
                else:
                    parts.append(f'<path d="{d}" fill="{color}" fill-rule="evenodd"/>')

    parts.append("</svg>")
    return "\n".join(parts), total_nodes


# -------------------------------------------------------------------------
# APPROACH B: Inner/Outer boundary compression
# -------------------------------------------------------------------------

def _compute_color_strength(soft_field, inner_iso=0.70, outer_iso=0.20):
    """For each pixel in the transition zone (between outer and inner),
    compute how 'hard' the transition is based on the gradient steepness
    of the soft field.

    Returns a map where:
      - High gradient = hard edge (boundary should be crisp, near inner)
      - Low gradient = soft blend (boundary should be gentler, near outer)
    """
    # Gradient magnitude of the soft field tells us how sharp the transition is
    gy = cv2.Sobel(soft_field.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    gx = cv2.Sobel(soft_field.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    grad_mag = np.sqrt(gx * gx + gy * gy)

    # Normalize to [0, 1]
    mx = grad_mag.max()
    if mx > 0:
        grad_mag = grad_mag / mx

    return grad_mag


def _compress_contour(inner_contour, outer_contour, strength_map, field_shape,
                      base_alpha=0.5, strength_power=1.0):
    """Given inner and outer contour point arrays, compress them together
    based on local color strength.

    For each inner point, find the closest outer point, then interpolate:
      final = outer + alpha * (inner - outer)
    where alpha varies based on how strong the edge is at that location.

    Strong edges (high gradient): alpha → 1.0 (snap to inner, tight boundary)
    Weak edges (low gradient):    alpha → base value (stay closer to outer)
    """
    from scipy.spatial import cKDTree

    if len(inner_contour) < 4 or len(outer_contour) < 4:
        return None

    # Build KD-tree of outer contour points for fast nearest-neighbor lookup
    tree = cKDTree(outer_contour)

    compressed = np.empty_like(inner_contour)
    h, w = field_shape

    for i, pt_inner in enumerate(inner_contour):
        # Find nearest outer point
        dist, idx = tree.query(pt_inner, k=1)
        pt_outer = outer_contour[idx]

        # Sample the strength map at the midpoint between inner and outer
        mid = (pt_inner + pt_outer) / 2.0
        # Clamp to valid image coordinates (row, col format for map_coordinates)
        row = np.clip(mid[1], 0, h - 1)
        col = np.clip(mid[0], 0, w - 1)

        # Bilinear interpolation of strength
        strength = float(map_coordinates(strength_map, [[row], [col]], order=1, mode='nearest')[0])

        # Alpha: blend between inner and outer
        # High strength → alpha close to 1 (use inner)
        # Low strength → alpha stays at base (use outer more)
        alpha = base_alpha + (1.0 - base_alpha) * (strength ** strength_power)

        compressed[i] = pt_outer + alpha * (pt_inner - pt_outer)

    return compressed


def _match_and_compress_contours(inner_contours, outer_contours, strength_map,
                                 field_shape, base_alpha, strength_power):
    """Match inner contours to outer contours by spatial overlap, then
    compress each pair. Unmatched outers are returned as-is.
    """
    from scipy.spatial import cKDTree

    if not inner_contours or not outer_contours:
        # If no inner contours, return outer; if no outer, return inner
        return outer_contours if outer_contours else inner_contours

    # For each inner contour, find the best-matching outer contour
    # by checking which outer contour's centroid is closest
    inner_centroids = [c.mean(axis=0) for c in inner_contours]
    outer_centroids = [c.mean(axis=0) for c in outer_contours]

    outer_tree = cKDTree(np.array(outer_centroids))

    used_outers = set()
    result = []

    for i_idx, i_contour in enumerate(inner_contours):
        # Find closest outer contour
        dist, o_idx = outer_tree.query(inner_centroids[i_idx], k=1)

        # Compress this pair
        compressed = _compress_contour(
            i_contour, outer_contours[o_idx], strength_map, field_shape,
            base_alpha=base_alpha, strength_power=strength_power,
        )
        if compressed is not None:
            result.append(compressed)
            used_outers.add(o_idx)
        else:
            result.append(i_contour)
            used_outers.add(o_idx)

    # Add unmatched outer contours as-is (they have no inner counterpart,
    # meaning they represent very faint/thin features)
    for o_idx, o_contour in enumerate(outer_contours):
        if o_idx not in used_outers:
            result.append(o_contour)

    return result


def make_svg_boundary_compress(pre, scale=4, inner_iso=0.65, outer_iso=0.20,
                                base_alpha=0.55, strength_power=1.5,
                                halo_iso=None, halo_opacity=0.40):
    """Inner/Outer boundary compression approach.

    1. Extract inner contour at high iso (firmly inside the cluster)
    2. Extract outer contour at low iso (where cluster presence begins)
    3. Sample the soft-field gradient (color strength) along the boundary
    4. Compress inner toward outer (or outer toward inner) based on
       local color strength → adaptive boundary placement
    5. Optionally add a faint halo at the outer boundary for AA
    """
    h, w, K = pre['h'], pre['w'], pre['K']
    S = scale
    ew_up = cv2.resize(pre['edge_weight'], (w * S, h * S), interpolation=cv2.INTER_LINEAR)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {w} {h}" width="{w}" height="{h}">',
        f'<rect width="{w}" height="{h}" fill="{pre["bg_hex"]}"/>',
    ]
    total_nodes = 0

    for ci in pre['order']:
        if ci == pre['bg_cluster']:
            continue
        color = _bgr_to_hex(pre['centers_u'][ci])

        # Build the smoothed soft field (same as production)
        soft_up = cv2.resize(pre['soft_fields'][ci], (w * S, h * S),
                             interpolation=cv2.INTER_LINEAR)
        sc = cv2.GaussianBlur(soft_up, (0, 0), sigmaX=0.6 * S)
        ss = cv2.GaussianBlur(soft_up, (0, 0), sigmaX=1.5 * S)
        soft = ew_up * sc + (1 - ew_up) * ss

        # Compute edge strength map from the soft field gradient
        strength_map = _compute_color_strength(soft, inner_iso, outer_iso)

        # Extract inner contours (firmly inside the cluster)
        raw_inner = find_contours(soft, inner_iso)
        inner_contours = []
        for c in raw_inner:
            if len(c) < 4: continue
            xy = c[:, ::-1].astype(np.float64)  # keep in upscaled coords
            area = abs(_polygon_area(xy / S))
            if area < 15: continue
            inner_contours.append(xy)

        # Extract outer contours (where any cluster presence begins)
        raw_outer = find_contours(soft, outer_iso)
        outer_contours = []
        for c in raw_outer:
            if len(c) < 4: continue
            xy = c[:, ::-1].astype(np.float64)
            area = abs(_polygon_area(xy / S))
            if area < 15: continue
            outer_contours.append(xy)

        # --- Compress inner and outer toward the true visual edge ---
        compressed = _match_and_compress_contours(
            inner_contours, outer_contours, strength_map,
            (h * S, w * S), base_alpha, strength_power,
        )

        # Optional: add a faint halo at the outer boundary for AA
        if halo_iso is not None and halo_opacity > 0:
            halo_contours = find_contours(soft, halo_iso)
            halo_parts = []
            for c in halo_contours:
                if len(c) < 4: continue
                xy = c[:, ::-1].astype(np.float64) / S
                if abs(_polygon_area(xy)) < 15: continue
                d = _fit_contour(xy, 0.08, 0.15, 60.0)
                if d:
                    halo_parts.append(d)
                    total_nodes += d.count("C") + d.count("L") + d.count("M")
            if halo_parts:
                d = " ".join(halo_parts)
                parts.append(f'<path d="{d}" fill="{color}" fill-rule="evenodd" opacity="{halo_opacity:.2f}"/>')

        # Fit the compressed contours and add as core (full opacity)
        core_parts = []
        for contour in compressed:
            xy = contour / S  # scale back to image coordinates
            if abs(_polygon_area(xy)) < 15: continue
            d = _fit_contour(xy, 0.08, 0.15, 60.0)
            if d:
                core_parts.append(d)
                total_nodes += d.count("C") + d.count("L") + d.count("M")

        if core_parts:
            d = " ".join(core_parts)
            parts.append(f'<path d="{d}" fill="{color}" fill-rule="evenodd"/>')

    parts.append("</svg>")
    return "\n".join(parts), total_nodes


# -------------------------------------------------------------------------
# APPROACH C: Per-pixel iso-level field (fully adaptive contour)
# -------------------------------------------------------------------------

def make_svg_adaptive_iso(pre, scale=4, inner_iso=0.65, outer_iso=0.20,
                          base_iso=0.47, halo_iso=0.20, halo_opacity=0.50):
    """Instead of compressing contour points, build an adaptive iso-level
    FIELD where each pixel gets its own iso threshold based on local
    color contrast strength.

    Where edges are hard: iso → closer to inner_iso (tight boundary)
    Where edges are soft: iso → closer to outer_iso (generous boundary)

    Then extract contours from: soft_field - adaptive_iso_field > 0
    This is equivalent to marching squares on a locally-varying threshold.
    """
    h, w, K = pre['h'], pre['w'], pre['K']
    S = scale
    ew_up = cv2.resize(pre['edge_weight'], (w * S, h * S), interpolation=cv2.INTER_LINEAR)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {w} {h}" width="{w}" height="{h}">',
        f'<rect width="{w}" height="{h}" fill="{pre["bg_hex"]}"/>',
    ]
    total_nodes = 0

    for ci in pre['order']:
        if ci == pre['bg_cluster']:
            continue
        color = _bgr_to_hex(pre['centers_u'][ci])

        soft_up = cv2.resize(pre['soft_fields'][ci], (w * S, h * S),
                             interpolation=cv2.INTER_LINEAR)
        sc = cv2.GaussianBlur(soft_up, (0, 0), sigmaX=0.6 * S)
        ss = cv2.GaussianBlur(soft_up, (0, 0), sigmaX=1.5 * S)
        soft = ew_up * sc + (1 - ew_up) * ss

        # Gradient magnitude of soft field = edge strength
        gx = cv2.Sobel(soft.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(soft.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(gx * gx + gy * gy)
        mx = grad.max()
        if mx > 0:
            grad_norm = grad / mx
        else:
            grad_norm = grad

        # Build per-pixel iso-level field:
        # Strong gradient → iso shifts toward inner (tighter)
        # Weak gradient → iso stays at base or shifts toward outer (looser)
        adaptive_iso = outer_iso + grad_norm * (inner_iso - outer_iso)

        # Smooth the iso field slightly to avoid jagged contours
        adaptive_iso = cv2.GaussianBlur(adaptive_iso, (0, 0), sigmaX=1.0 * S)

        # The "effective" field: shift soft field so the adaptive iso becomes 0
        # Then extract contour at level 0 = marching squares where
        # soft == adaptive_iso at each pixel
        shifted = soft - adaptive_iso

        # Optional: add halo at outer boundary
        if halo_iso is not None and halo_opacity > 0:
            halo_contours = find_contours(soft, halo_iso)
            halo_path_parts = []
            for c in halo_contours:
                if len(c) < 4: continue
                xy = c[:, ::-1].astype(np.float64) / S
                if abs(_polygon_area(xy)) < 15: continue
                d = _fit_contour(xy, 0.08, 0.15, 60.0)
                if d:
                    halo_path_parts.append(d)
                    total_nodes += d.count("C") + d.count("L") + d.count("M")
            if halo_path_parts:
                d = " ".join(halo_path_parts)
                parts.append(f'<path d="{d}" fill="{color}" fill-rule="evenodd" opacity="{halo_opacity:.2f}"/>')

        # Extract contour at level 0 (where soft == local adaptive iso)
        contours = find_contours(shifted, 0.0)
        path_parts = []
        for c in contours:
            if len(c) < 4: continue
            xy = c[:, ::-1].astype(np.float64) / S
            if abs(_polygon_area(xy)) < 15: continue
            d = _fit_contour(xy, 0.08, 0.15, 60.0)
            if d:
                path_parts.append(d)
                total_nodes += d.count("C") + d.count("L") + d.count("M")
        if path_parts:
            d = " ".join(path_parts)
            parts.append(f'<path d="{d}" fill="{color}" fill-rule="evenodd"/>')

    parts.append("</svg>")
    return "\n".join(parts), total_nodes


# -------------------------------------------------------------------------
# Main: run all approaches and compare
# -------------------------------------------------------------------------

if __name__ == "__main__":
    ref = cv2.imread("/Users/alexwaldmann/Desktop/SVG-gen/Ref.png")
    crop = ref[50:460, 486:1050]
    mahal = cv2.imread("/tmp/mahal_right.png")

    images = {"crop": crop}
    if mahal is not None:
        images["mahal"] = mahal

    configs = [
        # --- Baseline ---
        ("A  Production (dual iso)",
         lambda pre: make_svg_production(pre)),

        # --- Boundary compression variants ---
        ("B1 Compress a=0.55 p=1.5 +halo",
         lambda pre: make_svg_boundary_compress(pre, inner_iso=0.65, outer_iso=0.20,
                                                 base_alpha=0.55, strength_power=1.5,
                                                 halo_iso=0.15, halo_opacity=0.40)),
        ("B2 Compress a=0.50 p=1.0 +halo",
         lambda pre: make_svg_boundary_compress(pre, inner_iso=0.65, outer_iso=0.20,
                                                 base_alpha=0.50, strength_power=1.0,
                                                 halo_iso=0.15, halo_opacity=0.40)),
        ("B3 Compress a=0.60 p=2.0 +halo",
         lambda pre: make_svg_boundary_compress(pre, inner_iso=0.70, outer_iso=0.15,
                                                 base_alpha=0.60, strength_power=2.0,
                                                 halo_iso=0.12, halo_opacity=0.35)),
        ("B4 Compress a=0.45 p=1.0 nohalo",
         lambda pre: make_svg_boundary_compress(pre, inner_iso=0.65, outer_iso=0.25,
                                                 base_alpha=0.45, strength_power=1.0,
                                                 halo_iso=None, halo_opacity=0)),
        ("B5 Compress a=0.55 p=1.5 nohalo",
         lambda pre: make_svg_boundary_compress(pre, inner_iso=0.65, outer_iso=0.20,
                                                 base_alpha=0.55, strength_power=1.5,
                                                 halo_iso=None, halo_opacity=0)),

        # --- Adaptive iso-field variants ---
        ("C1 AdaptIso 0.25-0.55 +halo",
         lambda pre: make_svg_adaptive_iso(pre, inner_iso=0.55, outer_iso=0.25,
                                            halo_iso=0.18, halo_opacity=0.45)),
        ("C2 AdaptIso 0.20-0.65 +halo",
         lambda pre: make_svg_adaptive_iso(pre, inner_iso=0.65, outer_iso=0.20,
                                            halo_iso=0.15, halo_opacity=0.40)),
        ("C3 AdaptIso 0.30-0.60 +halo",
         lambda pre: make_svg_adaptive_iso(pre, inner_iso=0.60, outer_iso=0.30,
                                            halo_iso=0.20, halo_opacity=0.45)),
        ("C4 AdaptIso 0.20-0.65 nohalo",
         lambda pre: make_svg_adaptive_iso(pre, inner_iso=0.65, outer_iso=0.20,
                                            halo_iso=None, halo_opacity=0)),
        ("C5 AdaptIso 0.25-0.55 nohalo",
         lambda pre: make_svg_adaptive_iso(pre, inner_iso=0.55, outer_iso=0.25,
                                            halo_iso=None, halo_opacity=0)),
    ]

    # Header
    print(f"\n{'Config':<35s}", end="")
    for name in images:
        print(f"  {name:>6s}_blur {name:>6s}_raw {name:>6s}_MAE", end="")
    print("  nodes    KB    time")
    print("-" * (35 + 33 * len(images) + 24))

    for desc, fn in configs:
        line = f"  {desc:<33s}"
        all_blur, all_raw, all_mae = [], [], []
        t0 = time.time()

        for name, img in images.items():
            pre = precompute(img)
            svg, nodes = fn(pre)
            rend = render_svg(svg, pre['w'], pre['h'])
            sb = measure(img, rend, blur_sigma=1.5)
            sr = measure(img, rend, blur_sigma=0)
            mae = measure_mae(img, rend)
            all_blur.append(sb)
            all_raw.append(sr)
            all_mae.append(mae)
            line += f"  {sb:.4f}  {sr:.4f}  {mae:>5.2f}"

        elapsed = time.time() - t0
        kb = len(svg.encode()) / 1024
        line += f"  {nodes:>5d}  {kb:>5.0f}  {elapsed:>5.1f}s"
        print(line)

    print()
    print("Legend:")
    print("  A  = Current production (dual iso-contour 0.20@50% + 0.47@100%)")
    print("  B  = Inner/Outer boundary compression (trace both, compress by color strength)")
    print("  C  = Adaptive iso-field (per-pixel iso threshold from gradient strength)")
    print("  blur = blurred SSIM (sigma=1.5), raw = raw SSIM, MAE = mean absolute error")
    print("  +halo = includes faint outer halo for AA, nohalo = core path only")
