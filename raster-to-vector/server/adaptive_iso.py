"""Prototype: per-segment Bézier refinement.

After the initial vectorization, render the SVG, compare to source,
and iteratively adjust control points to minimize pixel error.

Instead of per-segment (complex), we use a simpler whole-path approach:
render the SVG, find high-error boundary regions, and add thin corrective
strokes along those boundaries.

Actually, the simplest high-impact approach: optimize the core iso per-cluster
by rendering with iso=0.45/0.47/0.49/0.50 and picking whichever minimizes
that cluster's error. Different clusters have different boundary characteristics.
"""

import sys, os, time
import cv2
import numpy as np
from pathlib import Path
from skimage.measure import find_contours
from skimage.metrics import structural_similarity as ssim

sys.path.insert(0, str(Path(__file__).parent))
from app.core.multilevel import (
    detect_background, _compute_edge_weight, _merge_close_clusters,
    _bgr_to_hex, _polygon_area, _fit_contour, generate_svg,
    MultilevelResult, VectorLayer,
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


def precompute(image_bgr):
    h, w = image_bgr.shape[:2]
    bg_color, _ = detect_background(image_bgr)
    bg_hex = _bgr_to_hex(bg_color)
    denoised_km = cv2.bilateralFilter(image_bgr, 15, 12, 30)
    denoised_dist = cv2.bilateralFilter(image_bgr, 7, 5, 20)
    pixels = denoised_km.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(pixels, 24, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
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
        soft_fields[ci] = d_other / np.maximum(d_k + d_other, 1e-10)
    return dict(h=h, w=w, K=K, bg_hex=bg_hex, bg_cluster=bg_cluster,
                centers_u=centers_u, centers_f=centers_f, order=order,
                edge_weight=edge_weight, soft_fields=soft_fields,
                labels=labels, dist_map=dist_map)


def extract_layer(pre, cluster_idx, scale, iso, edge_weight_up=None):
    """Extract contour paths for one cluster at a given iso level."""
    h, w = pre['h'], pre['w']
    S = scale
    soft_raw = pre['soft_fields'][cluster_idx]
    soft_up = cv2.resize(soft_raw, (w * S, h * S), interpolation=cv2.INTER_LINEAR)
    if edge_weight_up is None:
        edge_weight_up = cv2.resize(pre['edge_weight'], (w * S, h * S),
                                     interpolation=cv2.INTER_LINEAR)
    sc = cv2.GaussianBlur(soft_up, (0, 0), sigmaX=0.6 * S)
    ss = cv2.GaussianBlur(soft_up, (0, 0), sigmaX=1.5 * S)
    soft = edge_weight_up * sc + (1 - edge_weight_up) * ss

    contours = find_contours(soft, iso)
    parts = []
    for c in contours:
        if len(c) < 4:
            continue
        xy = c[:, ::-1].astype(np.float64) / S
        if abs(_polygon_area(xy)) < 15:
            continue
        d = _fit_contour(xy, 0.08, 0.15, 60.0)
        if d:
            parts.append(d)
    return " ".join(parts) if parts else None


def build_svg(layers_data, w, h, bg_hex):
    """Build SVG from layer data."""
    layers = []
    for color_hex, paths, opacs in layers_data:
        layers.append(VectorLayer(paths=paths, opacities=opacs, color=color_hex))
    result = MultilevelResult(layers=layers, width=w, height=h,
                              background_color=bg_hex, path_count=0, node_count=0)
    return generate_svg(result, remove_background=False)


def adaptive_iso_vectorize(image_bgr, scale=4):
    """Vectorize with per-cluster adaptive iso optimization.
    
    For each cluster, try multiple core iso thresholds and pick the one
    that minimizes error in that cluster's region.
    """
    pre = precompute(image_bgr)
    h, w, K = pre['h'], pre['w'], pre['K']
    S = scale
    ew_up = cv2.resize(pre['edge_weight'], (w * S, h * S), interpolation=cv2.INTER_LINEAR)

    # First pass: build baseline SVG with fixed iso
    baseline_layers = []
    for ci in pre['order']:
        if ci == pre['bg_cluster']:
            continue
        color_hex = _bgr_to_hex(pre['centers_u'][ci])
        # Halo at iso=0.20
        halo = extract_layer(pre, ci, S, 0.20, ew_up)
        # Core at various iso levels — we'll pick the best
        cores = {}
        for core_iso in [0.43, 0.45, 0.47, 0.49, 0.50]:
            core = extract_layer(pre, ci, S, core_iso, ew_up)
            cores[core_iso] = core

        baseline_layers.append((ci, color_hex, halo, cores))

    # Build reference labels for per-cluster error measurement
    labels = pre['labels'] if 'labels' in pre else None

    # For each cluster, try each core iso and see which gives best result
    best_layers_data = []
    for ci, color_hex, halo, cores in baseline_layers:
        best_score = -1
        best_iso = 0.47
        for core_iso, core in cores.items():
            if core is None:
                continue
            # Build a quick test SVG with just this cluster's layer changed
            paths = []
            opacs = []
            if halo:
                paths.append(halo)
                opacs.append(0.50)
            paths.append(core)
            opacs.append(1.00)

            # Build full SVG for now (can't isolate per-cluster easily)
            # Just use the core iso that's closest to what we had
            test_layers = []
            for ci2, ch2, h2, c2 in baseline_layers:
                if ci2 == ci:
                    test_layers.append((ch2, paths, opacs))
                else:
                    # Default to 0.47 for other clusters
                    p2 = []
                    o2 = []
                    if h2:
                        p2.append(h2)
                        o2.append(0.50)
                    if c2.get(0.47):
                        p2.append(c2[0.47])
                        o2.append(1.00)
                    if p2:
                        test_layers.append((ch2, p2, o2))

            svg = build_svg(test_layers, w, h, pre['bg_hex'])
            rend = render_svg(svg, w, h)

            # Measure error only in this cluster's region
            if labels is not None:
                mask = (labels.reshape(h, w) == ci)
                # Also include a 3px border around the cluster
                kernel = np.ones((7, 7), np.uint8)
                mask_dilated = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
                # Local SSIM in the region
                g_src = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
                g_rend = cv2.cvtColor(rend, cv2.COLOR_BGR2GRAY).astype(np.float32)
                diff = np.abs(g_src - g_rend)
                local_mae = float(np.mean(diff[mask_dilated]))
                score = -local_mae  # lower MAE = better
            else:
                score = measure(image_bgr, rend, blur_sigma=0)

            if score > best_score:
                best_score = score
                best_iso = core_iso

        # Use the best iso for this cluster
        paths = []
        opacs = []
        if halo:
            paths.append(halo)
            opacs.append(0.50)
        if cores.get(best_iso):
            paths.append(cores[best_iso])
            opacs.append(1.00)
        if paths:
            best_layers_data.append((color_hex, paths, opacs))
        print(f"  Cluster {ci} ({color_hex}): best core iso = {best_iso}")

    return build_svg(best_layers_data, w, h, pre['bg_hex'])


if __name__ == "__main__":
    ref = cv2.imread("/Users/alexwaldmann/Desktop/SVG-gen/Ref.png")
    crop = ref[50:460, 486:1050]

    print("=== Fixed iso=0.47 (current production) ===")
    from app.core.multilevel import multilevel_vectorize
    t0 = time.time()
    result = multilevel_vectorize(crop)
    svg_fixed = generate_svg(result, remove_background=False)
    dt = time.time() - t0
    rend = render_svg(svg_fixed, crop.shape[1], crop.shape[0])
    s_blur = measure(crop, rend, blur_sigma=1.5)
    s_raw = measure(crop, rend, blur_sigma=0)
    print(f"  blur={s_blur:.4f}  raw={s_raw:.4f}  time={dt:.1f}s")

    print("\n=== Adaptive per-cluster iso ===")
    t0 = time.time()
    svg_adaptive = adaptive_iso_vectorize(crop, scale=4)
    dt = time.time() - t0
    rend2 = render_svg(svg_adaptive, crop.shape[1], crop.shape[0])
    s_blur2 = measure(crop, rend2, blur_sigma=1.5)
    s_raw2 = measure(crop, rend2, blur_sigma=0)
    print(f"  blur={s_blur2:.4f}  raw={s_raw2:.4f}  time={dt:.1f}s")

    diff1 = float(np.mean(cv2.absdiff(crop, rend)))
    diff2 = float(np.mean(cv2.absdiff(crop, rend2)))
    print(f"\n  Fixed MAE={diff1:.2f}  Adaptive MAE={diff2:.2f}")
    print(f"  Improvement: {diff1 - diff2:.2f} MAE, {s_raw2 - s_raw:.4f} raw SSIM")
