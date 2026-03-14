"""Test graduated multi-level halo + higher superresolution.

Hypothesis: Instead of 2 iso levels (0.20@50%, 0.47@100%), 
use 3-5 graduated levels for smoother transition at zoom.
Also test 8x superresolution for tighter contour placement.
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
                centers_u=centers_u, order=order, edge_weight=edge_weight,
                soft_fields=soft_fields)


def make_svg(pre, scale=4, iso_levels=None):
    """General SVG builder with configurable iso levels."""
    if iso_levels is None:
        iso_levels = [(0.20, 0.50), (0.47, 1.00)]
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

        for iso, op in iso_levels:
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


if __name__ == "__main__":
    ref = cv2.imread("/Users/alexwaldmann/Desktop/SVG-gen/Ref.png")
    crop = ref[50:460, 486:1050]

    pre4 = precompute(crop)
    pre8 = None

    configs = [
        # --- Current production (baseline) ---
        ("4x  2-level (prod)",  4, [(0.20, 0.50), (0.47, 1.00)]),
        
        # --- Tighter halos ---
        ("4x  tight halo",     4, [(0.30, 0.40), (0.47, 1.00)]),
        ("4x  tight halo v2",  4, [(0.35, 0.35), (0.47, 1.00)]),
        
        # --- 3-level graduated ---
        ("4x  3-level grad",   4, [(0.15, 0.30), (0.30, 0.60), (0.47, 1.00)]),
        ("4x  3-level grad v2",4, [(0.20, 0.35), (0.35, 0.65), (0.47, 1.00)]),
        ("4x  3-level v3",     4, [(0.25, 0.30), (0.38, 0.55), (0.47, 1.00)]),
        
        # --- 4-level graduated ---
        ("4x  4-level grad",   4, [(0.10, 0.20), (0.20, 0.40), (0.35, 0.70), (0.47, 1.00)]),
        ("4x  4-level v2",     4, [(0.15, 0.25), (0.25, 0.45), (0.38, 0.70), (0.47, 1.00)]),
        
        # --- 5-level (smooth gradient) ---
        ("4x  5-level",        4, [(0.10, 0.15), (0.18, 0.30), (0.27, 0.50), (0.38, 0.75), (0.47, 1.00)]),
        
        # --- Single iso (no halo) ---
        ("4x  core only",      4, [(0.47, 1.00)]),
        
        # --- 8x superresolution ---
        ("8x  2-level (prod)",  8, [(0.20, 0.50), (0.47, 1.00)]),
        ("8x  3-level grad",    8, [(0.20, 0.35), (0.35, 0.65), (0.47, 1.00)]),
        ("8x  tight halo",     8, [(0.30, 0.40), (0.47, 1.00)]),
    ]

    print(f"{'Config':<25s}  blur   raw    nodes   KB")
    print("-" * 62)

    for desc, scale, levels in configs:
        pre = pre4 if scale == 4 else None
        if pre is None:
            pre = precompute(crop)  # re-compute with same settings; scale is post-factor
            if pre8 is None:
                pre8 = pre
            else:
                pre = pre8
        svg, nodes = make_svg(pre, scale=scale, iso_levels=levels)
        rend = render_svg(svg, pre['w'], pre['h'])
        sb = measure(crop, rend, blur_sigma=1.5)
        sr = measure(crop, rend, blur_sigma=0)
        kb = len(svg.encode()) / 1024
        print(f"  {desc:<23s}  {sb:.4f}  {sr:.4f}  {nodes:>6d}  {kb:>6.0f}")
