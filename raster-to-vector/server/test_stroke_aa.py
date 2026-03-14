"""Test: SVG stroke-based anti-aliasing vs opacity halo.

Current approach: dual iso-contour (outer halo at 50% opacity)
  - Looks great blurred, but zoom reveals visible ghost ring
  
New approach: single core contour + SVG stroke for antialiasing
  - The stroke naturally blends the shape edge with the background
  - Looks crisp at any zoom level
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
                centers_u=centers_u, order=order, edge_weight=edge_weight,
                soft_fields=soft_fields)


def make_svg_halo(pre, scale=4):
    """Current approach: dual iso-contour with opacity halo."""
    h, w, K = pre['h'], pre['w'], pre['K']
    S = scale
    ew_up = cv2.resize(pre['edge_weight'], (w * S, h * S), interpolation=cv2.INTER_LINEAR)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {w} {h}" width="{w}" height="{h}">',
        f'<rect width="{w}" height="{h}" fill="{pre["bg_hex"]}"/>',
    ]

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
                if d: path_parts.append(d)
            if path_parts:
                d = " ".join(path_parts)
                if op < 1.0:
                    parts.append(f'<path d="{d}" fill="{color}" fill-rule="evenodd" opacity="{op:.2f}"/>')
                else:
                    parts.append(f'<path d="{d}" fill="{color}" fill-rule="evenodd"/>')

    parts.append("</svg>")
    return "\n".join(parts)


def make_svg_stroke(pre, scale=4, stroke_width=1.0, stroke_opacity=0.5):
    """New approach: single core contour + SVG stroke for AA."""
    h, w, K = pre['h'], pre['w'], pre['K']
    S = scale
    ew_up = cv2.resize(pre['edge_weight'], (w * S, h * S), interpolation=cv2.INTER_LINEAR)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {w} {h}" width="{w}" height="{h}">',
        f'<rect width="{w}" height="{h}" fill="{pre["bg_hex"]}"/>',
    ]

    for ci in pre['order']:
        if ci == pre['bg_cluster']:
            continue
        color = _bgr_to_hex(pre['centers_u'][ci])
        soft_up = cv2.resize(pre['soft_fields'][ci], (w * S, h * S),
                             interpolation=cv2.INTER_LINEAR)
        sc = cv2.GaussianBlur(soft_up, (0, 0), sigmaX=0.6 * S)
        ss = cv2.GaussianBlur(soft_up, (0, 0), sigmaX=1.5 * S)
        soft = ew_up * sc + (1 - ew_up) * ss

        # Single core contour
        contours = find_contours(soft, 0.47)
        path_parts = []
        for c in contours:
            if len(c) < 4: continue
            xy = c[:, ::-1].astype(np.float64) / S
            if abs(_polygon_area(xy)) < 15: continue
            d = _fit_contour(xy, 0.08, 0.15, 60.0)
            if d: path_parts.append(d)
        if path_parts:
            d = " ".join(path_parts)
            # Core fill + stroke for anti-aliasing
            parts.append(
                f'<path d="{d}" fill="{color}" fill-rule="evenodd" '
                f'stroke="{color}" stroke-width="{stroke_width}" '
                f'stroke-opacity="{stroke_opacity}" stroke-linejoin="round"/>'
            )

    parts.append("</svg>")
    return "\n".join(parts)


def make_svg_stroke_expand(pre, scale=4, stroke_width=1.5, stroke_opacity=0.50):
    """Variant: stroke that expands slightly beyond the core for coverage."""
    h, w, K = pre['h'], pre['w'], pre['K']
    S = scale
    ew_up = cv2.resize(pre['edge_weight'], (w * S, h * S), interpolation=cv2.INTER_LINEAR)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {w} {h}" width="{w}" height="{h}">',
        f'<rect width="{w}" height="{h}" fill="{pre["bg_hex"]}"/>',
    ]

    for ci in pre['order']:
        if ci == pre['bg_cluster']:
            continue
        color = _bgr_to_hex(pre['centers_u'][ci])
        soft_up = cv2.resize(pre['soft_fields'][ci], (w * S, h * S),
                             interpolation=cv2.INTER_LINEAR)
        sc = cv2.GaussianBlur(soft_up, (0, 0), sigmaX=0.6 * S)
        ss = cv2.GaussianBlur(soft_up, (0, 0), sigmaX=1.5 * S)
        soft = ew_up * sc + (1 - ew_up) * ss

        # Core at 0.47, stroke extends outward
        contours = find_contours(soft, 0.47)
        path_parts = []
        for c in contours:
            if len(c) < 4: continue
            xy = c[:, ::-1].astype(np.float64) / S
            if abs(_polygon_area(xy)) < 15: continue
            d = _fit_contour(xy, 0.08, 0.15, 60.0)
            if d: path_parts.append(d)
        if path_parts:
            d = " ".join(path_parts)
            parts.append(
                f'<path d="{d}" fill="{color}" fill-rule="evenodd" '
                f'stroke="{color}" stroke-width="{stroke_width}" '
                f'stroke-opacity="{stroke_opacity}" paint-order="stroke fill" '
                f'stroke-linejoin="round"/>'
            )

    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    ref = cv2.imread("/Users/alexwaldmann/Desktop/SVG-gen/Ref.png")
    crop = ref[50:460, 486:1050]
    mahal = cv2.imread("/tmp/mahal_right.png")

    images = {"crop": crop}
    if mahal is not None:
        images["mahal"] = mahal

    configs = [
        ("Halo (current production)", lambda pre: make_svg_halo(pre)),
        ("Stroke w=0.5 op=0.50",     lambda pre: make_svg_stroke(pre, stroke_width=0.5, stroke_opacity=0.50)),
        ("Stroke w=0.8 op=0.50",     lambda pre: make_svg_stroke(pre, stroke_width=0.8, stroke_opacity=0.50)),
        ("Stroke w=1.0 op=0.50",     lambda pre: make_svg_stroke(pre, stroke_width=1.0, stroke_opacity=0.50)),
        ("Stroke w=1.0 op=0.40",     lambda pre: make_svg_stroke(pre, stroke_width=1.0, stroke_opacity=0.40)),
        ("Stroke w=1.0 op=0.30",     lambda pre: make_svg_stroke(pre, stroke_width=1.0, stroke_opacity=0.30)),
        ("Stroke w=1.5 op=0.50",     lambda pre: make_svg_stroke(pre, stroke_width=1.5, stroke_opacity=0.50)),
        ("Expand w=1.0 op=0.50",     lambda pre: make_svg_stroke_expand(pre, stroke_width=1.0, stroke_opacity=0.50)),
        ("Expand w=1.5 op=0.50",     lambda pre: make_svg_stroke_expand(pre, stroke_width=1.5, stroke_opacity=0.50)),
        ("Expand w=1.5 op=0.40",     lambda pre: make_svg_stroke_expand(pre, stroke_width=1.5, stroke_opacity=0.40)),
        ("Expand w=2.0 op=0.35",     lambda pre: make_svg_stroke_expand(pre, stroke_width=2.0, stroke_opacity=0.35)),
        ("Expand w=2.0 op=0.50",     lambda pre: make_svg_stroke_expand(pre, stroke_width=2.0, stroke_opacity=0.50)),
        ("Expand w=2.5 op=0.30",     lambda pre: make_svg_stroke_expand(pre, stroke_width=2.5, stroke_opacity=0.30)),
    ]

    print(f"{'Config':<35s}", end="")
    for name in images:
        print(f"  {name:>6s}_blur  {name:>6s}_raw", end="")
    print("   avg_blur  avg_raw")
    print("-" * (35 + 30 * len(images) + 22))

    for desc, fn in configs:
        all_blur = []
        all_raw = []
        line = f"  {desc:<33s}"
        for name, img in images.items():
            pre = precompute(img)
            svg = fn(pre)
            rend = render_svg(svg, pre['w'], pre['h'])
            sb = measure(img, rend, blur_sigma=1.5)
            sr = measure(img, rend, blur_sigma=0)
            all_blur.append(sb)
            all_raw.append(sr)
            line += f"  {sb:.4f}    {sr:.4f}"
        line += f"   {np.mean(all_blur):.4f}  {np.mean(all_raw):.4f}"
        print(line)
