"""Structural test: compare fundamentally different contour extraction strategies.

The parameter sweep shows we're stuck at SSIM ~0.982.  Theoretical ceiling
with pixel-perfect 5-color placement is 0.9985.  The entire gap is boundary
placement error.  This script tests strategies to close that gap:

A) Baseline — current production (soft field + dual sigma + dual iso)
B) Label-mask — trace K-means label boundaries directly (different sigma)
C) Upscale 2×/4× — compute soft field at higher resolution
D) Label-mask + upscale — combine B and C
E) Minimal sigma — soft field with very low blur (0.2) for tighter boundaries
"""

import sys, os, time
import cv2
import numpy as np
from pathlib import Path
from skimage.measure import find_contours
from skimage.metrics import structural_similarity as ssim

sys.path.insert(0, str(Path(__file__).parent))
from app.core.curve_fitting import fit_closed_bezier
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
    pil = Image.open(BytesIO(png)).convert("RGB")
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def measure(source, rendered, blur_sigma=1.5):
    g1 = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(rendered, cv2.COLOR_BGR2GRAY)
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

    return dict(h=h, w=w, K=K, bg_hex=bg_hex, bg_cluster=bg_cluster,
                centers_u=centers_u, centers_f=centers_f, dist_map=dist_map,
                labels=labels, order=order, edge_weight=edge_weight)


# ---- helpers ----
def _extract_contours(field, iso, min_area=15, scale=1, eps=0.15, me=0.2, ct=60.0):
    contours = find_contours(field, iso)
    parts = []
    for c in contours:
        if len(c) < 4:
            continue
        xy = c[:, ::-1].astype(np.float64) / scale
        if abs(_polygon_area(xy)) < min_area:
            continue
        d = _fit_contour(xy, eps, me, ct)
        if d:
            parts.append(d)
    return " ".join(parts) if parts else None


def _make_svg(layers, w, h, bg_hex):
    result = MultilevelResult(layers=layers, width=w, height=h,
                              background_color=bg_hex, path_count=0, node_count=0)
    return generate_svg(result, remove_background=False)


# ---- APPROACH A: Baseline (current production) ----
def approach_A(pre):
    h, w, K = pre['h'], pre['w'], pre['K']
    layers = []
    for ci in pre['order']:
        if ci == pre['bg_cluster']:
            continue
        d_k = pre['dist_map'][:, :, ci]
        omask = np.ones(K, dtype=bool); omask[ci] = False
        d_other = np.min(pre['dist_map'][:, :, omask], axis=2)
        soft_raw = d_other / np.maximum(d_k + d_other, 1e-10)
        sc = cv2.GaussianBlur(soft_raw, (0, 0), sigmaX=0.6)
        ss = cv2.GaussianBlur(soft_raw, (0, 0), sigmaX=1.5)
        soft = pre['edge_weight'] * sc + (1 - pre['edge_weight']) * ss

        paths, opacs = [], []
        for iso, op in [(0.20, 0.50), (0.50, 1.00)]:
            d = _extract_contours(soft, iso)
            if d:
                paths.append(d); opacs.append(op)
        if paths:
            layers.append(VectorLayer(paths=paths, opacities=opacs,
                                      color=_bgr_to_hex(pre['centers_u'][ci])))
    return _make_svg(layers, w, h, pre['bg_hex'])


# ---- APPROACH B: Label-mask contour (single iso, no halo) ----
def approach_label(pre, sigma=0.8, dilate=0):
    h, w, K = pre['h'], pre['w'], pre['K']
    layers = []
    for ci in pre['order']:
        if ci == pre['bg_cluster']:
            continue
        mask = (pre['labels'] == ci).astype(np.float32)
        if dilate > 0:
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.dilate((mask * 255).astype(np.uint8), kernel,
                              iterations=dilate).astype(np.float32) / 255.0
        if sigma > 0:
            soft = cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma)
        else:
            soft = mask
        d = _extract_contours(soft, 0.5)
        if d:
            layers.append(VectorLayer(paths=[d], opacities=[1.0],
                                      color=_bgr_to_hex(pre['centers_u'][ci])))
    return _make_svg(layers, w, h, pre['bg_hex'])


# ---- APPROACH C: Upscale soft field 2×/4× ----
def approach_upscale(pre, scale=2, sc_sigma=0.6, ss_sigma=1.5):
    h, w, K = pre['h'], pre['w'], pre['K']
    ew_up = cv2.resize(pre['edge_weight'], (w * scale, h * scale), interpolation=cv2.INTER_LINEAR)
    layers = []
    for ci in pre['order']:
        if ci == pre['bg_cluster']:
            continue
        d_k = pre['dist_map'][:, :, ci]
        omask = np.ones(K, dtype=bool); omask[ci] = False
        d_other = np.min(pre['dist_map'][:, :, omask], axis=2)
        soft_raw = d_other / np.maximum(d_k + d_other, 1e-10)
        soft_up = cv2.resize(soft_raw, (w * scale, h * scale), interpolation=cv2.INTER_LINEAR)
        sc = cv2.GaussianBlur(soft_up, (0, 0), sigmaX=sc_sigma * scale)
        ss = cv2.GaussianBlur(soft_up, (0, 0), sigmaX=ss_sigma * scale)
        soft = ew_up * sc + (1 - ew_up) * ss

        paths, opacs = [], []
        for iso, op in [(0.20, 0.50), (0.50, 1.00)]:
            d = _extract_contours(soft, iso, scale=scale)
            if d:
                paths.append(d); opacs.append(op)
        if paths:
            layers.append(VectorLayer(paths=paths, opacities=opacs,
                                      color=_bgr_to_hex(pre['centers_u'][ci])))
    return _make_svg(layers, w, h, pre['bg_hex'])


# ---- APPROACH D: Label-mask + upscale ----
def approach_label_upscale(pre, scale=2, sigma=0.8, dilate=0):
    h, w, K = pre['h'], pre['w'], pre['K']
    layers = []
    for ci in pre['order']:
        if ci == pre['bg_cluster']:
            continue
        mask = (pre['labels'] == ci).astype(np.float32)
        if dilate > 0:
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.dilate((mask * 255).astype(np.uint8), kernel,
                              iterations=dilate).astype(np.float32) / 255.0
        mask_up = cv2.resize(mask, (w * scale, h * scale), interpolation=cv2.INTER_LINEAR)
        if sigma > 0:
            soft = cv2.GaussianBlur(mask_up, (0, 0), sigmaX=sigma * scale)
        else:
            soft = mask_up
        d = _extract_contours(soft, 0.5, scale=scale)
        if d:
            layers.append(VectorLayer(paths=[d], opacities=[1.0],
                                      color=_bgr_to_hex(pre['centers_u'][ci])))
    return _make_svg(layers, w, h, pre['bg_hex'])


# ---- APPROACH E: Soft field, minimal sigma ----
def approach_minimal_sigma(pre, sigma=0.2):
    h, w, K = pre['h'], pre['w'], pre['K']
    layers = []
    for ci in pre['order']:
        if ci == pre['bg_cluster']:
            continue
        d_k = pre['dist_map'][:, :, ci]
        omask = np.ones(K, dtype=bool); omask[ci] = False
        d_other = np.min(pre['dist_map'][:, :, omask], axis=2)
        soft_raw = d_other / np.maximum(d_k + d_other, 1e-10)
        if sigma > 0:
            soft = cv2.GaussianBlur(soft_raw, (0, 0), sigmaX=sigma)
        else:
            soft = soft_raw
        d = _extract_contours(soft, 0.5)
        if d:
            layers.append(VectorLayer(paths=[d], opacities=[1.0],
                                      color=_bgr_to_hex(pre['centers_u'][ci])))
    return _make_svg(layers, w, h, pre['bg_hex'])


# ---- APPROACH F: Label-mask with halo (dilate for coverage, no opacity) ----
def approach_label_halo(pre, sigma=0.8):
    """Two-pass label: dilated contour underneath (lighter), exact contour on top."""
    h, w, K = pre['h'], pre['w'], pre['K']
    layers = []
    for ci in pre['order']:
        if ci == pre['bg_cluster']:
            continue
        mask = (pre['labels'] == ci).astype(np.float32)
        # Halo: dilate 1px
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate((mask * 255).astype(np.uint8), kernel,
                             iterations=1).astype(np.float32) / 255.0
        soft_halo = cv2.GaussianBlur(dilated, (0, 0), sigmaX=sigma)
        soft_core = cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma)

        paths, opacs = [], []
        d_halo = _extract_contours(soft_halo, 0.5)
        if d_halo:
            paths.append(d_halo); opacs.append(0.50)
        d_core = _extract_contours(soft_core, 0.5)
        if d_core:
            paths.append(d_core); opacs.append(1.00)
        if paths:
            layers.append(VectorLayer(paths=paths, opacities=opacs,
                                      color=_bgr_to_hex(pre['centers_u'][ci])))
    return _make_svg(layers, w, h, pre['bg_hex'])


# ---- APPROACH G: Hybrid — label mask for topology, soft field for sub-pixel ----
def approach_hybrid(pre, sigma=0.5):
    """Use label mask AND soft field together: mask provides the binary shape,
    soft field provides sub-pixel gradient for smooth contour extraction."""
    h, w, K = pre['h'], pre['w'], pre['K']
    layers = []
    for ci in pre['order']:
        if ci == pre['bg_cluster']:
            continue
        # Soft field
        d_k = pre['dist_map'][:, :, ci]
        omask = np.ones(K, dtype=bool); omask[ci] = False
        d_other = np.min(pre['dist_map'][:, :, omask], axis=2)
        soft_raw = d_other / np.maximum(d_k + d_other, 1e-10)
        # Label mask
        mask = (pre['labels'] == ci).astype(np.float32)
        # Hybrid: multiply soft field by mask to constrain to labeled region,
        # then blur for smooth contours
        hybrid = soft_raw * 0.5 + mask * 0.5  # blend
        soft = cv2.GaussianBlur(hybrid, (0, 0), sigmaX=sigma)
        d = _extract_contours(soft, 0.5)
        if d:
            layers.append(VectorLayer(paths=[d], opacities=[1.0],
                                      color=_bgr_to_hex(pre['centers_u'][ci])))
    return _make_svg(layers, w, h, pre['bg_hex'])


# ===========================================================================
if __name__ == "__main__":
    ref = cv2.imread(os.path.join(os.path.dirname(__file__), "../../Ref.png"))
    crop = ref[50:460, 486:1050]
    print(f"Image: crop {crop.shape[1]}×{crop.shape[0]}")

    pre = precompute(crop)
    print(f"Clusters after merge: {pre['K']}")
    print()

    tests = [
        ("A) Baseline (soft+dual iso+dual σ)", approach_A, [pre]),
        ("B) Label σ=0.5",                      approach_label, [pre], {'sigma': 0.5}),
        ("B) Label σ=0.8",                      approach_label, [pre], {'sigma': 0.8}),
        ("B) Label σ=1.0",                      approach_label, [pre], {'sigma': 1.0}),
        ("B) Label σ=0.3",                      approach_label, [pre], {'sigma': 0.3}),
        ("B) Label σ=0 (raw)",                  approach_label, [pre], {'sigma': 0}),
        ("C) Upscale 2× soft field",            approach_upscale, [pre], {'scale': 2}),
        ("C) Upscale 4× soft field",            approach_upscale, [pre], {'scale': 4}),
        ("D) Label 2× σ=0.5",                   approach_label_upscale, [pre], {'scale': 2, 'sigma': 0.5}),
        ("D) Label 2× σ=0.8",                   approach_label_upscale, [pre], {'scale': 2, 'sigma': 0.8}),
        ("D) Label 4× σ=0.5",                   approach_label_upscale, [pre], {'scale': 4, 'sigma': 0.5}),
        ("D) Label 4× σ=0.8",                   approach_label_upscale, [pre], {'scale': 4, 'sigma': 0.8}),
        ("E) Soft σ=0.2 (minimal)",             approach_minimal_sigma, [pre], {'sigma': 0.2}),
        ("E) Soft σ=0.3",                       approach_minimal_sigma, [pre], {'sigma': 0.3}),
        ("E) Soft σ=0 (raw)",                   approach_minimal_sigma, [pre], {'sigma': 0}),
        ("F) Label halo σ=0.8",                 approach_label_halo, [pre], {'sigma': 0.8}),
        ("F) Label halo σ=0.5",                 approach_label_halo, [pre], {'sigma': 0.5}),
        ("G) Hybrid label+soft σ=0.5",          approach_hybrid, [pre], {'sigma': 0.5}),
        ("G) Hybrid label+soft σ=0.8",          approach_hybrid, [pre], {'sigma': 0.8}),
        ("B) Label σ=0.8 dilate=1",             approach_label, [pre], {'sigma': 0.8, 'dilate': 1}),
    ]

    print(f"{'Approach':<45s}  SSIM   raw_SSIM")
    print("-" * 68)

    for entry in tests:
        desc = entry[0]
        fn = entry[1]
        args = entry[2]
        kwargs = entry[3] if len(entry) > 3 else {}
        t0 = time.time()
        svg = fn(*args, **kwargs)
        rendered = render_svg(svg, pre['w'], pre['h'])
        s_blur = measure(crop, rendered, blur_sigma=1.5)
        s_raw = measure(crop, rendered, blur_sigma=0)
        dt = time.time() - t0
        print(f"  {desc:<43s}  {s_blur:.4f}  {s_raw:.4f}  ({dt:.1f}s)")
