"""Push to 0.995: targeted experiments on the remaining 0.6% gap.

V1 4× upscale with h0.20@0.50 gives ~0.989 avg.  Theoretical ceiling = 0.9985.
Remaining gap is ~0.95%, caused by:
  1. Halo interference (50% opacity adds wrong color at boundaries)
  2. Curve fitting smoothing (Bézier approximation + approxPolyDP)
  3. Cluster merging (too few colors → wrong colors in some regions)

This script tests:
  A) Tighter curve fitting at 4× (eps=0.05, max_error=0.1)
  B) More clusters (K=36/48, merge=40) 
  C) Lower core iso (0.45/0.48 instead of 0.50) for better coverage
  D) 3-level smooth gradient halo
  E) Combination of best elements
"""

import sys, os
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


def precompute(image_bgr, num_k=24, merge_thresh=60.0):
    h, w = image_bgr.shape[:2]
    bg_color, _ = detect_background(image_bgr)
    bg_hex = _bgr_to_hex(bg_color)
    denoised_km = cv2.bilateralFilter(image_bgr, 15, 12, 30)
    denoised_dist = cv2.bilateralFilter(image_bgr, 7, 5, 20)
    pixels = denoised_km.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(pixels, num_k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    centers, labels = _merge_close_clusters(centers, labels.flatten(), h, w, threshold=merge_thresh)
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
    # Pre-compute soft fields
    soft_fields = {}
    for ci in order:
        if ci == bg_cluster:
            continue
        d_k = dist_map[:, :, ci]
        omask = np.ones(K, dtype=bool); omask[ci] = False
        d_other = np.min(dist_map[:, :, omask], axis=2)
        soft_fields[ci] = d_other / np.maximum(d_k + d_other, 1e-10)
    return dict(h=h, w=w, K=K, bg_hex=bg_hex, bg_cluster=bg_cluster,
                centers_u=centers_u, order=order, edge_weight=edge_weight,
                soft_fields=soft_fields)


def run_v1(pre, *, scale=4, sc_sigma=0.6, ss_sigma=1.5,
           iso_levels=(0.20, 0.50), iso_opacities=(0.50, 1.00),
           eps=0.15, max_error=0.2, corner=60.0):
    h, w, K = pre['h'], pre['w'], pre['K']
    ew_up = cv2.resize(pre['edge_weight'], (w * scale, h * scale), interpolation=cv2.INTER_LINEAR)
    layers = []
    for ci in pre['order']:
        if ci == pre['bg_cluster']:
            continue
        soft_up = cv2.resize(pre['soft_fields'][ci], (w * scale, h * scale),
                             interpolation=cv2.INTER_LINEAR)
        sc = cv2.GaussianBlur(soft_up, (0, 0), sigmaX=sc_sigma * scale)
        ss = cv2.GaussianBlur(soft_up, (0, 0), sigmaX=ss_sigma * scale)
        soft = ew_up * sc + (1 - ew_up) * ss
        paths, opacs = [], []
        for iso, op in zip(iso_levels, iso_opacities):
            contours = find_contours(soft, iso)
            parts = []
            for c in contours:
                if len(c) < 4: continue
                xy = c[:, ::-1].astype(np.float64) / scale
                if abs(_polygon_area(xy)) < 15: continue
                d = _fit_contour(xy, eps, max_error, corner)
                if d: parts.append(d)
            if parts:
                paths.append(" ".join(parts)); opacs.append(op)
        if paths:
            layers.append(VectorLayer(paths=paths, opacities=opacs,
                                      color=_bgr_to_hex(pre['centers_u'][ci])))
    result = MultilevelResult(layers=layers, width=w, height=h,
                              background_color=pre['bg_hex'], path_count=0, node_count=0)
    return generate_svg(result, remove_background=False)


if __name__ == "__main__":
    ref = cv2.imread(os.path.join(os.path.dirname(__file__), "../../Ref.png"))
    crop = ref[50:460, 486:1050]
    mahal = cv2.imread("/tmp/mahal_right.png")
    images = {"crop": crop}
    if mahal is not None:
        images["mahal"] = mahal

    configs = [
        # --- Baseline ---
        ("baseline s4",
         24, 60, dict(scale=4)),

        # --- A) Tighter curve fitting ---
        ("s4 eps=0.05 me=0.10",
         24, 60, dict(scale=4, eps=0.05, max_error=0.1)),
        ("s4 eps=0.08 me=0.15",
         24, 60, dict(scale=4, eps=0.08, max_error=0.15)),
        ("s4 eps=0.10 me=0.15",
         24, 60, dict(scale=4, eps=0.10, max_error=0.15)),

        # --- B) More clusters ---
        ("s4 K=36 m=40",
         36, 40, dict(scale=4)),
        ("s4 K=36 m=50",
         36, 50, dict(scale=4)),
        ("s4 K=48 m=40",
         48, 40, dict(scale=4)),
        ("s4 K=48 m=50",
         48, 50, dict(scale=4)),
        ("s4 K=24 m=40",
         24, 40, dict(scale=4)),

        # --- C) Lower core iso for gap coverage ---
        ("s4 core=0.48",
         24, 60, dict(scale=4, iso_levels=(0.20, 0.48), iso_opacities=(0.50, 1.00))),
        ("s4 core=0.45",
         24, 60, dict(scale=4, iso_levels=(0.20, 0.45), iso_opacities=(0.50, 1.00))),
        ("s4 core=0.42",
         24, 60, dict(scale=4, iso_levels=(0.20, 0.42), iso_opacities=(0.50, 1.00))),

        # --- D) 3-level gradient ---
        ("s4 3lvl [.15,.35,.50] [.25,.60,1]",
         24, 60, dict(scale=4, iso_levels=(0.15, 0.35, 0.50),
                       iso_opacities=(0.25, 0.60, 1.00))),
        ("s4 3lvl [.10,.30,.50] [.20,.50,1]",
         24, 60, dict(scale=4, iso_levels=(0.10, 0.30, 0.50),
                       iso_opacities=(0.20, 0.50, 1.00))),

        # --- E) Combinations ---
        ("s4 K=36 m=50 eps=0.10 me=0.15",
         36, 50, dict(scale=4, eps=0.10, max_error=0.15)),
        ("s4 K=36 m=50 core=0.48",
         36, 50, dict(scale=4, iso_levels=(0.20, 0.48), iso_opacities=(0.50, 1.00))),

        # --- F) Bigger upscale + tighter fit ---
        ("s6 eps=0.10 me=0.15",
         24, 60, dict(scale=6, eps=0.10, max_error=0.15)),
        ("s8 eps=0.10 me=0.15",
         24, 60, dict(scale=8, eps=0.10, max_error=0.15)),
    ]

    print(f"{'Config':<42s}", end="")
    for name in images:
        print(f"  {name:>8s}", end="")
    print("     avg")
    print("-" * (42 + 11 * (len(images) + 1)))

    for desc, num_k, merge_t, kwargs in configs:
        scores = {n: [] for n in images}
        for name, img in images.items():
            pre = precompute(img, num_k=num_k, merge_thresh=merge_t)
            svg = run_v1(pre, **kwargs)
            rend = render_svg(svg, pre['w'], pre['h'])
            scores[name].append(measure(img, rend))
        print(f"  {desc:<40s}", end="")
        avgs = []
        for name in images:
            m = np.mean(scores[name])
            avgs.append(m)
            print(f"  {m:.4f}", end="")
        print(f"  {np.mean(avgs):.4f}")
