"""Test: zero/minimal sigma at high upscale to eliminate both blur-shrinkage and halo artifacts.

Theory: Without Gaussian blur, the iso=0.5 contour sits exactly where soft_field=0.5
(which is exactly where K-means assigned the boundary). With 8× upscale, marching squares
traces this boundary with sub-pixel accuracy. No halo needed.

The bilateral denoising should already suppress SD static in the soft field.
Previous attempts at zero sigma (at 1×) failed, but 8× might be different.
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
    soft_fields = {}
    for ci in range(K):
        d_k = dist_map[:, :, ci]
        omask = np.ones(K, dtype=bool); omask[ci] = False
        d_other = np.min(dist_map[:, :, omask], axis=2)
        soft_fields[ci] = d_other / np.maximum(d_k + d_other, 1e-10)
    return dict(h=h, w=w, K=K, bg_hex=bg_hex, bg_cluster=bg_cluster,
                centers_u=centers_u, order=order, soft_fields=soft_fields)


def run(pre, *, scale=8, sigma=0.0, iso_levels=(0.50,), iso_opacities=(1.00,),
        eps=0.08, max_error=0.15, corner=60.0):
    """Simplified: uniform sigma (or zero), no dual-sigma, no edge weight."""
    h, w, K = pre['h'], pre['w'], pre['K']
    layers = []
    for ci in pre['order']:
        if ci == pre['bg_cluster']:
            continue
        soft_raw = pre['soft_fields'][ci]
        soft_up = cv2.resize(soft_raw, (w * scale, h * scale), interpolation=cv2.INTER_LINEAR)
        if sigma > 0:
            soft = cv2.GaussianBlur(soft_up, (0, 0), sigmaX=sigma * scale)
        else:
            soft = soft_up
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
        # Zero sigma, single iso, various core positions
        ("s8 σ=0 iso=0.50 single",
         dict(sigma=0, iso_levels=(0.50,), iso_opacities=(1.0,))),
        ("s8 σ=0 iso=0.48 single",
         dict(sigma=0, iso_levels=(0.48,), iso_opacities=(1.0,))),
        ("s8 σ=0 iso=0.47 single",
         dict(sigma=0, iso_levels=(0.47,), iso_opacities=(1.0,))),
        ("s8 σ=0 iso=0.45 single",
         dict(sigma=0, iso_levels=(0.45,), iso_opacities=(1.0,))),

        # Very low sigma, single iso
        ("s8 σ=0.1 iso=0.48 single",
         dict(sigma=0.1, iso_levels=(0.48,), iso_opacities=(1.0,))),
        ("s8 σ=0.15 iso=0.48 single",
         dict(sigma=0.15, iso_levels=(0.48,), iso_opacities=(1.0,))),
        ("s8 σ=0.2 iso=0.48 single",
         dict(sigma=0.2, iso_levels=(0.48,), iso_opacities=(1.0,))),
        ("s8 σ=0.3 iso=0.48 single",
         dict(sigma=0.3, iso_levels=(0.48,), iso_opacities=(1.0,))),

        # Zero sigma + thin halo 
        ("s8 σ=0 [0.35,0.50] [0.30,1]",
         dict(sigma=0, iso_levels=(0.35, 0.50), iso_opacities=(0.30, 1.0))),
        ("s8 σ=0 [0.40,0.50] [0.40,1]",
         dict(sigma=0, iso_levels=(0.40, 0.50), iso_opacities=(0.40, 1.0))),
        ("s8 σ=0 [0.30,0.48] [0.25,1]",
         dict(sigma=0, iso_levels=(0.30, 0.48), iso_opacities=(0.25, 1.0))),

        # Low sigma + standard halo
        ("s8 σ=0.2 [0.20,0.48] [0.50,1]",
         dict(sigma=0.2, iso_levels=(0.20, 0.48), iso_opacities=(0.50, 1.0))),
        ("s8 σ=0.3 [0.20,0.47] [0.50,1]",
         dict(sigma=0.3, iso_levels=(0.20, 0.47), iso_opacities=(0.50, 1.0))),

        # Reference: current best with dual sigma
        ("s8 dual-σ [0.20,0.47] [0.50,1] (ref)",
         dict(sigma=0.6, iso_levels=(0.20, 0.47), iso_opacities=(0.50, 1.0))),
    ]

    seeds = [42, 123, 999]
    print(f"{'Config':<42s}", end="")
    for name in images: print(f"  {name:>8s}", end="")
    print("     avg")
    print("-" * (42 + 11 * (len(images) + 1)))

    best = 0
    for desc, kwargs in configs:
        scores = {n: [] for n in images}
        for seed in seeds:
            for name, img in images.items():
                np.random.seed(seed)
                pre = precompute(img)
                svg = run(pre, **kwargs)
                rend = render_svg(svg, pre['w'], pre['h'])
                scores[name].append(measure(img, rend))
        print(f"  {desc:<40s}", end="")
        avgs = []
        for name in images:
            m = np.mean(scores[name])
            avgs.append(m)
            print(f"  {m:.4f}", end="")
        total = np.mean(avgs)
        if total > best: best = total
        print(f"  {total:.4f}")

    print(f"\nBest avg: {best:.4f}")
