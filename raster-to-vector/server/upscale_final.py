"""Final refine: top upscale combos across images, 3 K-means seeds each."""

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
    g1 = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(rend, cv2.COLOR_BGR2GRAY)
    if blur_sigma > 0:
        k = int(blur_sigma * 6) | 1
        g1 = cv2.GaussianBlur(g1, (k, k), blur_sigma)
        g2 = cv2.GaussianBlur(g2, (k, k), blur_sigma)
    return ssim(g1, g2)


def precompute(image_bgr, seed=42):
    h, w = image_bgr.shape[:2]
    bg_color, _ = detect_background(image_bgr)
    bg_hex = _bgr_to_hex(bg_color)

    denoised_km = cv2.bilateralFilter(image_bgr, 15, 12, 30)
    denoised_dist = cv2.bilateralFilter(image_bgr, 7, 5, 20)

    pixels = denoised_km.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)

    # Use seed for reproducibility
    np.random.seed(seed)
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


def upscale_run(pre, *, scale=4, sc_sigma=0.6, ss_sigma=1.5,
                iso_levels=(0.20, 0.50), iso_opacities=(0.50, 1.00)):
    h, w = pre['h'], pre['w']
    ew_up = cv2.resize(pre['edge_weight'], (w * scale, h * scale),
                       interpolation=cv2.INTER_LINEAR)
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
                if len(c) < 4:
                    continue
                xy = c[:, ::-1].astype(np.float64) / scale
                if abs(_polygon_area(xy)) < 15:
                    continue
                d = _fit_contour(xy, 0.15, 0.2, 60.0)
                if d:
                    parts.append(d)
            if parts:
                paths.append(" ".join(parts))
                opacs.append(op)
        if paths:
            layers.append(VectorLayer(paths=paths, opacities=opacs,
                                      color=_bgr_to_hex(pre['centers_u'][ci])))

    result = MultilevelResult(layers=layers, width=w, height=h,
                              background_color=pre['bg_hex'],
                              path_count=0, node_count=0)
    return generate_svg(result, remove_background=False)


if __name__ == "__main__":
    # Load images
    ref = cv2.imread(os.path.join(os.path.dirname(__file__), "../../Ref.png"))
    crop = ref[50:460, 486:1050]
    mahal = cv2.imread("/tmp/mahal_right.png")

    images = {"crop": crop}
    if mahal is not None:
        images["mahal"] = mahal

    # Configs to test
    configs = [
        # (name, kwargs)
        ("BASELINE: s4 sc0.6 ss1.5 h0.20@0.50",
         dict(scale=4, sc_sigma=0.6, ss_sigma=1.5,
              iso_levels=(0.20, 0.50), iso_opacities=(0.50, 1.00))),

        ("WINNER:  s4 sc0.6 ss1.5 h0.15@0.40",
         dict(scale=4, sc_sigma=0.6, ss_sigma=1.5,
              iso_levels=(0.15, 0.50), iso_opacities=(0.40, 1.00))),

        ("s6 sc0.6 ss1.5 h0.15@0.40",
         dict(scale=6, sc_sigma=0.6, ss_sigma=1.5,
              iso_levels=(0.15, 0.50), iso_opacities=(0.40, 1.00))),

        ("s8 sc0.6 ss1.5 h0.15@0.40",
         dict(scale=8, sc_sigma=0.6, ss_sigma=1.5,
              iso_levels=(0.15, 0.50), iso_opacities=(0.40, 1.00))),

        ("s4 sc0.8 ss0.8 h0.15@0.40",
         dict(scale=4, sc_sigma=0.8, ss_sigma=0.8,
              iso_levels=(0.15, 0.50), iso_opacities=(0.40, 1.00))),

        ("s4 sc0.6 ss1.0 h0.15@0.40",
         dict(scale=4, sc_sigma=0.6, ss_sigma=1.0,
              iso_levels=(0.15, 0.50), iso_opacities=(0.40, 1.00))),

        ("s4 sc0.6 ss1.5 h0.15@0.35",
         dict(scale=4, sc_sigma=0.6, ss_sigma=1.5,
              iso_levels=(0.15, 0.50), iso_opacities=(0.35, 1.00))),

        ("s4 sc0.6 ss1.5 h0.15@0.45",
         dict(scale=4, sc_sigma=0.6, ss_sigma=1.5,
              iso_levels=(0.15, 0.50), iso_opacities=(0.45, 1.00))),

        ("s4 sc0.6 ss1.5 h0.12@0.40",
         dict(scale=4, sc_sigma=0.6, ss_sigma=1.5,
              iso_levels=(0.12, 0.50), iso_opacities=(0.40, 1.00))),

        ("s4 sc0.6 ss1.5 h0.18@0.40",
         dict(scale=4, sc_sigma=0.6, ss_sigma=1.5,
              iso_levels=(0.18, 0.50), iso_opacities=(0.40, 1.00))),

        ("s6 sc0.8 ss0.8 h0.15@0.40",
         dict(scale=6, sc_sigma=0.8, ss_sigma=0.8,
              iso_levels=(0.15, 0.50), iso_opacities=(0.40, 1.00))),

        ("s6 sc0.6 ss1.0 h0.15@0.40",
         dict(scale=6, sc_sigma=0.6, ss_sigma=1.0,
              iso_levels=(0.15, 0.50), iso_opacities=(0.40, 1.00))),

        ("s4 sc0.5 ss0.8 h0.15@0.40",
         dict(scale=4, sc_sigma=0.5, ss_sigma=0.8,
              iso_levels=(0.15, 0.50), iso_opacities=(0.40, 1.00))),

        ("s4 sc0.4 ss0.8 h0.15@0.40",
         dict(scale=4, sc_sigma=0.4, ss_sigma=0.8,
              iso_levels=(0.15, 0.50), iso_opacities=(0.40, 1.00))),
    ]

    seeds = [42, 123, 999]
    print(f"{'Config':<42s}", end="")
    for name in images:
        print(f"  {name:>8s}", end="")
    print("     avg")
    print("-" * (42 + 11 * (len(images) + 1)))

    for desc, kwargs in configs:
        scores = {name: [] for name in images}
        for seed in seeds:
            for name, img in images.items():
                pre = precompute(img, seed=seed)
                svg = upscale_run(pre, **kwargs)
                rend = render_svg(svg, pre['w'], pre['h'])
                s = measure(img, rend)
                scores[name].append(s)

        print(f"  {desc:<40s}", end="")
        all_means = []
        for name in images:
            m = np.mean(scores[name])
            all_means.append(m)
            print(f"  {m:.4f}", end="")
        print(f"  {np.mean(all_means):.4f}")

    print("\nDone.")
