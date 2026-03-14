"""Focused sweep: optimize the upscale soft-field approach.

4× upscale already gave 0.9899 (vs 0.976 baseline).  Now refine:
- Scale factor (2×, 4×, 6×, 8×)  
- Sigma combinations
- ISO threshold + halo opacity
- Single iso vs dual iso
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

    # Pre-compute all raw soft fields once
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
        soft_raw = pre['soft_fields'][ci]
        soft_up = cv2.resize(soft_raw, (w * scale, h * scale),
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
    ref = cv2.imread(os.path.join(os.path.dirname(__file__), "../../Ref.png"))
    crop = ref[50:460, 486:1050]
    print(f"Image: crop {crop.shape[1]}×{crop.shape[0]}")

    pre = precompute(crop)
    print(f"Clusters: {pre['K']}\n")

    header = f"{'Config':<55s}  blur   raw"
    print(header)
    print("-" * 72)

    configs = []

    # Phase 1: Scale factor sweep (with current sigma/iso)
    for scale in [2, 3, 4, 6, 8]:
        configs.append((
            f"scale={scale}× sc=0.6 ss=1.5 dual-iso",
            dict(scale=scale, sc_sigma=0.6, ss_sigma=1.5,
                 iso_levels=(0.20, 0.50), iso_opacities=(0.50, 1.00))
        ))

    # Phase 2: Sigma sweep at 4× (best scale so far)
    for sc in [0.3, 0.4, 0.5, 0.6, 0.8]:
        for ss in [0.8, 1.0, 1.5]:
            configs.append((
                f"scale=4× sc={sc} ss={ss} dual-iso",
                dict(scale=4, sc_sigma=sc, ss_sigma=ss,
                     iso_levels=(0.20, 0.50), iso_opacities=(0.50, 1.00))
            ))

    # Phase 3: Single iso (no halo) at 4× with different sigmas
    for sc in [0.3, 0.5, 0.6, 0.8]:
        configs.append((
            f"scale=4× sc={sc} ss=1.0 SINGLE iso=0.5",
            dict(scale=4, sc_sigma=sc, ss_sigma=1.0,
                 iso_levels=(0.50,), iso_opacities=(1.00,))
        ))

    # Phase 4: Halo opacity sweep at 4×
    for halo_iso in [0.15, 0.20, 0.25, 0.30]:
        for halo_op in [0.30, 0.40, 0.50, 0.60]:
            configs.append((
                f"scale=4× halo_iso={halo_iso} op={halo_op}",
                dict(scale=4, sc_sigma=0.6, ss_sigma=1.5,
                     iso_levels=(halo_iso, 0.50), iso_opacities=(halo_op, 1.00))
            ))

    # Phase 5: 3-level iso at 4×
    for mid_iso in [0.30, 0.35, 0.40]:
        for halo_op in [0.30, 0.50]:
            configs.append((
                f"scale=4× 3-iso [{0.15},{mid_iso},0.50] op=[{halo_op},0.75,1.0]",
                dict(scale=4, sc_sigma=0.6, ss_sigma=1.5,
                     iso_levels=(0.15, mid_iso, 0.50),
                     iso_opacities=(halo_op, 0.75, 1.00))
            ))

    best_ssim = 0
    best_name = ""
    for desc, kwargs in configs:
        t0 = time.time()
        svg = upscale_run(pre, **kwargs)
        rendered = render_svg(svg, pre['w'], pre['h'])
        s_blur = measure(crop, rendered, blur_sigma=1.5)
        s_raw = measure(crop, rendered, blur_sigma=0)
        dt = time.time() - t0
        if s_blur > best_ssim:
            best_ssim = s_blur
            best_name = desc
        print(f"  {desc:<53s}  {s_blur:.4f}  {s_raw:.4f}  ({dt:.1f}s)")

    print(f"\n*** BEST: {best_name}  SSIM={best_ssim:.4f} ***")
