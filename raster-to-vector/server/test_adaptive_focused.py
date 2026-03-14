"""Focused test: adaptive iso-field with targeted parameter sweep.

The adaptive iso-field IS the inner/outer compression idea, done right:
- At each pixel, the soft-field gradient tells us how "hard" the edge is
- Hard edges → iso threshold shifts inward (tighter boundary)  
- Soft edges → iso threshold stays outward (generous boundary)
- This naturally compresses inner/outer boundaries toward the true visual edge

Previous results:
  Production (fixed iso):     crop raw=0.9474, mahal raw=0.9634
  C3 (0.30-0.60 +halo@0.20): crop raw=0.9484, mahal raw=0.9671  (best universal)
  C1 (0.25-0.55 +halo@0.18): crop raw=0.9507, mahal raw=0.9499  (best crop)

Sweep the promising region around C3.
"""

import sys, time
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
        soft_fields[ci] = d_other / np.maximum(d_k + d_other, 1e-10)
    return dict(h=h, w=w, K=K, bg_hex=bg_hex, bg_cluster=bg_cluster,
                centers_u=centers_u, order=order, edge_weight=edge_weight,
                soft_fields=soft_fields)


def make_svg(pre, scale=4, mode="production", inner_iso=0.55, outer_iso=0.25,
             halo_iso=0.20, halo_opacity=0.50, smooth_sigma=1.0):
    """Unified SVG builder. mode='production' or 'adaptive'."""
    h, w, K = pre['h'], pre['w'], pre['K']
    S = scale
    ew_up = cv2.resize(pre['edge_weight'], (w * S, h * S), interpolation=cv2.INTER_LINEAR)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">',
        f'<rect width="{w}" height="{h}" fill="{pre["bg_hex"]}"/>',
    ]
    nodes = 0

    for ci in pre['order']:
        if ci == pre['bg_cluster']: continue
        color = _bgr_to_hex(pre['centers_u'][ci])
        soft_up = cv2.resize(pre['soft_fields'][ci], (w * S, h * S), interpolation=cv2.INTER_LINEAR)
        sc = cv2.GaussianBlur(soft_up, (0, 0), sigmaX=0.6 * S)
        ss = cv2.GaussianBlur(soft_up, (0, 0), sigmaX=1.5 * S)
        soft = ew_up * sc + (1 - ew_up) * ss

        if mode == "production":
            iso_pairs = [(0.20, 0.50), (0.47, 1.00)]
            for iso, op in iso_pairs:
                pp = []
                for c in find_contours(soft, iso):
                    if len(c) < 4: continue
                    xy = c[:, ::-1].astype(np.float64) / S
                    if abs(_polygon_area(xy)) < 15: continue
                    d = _fit_contour(xy, 0.08, 0.15, 60.0)
                    if d: pp.append(d); nodes += d.count("C") + d.count("L") + d.count("M")
                if pp:
                    d = " ".join(pp)
                    if op < 1.0: parts.append(f'<path d="{d}" fill="{color}" fill-rule="evenodd" opacity="{op:.2f}"/>')
                    else: parts.append(f'<path d="{d}" fill="{color}" fill-rule="evenodd"/>')

        elif mode == "adaptive":
            # Gradient-based adaptive iso threshold
            gx = cv2.Sobel(soft.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(soft.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
            grad = np.sqrt(gx * gx + gy * gy)
            mx = grad.max()
            grad_norm = grad / mx if mx > 0 else grad
            # Per-pixel iso: outer where gradient is low, inner where high
            adaptive_iso = outer_iso + grad_norm * (inner_iso - outer_iso)
            adaptive_iso = cv2.GaussianBlur(adaptive_iso, (0, 0), sigmaX=smooth_sigma * S)
            shifted = soft - adaptive_iso

            # Halo
            if halo_iso is not None and halo_opacity > 0:
                pp = []
                for c in find_contours(soft, halo_iso):
                    if len(c) < 4: continue
                    xy = c[:, ::-1].astype(np.float64) / S
                    if abs(_polygon_area(xy)) < 15: continue
                    d = _fit_contour(xy, 0.08, 0.15, 60.0)
                    if d: pp.append(d); nodes += d.count("C") + d.count("L") + d.count("M")
                if pp:
                    parts.append(f'<path d="{" ".join(pp)}" fill="{color}" fill-rule="evenodd" opacity="{halo_opacity:.2f}"/>')

            # Core at shifted=0
            pp = []
            for c in find_contours(shifted, 0.0):
                if len(c) < 4: continue
                xy = c[:, ::-1].astype(np.float64) / S
                if abs(_polygon_area(xy)) < 15: continue
                d = _fit_contour(xy, 0.08, 0.15, 60.0)
                if d: pp.append(d); nodes += d.count("C") + d.count("L") + d.count("M")
            if pp:
                parts.append(f'<path d="{" ".join(pp)}" fill="{color}" fill-rule="evenodd"/>')

    parts.append("</svg>")
    return "\n".join(parts), nodes


if __name__ == "__main__":
    ref = cv2.imread("/Users/alexwaldmann/Desktop/SVG-gen/Ref.png")
    crop = ref[50:460, 486:1050]
    mahal = cv2.imread("/tmp/mahal_right.png")

    print("Pre-computing features...")
    pre_crop = precompute(crop)
    pre_mahal = precompute(mahal) if mahal is not None else None

    # Production baseline
    svg, _ = make_svg(pre_crop, mode="production")
    rend = render_svg(svg, pre_crop['w'], pre_crop['h'])
    pc_blur = measure(crop, rend, 1.5)
    pc_raw = measure(crop, rend, 0)

    pm_blur = pm_raw = 0
    if pre_mahal:
        svg, _ = make_svg(pre_mahal, mode="production")
        rend = render_svg(svg, pre_mahal['w'], pre_mahal['h'])
        pm_blur = measure(mahal, rend, 1.5)
        pm_raw = measure(mahal, rend, 0)

    print(f"\nProduction: crop blur={pc_blur:.4f} raw={pc_raw:.4f}  mahal blur={pm_blur:.4f} raw={pm_raw:.4f}")

    # Focused sweep
    configs = []
    for inner in [0.50, 0.55, 0.60]:
        for outer in [0.25, 0.30, 0.35]:
            if inner <= outer + 0.10: continue
            for h_iso in [0.15, 0.20, 0.25]:
                for h_op in [0.40, 0.45, 0.50]:
                    for sigma in [0.8, 1.0]:
                        configs.append((inner, outer, h_iso, h_op, sigma))

    print(f"\nSweeping {len(configs)} configs on BOTH images...")
    print(f"{'inner':>5s} {'outer':>5s} {'h_iso':>5s} {'h_op':>4s} {'sig':>4s}  "
          f"c_blur  c_raw   m_blur  m_raw   avg_Δraw")
    print("-" * 72)

    results = []

    for inner, outer, h_iso, h_op, sigma in configs:
        # Crop
        svg, _ = make_svg(pre_crop, mode="adaptive", inner_iso=inner, outer_iso=outer,
                          halo_iso=h_iso, halo_opacity=h_op, smooth_sigma=sigma)
        rend = render_svg(svg, pre_crop['w'], pre_crop['h'])
        cb = measure(crop, rend, 1.5)
        cr = measure(crop, rend, 0)

        # Mahal
        mb = mr = 0
        if pre_mahal:
            svg, _ = make_svg(pre_mahal, mode="adaptive", inner_iso=inner, outer_iso=outer,
                              halo_iso=h_iso, halo_opacity=h_op, smooth_sigma=sigma)
            rend = render_svg(svg, pre_mahal['w'], pre_mahal['h'])
            mb = measure(mahal, rend, 1.5)
            mr = measure(mahal, rend, 0)

        avg_dr = ((cr - pc_raw) + (mr - pm_raw)) / 2
        results.append((avg_dr, inner, outer, h_iso, h_op, sigma, cb, cr, mb, mr))

    results.sort(key=lambda x: -x[0])

    print("\nTop 15 by average raw SSIM improvement:")
    print(f"{'#':>3s} {'inner':>5s} {'outer':>5s} {'h_iso':>5s} {'h_op':>4s} {'sig':>4s}  "
          f"c_blur  c_raw   m_blur  m_raw   avg_Δraw")
    print("-" * 75)
    for rank, (avg_dr, inner, outer, h_iso, h_op, sigma, cb, cr, mb, mr) in enumerate(results[:15]):
        print(f"  {rank+1:>2d}  {inner:.2f}  {outer:.2f}  {h_iso:.2f}  {h_op:.1f}  {sigma:.1f}  "
              f"{cb:.4f}  {cr:.4f}  {mb:.4f}  {mr:.4f}  {avg_dr:+.4f}")

    print(f"\nBottom 5 (worst):")
    for rank, (avg_dr, inner, outer, h_iso, h_op, sigma, cb, cr, mb, mr) in enumerate(results[-5:]):
        print(f"  {rank+1:>2d}  {inner:.2f}  {outer:.2f}  {h_iso:.2f}  {h_op:.1f}  {sigma:.1f}  "
              f"{cb:.4f}  {cr:.4f}  {mb:.4f}  {mr:.4f}  {avg_dr:+.4f}")

    # Compare best to production
    best = results[0]
    print(f"\n{'='*75}")
    print(f"BEST CONFIG: inner={best[1]:.2f} outer={best[2]:.2f} halo_iso={best[3]:.2f} "
          f"halo_op={best[4]:.1f} sigma={best[5]:.1f}")
    print(f"  crop:  blur {best[6]:.4f} (Δ{best[6]-pc_blur:+.4f})  raw {best[7]:.4f} (Δ{best[7]-pc_raw:+.4f})")
    print(f"  mahal: blur {best[8]:.4f} (Δ{best[8]-pm_blur:+.4f})  raw {best[9]:.4f} (Δ{best[9]-pm_raw:+.4f})")
    print(f"  avg raw improvement: {best[0]:+.4f}")
