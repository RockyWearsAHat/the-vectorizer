"""Fine-tune the adaptive iso-field approach.

C3 (0.30-0.60 +halo@0.20/0.45) beat production on BOTH images:
  crop raw:  +0.0010 (0.9484 vs 0.9474)
  mahal raw: +0.0037 (0.9671 vs 0.9634)

C1 (0.25-0.55 +halo@0.18/0.45) was best on crop raw (+0.0033)
but worse on mahal.

Sweep the parameter space to find best universal setting.
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


def make_svg_production(pre, scale=4):
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
        for iso, op in [(0.20, 0.50), (0.47, 1.00)]:
            contours = find_contours(soft, iso)
            pp = []
            for c in contours:
                if len(c) < 4: continue
                xy = c[:, ::-1].astype(np.float64) / S
                if abs(_polygon_area(xy)) < 15: continue
                d = _fit_contour(xy, 0.08, 0.15, 60.0)
                if d: pp.append(d); nodes += d.count("C") + d.count("L") + d.count("M")
            if pp:
                d = " ".join(pp)
                if op < 1.0: parts.append(f'<path d="{d}" fill="{color}" fill-rule="evenodd" opacity="{op:.2f}"/>')
                else: parts.append(f'<path d="{d}" fill="{color}" fill-rule="evenodd"/>')
    parts.append("</svg>")
    return "\n".join(parts), nodes


def make_svg_adaptive(pre, scale=4, inner_iso=0.55, outer_iso=0.25,
                       halo_iso=0.18, halo_opacity=0.45, smooth_sigma=1.0):
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
        # Gradient-based adaptive iso
        gx = cv2.Sobel(soft.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(soft.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(gx * gx + gy * gy)
        mx = grad.max()
        grad_norm = grad / mx if mx > 0 else grad
        adaptive_iso = outer_iso + grad_norm * (inner_iso - outer_iso)
        adaptive_iso = cv2.GaussianBlur(adaptive_iso, (0, 0), sigmaX=smooth_sigma * S)
        shifted = soft - adaptive_iso
        # Halo
        if halo_iso is not None and halo_opacity > 0:
            for c in find_contours(soft, halo_iso):
                if len(c) < 4: continue
                xy = c[:, ::-1].astype(np.float64) / S
                if abs(_polygon_area(xy)) < 15: continue
                d = _fit_contour(xy, 0.08, 0.15, 60.0)
                if d:
                    parts.append(f'<path d="{d}" fill="{color}" fill-rule="evenodd" opacity="{halo_opacity:.2f}"/>')
                    nodes += d.count("C") + d.count("L") + d.count("M")
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
    svg_c, _ = make_svg_production(pre_crop)
    rend_c = render_svg(svg_c, pre_crop['w'], pre_crop['h'])
    prod_crop_blur = measure(crop, rend_c, 1.5)
    prod_crop_raw = measure(crop, rend_c, 0)

    prod_mahal_blur = prod_mahal_raw = 0
    if pre_mahal:
        svg_m, _ = make_svg_production(pre_mahal)
        rend_m = render_svg(svg_m, pre_mahal['w'], pre_mahal['h'])
        prod_mahal_blur = measure(mahal, rend_m, 1.5)
        prod_mahal_raw = measure(mahal, rend_m, 0)

    print(f"\nProduction baseline:  crop blur={prod_crop_blur:.4f} raw={prod_crop_raw:.4f}"
          f"  mahal blur={prod_mahal_blur:.4f} raw={prod_mahal_raw:.4f}")
    print()

    # Parameter sweep for adaptive iso-field
    configs = []
    for inner in [0.50, 0.55, 0.60, 0.65]:
        for outer in [0.20, 0.25, 0.30, 0.35]:
            if inner <= outer + 0.10: continue  # too narrow
            for halo_iso in [0.15, 0.20]:
                for halo_op in [0.40, 0.50]:
                    for sigma in [0.8, 1.0, 1.5]:
                        configs.append((inner, outer, halo_iso, halo_op, sigma))

    print(f"Sweeping {len(configs)} configurations on crop...")
    print(f"{'inner':>5s} {'outer':>5s} {'h_iso':>5s} {'h_op':>4s} {'sig':>4s}  "
          f"blur    raw    Δblur  Δraw")
    print("-" * 62)

    # Phase 1: sweep on crop only (fast)
    best_score = -1
    best_cfg = None
    top_results = []

    for inner, outer, h_iso, h_op, sigma in configs:
        svg, _ = make_svg_adaptive(pre_crop, inner_iso=inner, outer_iso=outer,
                                    halo_iso=h_iso, halo_opacity=h_op, smooth_sigma=sigma)
        rend = render_svg(svg, pre_crop['w'], pre_crop['h'])
        sb = measure(crop, rend, 1.5)
        sr = measure(crop, rend, 0)
        db = sb - prod_crop_blur
        dr = sr - prod_crop_raw

        # Score: weighted combo of blur and raw improvement
        score = db + 2 * dr  # raw matters more for zoom-in
        top_results.append((score, inner, outer, h_iso, h_op, sigma, sb, sr, db, dr))

    # Sort by score and show top 20
    top_results.sort(key=lambda x: -x[0])
    print("\nTop 20 configurations on crop:")
    print(f"{'#':>3s} {'inner':>5s} {'outer':>5s} {'h_iso':>5s} {'h_op':>4s} {'sig':>4s}  "
          f"blur    raw    Δblur   Δraw    score")
    print("-" * 72)
    for rank, (score, inner, outer, h_iso, h_op, sigma, sb, sr, db, dr) in enumerate(top_results[:20]):
        print(f"  {rank+1:>2d}  {inner:.2f}  {outer:.2f}  {h_iso:.2f}  {h_op:.1f}  {sigma:.1f}  "
              f"{sb:.4f}  {sr:.4f}  {db:+.4f}  {dr:+.4f}  {score:+.4f}")

    # Phase 2: test top 5 on mahal too
    if pre_mahal:
        print(f"\nTop 5 on both images:")
        print(f"{'#':>3s} {'inner':>5s} {'outer':>5s} {'h_iso':>5s} {'h_op':>4s} {'sig':>4s}  "
              f"crop_b  crop_r  mah_b   mah_r   avg_Δr")
        print("-" * 78)

        for rank in range(min(5, len(top_results))):
            _, inner, outer, h_iso, h_op, sigma, sb_c, sr_c, _, _ = top_results[rank]

            svg_m, _ = make_svg_adaptive(pre_mahal, inner_iso=inner, outer_iso=outer,
                                          halo_iso=h_iso, halo_opacity=h_op, smooth_sigma=sigma)
            rend_m = render_svg(svg_m, pre_mahal['w'], pre_mahal['h'])
            sb_m = measure(mahal, rend_m, 1.5)
            sr_m = measure(mahal, rend_m, 0)

            avg_dr = ((sr_c - prod_crop_raw) + (sr_m - prod_mahal_raw)) / 2
            print(f"  {rank+1:>2d}  {inner:.2f}  {outer:.2f}  {h_iso:.2f}  {h_op:.1f}  {sigma:.1f}  "
                  f"{sb_c:.4f}  {sr_c:.4f}  {sb_m:.4f}  {sr_m:.4f}  {avg_dr:+.4f}")

    print("\nDone.")
