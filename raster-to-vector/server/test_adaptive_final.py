"""Final refinement: narrow sweep around the winner.

Winner: inner=0.60 outer=0.30 halo_iso=0.22 halo_op=0.50 sigma=1.0
  crop raw +0.0029, mahal raw +0.0030  avg +0.0029

Test small variations to confirm optimality.
"""

import sys, time, os
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


def meas(src, rend, sigma=1.5):
    g1, g2 = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY), cv2.cvtColor(rend, cv2.COLOR_BGR2GRAY)
    if sigma > 0:
        k = int(sigma * 6) | 1
        g1 = cv2.GaussianBlur(g1, (k, k), sigma)
        g2 = cv2.GaussianBlur(g2, (k, k), sigma)
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


def build_svg(pre, mode, inner_iso=0.60, outer_iso=0.30,
              halo_iso=0.22, halo_op=0.50, sigma=1.0, scale=4):
    h, w = pre['h'], pre['w']
    S = scale
    ew_up = cv2.resize(pre['edge_weight'], (w*S, h*S), interpolation=cv2.INTER_LINEAR)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">',
        f'<rect width="{w}" height="{h}" fill="{pre["bg_hex"]}"/>',
    ]
    nodes = 0
    for ci in pre['order']:
        if ci == pre['bg_cluster']: continue
        color = _bgr_to_hex(pre['centers_u'][ci])
        soft_up = cv2.resize(pre['soft_fields'][ci], (w*S, h*S), interpolation=cv2.INTER_LINEAR)
        sc = cv2.GaussianBlur(soft_up, (0,0), sigmaX=0.6*S)
        ss = cv2.GaussianBlur(soft_up, (0,0), sigmaX=1.5*S)
        soft = ew_up * sc + (1-ew_up) * ss

        if mode == "prod":
            for iso, op in [(0.20, 0.50), (0.47, 1.00)]:
                pp = []
                for c in find_contours(soft, iso):
                    if len(c)<4: continue
                    xy = c[:,::-1].astype(np.float64)/S
                    if abs(_polygon_area(xy))<15: continue
                    d = _fit_contour(xy, 0.08, 0.15, 60.0)
                    if d: pp.append(d); nodes += d.count("C")+d.count("L")+d.count("M")
                if pp:
                    d = " ".join(pp)
                    if op<1.0: parts.append(f'<path d="{d}" fill="{color}" fill-rule="evenodd" opacity="{op:.2f}"/>')
                    else: parts.append(f'<path d="{d}" fill="{color}" fill-rule="evenodd"/>')
        else:
            gx = cv2.Sobel(soft.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(soft.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
            grad = np.sqrt(gx*gx + gy*gy)
            mx = grad.max()
            gn = grad/mx if mx>0 else grad
            aiso = outer_iso + gn * (inner_iso - outer_iso)
            aiso = cv2.GaussianBlur(aiso, (0,0), sigmaX=sigma*S)
            shifted = soft - aiso
            if halo_iso is not None and halo_op > 0:
                pp = []
                for c in find_contours(soft, halo_iso):
                    if len(c)<4: continue
                    xy = c[:,::-1].astype(np.float64)/S
                    if abs(_polygon_area(xy))<15: continue
                    d = _fit_contour(xy, 0.08, 0.15, 60.0)
                    if d: pp.append(d); nodes += d.count("C")+d.count("L")+d.count("M")
                if pp:
                    parts.append(f'<path d="{" ".join(pp)}" fill="{color}" fill-rule="evenodd" opacity="{halo_op:.2f}"/>')
            pp = []
            for c in find_contours(shifted, 0.0):
                if len(c)<4: continue
                xy = c[:,::-1].astype(np.float64)/S
                if abs(_polygon_area(xy))<15: continue
                d = _fit_contour(xy, 0.08, 0.15, 60.0)
                if d: pp.append(d); nodes += d.count("C")+d.count("L")+d.count("M")
            if pp:
                parts.append(f'<path d="{" ".join(pp)}" fill="{color}" fill-rule="evenodd"/>')
    parts.append("</svg>")
    return "\n".join(parts), nodes


if __name__ == "__main__":
    ref = cv2.imread("/Users/alexwaldmann/Desktop/SVG-gen/Ref.png")
    crop = ref[50:460, 486:1050]
    mahal = cv2.imread("/tmp/mahal_right.png")

    pre_c = precompute(crop)
    pre_m = precompute(mahal) if mahal is not None else None

    tests = [
        ("Production (baseline)",  "prod",  {}),
        # Winner from round 1
        ("0.30-0.60 h=0.22/0.50",  "ada", dict(inner_iso=0.60, outer_iso=0.30, halo_iso=0.22, halo_op=0.50, sigma=1.0)),
        # Fine-tune inner
        ("0.30-0.58 h=0.22/0.50",  "ada", dict(inner_iso=0.58, outer_iso=0.30, halo_iso=0.22, halo_op=0.50, sigma=1.0)),
        ("0.30-0.62 h=0.22/0.50",  "ada", dict(inner_iso=0.62, outer_iso=0.30, halo_iso=0.22, halo_op=0.50, sigma=1.0)),
        # Fine-tune outer
        ("0.28-0.60 h=0.22/0.50",  "ada", dict(inner_iso=0.60, outer_iso=0.28, halo_iso=0.22, halo_op=0.50, sigma=1.0)),
        ("0.32-0.60 h=0.22/0.50",  "ada", dict(inner_iso=0.60, outer_iso=0.32, halo_iso=0.22, halo_op=0.50, sigma=1.0)),
        # Fine-tune halo iso
        ("0.30-0.60 h=0.20/0.50",  "ada", dict(inner_iso=0.60, outer_iso=0.30, halo_iso=0.20, halo_op=0.50, sigma=1.0)),
        ("0.30-0.60 h=0.24/0.50",  "ada", dict(inner_iso=0.60, outer_iso=0.30, halo_iso=0.24, halo_op=0.50, sigma=1.0)),
        # Fine-tune halo opacity
        ("0.30-0.60 h=0.22/0.45",  "ada", dict(inner_iso=0.60, outer_iso=0.30, halo_iso=0.22, halo_op=0.45, sigma=1.0)),
        ("0.30-0.60 h=0.22/0.55",  "ada", dict(inner_iso=0.60, outer_iso=0.30, halo_iso=0.22, halo_op=0.55, sigma=1.0)),
        # Fine-tune sigma
        ("0.30-0.60 h=0.22/0.50 s=0.8",  "ada", dict(inner_iso=0.60, outer_iso=0.30, halo_iso=0.22, halo_op=0.50, sigma=0.8)),
        ("0.30-0.60 h=0.22/0.50 s=1.2",  "ada", dict(inner_iso=0.60, outer_iso=0.30, halo_iso=0.22, halo_op=0.50, sigma=1.2)),
    ]

    print(f"{'Config':<37s}  c_blur  c_raw   m_blur  m_raw   nodes  avg_Δr")
    print("-" * 82)

    prod_cr = prod_mr = 0
    for name, mode, kw in tests:
        svg_c, nodes_c = build_svg(pre_c, mode, **kw)
        rend_c = render_svg(svg_c, pre_c['w'], pre_c['h'])
        cb = meas(crop, rend_c, 1.5)
        cr = meas(crop, rend_c, 0)
        mb = mr = 0
        nodes_m = 0
        if pre_m:
            svg_m, nodes_m = build_svg(pre_m, mode, **kw)
            rend_m = render_svg(svg_m, pre_m['w'], pre_m['h'])
            mb = meas(mahal, rend_m, 1.5)
            mr = meas(mahal, rend_m, 0)
        if name.startswith("Prod"):
            prod_cr, prod_mr = cr, mr
            avg_dr = 0
        else:
            avg_dr = ((cr - prod_cr) + (mr - prod_mr)) / 2
        flag = " ***" if avg_dr > 0.002 else ""
        print(f"  {name:<35s}  {cb:.4f}  {cr:.4f}  {mb:.4f}  {mr:.4f}  {nodes_c+nodes_m:>5d}  {avg_dr:+.4f}{flag}")

    print("\nDone.")
