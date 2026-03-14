"""Precise sweep of iso thresholds, halo opacity, and soft sigma.

Theoretical ceiling: 0.9985 (pixel-perfect quantized)
Current: ~0.983
Gap: 0.015 — entirely shape/boundary placement

Using sigma_crisp=0.8, sigma_smooth=1.0 (better than current 0.6/1.5).
"""
import cv2
import numpy as np
from app.core.multilevel import (
    _merge_close_clusters, _compute_edge_weight, detect_background,
    _bgr_to_hex, _polygon_area, _fit_contour, VectorLayer, MultilevelResult,
    generate_svg,
)
from app.core.comparison import compare
from skimage.measure import find_contours


ref = cv2.imread("/Users/alexwaldmann/Desktop/SVG-gen/Ref.png")
h, w = ref.shape[:2]
crop = ref[0:min(410, h), max(0, (w-564)//2):max(0, (w-564)//2)+564]
mahal = cv2.imread("/tmp/mahal_right.png")
images = {"crop": crop, "mahal": mahal}

# Precompute per image (fix K-means seed for consistency)
precomp = {}
for name, img in images.items():
    ih, iw = img.shape[:2]
    bg_color, _ = detect_background(img)
    bg_hex = _bgr_to_hex(bg_color)
    edge_weight = _compute_edge_weight(img)
    denoised_km = cv2.bilateralFilter(img, 15, 12, 30)
    denoised_dist = cv2.bilateralFilter(img, 7, 5, 20)

    pixels = denoised_km.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(pixels, 24, None, criteria, 6, cv2.KMEANS_PP_CENTERS)
    centers, labels = _merge_close_clusters(centers, labels.flatten(), ih, iw, threshold=60.0)
    nc = len(centers)
    cu = centers.astype(np.uint8)
    cf = centers.astype(np.float32)

    bg_dists = np.array([np.linalg.norm(cf[k] - bg_color.astype(np.float32)) for k in range(nc)])
    bci = int(np.argmin(bg_dists))
    bg_cluster = bci if bg_dists[bci] < 40.0 else -1

    p3d = denoised_dist.astype(np.float32)
    dm = np.empty((ih, iw, nc), dtype=np.float32)
    for k in range(nc):
        diff = p3d - cf[k]
        dm[:, :, k] = np.sqrt(np.sum(diff * diff, axis=2))

    grays = np.array([int(cv2.cvtColor(c.reshape(1,1,3), cv2.COLOR_BGR2GRAY)[0,0]) for c in cu])
    order = np.argsort(-grays)

    precomp[name] = {
        "img": img, "ih": ih, "iw": iw, "bg_hex": bg_hex,
        "cu": cu, "nc": nc, "bg_cluster": bg_cluster,
        "order": order, "dm": dm, "edge_weight": edge_weight,
    }


def run_config(sigma_c, sigma_s, iso_levels, iso_opacities):
    ssims = {}
    for name, pc in precomp.items():
        layers = []
        for ci in pc["order"]:
            if ci == pc["bg_cluster"]:
                continue
            ch = _bgr_to_hex(pc["cu"][ci])
            dk = pc["dm"][:, :, ci]
            om = np.ones(pc["nc"], dtype=bool); om[ci] = False
            do = np.min(pc["dm"][:, :, om], axis=2)
            den = dk + do; den = np.where(den < 1e-10, 1e-10, den)
            sr = do / den
            sc = cv2.GaussianBlur(sr, (0,0), sigmaX=sigma_c)
            ss = cv2.GaussianBlur(sr, (0,0), sigmaX=sigma_s)
            soft = pc["edge_weight"] * sc + (1.0 - pc["edge_weight"]) * ss

            lp, lo_list = [], []
            for iso, op in zip(iso_levels, iso_opacities):
                cl = find_contours(soft, iso)
                ip = []
                for c in cl:
                    if len(c) < 4: continue
                    xy = c[:, ::-1].astype(np.float64)
                    if abs(_polygon_area(xy)) < 15: continue
                    d = _fit_contour(xy, 0.15, 0.2, 60.0)
                    if d: ip.append(d)
                if ip:
                    lp.append(" ".join(ip)); lo_list.append(op)
            if lp:
                layers.append(VectorLayer(paths=lp, opacities=lo_list, color=ch))

        mr = MultilevelResult(layers=layers, width=pc["iw"], height=pc["ih"],
                              background_color=pc["bg_hex"], path_count=0, node_count=0)
        svg = generate_svg(mr, remove_background=False)
        comp = compare(pc["img"], svg)
        ssims[name] = comp.ssim_score
    return ssims


print(f"{'Config':>45s}  {'crop':>6s}  {'mahal':>6s}  {'avg':>6s}")
print("-" * 70)

# Phase 1: sigma sweep with current iso config
print("--- Phase 1: sigma sweep ---")
for sc in [0.6, 0.7, 0.8, 0.9, 1.0]:
    for ss in [0.8, 1.0, 1.2, 1.5]:
        s = run_config(sc, ss, [0.20, 0.50], [0.50, 1.00])
        avg = np.mean(list(s.values()))
        marker = " ***" if avg >= 0.984 else ""
        print(f"  sc={sc:.1f} ss={ss:.1f} iso=[0.20,0.50] op=[0.50,1.00]"
              f"  {s['crop']:.4f}  {s['mahal']:.4f}  {avg:.4f}{marker}")

# Phase 2: iso sweep with best sigma
print("\n--- Phase 2: iso sweep (sigma_c=0.8, sigma_s=1.0) ---")
for halo_iso in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
    for halo_op in [0.30, 0.40, 0.50, 0.60]:
        s = run_config(0.8, 1.0, [halo_iso, 0.50], [halo_op, 1.00])
        avg = np.mean(list(s.values()))
        marker = " ***" if avg >= 0.984 else ""
        print(f"  sc=0.8 ss=1.0 iso=[{halo_iso:.2f},0.50] op=[{halo_op:.2f},1.00]"
              f"  {s['crop']:.4f}  {s['mahal']:.4f}  {avg:.4f}{marker}")

# Phase 3: 3-level with best sigma (most promising combos only)
print("\n--- Phase 3: 3-level iso sweep ---")
for o_iso in [0.10, 0.15, 0.20]:
    for m_iso in [0.30, 0.35, 0.40]:
        for o_op in [0.25, 0.35, 0.45]:
            m_op = (o_op + 1.0) / 2  # midpoint
            s = run_config(0.8, 1.0, [o_iso, m_iso, 0.50], [o_op, m_op, 1.00])
            avg = np.mean(list(s.values()))
            if avg >= 0.984:
                print(f"  iso=[{o_iso:.2f},{m_iso:.2f},0.50] "
                      f"op=[{o_op:.2f},{m_op:.2f},1.00]"
                      f"  {s['crop']:.4f}  {s['mahal']:.4f}  {avg:.4f} ***")
