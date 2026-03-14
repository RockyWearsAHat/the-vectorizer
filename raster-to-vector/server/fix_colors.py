"""Fix the color resolution problem.

Diagnosis: only 5 colors for the whole image. Need 8-12+.
Problem: more clusters = more overlapping halos = artifacts.
Solution: try various merge thresholds + halo management.
"""
import cv2
import numpy as np
import time
from app.core.multilevel import (
    _merge_close_clusters, _compute_edge_weight, detect_background,
    _bgr_to_hex, _polygon_area, _fit_contour, VectorLayer, MultilevelResult,
    generate_svg,
)
from app.core.comparison import compare
from skimage.measure import find_contours


def run_pipeline_custom(img, merge_thresh, halo_opacity, iso_levels, iso_opacities,
                        num_levels=24, min_contour_area=15):
    h, w = img.shape[:2]
    bg_color, _ = detect_background(img)
    bg_hex = _bgr_to_hex(bg_color)
    edge_weight = _compute_edge_weight(img)

    denoised_km = cv2.bilateralFilter(img, 15, 12, 30)
    denoised_dist = cv2.bilateralFilter(img, 7, 5, 20)

    pixels = denoised_km.reshape(-1, 3).astype(np.float32)
    K = max(2, min(num_levels, 64))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(pixels, K, None, criteria, 6, cv2.KMEANS_PP_CENTERS)
    centers, labels = _merge_close_clusters(centers, labels.flatten(), h, w, threshold=merge_thresh)
    nc = len(centers)
    cu = centers.astype(np.uint8)
    cf = centers.astype(np.float32)

    bg_dists = np.array([np.linalg.norm(cf[k] - bg_color.astype(np.float32)) for k in range(nc)])
    bci = int(np.argmin(bg_dists))
    bg_cluster = bci if bg_dists[bci] < 40.0 else -1

    p3d = denoised_dist.astype(np.float32)
    dm = np.empty((h, w, nc), dtype=np.float32)
    for k in range(nc):
        diff = p3d - cf[k]
        dm[:, :, k] = np.sqrt(np.sum(diff * diff, axis=2))

    grays = np.array([int(cv2.cvtColor(c.reshape(1,1,3), cv2.COLOR_BGR2GRAY)[0,0]) for c in cu])
    order = np.argsort(-grays)

    layers = []
    for ci in order:
        if ci == bg_cluster:
            continue
        ch = _bgr_to_hex(cu[ci])
        dk = dm[:, :, ci]
        om = np.ones(nc, dtype=bool); om[ci] = False
        do = np.min(dm[:, :, om], axis=2)
        den = dk + do; den = np.where(den < 1e-10, 1e-10, den)
        sr = do / den
        sc = cv2.GaussianBlur(sr, (0,0), sigmaX=0.6)
        ss = cv2.GaussianBlur(sr, (0,0), sigmaX=1.5)
        soft = edge_weight * sc + (1.0 - edge_weight) * ss

        lp, lo = [], []
        for iso, op in zip(iso_levels, iso_opacities):
            cl = find_contours(soft, iso)
            ip = []
            for c in cl:
                if len(c) < 4: continue
                xy = c[:, ::-1].astype(np.float64)
                if abs(_polygon_area(xy)) < min_contour_area: continue
                d = _fit_contour(xy, 0.15, 0.2, 60.0)
                if d: ip.append(d)
            if ip:
                lp.append(" ".join(ip)); lo.append(op)
        if lp:
            layers.append(VectorLayer(paths=lp, opacities=lo, color=ch))

    mr = MultilevelResult(layers=layers, width=w, height=h, background_color=bg_hex, path_count=0, node_count=0)
    svg = generate_svg(mr, remove_background=False)
    comp = compare(img, svg)
    return comp.ssim_score, comp.mae, nc, len(layers), svg


# Load images
ref_img = cv2.imread("/Users/alexwaldmann/Desktop/SVG-gen/Ref.png")
h, w = ref_img.shape[:2]
crop = ref_img[0:min(410, h), max(0, (w-564)//2):max(0, (w-564)//2)+564]
mahal = cv2.imread("/tmp/mahal_right.png")
images = {"crop": crop, "mahal": mahal}

# Strategy: increase starting K + lower merge threshold to keep more colors
configs = [
    # (name, K, merge_thresh, iso_levels, iso_opacities)
    ("CURRENT: K24 m60", 24, 60, [0.20, 0.50], [0.50, 1.00]),
    ("K24 m30",          24, 30, [0.20, 0.50], [0.50, 1.00]),
    ("K24 m20",          24, 20, [0.20, 0.50], [0.50, 1.00]),
    ("K32 m30",          32, 30, [0.20, 0.50], [0.50, 1.00]),
    ("K32 m20",          32, 20, [0.20, 0.50], [0.50, 1.00]),
    ("K48 m30",          48, 30, [0.20, 0.50], [0.50, 1.00]),
    ("K48 m20",          48, 20, [0.20, 0.50], [0.50, 1.00]),
    # More clusters + 3 iso levels for smoother transitions
    ("K32 m25 3iso",     32, 25, [0.15, 0.35, 0.50], [0.35, 0.65, 1.00]),
    ("K48 m25 3iso",     48, 25, [0.15, 0.35, 0.50], [0.35, 0.65, 1.00]),
    # Core only (no halo) with many colors
    ("K32 m20 core",     32, 20, [0.50], [1.00]),
    ("K48 m20 core",     48, 20, [0.50], [1.00]),
    # Very fine: lots of clusters, minimal merge
    ("K64 m25",          64, 25, [0.20, 0.50], [0.50, 1.00]),
    ("K64 m15",          64, 15, [0.20, 0.50], [0.50, 1.00]),
]

print(f"{'Config':>22s}  {'crop':>6s}  {'mahal':>6s}  {'avg':>6s}  {'c_cl':>4s}  {'m_cl':>4s}  {'c_ly':>4s}  {'m_ly':>4s}  {'time':>5s}")
print("-" * 85)

for name, K, mt, iso_l, iso_o in configs:
    ssims, clusters, layer_counts = {}, {}, {}
    t0 = time.time()
    for img_name, img in images.items():
        if img is None: continue
        ssim, mae, nc, nl, svg = run_pipeline_custom(
            img, mt, iso_o[0], iso_l, iso_o, num_levels=K)
        ssims[img_name] = ssim
        clusters[img_name] = nc
        layer_counts[img_name] = nl
        if img_name == "crop" and name.startswith("K48 m20"):
            with open(f"/tmp/diag_{name.replace(' ', '_')}.svg", "w") as f:
                f.write(svg)
    dt = time.time() - t0
    avg = np.mean(list(ssims.values()))
    print(f"  {name:>20s}  {ssims.get('crop',0):.4f}  {ssims.get('mahal',0):.4f}  {avg:.4f}  "
          f"{clusters.get('crop','?'):>4}  {clusters.get('mahal','?'):>4}  "
          f"{layer_counts.get('crop','?'):>4}  {layer_counts.get('mahal','?'):>4}  {dt:.1f}s")
