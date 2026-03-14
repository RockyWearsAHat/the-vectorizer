"""Investigate: what if we fix the halo problem so more clusters actually help?

The issue: overlapping halos from different layers create color interference.
Approach: instead of uniform halo opacity, use per-pixel opacity based on
the actual soft membership value at that point.

Also test: single iso (no halo) + more clusters — the "precision" approach.
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


def run_pipeline_multi_iso(img, merge_thresh, num_levels, n_iso, min_contour_area=15):
    """Pipeline with N iso levels, each with opacity = iso_level."""
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

    # Build iso levels: evenly spaced from 0.1 to 0.5
    if n_iso == 1:
        iso_levels = [0.50]
        iso_opacities = [1.00]
    else:
        # e.g. n_iso=5: [0.10, 0.20, 0.30, 0.40, 0.50]
        # Opacities: the outermost has lowest, innermost = 1.0
        iso_levels = [0.10 + 0.40 * i / (n_iso - 1) for i in range(n_iso)]
        # Linear opacity: outer ring is faint, inner core is full
        iso_opacities = [0.20 + 0.80 * i / (n_iso - 1) for i in range(n_iso)]

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


# Load
ref_img = cv2.imread("/Users/alexwaldmann/Desktop/SVG-gen/Ref.png")
h, w = ref_img.shape[:2]
crop = ref_img[0:min(410, h), max(0, (w-564)//2):max(0, (w-564)//2)+564]
mahal = cv2.imread("/tmp/mahal_right.png")
images = {"crop": crop, "mahal": mahal}

configs = [
    # (name, K, merge, n_iso)
    ("CURRENT K24m60 2iso",   24, 60, 2),
    # Single iso (no halo) - cleaner with more colors
    ("K24 m60 1iso",          24, 60, 1),
    ("K24 m40 1iso",          24, 40, 1),
    ("K24 m30 1iso",          24, 30, 1),
    ("K32 m30 1iso",          32, 30, 1),
    ("K48 m30 1iso",          48, 30, 1),
    ("K48 m20 1iso",          48, 20, 1),
    # 3 iso levels with more colors
    ("K24 m40 3iso",          24, 40, 3),
    ("K32 m40 3iso",          32, 40, 3),
    # 5 iso levels (smooth gradient) with few colors
    ("K24 m60 5iso",          24, 60, 5),
    ("K24 m50 5iso",          24, 50, 5),
    ("K24 m40 5iso",          24, 40, 5),
    # 5 iso with more starting colors
    ("K32 m40 5iso",          32, 40, 5),
    ("K48 m40 5iso",          48, 40, 5),
    # 7 iso levels maximum smoothness
    ("K24 m60 7iso",          24, 60, 7),
    ("K24 m50 7iso",          24, 50, 7),
]

print(f"{'Config':>22s}  {'crop':>6s}  {'mahal':>6s}  {'avg':>6s}  {'cl':>3s}  {'ly':>3s}  {'time':>5s}")
print("-" * 60)

best_avg = 0
best_name = ""

for name, K, mt, n_iso in configs:
    ssims, ncs, nls = {}, {}, {}
    t0 = time.time()
    for img_name, img in images.items():
        if img is None: continue
        ssim, mae, nc, nl, svg = run_pipeline_multi_iso(img, mt, K, n_iso)
        ssims[img_name] = ssim
        ncs[img_name] = nc
        nls[img_name] = nl
    dt = time.time() - t0
    avg = np.mean(list(ssims.values()))
    if avg > best_avg:
        best_avg = avg
        best_name = name
    marker = " ***" if avg > 0.988 else ""
    print(f"  {name:>20s}  {ssims.get('crop',0):.4f}  {ssims.get('mahal',0):.4f}  {avg:.4f}  "
          f"{ncs.get('crop','?'):>3}  {nls.get('crop','?'):>3}  {dt:.1f}s{marker}")

print(f"\nBest: {best_name} = {best_avg:.4f}")
