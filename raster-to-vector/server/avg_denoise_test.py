"""Average over 5 runs per config to get stable denoise comparison."""
import cv2
import numpy as np
from app.core.multilevel import (
    _merge_close_clusters, _compute_edge_weight, detect_background,
    _bgr_to_hex, _polygon_area, _fit_contour, VectorLayer, MultilevelResult,
    generate_svg,
)
from app.core.comparison import compare
from skimage.measure import find_contours


def run_pipeline(img, denoised_km, denoised_dist):
    h, w = img.shape[:2]
    bg_color, _ = detect_background(img)
    bg_hex = _bgr_to_hex(bg_color)
    edge_weight = _compute_edge_weight(img)

    pixels = denoised_km.reshape(-1, 3).astype(np.float32)
    K = 24
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(pixels, K, None, criteria, 6, cv2.KMEANS_PP_CENTERS)
    centers, labels = _merge_close_clusters(centers, labels.flatten(), h, w, threshold=60.0)
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
        for iso, op in [(0.20, 0.50), (0.50, 1.00)]:
            cl = find_contours(soft, iso)
            ip = []
            for c in cl:
                if len(c) < 4: continue
                xy = c[:, ::-1].astype(np.float64)
                if abs(_polygon_area(xy)) < 15: continue
                d = _fit_contour(xy, 0.15, 0.2, 60.0)
                if d: ip.append(d)
            if ip:
                lp.append(" ".join(ip)); lo.append(op)
        if lp:
            layers.append(VectorLayer(paths=lp, opacities=lo, color=ch))

    mr = MultilevelResult(layers=layers, width=w, height=h, background_color=bg_hex, path_count=0, node_count=0)
    svg = generate_svg(mr, remove_background=False)
    comp = compare(img, svg)
    return comp.ssim_score


ref = cv2.imread("/Users/alexwaldmann/Desktop/SVG-gen/Ref.png")
h, w = ref.shape[:2]
crop = ref[0:min(410,h), max(0,(w-564)//2):max(0,(w-564)//2)+564]
mahal = cv2.imread("/tmp/mahal_right.png")
images = {"crop": crop, "mahal": mahal}

configs = [
    ("A: km(7/10/10) + orig_dist", 7, 10, 10, 0, 0, 0),
    ("B: km(7/10/10) + tight_dist(7/5/20)", 7, 10, 10, 7, 5, 20),
    ("C: km(15/12/30) + tight_dist(7/5/20)", 15, 12, 30, 7, 5, 20),
]

for name, km_d, km_sc, km_ss, dt_d, dt_sc, dt_ss in configs:
    crop_ssims = []
    mahal_ssims = []
    for run in range(5):
        for img_name, img in images.items():
            if img is None:
                continue
            dn_km = cv2.bilateralFilter(img, km_d, km_sc, km_ss)
            dn_dist = img if dt_d == 0 else cv2.bilateralFilter(img, dt_d, dt_sc, dt_ss)
            ssim = run_pipeline(img, dn_km, dn_dist)
            if img_name == "crop":
                crop_ssims.append(ssim)
            else:
                mahal_ssims.append(ssim)
    ca, ma = np.mean(crop_ssims), np.mean(mahal_ssims)
    cs, ms = np.std(crop_ssims), np.std(mahal_ssims)
    print(f"{name}")
    print(f"  crop:  {ca:.4f} +/-{cs:.4f}  [{min(crop_ssims):.4f} - {max(crop_ssims):.4f}]")
    print(f"  mahal: {ma:.4f} +/-{ms:.4f}  [{min(mahal_ssims):.4f} - {max(mahal_ssims):.4f}]")
    print(f"  avg:   {(ca+ma)/2:.4f}")
    print()
