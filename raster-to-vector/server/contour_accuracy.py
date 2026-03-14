"""Investigate contour placement accuracy.

The SSIM gap is 100% shape error, not color error.
5 colors at pixel-perfect placement = 0.9986.
Current = 0.983.

Questions:
1. Is the soft field correct? (close to ideal membership)
2. Is the iso-contour threshold optimal?
3. Are the Bezier fits shifting boundaries?
4. Is the background detection cutting off regions?
"""
import cv2
import numpy as np
from app.core.multilevel import (
    _merge_close_clusters, _compute_edge_weight, detect_background,
    _bgr_to_hex, VectorLayer, MultilevelResult, generate_svg,
)
from app.core.comparison import compare
from skimage.metrics import structural_similarity
import cairosvg
from io import BytesIO
from PIL import Image


def render_svg(svg_str, width, height):
    png = cairosvg.svg2png(bytestring=svg_str.encode(), output_width=width, output_height=height)
    arr = np.array(Image.open(BytesIO(png)))
    return cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2BGR)


ref = cv2.imread("/Users/alexwaldmann/Desktop/SVG-gen/Ref.png")
h, w = ref.shape[:2]
crop = ref[0:min(410, h), max(0, (w-564)//2):max(0, (w-564)//2)+564]
ih, iw = crop.shape[:2]

bg_color, _ = detect_background(crop)
bg_hex = _bgr_to_hex(bg_color)
edge_weight = _compute_edge_weight(crop)

denoised_km = cv2.bilateralFilter(crop, 15, 12, 30)
denoised_dist = cv2.bilateralFilter(crop, 7, 5, 20)

pixels = denoised_km.reshape(-1, 3).astype(np.float32)
K = 24
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
_, labels, centers = cv2.kmeans(pixels, K, None, criteria, 6, cv2.KMEANS_PP_CENTERS)
centers, labels = _merge_close_clusters(centers, labels.flatten(), ih, iw, threshold=60.0)
nc = len(centers)
cu = centers.astype(np.uint8)
cf = centers.astype(np.float32)

bg_dists = np.array([np.linalg.norm(cf[k] - bg_color.astype(np.float32)) for k in range(nc)])
bci = int(np.argmin(bg_dists))
bg_cluster = bci if bg_dists[bci] < 40.0 else -1

print(f"Clusters: {nc}, bg_cluster: {bg_cluster}")
for k in range(nc):
    hex_c = _bgr_to_hex(cu[k])
    is_bg = " (BG)" if k == bg_cluster else ""
    gray = int(cv2.cvtColor(cu[k].reshape(1,1,3), cv2.COLOR_BGR2GRAY)[0,0])
    print(f"  Cluster {k}: {hex_c} gray={gray}{is_bg}")

# Compute soft fields
p3d = denoised_dist.astype(np.float32)
dm = np.empty((ih, iw, nc), dtype=np.float32)
for k in range(nc):
    diff = p3d - cf[k]
    dm[:, :, k] = np.sqrt(np.sum(diff * diff, axis=2))

print("\n=== Experiment: pixel-perfect label map vs SVG ===")

# Create pixel-perfect quantized image (the theoretical ceiling)
# Assign each pixel to nearest cluster center
pixel_labels = np.argmin(dm, axis=2)  # shape (h, w)
quantized = cu[pixel_labels]  # shape (h, w, 3)

src_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
src_b = cv2.GaussianBlur(src_gray, (0,0), 1.5)
q_gray = cv2.cvtColor(quantized, cv2.COLOR_BGR2GRAY).astype(np.float32)
q_b = cv2.GaussianBlur(q_gray, (0,0), 1.5)
q_ssim = structural_similarity(src_b, q_b, data_range=255)
print(f"  Pixel-perfect quantized SSIM: {q_ssim:.4f}")

# Now test: what if we create SVG from the pixel-perfect assignment
# but render it as a raster (direct array, no SVG intermediary)?
# This measures: raster quantization SSIM vs SVG rendering SSIM
# The difference = shape/boundary error from the SVG pipeline

# Create a synthetic "SVG" that's just the quantized raster
# This bypasses all contour extraction and curve fitting
from app.core.multilevel import _fit_contour, _polygon_area
from skimage.measure import find_contours

grays = np.array([int(cv2.cvtColor(c.reshape(1,1,3), cv2.COLOR_BGR2GRAY)[0,0]) for c in cu])
order = np.argsort(-grays)

print("\n=== Experiment: vary simplify_epsilon and max_error ===")
# Maybe the curve fitting is too aggressive with simplification

for eps in [0.05, 0.10, 0.15, 0.30, 0.50, 1.0]:
    for max_err in [0.1, 0.2, 0.5, 1.0]:
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
                    d = _fit_contour(xy, eps, max_err, 60.0)
                    if d: ip.append(d)
                if ip:
                    lp.append(" ".join(ip)); lo.append(op)
            if lp:
                layers.append(VectorLayer(paths=lp, opacities=lo, color=ch))

        mr = MultilevelResult(layers=layers, width=iw, height=ih,
                              background_color=bg_hex, path_count=0, node_count=0)
        svg = generate_svg(mr, remove_background=False)
        comp = compare(crop, svg)
        print(f"  eps={eps:.2f} maxerr={max_err:.1f}: SSIM={comp.ssim_score:.4f} MAE={comp.mae:.2f}")
    # Skip less interesting combos for speed
    if eps > 0.3:
        break

print("\n=== Experiment: vary soft field sigma ===")
for sigma_c in [0.3, 0.5, 0.6, 0.8, 1.0]:
    for sigma_s in [1.0, 1.5, 2.0, 3.0]:
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
            sc = cv2.GaussianBlur(sr, (0,0), sigmaX=sigma_c)
            ss = cv2.GaussianBlur(sr, (0,0), sigmaX=sigma_s)
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

        mr = MultilevelResult(layers=layers, width=iw, height=ih,
                              background_color=bg_hex, path_count=0, node_count=0)
        svg = generate_svg(mr, remove_background=False)
        comp = compare(crop, svg)
        print(f"  sigma_crisp={sigma_c:.1f} sigma_smooth={sigma_s:.1f}: SSIM={comp.ssim_score:.4f}")
