"""Adaptive edge/fill processing prototype.

Strategy:
  1. Compute edge-density map (gradient magnitude, locally averaged)
  2. Build TWO soft fields per cluster:
     - Crisp (sigma=0.6): preserves thin lines
     - Smooth (sigma=1.5): suppresses noise in flat fills
  3. Blend based on local edge density:
     soft_final = w * soft_crisp + (1-w) * soft_smooth
  4. Skip background cluster entirely (huge perf win)
  5. Extract contours from blended field → Archimedes squeeze
"""
import sys, os, cv2, numpy as np, time
sys.path.insert(0, os.path.dirname(__file__))
from app.core.comparison import compare
from app.core.multilevel import (
    detect_background, _bgr_to_hex, _merge_close_clusters,
    _fit_contour, _polygon_area, VectorLayer, MultilevelResult, generate_svg,
)
from skimage.measure import find_contours


def compute_edge_weight(image_bgr, blur_radius=15):
    """Build a [0,1] edge-density map. 1 = edge-rich, 0 = flat."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # Sobel gradient magnitude
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    # Local average (box filter) → edge density in neighbourhood
    density = cv2.blur(mag, (blur_radius, blur_radius))
    # Normalize to [0, 1]
    mx = density.max()
    if mx > 0:
        density = density / mx
    # Apply a soft threshold to separate edge from flat
    # Pixels above 0.15 density are "edge-rich"
    weight = np.clip((density - 0.05) / 0.20, 0.0, 1.0).astype(np.float32)
    return weight


def vectorize_adaptive(image_bgr, bilateral_sc=10, merge_thresh=80,
                       sigma_crisp=0.6, sigma_smooth=1.5,
                       iso_levels=[0.20, 0.50], iso_opacities=[0.55, 1.00],
                       simplify_epsilon=0.15, max_error=0.2,
                       corner_threshold=60.0, min_contour_area=15,
                       edge_blur_radius=15, skip_bg=True):
    h, w = image_bgr.shape[:2]
    if len(image_bgr.shape) == 2:
        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    bg_color, _ = detect_background(image_bgr)
    bg_hex = _bgr_to_hex(bg_color)

    # Edge weight map
    edge_w = compute_edge_weight(image_bgr, edge_blur_radius)

    # Bilateral denoise (gentler sc=10 preserves thin lines)
    denoised = cv2.bilateralFilter(image_bgr, 7, bilateral_sc, bilateral_sc)

    # K-means on denoised
    pixels = denoised.reshape(-1, 3).astype(np.float32)
    K = 24
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(pixels, K, None, criteria, 6, cv2.KMEANS_PP_CENTERS)
    centers, labels = _merge_close_clusters(centers, labels.flatten(), h, w, threshold=merge_thresh)
    K = len(centers)
    centers_u = centers.astype(np.uint8)
    centers_f = centers.astype(np.float32)

    # Identify background cluster (closest to bg_color)
    bg_dists = np.array([np.linalg.norm(centers_f[k] - bg_color.astype(np.float32)) for k in range(K)])
    bg_cluster = int(np.argmin(bg_dists))

    # Distance map from ORIGINAL image (sharper soft field on thin lines)
    pixels_3d = image_bgr.astype(np.float32)
    dist_map = np.empty((h, w, K), dtype=np.float32)
    for k in range(K):
        diff = pixels_3d - centers_f[k]
        dist_map[:, :, k] = np.sqrt(np.sum(diff * diff, axis=2))

    grays = np.array([
        int(cv2.cvtColor(c.reshape(1,1,3), cv2.COLOR_BGR2GRAY)[0,0])
        for c in centers_u
    ])
    order = np.argsort(-grays)

    layers = []
    total_paths = 0
    total_nodes = 0
    skipped = 0

    for cluster_idx in order:
        # Skip background cluster
        if skip_bg and cluster_idx == bg_cluster:
            skipped += 1
            continue

        color_hex = _bgr_to_hex(centers_u[cluster_idx])
        d_k = dist_map[:, :, cluster_idx]
        other_mask = np.ones(K, dtype=bool)
        other_mask[cluster_idx] = False
        d_other = np.min(dist_map[:, :, other_mask], axis=2)
        denom = d_k + d_other
        denom = np.where(denom < 1e-10, 1e-10, denom)
        soft_raw = d_other / denom

        # ADAPTIVE: two soft fields with different sigma
        soft_crisp = cv2.GaussianBlur(soft_raw, (0, 0), sigmaX=sigma_crisp)
        soft_smooth = cv2.GaussianBlur(soft_raw, (0, 0), sigmaX=sigma_smooth)

        # Blend: edge-rich areas get crisp, flat areas get smooth
        soft = edge_w * soft_crisp + (1.0 - edge_w) * soft_smooth

        layer_paths = []
        layer_opacities = []
        for iso, opacity in zip(iso_levels, iso_opacities):
            contour_list = find_contours(soft, iso)
            iso_parts = []
            for contour in contour_list:
                if len(contour) < 4:
                    continue
                xy = contour[:, ::-1].astype(np.float64)
                area = abs(_polygon_area(xy))
                if area < min_contour_area:
                    continue
                d = _fit_contour(xy, simplify_epsilon, max_error, corner_threshold)
                if d:
                    iso_parts.append(d)
            if iso_parts:
                combined = " ".join(iso_parts)
                layer_paths.append(combined)
                layer_opacities.append(opacity)
                total_paths += 1
                total_nodes += combined.count("C") + combined.count("M")

        if layer_paths:
            layers.append(VectorLayer(paths=layer_paths, opacities=layer_opacities, color=color_hex))

    return MultilevelResult(
        layers=layers, width=w, height=h, background_color=bg_hex,
        path_count=total_paths, node_count=total_nodes,
    ), skipped


# ---- Test ----
ref = cv2.imread("/Users/alexwaldmann/Desktop/SVG-gen/Ref.png")
crop = ref[200:610, 400:964]
mahal = cv2.imread("/tmp/mahal_right.png")
images = [("crop", crop), ("mahal", mahal), ("ref", ref)]

configs = [
    # label, overrides
    ("baseline_prod",  dict(bilateral_sc=20, merge_thresh=80, sigma_crisp=1.0, sigma_smooth=1.0,
                            simplify_epsilon=0.3, max_error=0.3, min_contour_area=30,
                            skip_bg=False)),  # current production (uniform sigma=1.0)
    ("adaptive_v1",    dict()),  # default adaptive params
    ("adapt_s0.5/1.5", dict(sigma_crisp=0.5, sigma_smooth=1.5)),
    ("adapt_s0.8/1.2", dict(sigma_crisp=0.8, sigma_smooth=1.2)),
    ("adapt_blur20",   dict(edge_blur_radius=20)),
    ("adapt_blur10",   dict(edge_blur_radius=10)),
    ("adapt_3iso",     dict(iso_levels=[0.15, 0.35, 0.55], iso_opacities=[0.30, 0.65, 1.00])),
    ("adapt_3iso+0.5", dict(sigma_crisp=0.5, sigma_smooth=1.5,
                            iso_levels=[0.15, 0.35, 0.55], iso_opacities=[0.30, 0.65, 1.00])),
    ("adapt_m50",      dict(merge_thresh=50)),
    ("adapt_m50+3iso", dict(merge_thresh=50,
                            iso_levels=[0.15, 0.35, 0.55], iso_opacities=[0.30, 0.65, 1.00])),
]

print(f"{'config':<20}", end="")
for name, _ in images:
    print(f" {name:>6}", end="")
print(f"  {'avg':>6} {'paths':>5} {'nodes':>6} {'skip':>4} {'ms':>6}")
print("-" * 85)

for label, overrides in configs:
    ssims = []
    line = f"{label:<20}"
    last_result = None
    last_skip = 0
    t0 = time.time()
    for img_name, img in images:
        result, skipped = vectorize_adaptive(img, **overrides)
        svg = generate_svg(result, remove_background=False)
        m = compare(img, svg)
        ssims.append(m.ssim_score)
        last_result = result
        last_skip = skipped
    elapsed = (time.time() - t0) * 1000 / len(images)
    avg = np.mean(ssims)
    line += " ".join(f" {s:.4f}" for s in ssims)
    line += f"  {avg:.4f} {last_result.path_count:>5} {last_result.node_count:>6} {last_skip:>4} {elapsed:>6.0f}"
    print(line)
