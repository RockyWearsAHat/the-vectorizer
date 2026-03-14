"""Combine EVERY winning param into one config + try adaptive approaches.

Error budget analysis:
  - Background color mismatch: ~1% (flat fill vs subtle gradient)
  - Edge anti-aliasing: ~1% (2-level halo can't perfectly reconstruct gradient)
  - Curve fitting tolerance: ~0.5% (Beziers approximate the contour)
  - Too few clusters: ~0.5% (merge=80 collapses subtle tones)

Strategy: attack ALL of these simultaneously.
"""
import sys, os, cv2, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from app.core.comparison import compare
from app.core.multilevel import (
    detect_background, _bgr_to_hex, _merge_close_clusters,
    _fit_contour, _polygon_area, VectorLayer, MultilevelResult, generate_svg,
)
from skimage.measure import find_contours

ref = cv2.imread("/Users/alexwaldmann/Desktop/SVG-gen/Ref.png")
crop = ref[200:610, 400:964]
mahal = cv2.imread("/tmp/mahal_right.png")

def vectorize_ultimate(image_bgr, bilateral_d=7, bilateral_sc=10, bilateral_ss=10,
                       merge_thresh=80, sigma=1.0,
                       iso_levels=[0.20, 0.50], iso_opacities=[0.55, 1.00],
                       simplify_epsilon=0.15, max_error=0.2, corner_threshold=60.0,
                       min_contour_area=30, dist_from_original=True,
                       num_levels=24):
    h, w = image_bgr.shape[:2]
    if len(image_bgr.shape) == 2:
        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    bg_color, _ = detect_background(image_bgr)
    bg_hex = _bgr_to_hex(bg_color)

    if bilateral_d > 0:
        denoised = cv2.bilateralFilter(image_bgr, bilateral_d, bilateral_sc, bilateral_ss)
    else:
        denoised = image_bgr.copy()

    pixels = denoised.reshape(-1, 3).astype(np.float32)
    K = max(2, min(num_levels, 64))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(pixels, K, None, criteria, 6, cv2.KMEANS_PP_CENTERS)
    centers, labels = _merge_close_clusters(centers, labels.flatten(), h, w, threshold=merge_thresh)
    K = len(centers)
    centers_u = centers.astype(np.uint8)
    centers_f = centers.astype(np.float32)

    # Use original image for distance = sharper soft field on thin lines
    source = image_bgr.astype(np.float32) if dist_from_original else denoised.astype(np.float32)
    dist_map = np.empty((h, w, K), dtype=np.float32)
    for k in range(K):
        diff = source - centers_f[k]
        dist_map[:, :, k] = np.sqrt(np.sum(diff * diff, axis=2))

    grays = np.array([
        int(cv2.cvtColor(c.reshape(1,1,3), cv2.COLOR_BGR2GRAY)[0,0])
        for c in centers_u
    ])
    order = np.argsort(-grays)

    layers = []
    total_paths = 0
    total_nodes = 0
    for cluster_idx in order:
        color_hex = _bgr_to_hex(centers_u[cluster_idx])
        d_k = dist_map[:, :, cluster_idx]
        other_mask = np.ones(K, dtype=bool)
        other_mask[cluster_idx] = False
        d_other = np.min(dist_map[:, :, other_mask], axis=2)
        denom = d_k + d_other
        denom = np.where(denom < 1e-10, 1e-10, denom)
        soft = d_other / denom
        soft = cv2.GaussianBlur(soft, (0, 0), sigmaX=sigma)

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
    )

# ----- COMBINATIONS -----
configs = [
    ("1_baseline",  dict()),  # Current production
    
    # Best from all sweeps combined
    ("2_BEST_all",  dict(bilateral_sc=10, bilateral_ss=10, dist_from_original=True,
                         simplify_epsilon=0.15, max_error=0.2, min_contour_area=15)),
    
    # More clusters (lower merge = finer color steps)
    ("3_moreClust", dict(bilateral_sc=10, bilateral_ss=10, dist_from_original=True,
                         simplify_epsilon=0.15, max_error=0.2, min_contour_area=15,
                         merge_thresh=50)),
    
    # More clusters + 3 iso levels
    ("4_3iso+clust", dict(bilateral_sc=10, bilateral_ss=10, dist_from_original=True,
                          simplify_epsilon=0.15, max_error=0.2, min_contour_area=15,
                          merge_thresh=50,
                          iso_levels=[0.15, 0.35, 0.55], iso_opacities=[0.30, 0.65, 1.00])),
    
    # More K-means initial clusters (48 instead of 24) + tight merge
    ("5_K48+m50",   dict(bilateral_sc=10, bilateral_ss=10, dist_from_original=True,
                         simplify_epsilon=0.15, max_error=0.2, min_contour_area=15,
                         num_levels=48, merge_thresh=50)),
    
    # K48 + 3 iso
    ("6_K48+3iso",  dict(bilateral_sc=10, bilateral_ss=10, dist_from_original=True,
                         simplify_epsilon=0.15, max_error=0.2, min_contour_area=15,
                         num_levels=48, merge_thresh=50,
                         iso_levels=[0.15, 0.35, 0.55], iso_opacities=[0.30, 0.65, 1.00])),
    
    # More clusters + tighter sigma for even crisper lines
    ("7_sigma0.8",  dict(bilateral_sc=10, bilateral_ss=10, dist_from_original=True,
                         simplify_epsilon=0.15, max_error=0.2, min_contour_area=15,
                         merge_thresh=50, sigma=0.8)),
    
    # K48 + merge40 (even more clusters)
    ("8_K48+m40",   dict(bilateral_sc=10, bilateral_ss=10, dist_from_original=True,
                         simplify_epsilon=0.15, max_error=0.2, min_contour_area=15,
                         num_levels=48, merge_thresh=40)),
    
    # K48 + merge40 + 3 iso  
    ("9_K48m40+3i", dict(bilateral_sc=10, bilateral_ss=10, dist_from_original=True,
                         simplify_epsilon=0.15, max_error=0.2, min_contour_area=15,
                         num_levels=48, merge_thresh=40,
                         iso_levels=[0.15, 0.35, 0.55], iso_opacities=[0.30, 0.65, 1.00])),
    
    # Extreme: K64 + merge30 + 3iso + area10
    ("10_EXTREME",  dict(bilateral_sc=10, bilateral_ss=10, dist_from_original=True,
                         simplify_epsilon=0.10, max_error=0.15, min_contour_area=10,
                         num_levels=64, merge_thresh=30,
                         iso_levels=[0.15, 0.35, 0.55], iso_opacities=[0.30, 0.65, 1.00])),
]

defaults = dict(bilateral_d=7, bilateral_sc=20, bilateral_ss=20, merge_thresh=80,
                sigma=1.0, iso_levels=[0.20, 0.50], iso_opacities=[0.55, 1.00],
                simplify_epsilon=0.3, max_error=0.3, corner_threshold=60.0,
                min_contour_area=30, dist_from_original=False, num_levels=24)

images = [("crop", crop), ("mahal", mahal), ("ref", ref)]

print(f"{'config':<16}", end="")
for name, _ in images:
    print(f" {name:>6}", end="")
print(f"  {'avg':>6} {'K':>3} {'paths':>5} {'nodes':>6}")
print("-" * 75)

for label, overrides in configs:
    params = {**defaults, **overrides}
    ssims = []
    line = f"{label:<16}"
    last_result = None
    K_final = 0
    for img_name, img in images:
        result = vectorize_ultimate(img, **params)
        svg = generate_svg(result, remove_background=False)
        m = compare(img, svg)
        ssims.append(m.ssim_score)
        last_result = result
        K_final = max(K_final, len(result.layers))
        line += f" {m.ssim_score:.4f}"
    avg = np.mean(ssims)
    line += f"  {avg:.4f} {K_final:>3} {last_result.path_count:>5} {last_result.node_count:>6}"
    print(line)
