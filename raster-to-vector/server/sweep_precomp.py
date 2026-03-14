"""Sweep pre-quantization params: bilateral filter + merge threshold."""
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

def vectorize_custom(image_bgr, bilateral_d, bilateral_sc, bilateral_ss,
                     merge_thresh, sigma, iso_levels, iso_opacities,
                     simplify_epsilon=0.3, max_error=0.3, corner_threshold=60.0,
                     min_contour_area=30):
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
    K = 24
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(pixels, K, None, criteria, 6, cv2.KMEANS_PP_CENTERS)
    centers, labels = _merge_close_clusters(centers, labels.flatten(), h, w, threshold=merge_thresh)
    K = len(centers)
    centers_u = centers.astype(np.uint8)
    centers_f = centers.astype(np.float32)

    # Distance map uses ORIGINAL image (not denoised) for soft field accuracy
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


configs = [
    # label, bilateral_d, bilateral_sc, bilateral_ss, merge_thresh, sigma, iso_levels, iso_opacities
    ("baseline",       7, 20, 20, 80, 1.0, [0.20,0.50], [0.55,1.00]),
    ("bilat_d5",       5, 20, 20, 80, 1.0, [0.20,0.50], [0.55,1.00]),
    ("bilat_d3",       3, 20, 20, 80, 1.0, [0.20,0.50], [0.55,1.00]),
    ("bilat_sc10",     7, 10, 10, 80, 1.0, [0.20,0.50], [0.55,1.00]),
    ("bilat_sc10_d5",  5, 10, 10, 80, 1.0, [0.20,0.50], [0.55,1.00]),
    ("no_bilat",       0,  0,  0, 80, 1.0, [0.20,0.50], [0.55,1.00]),
    ("merge60",        7, 20, 20, 60, 1.0, [0.20,0.50], [0.55,1.00]),
    ("merge60_d5",     5, 20, 20, 60, 1.0, [0.20,0.50], [0.55,1.00]),
    ("merge60_noB",    0,  0,  0, 60, 1.0, [0.20,0.50], [0.55,1.00]),
    ("d5+sc10+m60",    5, 10, 10, 60, 1.0, [0.20,0.50], [0.55,1.00]),
]

images = [("crop", crop), ("mahal", mahal)]

print(f"{'config':<18} ", end="")
for name, _ in images:
    print(f"  {name+'_SSIM':>11} {name+'_MAE':>9}", end="")
print(f"  {'avg_SSIM':>8}")
print("-" * 80)

for label, bd, bsc, bss, mt, sigma, iso_l, iso_o in configs:
    ssims = []
    line = f"{label:<18} "
    for img_name, img in images:
        result = vectorize_custom(img, bd, bsc, bss, mt, sigma, iso_l, iso_o)
        svg = generate_svg(result, remove_background=False)
        m = compare(img, svg)
        ssims.append(m.ssim_score)
        line += f"  {m.ssim_score:>11.4f} {m.mae:>9.2f}"
    avg = np.mean(ssims)
    line += f"  {avg:>8.4f}"
    print(line)
