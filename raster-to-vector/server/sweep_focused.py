"""Focused sweep: bilateral + distance-source + curve fitting params."""
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
                     min_contour_area=30, dist_from_original=False):
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

    # Key choice: compute soft field from original or denoised pixels
    source_pixels = image_bgr.astype(np.float32) if dist_from_original else denoised.astype(np.float32)
    dist_map = np.empty((h, w, K), dtype=np.float32)
    for k in range(K):
        diff = source_pixels - centers_f[k]
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

# Focused combos based on first sweep winners
configs = [
    # label, bilat_d, sc, ss, merge, sigma, iso_l, iso_o, simp_eps, max_err, dist_orig
    ("baseline",          7, 20, 20, 80, 1.0, [.20,.50], [.55,1.], .3,  .3,  False),
    # Reduce bilateral sigma_color (preserves thin line contrast)
    ("sc10",              7, 10, 10, 80, 1.0, [.20,.50], [.55,1.], .3,  .3,  False),
    # Distance from original (sharper soft field on lines)
    ("dist_orig",         7, 20, 20, 80, 1.0, [.20,.50], [.55,1.], .3,  .3,  True),
    ("sc10+dist_orig",    7, 10, 10, 80, 1.0, [.20,.50], [.55,1.], .3,  .3,  True),
    # Tighter curve fitting (less simplification = closer to contour)
    ("sc10+eps.15",       7, 10, 10, 80, 1.0, [.20,.50], [.55,1.], .15, .2,  False),
    ("sc10+orig+eps.15",  7, 10, 10, 80, 1.0, [.20,.50], [.55,1.], .15, .2,  True),
    # Smaller min area (capture small thin-line fragments)
    ("sc10+orig+area15",  7, 10, 10, 80, 1.0, [.20,.50], [.55,1.], .15, .2,  True),
    # No bilateral at all + distance from original
    ("noBilat+orig",      0,  0,  0, 80, 1.0, [.20,.50], [.55,1.], .3,  .3,  True),
    ("noBilat+orig+eps",  0,  0,  0, 80, 1.0, [.20,.50], [.55,1.], .15, .2,  True),
    # Best combo: sc10 + original dist + tight fit + lower merge
    ("BEST_combo",        7, 10, 10, 80, 1.0, [.20,.50], [.55,1.], .15, .2,  True),
]

images = [("crop", crop), ("mahal", mahal)]

print(f"{'config':<22} ", end="")
for name, _ in images:
    print(f" {name+'_SSIM':>10} {name+'_MAE':>8}", end="")
print(f"  {'avg':>6}")
print("-" * 75)

for row in configs:
    label = row[0]
    bd, bsc, bss, mt, sigma = row[1], row[2], row[3], row[4], row[5]
    iso_l, iso_o = row[6], row[7]
    simp_eps, max_err, dist_orig = row[8], row[9], row[10]
    min_area = 15 if "area15" in label else 30

    ssims = []
    line = f"{label:<22} "
    for img_name, img in images:
        result = vectorize_custom(img, bd, bsc, bss, mt, sigma, iso_l, iso_o,
                                  simplify_epsilon=simp_eps, max_error=max_err,
                                  min_contour_area=min_area, dist_from_original=dist_orig)
        svg = generate_svg(result, remove_background=False)
        m = compare(img, svg)
        ssims.append(m.ssim_score)
        line += f" {m.ssim_score:>10.4f} {m.mae:>8.2f}"
    avg = np.mean(ssims)
    line += f"  {avg:>6.4f}"
    print(line)
