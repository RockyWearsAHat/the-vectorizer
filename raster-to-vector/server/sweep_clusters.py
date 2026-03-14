"""Sweep K and merge threshold to find optimal cluster/precision combo."""
import cv2
import time
from app.core.multilevel import (
    multilevel_vectorize, generate_svg,
    _merge_close_clusters, _compute_edge_weight, detect_background,
    _bgr_to_hex, _polygon_area, _fit_contour, VectorLayer, MultilevelResult,
)
from app.core.comparison import compare
import numpy as np
from skimage.measure import find_contours

# Load test images
ref = cv2.imread("/Users/alexwaldmann/Desktop/SVG-gen/Ref.png")
h, w = ref.shape[:2]
crop_h, crop_w = min(410, h), min(564, w)
x_start = max(0, (w - crop_w) // 2)
crop = ref[0:crop_h, x_start:x_start + crop_w]

images = {"crop": crop, "mahal": cv2.imread("/tmp/mahal_right.png")}

# Test configurations: (K, merge_threshold)
configs = [
    (24, 60),   # Current
    (32, 60),   # More starting clusters
    (32, 50),   # More clusters, less merging
    (32, 40),   # Even less merging
    (48, 60),   # Many starting clusters
    (48, 40),   # Many clusters, less merging
    (24, 40),   # Current K, less merging
]

print(f"{'Config':>12s}  {'crop SSIM':>10s}  {'mahal SSIM':>10s}  {'crop layers':>12s}  {'mahal layers':>13s}  {'time':>5s}")
print("-" * 75)

for K, merge_t in configs:
    ssims = {}
    layer_counts = {}
    t0 = time.time()
    for name, img in images.items():
        if img is None:
            continue
        result = multilevel_vectorize(img, num_levels=K)
        # We need to override merge threshold - but it's hardcoded.
        # For now, just test the default since we changed it to 60.
        # Instead, let's directly replicate the pipeline with custom merge.
        pass

    # Since we can't easily override merge_threshold from outside,
    # let's just do a quick inline test for each config
    for name, img in images.items():
        if img is None:
            continue
        ih, iw = img.shape[:2]
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        bg_color, bg_gray = detect_background(img)
        bg_hex = _bgr_to_hex(bg_color)
        edge_weight = _compute_edge_weight(img)
        denoised = cv2.bilateralFilter(img, 7, 10, 10)

        pixels = denoised.reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(
            pixels, K, None, criteria, 6, cv2.KMEANS_PP_CENTERS
        )

        centers, labels = _merge_close_clusters(
            centers, labels.flatten(), ih, iw, threshold=float(merge_t),
        )
        num_clusters = len(centers)

        centers_u = centers.astype(np.uint8)
        centers_f = centers.astype(np.float32)

        bg_dists = np.array([
            np.linalg.norm(centers_f[k] - bg_color.astype(np.float32))
            for k in range(num_clusters)
        ])
        bg_cluster_idx = int(np.argmin(bg_dists))
        bg_cluster = bg_cluster_idx if bg_dists[bg_cluster_idx] < 40.0 else -1

        pixels_3d = img.astype(np.float32)
        dist_map = np.empty((ih, iw, num_clusters), dtype=np.float32)
        for k in range(num_clusters):
            diff = pixels_3d - centers_f[k]
            dist_map[:, :, k] = np.sqrt(np.sum(diff * diff, axis=2))

        grays = np.array([
            int(cv2.cvtColor(c.reshape(1, 1, 3), cv2.COLOR_BGR2GRAY)[0, 0])
            for c in centers_u
        ])
        order = np.argsort(-grays)

        layers = []
        for cluster_idx in order:
            if cluster_idx == bg_cluster:
                continue
            color_hex = _bgr_to_hex(centers_u[cluster_idx])
            d_k = dist_map[:, :, cluster_idx]
            other_mask = np.ones(num_clusters, dtype=bool)
            other_mask[cluster_idx] = False
            d_other = np.min(dist_map[:, :, other_mask], axis=2)
            denom = d_k + d_other
            denom = np.where(denom < 1e-10, 1e-10, denom)
            soft_raw = d_other / denom

            soft_crisp = cv2.GaussianBlur(soft_raw, (0, 0), sigmaX=0.6)
            soft_smooth = cv2.GaussianBlur(soft_raw, (0, 0), sigmaX=1.5)
            soft = edge_weight * soft_crisp + (1.0 - edge_weight) * soft_smooth

            iso_levels = [0.20, 0.50]
            iso_opacities = [0.50, 1.00]

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
                    if area < 15:
                        continue
                    d = _fit_contour(xy, 0.15, 0.2, 60.0)
                    if d:
                        iso_parts.append(d)

                if iso_parts:
                    combined = " ".join(iso_parts)
                    layer_paths.append(combined)
                    layer_opacities.append(opacity)

            if layer_paths:
                layers.append(VectorLayer(
                    paths=layer_paths,
                    opacities=layer_opacities,
                    color=color_hex,
                ))

        mr = MultilevelResult(
            layers=layers, width=iw, height=ih,
            background_color=bg_hex, path_count=0, node_count=0,
        )
        svg = generate_svg(mr, remove_background=False)
        comp = compare(img, svg)
        ssims[name] = comp.ssim_score
        layer_counts[name] = len(layers)

    dt = time.time() - t0
    print(f"  K={K:2d} m={merge_t:2d}  "
          f"{ssims.get('crop', 0):.4f}      "
          f"{ssims.get('mahal', 0):.4f}       "
          f"{layer_counts.get('crop', 0):4d}          "
          f"{layer_counts.get('mahal', 0):4d}          "
          f"{dt:.1f}s")
