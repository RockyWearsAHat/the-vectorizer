"""Test adaptive halo opacity — scale halo strength by edge density.

The problem: halo at 55% bleeds dark color into light zones (-34 bias)
and light color into dark zones (+27 bias) in flat regions far from edges.

Solution: multiply halo opacity by edge_weight. Near edges (weight~1)
the halo renders at full opacity. In flat regions (weight~0) the halo
fades out, eliminating the bleeding artifacts.
"""
import sys, os, cv2, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from app.core.comparison import compare
from app.core.multilevel import (
    detect_background, _bgr_to_hex, _merge_close_clusters, _compute_edge_weight,
    _fit_contour, _polygon_area, VectorLayer, MultilevelResult, generate_svg,
)
from skimage.measure import find_contours

ref = cv2.imread("/Users/alexwaldmann/Desktop/SVG-gen/Ref.png")
crop = ref[200:610, 400:964]
mahal = cv2.imread("/tmp/mahal_right.png")


def vectorize_adaptive_halo(image_bgr, merge_thresh=80,
                            sigma_crisp=0.6, sigma_smooth=1.5,
                            halo_iso=0.20, halo_opacity=0.55, core_iso=0.50,
                            use_3iso=False, mid_iso=0.35, mid_opacity=0.65,
                            simplify_epsilon=0.15, max_error=0.2,
                            corner_threshold=60.0, min_contour_area=15):
    h, w = image_bgr.shape[:2]
    if len(image_bgr.shape) == 2:
        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    bg_color, _ = detect_background(image_bgr)
    bg_hex = _bgr_to_hex(bg_color)
    edge_weight = _compute_edge_weight(image_bgr)

    denoised = cv2.bilateralFilter(image_bgr, 7, 10, 10)
    pixels = denoised.reshape(-1, 3).astype(np.float32)
    K = 24
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(pixels, K, None, criteria, 6, cv2.KMEANS_PP_CENTERS)
    centers, labels = _merge_close_clusters(centers, labels.flatten(), h, w, threshold=merge_thresh)
    K = len(centers)
    centers_u = centers.astype(np.uint8)
    centers_f = centers.astype(np.float32)

    bg_dists = np.array([np.linalg.norm(centers_f[k] - bg_color.astype(np.float32)) for k in range(K)])
    bg_ci = int(np.argmin(bg_dists))
    bg_cluster = bg_ci if bg_dists[bg_ci] < 40.0 else -1

    pixels_3d = image_bgr.astype(np.float32)
    dist_map = np.empty((h, w, K), dtype=np.float32)
    for k in range(K):
        diff = pixels_3d - centers_f[k]
        dist_map[:, :, k] = np.sqrt(np.sum(diff * diff, axis=2))

    grays = np.array([int(cv2.cvtColor(c.reshape(1,1,3), cv2.COLOR_BGR2GRAY)[0,0]) for c in centers_u])
    order = np.argsort(-grays)

    if use_3iso:
        iso_levels = [halo_iso, mid_iso, core_iso]
        iso_opacities = [halo_opacity, mid_opacity, 1.00]
    else:
        iso_levels = [halo_iso, core_iso]
        iso_opacities = [halo_opacity, 1.00]

    layers = []
    total_paths = 0
    total_nodes = 0
    for cluster_idx in order:
        if cluster_idx == bg_cluster:
            continue
        color_hex = _bgr_to_hex(centers_u[cluster_idx])
        d_k = dist_map[:, :, cluster_idx]
        other_mask = np.ones(K, dtype=bool)
        other_mask[cluster_idx] = False
        d_other = np.min(dist_map[:, :, other_mask], axis=2)
        denom = d_k + d_other
        denom = np.where(denom < 1e-10, 1e-10, denom)
        soft_raw = d_other / denom
        soft_crisp = cv2.GaussianBlur(soft_raw, (0, 0), sigmaX=sigma_crisp)
        soft_smooth = cv2.GaussianBlur(soft_raw, (0, 0), sigmaX=sigma_smooth)
        soft = edge_weight * soft_crisp + (1.0 - edge_weight) * soft_smooth

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


images = [("crop", crop), ("mahal", mahal), ("ref", ref)]

configs = [
    ("CURRENT",             dict()),
    # merge60 was consistently good
    ("m60",                 dict(merge_thresh=60)),
    # m60 + lower opacity
    ("m60+op45",            dict(merge_thresh=60, halo_opacity=0.45)),
    ("m60+op40",            dict(merge_thresh=60, halo_opacity=0.40)),
    ("m60+op50",            dict(merge_thresh=60, halo_opacity=0.50)),
    # m60 + 3iso for smoother gradient
    ("m60+3iso",            dict(merge_thresh=60, use_3iso=True, halo_opacity=0.35, mid_opacity=0.65)),
    ("m60+3iso+hi_mid",     dict(merge_thresh=60, use_3iso=True, halo_opacity=0.30, mid_opacity=0.70)),
    # m60 + tighter halo iso
    ("m60+iso25+op50",      dict(merge_thresh=60, halo_iso=0.25, halo_opacity=0.50)),
    ("m60+iso22+op50",      dict(merge_thresh=60, halo_iso=0.22, halo_opacity=0.50)),
    # The theoretical best: m60 + 3-iso with balanced opacities
    ("m60+3iso_balanced",   dict(merge_thresh=60, use_3iso=True, 
                                 halo_iso=0.18, halo_opacity=0.30,
                                 mid_iso=0.35, mid_opacity=0.60)),
]

print(f"{'config':<22}", end="")
for name, _ in images:
    print(f" {name:>7}", end="")
print(f"  {'avg':>7} {'K':>3} {'paths':>5}")
print("-" * 68)

for label, overrides in configs:
    ssims = []
    line = f"{label:<22}"
    last_result = None
    K_count = 0
    for img_name, img in images:
        result = vectorize_adaptive_halo(img, **overrides)
        svg = generate_svg(result, remove_background=False)
        m = compare(img, svg)
        ssims.append(m.ssim_score)
        last_result = result
        K_count = max(K_count, len(result.layers))
    avg = np.mean(ssims)
    line += " ".join(f" {s:>7.4f}" for s in ssims)
    line += f"  {avg:>7.4f} {K_count:>3} {last_result.path_count:>5}"
    print(line)
