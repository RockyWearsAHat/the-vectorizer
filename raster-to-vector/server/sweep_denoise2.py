"""Sweep v2: push color-gated denoising harder.

User's idea: ONLY blur between very similar shades. 
Try very large spatial kernels with very tight color gates,
LAB-space bilateral, and color pre-quantization.
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


def run_pipeline(img, denoised):
    """Run full vectorization with custom denoised input for K-means."""
    h, w = img.shape[:2]
    bg_color, _ = detect_background(img)
    bg_hex = _bgr_to_hex(bg_color)
    edge_weight = _compute_edge_weight(img)

    pixels = denoised.reshape(-1, 3).astype(np.float32)
    K = 24
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(
        pixels, K, None, criteria, 6, cv2.KMEANS_PP_CENTERS
    )
    centers, labels = _merge_close_clusters(
        centers, labels.flatten(), h, w, threshold=60.0,
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

    # Distance from ORIGINAL image
    pixels_3d = img.astype(np.float32)
    dist_map = np.empty((h, w, num_clusters), dtype=np.float32)
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
                paths=layer_paths, opacities=layer_opacities, color=color_hex,
            ))

    mr = MultilevelResult(
        layers=layers, width=w, height=h,
        background_color=bg_hex, path_count=0, node_count=0,
    )
    svg = generate_svg(mr, remove_background=False)
    comp = compare(img, svg)
    return comp.ssim_score, comp.mae, len(layers), num_clusters


def run_pipeline_denoise_both(img, denoised_kmeans, denoised_dist):
    """Run pipeline with separate denoised inputs for K-means and distance map."""
    h, w = img.shape[:2]
    bg_color, _ = detect_background(img)
    bg_hex = _bgr_to_hex(bg_color)
    edge_weight = _compute_edge_weight(img)

    pixels = denoised_kmeans.reshape(-1, 3).astype(np.float32)
    K = 24
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(
        pixels, K, None, criteria, 6, cv2.KMEANS_PP_CENTERS
    )
    centers, labels = _merge_close_clusters(
        centers, labels.flatten(), h, w, threshold=60.0,
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

    # Distance from DENOISED image (mild) for smoother soft fields
    pixels_3d = denoised_dist.astype(np.float32)
    dist_map = np.empty((h, w, num_clusters), dtype=np.float32)
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
                paths=layer_paths, opacities=layer_opacities, color=color_hex,
            ))

    mr = MultilevelResult(
        layers=layers, width=w, height=h,
        background_color=bg_hex, path_count=0, node_count=0,
    )
    svg = generate_svg(mr, remove_background=False)
    comp = compare(img, svg)
    return comp.ssim_score, comp.mae, len(layers), num_clusters


# Load images
ref = cv2.imread("/Users/alexwaldmann/Desktop/SVG-gen/Ref.png")
h, w = ref.shape[:2]
crop = ref[0:min(410, h), max(0, (w - 564) // 2):max(0, (w - 564) // 2) + 564]
mahal = cv2.imread("/tmp/mahal_right.png")
images = {"crop": crop, "mahal": mahal}


# --- Strategy definitions ---
strategies = []

# 1. Baseline: current (for reference)
strategies.append(("current bi 7/10/10",
    lambda img: cv2.bilateralFilter(img, 7, 10, 10), None))

# 2. Very tight color gate: only blur within ~5 BGR units (SD noise range)
strategies.append(("bi 15/5/30 (sc=5)",
    lambda img: cv2.bilateralFilter(img, 15, 5, 30), None))

# 3. Slightly wider color gatestrategy
strategies.append(("bi 15/8/30 (sc=8)",
    lambda img: cv2.bilateralFilter(img, 15, 8, 30), None))

# 4. Even wider spatial, very tight color
strategies.append(("bi 31/8/50",
    lambda img: cv2.bilateralFilter(img, 31, 8, 50), None))

# 5. LAB-space: convert to LAB, bilateral on L channel, convert back
def denoise_lab(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.bilateralFilter(lab[:, :, 0], 15, 10, 30)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
strategies.append(("LAB bilateral L-ch", denoise_lab, None))

# 6. LAB-space bilateral on all channels
def denoise_lab_all(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab = cv2.bilateralFilter(lab, 15, 10, 30)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
strategies.append(("LAB bilateral all", denoise_lab_all, None))

# 7. Color quantize: round each channel to nearest 4
def denoise_quantize4(img):
    return (img.astype(np.float32) / 4).round().astype(np.float32) * 4
strategies.append(("quantize/4",
    lambda img: np.clip(denoise_quantize4(img), 0, 255).astype(np.uint8), None))

# 8. Color quantize + light bilateral
def denoise_quant_bi(img):
    q = np.clip((img.astype(np.float32) / 4).round() * 4, 0, 255).astype(np.uint8)
    return cv2.bilateralFilter(q, 7, 10, 10)
strategies.append(("quantize/4 + bi 7/10/10", denoise_quant_bi, None))

# 9. Two-stage: strong bilateral for K-means, mild for distance map
# This is the "denoise_both" approach
strategies.append(("DUAL: bi15/12/30 km + bi7/5/20 dist", "dual", None))

# 10. Strong bilateral for K-means, original for distance (current flow but stronger denoise)
strategies.append(("bi 15/10/30 (km only)",
    lambda img: cv2.bilateralFilter(img, 15, 10, 30), None))


print(f"{'Strategy':>35s}  {'crop':>6s}  {'mahal':>6s}  {'avg':>6s}  {'clusters':>8s}  {'time':>5s}")
print("-" * 80)

for entry in strategies:
    strat_name = entry[0]
    ssims = {}
    cluster_info = {}
    t0 = time.time()

    for name, img in images.items():
        if img is None:
            continue

        if entry[1] == "dual":
            # Two-stage approach
            dn_km = cv2.bilateralFilter(img, 15, 12, 30)
            dn_dist = cv2.bilateralFilter(img, 7, 5, 20)
            ssim, mae, nlayers, nclusters = run_pipeline_denoise_both(img, dn_km, dn_dist)
        else:
            denoised = entry[1](img)
            ssim, mae, nlayers, nclusters = run_pipeline(img, denoised)
        ssims[name] = ssim
        cluster_info[name] = nclusters

    dt = time.time() - t0
    avg = np.mean(list(ssims.values()))
    cl = f"{cluster_info.get('crop', '?')}/{cluster_info.get('mahal', '?')}"
    print(f"  {strat_name:>33s}  {ssims.get('crop', 0):.4f}  {ssims.get('mahal', 0):.4f}  {avg:.4f}  {cl:>8s}  {dt:.1f}s")
