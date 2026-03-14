"""Sweep sigma / iso params to find crisp-line setting."""
import sys, os, cv2, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from app.core.multilevel import multilevel_vectorize, generate_svg
from app.core.comparison import compare

# Use the Ref crop (has thin lines + broad fills, closest to user's line-art)
ref = cv2.imread("/Users/alexwaldmann/Desktop/SVG-gen/Ref.png")
crop = ref[200:610, 400:964]   # 564x410 crop
h, w = crop.shape[:2]

configs = [
    # (label, sigma, iso_levels, iso_opacities)
    ("baseline",        1.5, [0.20, 0.50], [0.55, 1.00]),
    ("sigma1.0",        1.0, [0.20, 0.50], [0.55, 1.00]),
    ("sigma0.8",        0.8, [0.20, 0.50], [0.55, 1.00]),
    ("sigma0.6",        0.6, [0.20, 0.50], [0.55, 1.00]),
    ("iso25",           1.5, [0.25, 0.50], [0.45, 1.00]),
    ("iso30",           1.5, [0.30, 0.50], [0.40, 1.00]),
    ("s1.0+iso25",      1.0, [0.25, 0.50], [0.45, 1.00]),
    ("s1.0+iso30",      1.0, [0.30, 0.50], [0.40, 1.00]),
    ("s0.8+iso25",      0.8, [0.25, 0.50], [0.45, 1.00]),
    ("s0.8+iso30",      0.8, [0.30, 0.50], [0.40, 1.00]),
    ("noHalo",          1.0, [0.50],        [1.00]),       # no halo at all
]

# We'll monkey-patch the sigma and iso params
import app.core.multilevel as ml

_orig = ml.multilevel_vectorize

def make_patched(sigma, iso_levels, iso_opacities):
    import types
    src = open(ml.__file__).read()

    def patched(image_bgr, **kw):
        # Temporarily replace the values in the module source
        old_blur = cv2.GaussianBlur
        def new_blur(img, ksize, sigmaX, **bkw):
            return old_blur(img, ksize, sigma, **bkw)

        orig_find = ml.find_contours

        # We need to patch at a finer level - let's just modify the source inline
        # Actually let's use a simpler approach: modify the function's constants
        return _orig(image_bgr, **kw)
    return patched

# The simpler approach: directly modify the source code and re-exec
# Actually, easiest: just copy the vectorize function and parameterize it

from skimage.measure import find_contours
from app.core.curve_fitting import fit_closed_bezier
from app.core.multilevel import (
    detect_background, _bgr_to_hex, _merge_close_clusters,
    _fit_contour, _polygon_area, VectorLayer, MultilevelResult, generate_svg,
)

def vectorize_custom(image_bgr, sigma, iso_levels, iso_opacities, **kw):
    h, w = image_bgr.shape[:2]
    if len(image_bgr.shape) == 2:
        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    bg_color, bg_gray = detect_background(image_bgr)
    bg_hex = _bgr_to_hex(bg_color)

    denoised = cv2.bilateralFilter(image_bgr, 7, 20, 20)
    pixels = denoised.reshape(-1, 3).astype(np.float32)
    K = 24
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(pixels, K, None, criteria, 6, cv2.KMEANS_PP_CENTERS)
    centers, labels = _merge_close_clusters(centers, labels.flatten(), h, w, threshold=80.0)
    K = len(centers)
    centers_u = centers.astype(np.uint8)
    centers_f = centers.astype(np.float32)

    pixels_3d = denoised.astype(np.float32)
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
    min_contour_area = 30
    simplify_epsilon = 0.3
    max_error = 0.3
    corner_threshold = 60.0

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


print(f"{'config':<18} {'SSIM':>6} {'MAE':>6} {'paths':>5}")
print("-" * 42)

for label, sigma, iso_levels, iso_opacities in configs:
    result = vectorize_custom(crop, sigma, iso_levels, iso_opacities)
    svg = generate_svg(result, remove_background=False)
    metrics = compare(crop, svg)
    print(f"{label:<18} {metrics.ssim_score:.4f} {metrics.mae:.2f} {result.path_count:>5}")
