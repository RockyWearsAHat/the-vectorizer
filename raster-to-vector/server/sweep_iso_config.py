"""Sweep multi-iso configurations to find optimal squeeze levels."""
import cv2
import numpy as np
import app.core.multilevel as ml
from app.core.multilevel import multilevel_vectorize, generate_svg
from app.core.comparison import compare

# Load images
img = cv2.imread("/Users/alexwaldmann/Desktop/SVG-gen/Ref.png")
h, w = img.shape[:2]
crop_h, crop_w = min(410, h), min(564, w)
x_start = max(0, (w - crop_w) // 2)
crop = img[0:crop_h, x_start:x_start+crop_w]

mahal = cv2.imread("/tmp/mahal_right.png")

configs = [
    # (name, iso_levels, iso_opacities)
    ("current",   [0.10, 0.20, 0.30, 0.40, 0.50, 0.65, 0.80],
                  [0.15, 0.30, 0.45, 0.60, 1.00, 1.00, 1.00]),
    
    # fewer levels, focus on outer transition
    ("5-level",   [0.10, 0.20, 0.35, 0.50, 0.70],
                  [0.20, 0.35, 0.55, 1.00, 1.00]),
    
    # 4 levels, tighter
    ("4-level",   [0.10, 0.25, 0.40, 0.55],
                  [0.20, 0.40, 0.65, 1.00]),
    
    # 6 levels, smooth ramp
    ("6-ramp",    [0.08, 0.16, 0.25, 0.35, 0.50, 0.70],
                  [0.12, 0.25, 0.40, 0.60, 1.00, 1.00]),
    
    # 5 levels, strong outer
    ("5-strong",  [0.08, 0.18, 0.30, 0.45, 0.60],
                  [0.25, 0.40, 0.60, 0.85, 1.00]),
    
    # original 2-level for comparison
    ("2-level",   [0.20, 0.50],
                  [0.55, 1.00]),
]

for name, isos, opacs in configs:
    # Monkey-patch the iso_levels and opacities
    # Need to patch the vectorize function
    orig_vectorize = ml.multilevel_vectorize.__wrapped__ if hasattr(ml.multilevel_vectorize, '__wrapped__') else None
    
    # The simplest approach: override the module-level constants
    # Actually, let's just patch the function call
    # Store original find_contours
    orig_find = ml.find_contours
    
    # We need to edit the actual source... let's use exec trickery
    # Actually the simplest: since the iso_levels are hardcoded, 
    # we need to write a proper test. Let me just read and re-exec
    pass

# Actually, let me just directly modify the iso arrays in the source
# For each config, I'll write a small wrapper
from app.core.multilevel import (
    VectorLayer, MultilevelResult, detect_background, 
    _merge_close_clusters, _bgr_to_hex, _polygon_area, _fit_contour
)
from skimage.measure import find_contours as _find_contours

def test_config(image, isos, opacs, min_contour_area=30):
    h, w = image.shape[:2]
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    bg_color, bg_gray = detect_background(image)
    bg_hex = _bgr_to_hex(bg_color)
    denoised = cv2.bilateralFilter(image, 7, 20, 20)
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
    grays = np.array([int(cv2.cvtColor(c.reshape(1,1,3), cv2.COLOR_BGR2GRAY)[0,0]) for c in centers_u])
    order = np.argsort(-grays)
    
    layers = []
    for cluster_idx in order:
        color_hex = _bgr_to_hex(centers_u[cluster_idx])
        d_k = dist_map[:, :, cluster_idx]
        other_mask = np.ones(K, dtype=bool)
        other_mask[cluster_idx] = False
        d_other = np.min(dist_map[:, :, other_mask], axis=2)
        denom = d_k + d_other
        denom = np.where(denom < 1e-10, 1e-10, denom)
        soft = d_other / denom
        soft = cv2.GaussianBlur(soft, (0, 0), sigmaX=1.5)
        
        layer_paths = []
        layer_opacities = []
        for iso, opacity in zip(isos, opacs):
            contour_list = _find_contours(soft, iso)
            iso_parts = []
            for contour in contour_list:
                if len(contour) < 4: continue
                xy = contour[:, ::-1].astype(np.float64)
                if abs(_polygon_area(xy)) < min_contour_area: continue
                d = _fit_contour(xy, 0.3, 0.3, 60.0)
                if d: iso_parts.append(d)
            if iso_parts:
                layer_paths.append(" ".join(iso_parts))
                layer_opacities.append(opacity)
        if layer_paths:
            layers.append(VectorLayer(paths=layer_paths, opacities=layer_opacities, color=color_hex))
    
    result = MultilevelResult(layers=layers, width=w, height=h, background_color=bg_hex, path_count=0, node_count=0)
    return result

def gen_svg(result, remove_bg=False):
    w, h = result.width, result.height
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">']
    if not remove_bg:
        parts.append(f'<rect width="{w}" height="{h}" fill="{result.background_color}"/>')
    for layer in result.layers:
        for path_d, opacity in zip(layer.paths, layer.opacities):
            if not path_d: continue
            if opacity >= 1.0:
                parts.append(f'<path d="{path_d}" fill="{layer.color}" fill-rule="evenodd"/>')
            else:
                parts.append(f'<path d="{path_d}" fill="{layer.color}" fill-rule="evenodd" opacity="{opacity:.2f}"/>')
    parts.append("</svg>")
    return "\n".join(parts)

print(f"{'Config':<12} {'Crop SSIM':>9} {'Crop MAE':>9} {'Mahal SSIM':>10} {'Mahal MAE':>10}")
print("-" * 55)
for name, isos, opacs in configs:
    r_crop = test_config(crop, isos, opacs)
    svg_crop = gen_svg(r_crop, remove_bg=False)
    c_crop = compare(crop, svg_crop)
    
    if mahal is not None:
        r_mahal = test_config(mahal, isos, opacs)
        svg_mahal = gen_svg(r_mahal, remove_bg=False)
        c_mahal = compare(mahal, svg_mahal)
        print(f"{name:<12} {c_crop.ssim_score:9.4f} {c_crop.mae:9.2f} {c_mahal.ssim_score:10.4f} {c_mahal.mae:10.2f}")
    else:
        print(f"{name:<12} {c_crop.ssim_score:9.4f} {c_crop.mae:9.2f}")
