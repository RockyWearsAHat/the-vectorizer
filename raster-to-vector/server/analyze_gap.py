"""Detailed analysis of remaining quality gap."""
import cv2
import numpy as np
from app.core.multilevel import multilevel_vectorize, generate_svg
from app.core.comparison import compare

img = cv2.imread("/Users/alexwaldmann/Desktop/SVG-gen/Ref.png")
h, w = img.shape[:2]
crop_h, crop_w = min(393, h), min(544, w)
x_start = max(0, (w - crop_w) // 2)
crop = img[0:crop_h, x_start:x_start+crop_w]
print(f"Crop: {crop.shape[1]}x{crop.shape[0]}\n")

# Current result
result = multilevel_vectorize(crop, num_levels=24)
svg = generate_svg(result, remove_background=False)
comp = compare(crop, svg)
print(f"Current: layers={len(result.layers)} SSIM={comp.ssim_score:.4f} MAE={comp.mae:.2f}")
for i, layer in enumerate(result.layers):
    print(f"  layer {i}: {layer.color}")

# Analyze the diff map - where is the error?
diff = comp.diff_map  # absolute difference grayscale
total_error = diff.astype(float).sum()

# Break down error by region
src_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
# Dark regions (text/outlines): gray < 100
# Mid regions (floral shading): 100 <= gray < 200
# Light regions (background): gray >= 200
for name, lo, hi in [("Dark (<100)", 0, 100), ("Mid (100-200)", 100, 200), ("Light (>200)", 200, 256)]:
    mask = (src_gray >= lo) & (src_gray < hi)
    pix_count = mask.sum()
    region_error = diff[mask].astype(float).sum()
    pct_pixels = 100.0 * pix_count / (crop.shape[0] * crop.shape[1])
    pct_error = 100.0 * region_error / total_error if total_error > 0 else 0
    mean_err = diff[mask].astype(float).mean() if pix_count > 0 else 0
    print(f"  {name:15s}: {pct_pixels:5.1f}% pixels, {pct_error:5.1f}% of error, mean_diff={mean_err:.1f}")

# What if we add more iso levels?
print("\n--- Multi-iso sweep ---")
import app.core.multilevel as ml
from app.core.multilevel import VectorLayer

def gen_multi_iso(result, iso_config, remove_bg=False):
    """Generate SVG with custom iso-level rendering."""
    w, h = result.width, result.height
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {w} {h}" width="{w}" height="{h}">',
    ]
    if not remove_bg:
        parts.append(f'<rect width="{w}" height="{h}" fill="{result.background_color}"/>')
    
    for layer in result.layers:
        # Render paths from outermost (lowest opacity) to innermost (full)
        for i, (path_d, opacity) in enumerate(zip(layer.paths, iso_config)):
            if path_d:
                if opacity >= 1.0:
                    parts.append(f'<path d="{path_d}" fill="{layer.color}" fill-rule="evenodd"/>')
                else:
                    parts.append(f'<path d="{path_d}" fill="{layer.color}" fill-rule="evenodd" opacity="{opacity:.2f}"/>')
    parts.append("</svg>")
    return "\n".join(parts)

# Test: what if we just use the current 2-level but optimize opacity further?
for opacity in [0.3, 0.4, 0.5, 0.55, 0.6, 0.7]:
    svg = gen_multi_iso(result, [opacity, 1.0], remove_bg=False)
    comp = compare(crop, svg)
    print(f"  2-iso opacity={opacity:.2f}: SSIM={comp.ssim_score:.4f} MAE={comp.mae:.2f}")

# Test: what about merge threshold?
print("\n--- Merge threshold sweep ---")
for thresh in [50, 60, 70, 80, 90, 100]:
    result_t = multilevel_vectorize(crop, num_levels=24)
    # Hack: re-run with different threshold
    # Since we can't easily pass threshold, use monkey-patching
    orig = ml._merge_close_clusters
    def patched(centers, labels_flat, h, w, threshold=thresh):
        return orig(centers, labels_flat, h, w, threshold=threshold)
    ml._merge_close_clusters = patched
    result_t = multilevel_vectorize(crop, num_levels=24)
    ml._merge_close_clusters = orig
    svg_t = generate_svg(result_t, remove_background=False)
    comp_t = compare(crop, svg_t)
    n_layers = len(result_t.layers)
    print(f"  thresh={thresh:3d}: layers={n_layers} SSIM={comp_t.ssim_score:.4f} MAE={comp_t.mae:.2f}")
