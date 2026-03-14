"""Sweep soft-field sigma and SVG blur to find optimal edge reconstruction."""
import cv2
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

# We need to monkey-patch to try different parameters
import app.core.multilevel as ml
from app.core.multilevel import MultilevelResult, VectorLayer
from app.core.comparison import compare

img = cv2.imread("/Users/alexwaldmann/Desktop/SVG-gen/Ref.png")
h, w = img.shape[:2]
# Simulate crop similar to user's upload
crop_h, crop_w = min(410, h), min(564, w)
x_start = max(0, (w - crop_w) // 2)
crop = img[0:crop_h, x_start:x_start+crop_w]
print(f"Testing on crop {crop.shape[1]}x{crop.shape[0]}\n")

# Save original generate_svg
_orig_gen = ml.generate_svg

def gen_svg_with_blur(result, *, remove_background=True, blur_std=0.0):
    """Generate SVG with optional feGaussianBlur."""
    w, h = result.width, result.height
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {w} {h}" width="{w}" height="{h}">',
    ]
    if blur_std > 0:
        parts.append(
            f'<defs><filter id="aa" color-interpolation-filters="sRGB">'
            f'<feGaussianBlur stdDeviation="{blur_std}"/>'
            f'</filter></defs>'
        )
    if not remove_background:
        parts.append(f'<rect width="{w}" height="{h}" fill="{result.background_color}"/>')
    
    filt = ' filter="url(#aa)"' if blur_std > 0 else ''
    for layer in result.layers:
        parts.append(f'<g fill="{layer.color}" fill-rule="evenodd"{filt}>')
        for d in layer.paths:
            parts.append(f'<path d="{d}"/>')
        parts.append("</g>")
    parts.append("</svg>")
    return "\n".join(parts)


# Test different soft-field sigmas and SVG blur combos
print(f"{'Sigma':>6} {'SVGblur':>7} {'Nodes':>6} {'SSIM':>7} {'MAE':>6} {'Diff%':>6}")
print("-" * 50)

for sigma in [1.0, 1.5, 2.0, 2.5]:
    # Patch the Gaussian blur sigma in the source
    orig_blur = cv2.GaussianBlur
    
    call_count = [0]
    def patched_blur(src, ksize, sigmaX=0, **kw):
        # The soft field blur is the one called with sigmaX
        # and (0,0) kernel. We want to change only that one.
        if ksize == (0, 0) and call_count[0] < 100:
            call_count[0] += 1
            return orig_blur(src, ksize, sigmaX=sigma, **kw)
        return orig_blur(src, ksize, sigmaX=sigmaX, **kw)
    
    cv2.GaussianBlur = patched_blur
    result = ml.multilevel_vectorize(crop, num_levels=24, min_contour_area=30)
    cv2.GaussianBlur = orig_blur
    
    for svg_blur in [0.0, 0.4, 0.6, 0.8]:
        svg = gen_svg_with_blur(result, remove_background=False, blur_std=svg_blur)
        comp = compare(crop, svg)
        nodes = sum(d.count("C") + d.count("M") for layer in result.layers for d in layer.paths)
        print(f"{sigma:6.1f} {svg_blur:7.1f} {nodes:6d} {comp.ssim_score:7.4f} {comp.mae:6.2f} {comp.pixel_diff_ratio*100:5.1f}%")

# Also test min_contour_area effect with the best sigma
print(f"\nMin area sweep (sigma=1.5, no SVG blur):")
for min_area in [8, 15, 30, 50, 100]:
    cv2.GaussianBlur = patched_blur
    call_count[0] = 0
    result = ml.multilevel_vectorize(crop, num_levels=24, min_contour_area=min_area)
    cv2.GaussianBlur = orig_blur
    svg = gen_svg_with_blur(result, remove_background=False, blur_std=0.0)
    comp = compare(crop, svg)
    nodes = sum(d.count("C") + d.count("M") for layer in result.layers for d in layer.paths)
    paths = result.path_count
    print(f"  area>={min_area:3d}: paths={paths:3d} nodes={nodes:5d} SSIM={comp.ssim_score:.4f} MAE={comp.mae:.2f}")
