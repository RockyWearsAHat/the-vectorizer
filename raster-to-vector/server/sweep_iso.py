"""Sweep halo iso-contour level."""
import cv2
import numpy as np
import app.core.multilevel as ml
from app.core.multilevel import multilevel_vectorize, generate_svg
from app.core.comparison import compare

img = cv2.imread("/Users/alexwaldmann/Desktop/SVG-gen/Ref.png")
h, w = img.shape[:2]
crop_h, crop_w = min(410, h), min(564, w)
x_start = max(0, (w - crop_w) // 2)
crop = img[0:crop_h, x_start:x_start+crop_w]

# Patch the halo iso-level
original_find = ml.find_contours
print(f"{'HaloISO':>8} {'Opacity':>8} {'SSIM':>7} {'MAE':>6}")
print("-" * 40)

# For each halo iso, re-run the pipeline
# Easier: just modify the source and re-import
for halo_iso in [0.15, 0.2, 0.25, 0.3, 0.35]:
    # Monkey-patch: intercept find_contours calls
    call_idx = [0]
    def patched_find(soft, iso, halo_iso=halo_iso):
        call_idx[0] += 1
        if call_idx[0] % 2 == 0:  # Even calls are halo (second in the pair)
            return original_find(soft, halo_iso)
        return original_find(soft, iso)  # Odd calls are core (first in pair)
    
    ml.find_contours = patched_find
    try:
        result = multilevel_vectorize(crop, num_levels=24)
        svg = generate_svg(result, remove_background=False)
        comp = compare(crop, svg)
        print(f"{halo_iso:8.2f}     0.55 {comp.ssim_score:7.4f} {comp.mae:6.2f}")
    finally:
        ml.find_contours = original_find
