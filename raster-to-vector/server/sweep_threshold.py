"""Sweep merge thresholds and K values on the cropped region."""
import cv2
import numpy as np
from app.core.multilevel import multilevel_vectorize, generate_svg
from app.core.comparison import compare

img = cv2.imread("/Users/alexwaldmann/Desktop/SVG-gen/Ref.png")
h, w = img.shape[:2]
crop_h, crop_w = min(410, h), min(564, w)
x_start = max(0, (w - crop_w) // 2)
crop = img[0:crop_h, x_start:x_start + crop_w]

print(f"Cropped region: {crop.shape[1]}x{crop.shape[0]}")
print()

# To test threshold, we need to pass it through. Monkey-patch for sweep.
import app.core.multilevel as ml

original_merge = ml._merge_close_clusters

def test_threshold(threshold, K=24):
    """Run pipeline with given merge threshold."""
    def patched_merge(centers, labels_flat, h, w, threshold=threshold):
        return original_merge(centers, labels_flat, h, w, threshold=threshold)
    ml._merge_close_clusters = patched_merge
    try:
        result = multilevel_vectorize(crop, num_levels=K)
        svg = generate_svg(result, remove_background=False)
        comp = compare(crop, svg)
        colors = [l.color for l in result.layers]
        return len(result.layers), comp.ssim_score, comp.mae, comp.pixel_diff_ratio, colors
    finally:
        ml._merge_close_clusters = original_merge

print(f"{'Thresh':>6} {'K':>3} {'Layers':>6} {'SSIM':>7} {'MAE':>6} {'Diff%':>6}  Colors")
print("-" * 80)
for threshold in [30, 40, 50, 60, 70, 80, 100]:
    layers, ssim_val, mae, diff, colors = test_threshold(threshold, K=24)
    colors_str = " ".join(colors)
    print(f"{threshold:6d}  24 {layers:6d} {ssim_val:7.4f} {mae:6.2f} {diff*100:5.1f}%  {colors_str}")

print()
print("Direct K-means (no merge, threshold=999):")
for K in [3, 4, 5, 6, 8]:
    layers, ssim_val, mae, diff, colors = test_threshold(999, K=K)
    colors_str = " ".join(colors)
    print(f"   K={K:2d}      {layers:6d} {ssim_val:7.4f} {mae:6.2f} {diff*100:5.1f}%  {colors_str}")
