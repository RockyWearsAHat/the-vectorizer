"""Quick test of the cropped region."""
import cv2
from app.core.multilevel import multilevel_vectorize, generate_svg
from app.core.comparison import compare

img = cv2.imread("/Users/alexwaldmann/Desktop/SVG-gen/Ref.png")
h, w = img.shape[:2]
crop_h, crop_w = min(410, h), min(564, w)
x_start = max(0, (w - crop_w) // 2)
crop = img[0:crop_h, x_start:x_start+crop_w]

result = multilevel_vectorize(crop, num_levels=24)
svg = generate_svg(result, remove_background=False)
comp = compare(crop, svg)
print(f"Crop {crop.shape[1]}x{crop.shape[0]}: layers={len(result.layers)} "
      f"paths={result.path_count} SSIM={comp.ssim_score:.4f} MAE={comp.mae:.2f} "
      f"diff={comp.pixel_diff_ratio*100:.1f}%")

# Save for inspection
with open("/tmp/crop_feathered.svg", "w") as f:
    f.write(svg)
svg_nobg = generate_svg(result, remove_background=True)
with open("/tmp/crop_feathered_nobg.svg", "w") as f:
    f.write(svg_nobg)
print("Saved /tmp/crop_feathered.svg and /tmp/crop_feathered_nobg.svg")
