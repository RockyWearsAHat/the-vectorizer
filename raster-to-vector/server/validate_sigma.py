"""Validate sigma=1.0 across crop, mahal, ref."""
import sys, os, cv2, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from app.core.comparison import compare
from sweep_crisp import vectorize_custom
from app.core.multilevel import generate_svg

ref = cv2.imread("/Users/alexwaldmann/Desktop/SVG-gen/Ref.png")
crop = ref[200:610, 400:964]
mahal = cv2.imread("/tmp/mahal_right.png")

images = [("crop", crop), ("ref", ref), ("mahal", mahal)]
configs = [
    ("baseline s=1.5", 1.5, [0.20, 0.50], [0.55, 1.00]),
    ("crisp   s=1.0",  1.0, [0.20, 0.50], [0.55, 1.00]),
]

print(f"{'config':<18} {'image':<8} {'SSIM':>6} {'MAE':>6} {'paths':>5}")
print("-" * 50)

for label, sigma, iso_l, iso_o in configs:
    for img_name, img in images:
        result = vectorize_custom(img, sigma, iso_l, iso_o)
        svg = generate_svg(result, remove_background=False)
        m = compare(img, svg)
        print(f"{label:<18} {img_name:<8} {m.ssim_score:.4f} {m.mae:.2f} {result.path_count:>5}")
    print()
