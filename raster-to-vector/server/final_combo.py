"""Final combo sweep — combine best individual params."""
import sys, os, cv2, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from app.core.comparison import compare
from test_adaptive import vectorize_adaptive
from app.core.multilevel import generate_svg

ref = cv2.imread("/Users/alexwaldmann/Desktop/SVG-gen/Ref.png")
crop = ref[200:610, 400:964]
mahal = cv2.imread("/tmp/mahal_right.png")
images = [("crop", crop), ("mahal", mahal), ("ref", ref)]

# v1_default was avg 0.9806 crop=0.975, mahal=0.983, ref=0.984
# smooth1.8/2.0 helped mahal/ref, area10 helped mahal
# Let's combine winners
configs = [
    ("v1_baseline",         dict()),
    ("smooth1.8+area10",    dict(sigma_smooth=1.8, min_contour_area=10)),
    ("smooth2.0+area10",    dict(sigma_smooth=2.0, min_contour_area=10)),
    # 3-iso with adaptive (prev 3iso was good on crop)
    ("3iso+smooth1.8",      dict(sigma_smooth=1.8,
                                 iso_levels=[0.15, 0.35, 0.55],
                                 iso_opacities=[0.30, 0.65, 1.00])),
    ("3iso+s1.8+area10",    dict(sigma_smooth=1.8, min_contour_area=10,
                                 iso_levels=[0.15, 0.35, 0.55],
                                 iso_opacities=[0.30, 0.65, 1.00])),
    # More clusters with adaptive
    ("m50+s1.8+area10",     dict(merge_thresh=50, sigma_smooth=1.8, min_contour_area=10)),
    ("m50+3iso+s1.8",       dict(merge_thresh=50, sigma_smooth=1.8,
                                 iso_levels=[0.15, 0.35, 0.55],
                                 iso_opacities=[0.30, 0.65, 1.00])),
    # Try wider halo
    ("wider_halo",          dict(sigma_smooth=1.8, min_contour_area=10,
                                 iso_levels=[0.15, 0.50],
                                 iso_opacities=[0.45, 1.00])),
    # Try 4-iso with adaptive
    ("4iso_adaptive",       dict(sigma_smooth=1.8, min_contour_area=10,
                                 iso_levels=[0.10, 0.25, 0.40, 0.55],
                                 iso_opacities=[0.20, 0.45, 0.70, 1.00])),
]

print(f"{'config':<22}", end="")
for name, _ in images:
    print(f" {name:>7}", end="")
print(f"  {'avg':>7} {'paths':>5} {'nodes':>6}")
print("-" * 72)

for label, overrides in configs:
    ssims = []
    line = f"{label:<22}"
    last_result = None
    for img_name, img in images:
        result, _ = vectorize_adaptive(img, **overrides)
        svg = generate_svg(result, remove_background=False)
        m = compare(img, svg)
        ssims.append(m.ssim_score)
        last_result = result
    avg = np.mean(ssims)
    line += " ".join(f" {s:>7.4f}" for s in ssims)
    line += f"  {avg:>7.4f} {last_result.path_count:>5} {last_result.node_count:>6}"
    print(line)
