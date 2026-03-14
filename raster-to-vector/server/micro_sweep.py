"""Micro-sweep around m60+op50 winner."""
import sys, os, cv2, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from app.core.comparison import compare
from fix_artifacts2 import vectorize_adaptive_halo
from app.core.multilevel import generate_svg

ref = cv2.imread("/Users/alexwaldmann/Desktop/SVG-gen/Ref.png")
crop = ref[200:610, 400:964]
mahal = cv2.imread("/tmp/mahal_right.png")
images = [("crop", crop), ("mahal", mahal), ("ref", ref)]

configs = [
    ("m60+op50",          dict(merge_thresh=60, halo_opacity=0.50)),
    ("m60+op48",          dict(merge_thresh=60, halo_opacity=0.48)),
    ("m60+op52",          dict(merge_thresh=60, halo_opacity=0.52)),
    ("m65+op50",          dict(merge_thresh=65, halo_opacity=0.50)),
    ("m70+op50",          dict(merge_thresh=70, halo_opacity=0.50)),
    ("m55+op50",          dict(merge_thresh=55, halo_opacity=0.50)),
    ("m65+op52",          dict(merge_thresh=65, halo_opacity=0.52)),
    ("m70+op48",          dict(merge_thresh=70, halo_opacity=0.48)),
    ("m60+op50+sc0.5",    dict(merge_thresh=60, halo_opacity=0.50, sigma_crisp=0.5)),
    ("m60+op50+sc0.7",    dict(merge_thresh=60, halo_opacity=0.50, sigma_crisp=0.7)),
    ("m60+op50+ss1.8",    dict(merge_thresh=60, halo_opacity=0.50, sigma_smooth=1.8)),
]

print(f"{'config':<22}", end="")
for name, _ in images:
    print(f" {name:>7}", end="")
print(f"  {'avg':>7} {'K':>3}")
print("-" * 60)

for label, overrides in configs:
    ssims = []
    K_count = 0
    for _, img in images:
        result = vectorize_adaptive_halo(img, **overrides)
        svg = generate_svg(result, remove_background=False)
        m = compare(img, svg)
        ssims.append(m.ssim_score)
        K_count = max(K_count, len(result.layers))
    avg = np.mean(ssims)
    line = f"{label:<22}"
    line += " ".join(f" {s:>7.4f}" for s in ssims)
    line += f"  {avg:>7.4f} {K_count:>3}"
    print(line)
