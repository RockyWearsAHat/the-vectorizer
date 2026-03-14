"""Final push: best combos including simplify_eps=0.1 and adaptive sigma."""
import sys, os, cv2, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from app.core.comparison import compare
from app.core.multilevel import generate_svg
from gap_analysis import vectorize_full, test_config

ref = cv2.imread("/Users/alexwaldmann/Desktop/SVG-gen/Ref.png")
crop = ref[200:610, 400:964]
mahal = cv2.imread("/tmp/mahal_right.png")

combos = [
    # label, kwargs
    ("baseline",                       dict()),
    ("se0.1",                          dict(simplify_eps=0.1)),
    ("se0.1+area10",                   dict(simplify_eps=0.1, min_area=10)),
    ("se0.1+area10+me0.2",             dict(simplify_eps=0.1, min_area=10, max_err=0.2)),
    ("se0.1+me0.2",                    dict(simplify_eps=0.1, max_err=0.2)),
    ("se0.1+mt60",                     dict(simplify_eps=0.1, merge_thresh=60)),
    ("se0.1+mt60+area10",              dict(simplify_eps=0.1, merge_thresh=60, min_area=10)),
    ("se0.1+mt60+area10+me0.2",        dict(simplify_eps=0.1, merge_thresh=60, min_area=10, max_err=0.2)),
    ("se0.15+mt60+area10+me0.2",       dict(simplify_eps=0.15, merge_thresh=60, min_area=10, max_err=0.2)),
    ("se0.1+mt50+area10+me0.2",        dict(simplify_eps=0.1, merge_thresh=50, min_area=10, max_err=0.2)),
    ("se0.1+K32+mt60+area10",          dict(simplify_eps=0.1, K=32, merge_thresh=60, min_area=10)),
    ("se0.1+K48+mt50+area10",          dict(simplify_eps=0.1, K=48, merge_thresh=50, min_area=10)),
    ("se0.1+bilat10+mt60+area10",      dict(simplify_eps=0.1, bilateral_sc=10, merge_thresh=60, min_area=10)),
]

print(f"{'combo':<36} {'crop':>6} {'ref':>6} {'mahal':>6} {'avg':>6} {'K':>8}")
print("-" * 75)

for label, kw in combos:
    results = []
    for img_name, img in [("crop", crop), ("ref", ref), ("mahal", mahal)]:
        s, mae, keff, paths = test_config(img, label, **kw.copy())
        results.append((s, mae, keff, paths))
    avg_ssim = np.mean([r[0] for r in results])
    ks = f"{results[0][2]}/{results[1][2]}/{results[2][2]}"
    print(f"  {label:<34} {results[0][0]:.4f} {results[1][0]:.4f} {results[2][0]:.4f} {avg_ssim:.4f} {ks:>8}")
