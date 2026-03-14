"""Fine-tune adaptive params around the v1 winner."""
import sys, os, cv2, numpy as np, time
sys.path.insert(0, os.path.dirname(__file__))
from app.core.comparison import compare
from test_adaptive import vectorize_adaptive, compute_edge_weight
from app.core.multilevel import generate_svg

ref = cv2.imread("/Users/alexwaldmann/Desktop/SVG-gen/Ref.png")
crop = ref[200:610, 400:964]
mahal = cv2.imread("/tmp/mahal_right.png")
images = [("crop", crop), ("mahal", mahal), ("ref", ref)]

# Fine-tune around adaptive_v1 winner (sigma_crisp=0.6, sigma_smooth=1.5, blur=15)
configs = [
    ("v1_default",     dict()),
    # Sigma crisp sweep
    ("crisp0.4",       dict(sigma_crisp=0.4)),
    ("crisp0.5",       dict(sigma_crisp=0.5)),
    ("crisp0.7",       dict(sigma_crisp=0.7)),
    ("crisp0.8",       dict(sigma_crisp=0.8)),
    # Sigma smooth sweep  
    ("smooth1.2",      dict(sigma_smooth=1.2)),
    ("smooth1.8",      dict(sigma_smooth=1.8)),
    ("smooth2.0",      dict(sigma_smooth=2.0)),
    # Best crisp + smooth combos
    ("c0.5+s1.2",      dict(sigma_crisp=0.5, sigma_smooth=1.2)),
    ("c0.7+s1.8",      dict(sigma_crisp=0.7, sigma_smooth=1.8)),
    ("c0.5+s1.8",      dict(sigma_crisp=0.5, sigma_smooth=1.8)),
    # Edge threshold sensitivity
    ("c0.6+blur12",    dict(edge_blur_radius=12)),
    ("c0.6+blur18",    dict(edge_blur_radius=18)),
    # min_area sweep
    ("area10",         dict(min_contour_area=10)),
    ("area20",         dict(min_contour_area=20)),
    # Epsilon sweep (curve fitting tightness)
    ("eps0.10",        dict(simplify_epsilon=0.10, max_error=0.15)),
    ("eps0.20",        dict(simplify_epsilon=0.20, max_error=0.25)),
]

print(f"{'config':<18}", end="")
for name, _ in images:
    print(f" {name:>6}", end="")
print(f"  {'avg':>6} {'paths':>5} {'nodes':>6}")
print("-" * 70)

for label, overrides in configs:
    ssims = []
    line = f"{label:<18}"
    last_result = None
    for img_name, img in images:
        result, _ = vectorize_adaptive(img, **overrides)
        svg = generate_svg(result, remove_background=False)
        m = compare(img, svg)
        ssims.append(m.ssim_score)
        last_result = result
    avg = np.mean(ssims)
    line += " ".join(f" {s:.4f}" for s in ssims)
    line += f"  {avg:.4f} {last_result.path_count:>5} {last_result.node_count:>6}"
    print(line)
