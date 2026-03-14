"""Final validation of top combos on all 3 images."""
import sys, os, cv2, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from app.core.comparison import compare
from sweep_focused import vectorize_custom
from app.core.multilevel import generate_svg

ref = cv2.imread("/Users/alexwaldmann/Desktop/SVG-gen/Ref.png")
crop = ref[200:610, 400:964]
mahal = cv2.imread("/tmp/mahal_right.png")

images = [("crop", crop), ("ref", ref), ("mahal", mahal)]

# Top candidates
configs = [
    # label, d, sc, ss, merge, sigma, iso_l, iso_o, eps, maxerr, dist_orig, min_area
    ("baseline",      7,20,20, 80, 1.0, [.20,.50],[.55,1.], .3, .3, False, 30),
    ("BEST_combo",    7,10,10, 80, 1.0, [.20,.50],[.55,1.], .15,.2, True,  30),
    ("BEST+area15",   7,10,10, 80, 1.0, [.20,.50],[.55,1.], .15,.2, True,  15),
    ("BEST+area10",   7,10,10, 80, 1.0, [.20,.50],[.55,1.], .15,.2, True,  10),
]

print(f"{'config':<18}", end="")
for name, _ in images:
    print(f" {name+'/SSIM':>10} {name+'/MAE':>8}", end="")
print(f"  {'avg':>6} {'nodes':>6}")
print("-" * 85)

for row in configs:
    label = row[0]
    bd,bsc,bss,mt,sigma = row[1],row[2],row[3],row[4],row[5]
    iso_l,iso_o = row[6],row[7]
    eps,me,dorig,mina = row[8],row[9],row[10],row[11]

    ssims = []
    line = f"{label:<18}"
    nc = 0
    for img_name, img in images:
        result = vectorize_custom(img, bd, bsc, bss, mt, sigma, iso_l, iso_o,
                                  simplify_epsilon=eps, max_error=me,
                                  min_contour_area=mina, dist_from_original=dorig)
        svg = generate_svg(result, remove_background=False)
        m = compare(img, svg)
        ssims.append(m.ssim_score)
        nc = max(nc, result.node_count)
        line += f" {m.ssim_score:>10.4f} {m.mae:>8.2f}"
    avg = np.mean(ssims)
    line += f"  {avg:>6.4f} {nc:>6}"
    print(line)
