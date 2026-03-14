"""Test different contour precision parameters."""
import cv2, numpy as np
from app.core.multilevel import multilevel_vectorize, generate_svg
from app.core.comparison import compare

img = cv2.imread('/Users/alexwaldmann/Desktop/SVG-gen/Ref.png')
crop = img[50:460, 486:1050]
mahal = cv2.imread('/tmp/mahal_right.png')

configs = [
    ('eps=0.30 err=0.50', dict(simplify_epsilon=0.30, max_error=0.50)),
    ('eps=0.20 err=0.40', dict(simplify_epsilon=0.20, max_error=0.40)),
    ('eps=0.15 err=0.35', dict(simplify_epsilon=0.15, max_error=0.35)),
    ('eps=0.10 err=0.30', dict(simplify_epsilon=0.10, max_error=0.30)),
]

for label, kw in configs:
    scores_c = []
    scores_m = []
    for _ in range(5):
        r = multilevel_vectorize(crop, num_levels=24, **kw)
        s = generate_svg(r, remove_background=False)
        c = compare(crop, s)
        scores_c.append(c.ssim_score)
        r2 = multilevel_vectorize(mahal, num_levels=24, **kw)
        s2 = generate_svg(r2, remove_background=False)
        c2 = compare(mahal, s2)
        scores_m.append(c2.ssim_score)
    print(f'{label}: crop={np.mean(scores_c):.4f} mahal={np.mean(scores_m):.4f} nodes_c={r.node_count} nodes_m={r2.node_count}')
