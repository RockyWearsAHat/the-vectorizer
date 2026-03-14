"""Generate SVGs at different sigma for visual comparison in browser."""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np
from app.core.multilevel import multilevel_vectorize, generate_svg

ref = cv2.imread(os.path.join(os.path.dirname(__file__), "../../Ref.png"))
crop = ref[50:460, 486:1050]

np.random.seed(42)

for sigma in [0.0, 0.35, 0.5, 0.8]:
    np.random.seed(42)
    result = multilevel_vectorize(crop, smooth_sigma=sigma)
    svg = generate_svg(result, remove_background=False)
    fname = f"/tmp/svg_sigma_{sigma:.2f}.svg"
    with open(fname, "w") as f:
        f.write(svg)
    print(f"sigma={sigma:.2f}: {result.node_count} nodes -> {fname}")

print("\nOpen these in a browser and zoom in to compare edge quality:")
print("  open /tmp/svg_sigma_0.00.svg  # pixelated baseline")
print("  open /tmp/svg_sigma_0.35.svg  # light smoothing")
print("  open /tmp/svg_sigma_0.50.svg  # moderate smoothing")
print("  open /tmp/svg_sigma_0.80.svg  # heavy smoothing")
