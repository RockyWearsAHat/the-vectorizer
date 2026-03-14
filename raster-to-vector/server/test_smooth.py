"""Visual comparison: before/after contour smoothing.

Renders crop at 3x zoom and saves zoomed edge region to see pixel staircase vs smooth.
Also measures SSIM impact.
"""
import sys, os, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np
from io import BytesIO
from PIL import Image
import cairosvg
from skimage.metrics import structural_similarity as ssim

from app.core.multilevel import multilevel_vectorize, generate_svg

ref = cv2.imread(os.path.join(os.path.dirname(__file__), "../../Ref.png"))
crop = ref[50:460, 486:1050]

def render_svg(svg_str, w, h, scale=1):
    """Render SVG at scale× resolution."""
    png = cairosvg.svg2png(bytestring=svg_str.encode(),
                           output_width=w*scale, output_height=h*scale)
    return cv2.cvtColor(np.array(Image.open(BytesIO(png)).convert("RGB")), cv2.COLOR_RGB2BGR)

def measure(src, rend, blur_sigma=1.5):
    g1 = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(rend, cv2.COLOR_BGR2GRAY)
    if blur_sigma > 0:
        k = int(blur_sigma * 6) | 1
        g1 = cv2.GaussianBlur(g1, (k, k), blur_sigma)
        g2 = cv2.GaussianBlur(g2, (k, k), blur_sigma)
    return ssim(g1, g2)

h, w = crop.shape[:2]

# Generate SVG
print("Generating SVG...")
result = multilevel_vectorize(crop)
svg = generate_svg(result, remove_background=False)

# Render at 1x for SSIM
rend_1x = render_svg(svg, w, h, scale=1)
s_blur = measure(crop, rend_1x, 1.5)
s_raw = measure(crop, rend_1x, 0)
print(f"SSIM: blur={s_blur:.4f}  raw={s_raw:.4f}  paths={result.path_count}  nodes={result.node_count}")

# Render at 4x for visual inspection
rend_4x = render_svg(svg, w, h, scale=4)
cv2.imwrite("/tmp/smooth_4x_full.png", rend_4x)
print(f"Saved 4x render: /tmp/smooth_4x_full.png ({rend_4x.shape[1]}x{rend_4x.shape[0]})")

# Also save a cropped region showing the diagonal edge area
# (approximate location of the V-shaped diagonal from the screenshots)
# The screenshots appear to show the "M" letter area
# Let's crop a few interesting edge regions at 4x
regions = [
    ("top_left", (0, 0, 200, 200)),      # likely has diagonal strokes
    ("center", (150, 100, 400, 350)),     # main letter area
]
for name, (x1, y1, x2, y2) in regions:
    # Scale to 4x coords
    patch = rend_4x[y1*4:y2*4, x1*4:x2*4]
    if patch.size > 0:
        cv2.imwrite(f"/tmp/smooth_4x_{name}.png", patch)
        print(f"Saved /tmp/smooth_4x_{name}.png ({patch.shape[1]}x{patch.shape[0]})")

# Save the SVG itself for browser inspection
with open("/tmp/smooth_test.svg", "w") as f:
    f.write(svg)
print("Saved SVG: /tmp/smooth_test.svg")
print("\nOpen the SVG in a browser and zoom in to inspect edge quality.")
