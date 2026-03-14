"""5-run validation of production config with min_contour_area=1."""
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
mahal = cv2.imread("/tmp/mahal_right.png")

def render_svg(svg_str, w, h):
    png = cairosvg.svg2png(bytestring=svg_str.encode(), output_width=w, output_height=h)
    return cv2.cvtColor(np.array(Image.open(BytesIO(png)).convert("RGB")), cv2.COLOR_RGB2BGR)

def measure(src, rend, blur_sigma=1.5):
    g1 = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(rend, cv2.COLOR_BGR2GRAY)
    if blur_sigma > 0:
        k = int(blur_sigma * 6) | 1
        g1 = cv2.GaussianBlur(g1, (k, k), blur_sigma)
        g2 = cv2.GaussianBlur(g2, (k, k), blur_sigma)
    return ssim(g1, g2)

print("=" * 70)
print("PRODUCTION VALIDATION — min_contour_area=1 (5 runs)")
print("=" * 70)

for name, img in [("crop", crop), ("mahal", mahal)]:
    h, w = img.shape[:2]
    print(f"\n--- {name} ({w}x{h}) ---")
    blurs, raws, paths_list, nodes_list, times = [], [], [], [], []
    for i in range(5):
        t0 = time.time()
        result = multilevel_vectorize(img)
        svg = generate_svg(result, remove_background=False)
        dt = time.time() - t0
        rend = render_svg(svg, w, h)
        b = measure(img, rend, 1.5)
        r = measure(img, rend, 0)
        blurs.append(b)
        raws.append(r)
        paths_list.append(result.path_count)
        nodes_list.append(result.node_count)
        times.append(dt)
        print(f"  run {i+1}: blur={b:.4f}  raw={r:.4f}  paths={result.path_count}  nodes={result.node_count}  {dt:.1f}s")
    print(f"  AVG: blur={np.mean(blurs):.4f}±{np.std(blurs):.4f}  raw={np.mean(raws):.4f}±{np.std(raws):.4f}")
    print(f"       paths={np.mean(paths_list):.0f}  nodes={np.mean(nodes_list):.0f}  time={np.mean(times):.1f}s")

print("\n\nTheoretical ceilings (K=5 quantized → same rasterization):")
print("  crop K=5:  raw=0.9734  blur=0.9977")
print("  mahal K=5: raw=0.9721  blur=0.9960")
print("\nDONE")
