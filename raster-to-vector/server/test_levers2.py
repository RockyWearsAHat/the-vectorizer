"""Focused test: min_contour_area=1 and =2, plus epsilon=0.03 combo."""
import sys, os, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
os.environ.setdefault("DYLD_LIBRARY_PATH", "/opt/homebrew/opt/cairo/lib")

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

def run_test(img, name, n=3, **kwargs):
    h, w = img.shape[:2]
    blurs, raws = [], []
    for i in range(n):
        t0 = time.time()
        result = multilevel_vectorize(img, **kwargs)
        svg = generate_svg(result, remove_background=False)
        rend = render_svg(svg, w, h)
        b = measure(img, rend, blur_sigma=1.5)
        r = measure(img, rend, blur_sigma=0)
        dt = time.time() - t0
        blurs.append(b)
        raws.append(r)
        print(f"    run {i+1}: blur={b:.4f} raw={r:.4f} paths={result.path_count} nodes={result.node_count} {dt:.1f}s")
    print(f"  {name} AVG: blur={np.mean(blurs):.4f}  raw={np.mean(raws):.4f}")
    return np.mean(blurs), np.mean(raws)

print("=" * 70)
print("FOCUSED LEVER TESTS")
print("=" * 70)

print("\n--- BASELINE ---")
for name, img in [("crop", crop), ("mahal", mahal)]:
    run_test(img, name)

print("\n--- min_contour_area=2 ---")
for name, img in [("crop", crop), ("mahal", mahal)]:
    run_test(img, name, min_contour_area=2)

print("\n--- min_contour_area=1 ---")
for name, img in [("crop", crop), ("mahal", mahal)]:
    run_test(img, name, min_contour_area=1)

print("\n--- min_contour_area=1 + simplify_epsilon=0.03 ---")
for name, img in [("crop", crop), ("mahal", mahal)]:
    run_test(img, name, min_contour_area=1, simplify_epsilon=0.03)

print("\n--- min_contour_area=1 + contour_scale=8 ---")
for name, img in [("crop", crop), ("mahal", mahal)]:
    run_test(img, name, min_contour_area=1, contour_scale=8)

print("\nDONE")
