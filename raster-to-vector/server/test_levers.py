"""Test optimization levers for SSIM accuracy.

Tries variations of key parameters to find what moves the needle.
"""
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

# --- Load test images ---
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

def run_one(img, **kwargs):
    h, w = img.shape[:2]
    result = multilevel_vectorize(img, **kwargs)
    svg = generate_svg(result, remove_background=False)
    rend = render_svg(svg, w, h)
    s_blur = measure(img, rend, blur_sigma=1.5)
    s_raw = measure(img, rend, blur_sigma=0)
    return s_blur, s_raw

def avg_runs(img, name, n=3, **kwargs):
    scores = [run_one(img, **kwargs) for _ in range(n)]
    blur_avg = np.mean([s[0] for s in scores])
    raw_avg = np.mean([s[1] for s in scores])
    return blur_avg, raw_avg

print("=" * 70)
print("OPTIMIZATION LEVER TESTS (3-run avg each)")
print("=" * 70)

# Baseline: current best config
print("\n--- BASELINE (current defaults) ---")
for name, img in [("crop", crop), ("mahal", mahal)]:
    b, r = avg_runs(img, name)
    print(f"  {name}: blur={b:.4f}  raw={r:.4f}")

# Lever 1: Increase K from 24 to 32
print("\n--- LEVER 1: K=32 (more initial clusters) ---")
for name, img in [("crop", crop), ("mahal", mahal)]:
    b, r = avg_runs(img, name, num_levels=32)
    print(f"  {name}: blur={b:.4f}  raw={r:.4f}")

# Lever 2: contour_scale=8 (higher superresolution)
print("\n--- LEVER 2: contour_scale=8 ---")
for name, img in [("crop", crop), ("mahal", mahal)]:
    b, r = avg_runs(img, name, contour_scale=8)
    print(f"  {name}: blur={b:.4f}  raw={r:.4f}")

# Lever 3: Tighter max_error=0.10 (more precise Béziers)
print("\n--- LEVER 3: max_error=0.10 ---")
for name, img in [("crop", crop), ("mahal", mahal)]:
    b, r = avg_runs(img, name, max_error=0.10)
    print(f"  {name}: blur={b:.4f}  raw={r:.4f}")

# Lever 4: Smaller simplify_epsilon=0.03
print("\n--- LEVER 4: simplify_epsilon=0.03 ---")
for name, img in [("crop", crop), ("mahal", mahal)]:
    b, r = avg_runs(img, name, simplify_epsilon=0.03)
    print(f"  {name}: blur={b:.4f}  raw={r:.4f}")

# Lever 5: Smaller min_contour_area=1 (keep tiny contours)
print("\n--- LEVER 5: min_contour_area=1 ---")
for name, img in [("crop", crop), ("mahal", mahal)]:
    b, r = avg_runs(img, name, min_contour_area=1)
    print(f"  {name}: blur={b:.4f}  raw={r:.4f}")

# Lever 6: Combined: K=32 + scale=8
print("\n--- LEVER 6: K=32 + contour_scale=8 ---")
for name, img in [("crop", crop), ("mahal", mahal)]:
    b, r = avg_runs(img, name, num_levels=32, contour_scale=8)
    print(f"  {name}: blur={b:.4f}  raw={r:.4f}")

# Lever 7: Combined: max_error=0.10 + epsilon=0.03
print("\n--- LEVER 7: max_error=0.10 + epsilon=0.03 ---")
for name, img in [("crop", crop), ("mahal", mahal)]:
    b, r = avg_runs(img, name, max_error=0.10, simplify_epsilon=0.03)
    print(f"  {name}: blur={b:.4f}  raw={r:.4f}")

print("\n" + "=" * 70)
print("DONE")
