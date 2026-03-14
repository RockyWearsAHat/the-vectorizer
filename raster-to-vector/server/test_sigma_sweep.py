"""Sweep smooth_sigma to find optimal smoothing vs accuracy tradeoff."""
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

print("=" * 72)
print("SMOOTH SIGMA SWEEP (3-run avg, crop + mahal)")
print("sigma  | crop_blur  crop_raw  | mahal_blur mahal_raw | nodes")
print("-" * 72)

for sigma in [0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0, 1.2]:
    results = {}
    for name, img in [("crop", crop), ("mahal", mahal)]:
        h, w = img.shape[:2]
        blurs, raws, ns = [], [], []
        for _ in range(3):
            result = multilevel_vectorize(img, smooth_sigma=sigma)
            svg = generate_svg(result, remove_background=False)
            rend = render_svg(svg, w, h)
            blurs.append(measure(img, rend, 1.5))
            raws.append(measure(img, rend, 0))
            ns.append(result.node_count)
        results[name] = (np.mean(blurs), np.mean(raws), int(np.mean(ns)))

    c = results["crop"]
    m = results["mahal"]
    print(f" {sigma:.2f}  | {c[0]:.4f}    {c[1]:.4f}   | {m[0]:.4f}     {m[1]:.4f}   | {c[2]:>5d}/{m[2]:>5d}")

print("=" * 72)
print("sigma=0 = no smoothing (pixel-perfect baseline)")
print("Target: minimal raw SSIM loss from sigma=0, visually smooth edges")
