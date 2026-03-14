"""Zoom-in diagnostic: render the MB monogram SVG, zoom into critical areas,
measure per-pixel error to identify exactly what type of artifacts appear.

Generates side-by-side comparison patches at 4× zoom for:
  - Serif corners (sharp edges)
  - Thin flower outlines (fine strokes)
  - Boundary transitions (where colors meet)
"""

import sys, os
import cv2
import numpy as np
from pathlib import Path
from skimage.metrics import structural_similarity as ssim

sys.path.insert(0, str(Path(__file__).parent))
from app.core.multilevel import multilevel_vectorize, generate_svg


def render_svg(svg_str, w, h):
    import cairosvg
    from io import BytesIO
    from PIL import Image
    png = cairosvg.svg2png(bytestring=svg_str.encode(), output_width=w, output_height=h)
    return cv2.cvtColor(np.array(Image.open(BytesIO(png)).convert("RGB")), cv2.COLOR_RGB2BGR)


def measure(src, rend, blur_sigma=1.5):
    g1, g2 = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY), cv2.cvtColor(rend, cv2.COLOR_BGR2GRAY)
    if blur_sigma > 0:
        k = int(blur_sigma * 6) | 1
        g1 = cv2.GaussianBlur(g1, (k, k), blur_sigma)
        g2 = cv2.GaussianBlur(g2, (k, k), blur_sigma)
    return ssim(g1, g2)


if __name__ == "__main__":
    img = cv2.imread("/Users/alexwaldmann/Desktop/SVG-gen/Ref.png")
    if img is None:
        print("Cannot find Ref.png")
        sys.exit(1)

    h, w = img.shape[:2]
    print(f"Source: {w}×{h}")

    # Vectorize
    result = multilevel_vectorize(img)
    svg = generate_svg(result, remove_background=False)
    rend = render_svg(svg, w, h)

    s_blur = measure(img, rend, blur_sigma=1.5)
    s_raw = measure(img, rend, blur_sigma=0)
    print(f"Full image: blur_SSIM={s_blur:.4f}  raw_SSIM={s_raw:.4f}")
    print(f"Clusters: {len(result.layers)+1}, Paths: {result.path_count}, Nodes: {result.node_count}")

    # Error analysis
    diff = cv2.absdiff(img, rend)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    # Find worst error patches (128×128 blocks)
    bs = 128
    worst_patches = []
    for y in range(0, h - bs, bs // 2):
        for x in range(0, w - bs, bs // 2):
            patch = gray_diff[y:y+bs, x:x+bs]
            mae = float(np.mean(patch))
            max_err = float(np.max(patch))
            worst_patches.append((mae, max_err, x, y))

    worst_patches.sort(key=lambda t: -t[0])

    print(f"\nTop 10 worst 128×128 patches (by mean absolute error):")
    print(f"{'Rank':>4} {'MAE':>6} {'Max':>5} {'Location':>12}  Type")
    print("-" * 50)
    for i, (mae, mx, x, y) in enumerate(worst_patches[:10]):
        # Classify based on position
        cx, cy = x + bs // 2, y + bs // 2
        region = "interior"
        if cx < w * 0.1 or cx > w * 0.9 or cy < h * 0.1 or cy > h * 0.9:
            region = "border"
        elif 400 < cx < 700 and 200 < cy < 800:
            region = "letter_area"
        else:
            region = "flower_area"
        print(f"  {i+1:2d}  {mae:5.1f}  {mx:4.0f}  ({x:4d},{y:4d})   {region}")

    # Per-channel analysis of high-error regions
    print(f"\nError breakdown (full image):")
    print(f"  MAE per channel: B={np.mean(diff[:,:,0]):.1f} G={np.mean(diff[:,:,1]):.1f} R={np.mean(diff[:,:,2]):.1f}")
    print(f"  Max error: {np.max(gray_diff)}")
    print(f"  Pixels with error > 20: {np.count_nonzero(gray_diff > 20)} ({100*np.count_nonzero(gray_diff > 20)/(h*w):.1f}%)")
    print(f"  Pixels with error > 50: {np.count_nonzero(gray_diff > 50)} ({100*np.count_nonzero(gray_diff > 50)/(h*w):.1f}%)")
    print(f"  Pixels with error > 100: {np.count_nonzero(gray_diff > 100)} ({100*np.count_nonzero(gray_diff > 100)/(h*w):.1f}%)")

    # Classify error by source vs render brightness
    g_src = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.int16)
    g_rend = cv2.cvtColor(rend, cv2.COLOR_BGR2GRAY).astype(np.int16)
    signed_diff = g_rend - g_src  # positive = SVG brighter, negative = SVG darker

    too_bright = np.count_nonzero(signed_diff > 20)
    too_dark = np.count_nonzero(signed_diff < -20)
    print(f"\n  SVG too bright (>20): {too_bright} px ({100*too_bright/(h*w):.2f}%)")
    print(f"  SVG too dark  (<-20): {too_dark} px ({100*too_dark/(h*w):.2f}%)")

    # Save error heatmap
    heatmap = cv2.applyColorMap((gray_diff * 3).clip(0, 255).astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite("/tmp/error_heatmap.png", heatmap)
    cv2.imwrite("/tmp/svg_rendered.png", rend)
    print(f"\nSaved /tmp/error_heatmap.png and /tmp/svg_rendered.png")

    # Check SVG size
    svg_kb = len(svg.encode()) / 1024
    print(f"SVG file size: {svg_kb:.0f} KB")
