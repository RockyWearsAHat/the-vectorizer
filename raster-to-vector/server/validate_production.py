"""Validate the production pipeline after applying 4× superresolution.

Tests on crop and mahal images, 3 seeds each, measures both blur and raw SSIM.
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
    ref = cv2.imread(os.path.join(os.path.dirname(__file__), "../../Ref.png"))
    crop = ref[50:460, 486:1050]
    mahal = cv2.imread("/tmp/mahal_right.png")

    images = {"crop": crop}
    if mahal is not None:
        images["mahal"] = mahal

    print("=== Production Pipeline Validation (4× superresolution) ===\n")

    seeds = [42, 123, 999]

    for name, img in images.items():
        h, w = img.shape[:2]
        print(f"--- {name} ({w}×{h}) ---")
        blur_scores = []
        raw_scores = []
        for seed in seeds:
            np.random.seed(seed)
            import time
            t0 = time.time()
            result = multilevel_vectorize(img)
            svg = generate_svg(result, remove_background=False)
            dt = time.time() - t0
            rend = render_svg(svg, w, h)
            s_blur = measure(img, rend, blur_sigma=1.5)
            s_raw = measure(img, rend, blur_sigma=0)
            blur_scores.append(s_blur)
            raw_scores.append(s_raw)
            print(f"  seed={seed}: blur_SSIM={s_blur:.4f}  raw_SSIM={s_raw:.4f}  "
                  f"paths={result.path_count}  nodes={result.node_count}  "
                  f"time={dt:.1f}s")
        print(f"  AVG: blur_SSIM={np.mean(blur_scores):.4f} ± {np.std(blur_scores):.4f}")
        print(f"  AVG: raw_SSIM={np.mean(raw_scores):.4f} ± {np.std(raw_scores):.4f}")
        print()
