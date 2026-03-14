"""Quick test: 4× vs 8× scale in production, plus full Ref.png timing."""

import sys, os, time
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
    images["Ref_full"] = ref

    for scale in [4, 8]:
        print(f"\n=== contour_scale={scale}× ===")
        for name, img in images.items():
            h, w = img.shape[:2]
            t0 = time.time()
            result = multilevel_vectorize(img, contour_scale=scale)
            svg = generate_svg(result, remove_background=False)
            dt = time.time() - t0
            rend = render_svg(svg, w, h)
            s = measure(img, rend)
            s_raw = measure(img, rend, blur_sigma=0)
            print(f"  {name:12s} ({w}×{h}): blur={s:.4f} raw={s_raw:.4f} "
                  f"paths={result.path_count} nodes={result.node_count} time={dt:.1f}s")
