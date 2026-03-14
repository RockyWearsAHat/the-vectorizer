"""Full validation: blur + raw SSIM for the new adaptive iso production."""
import cv2, time
import numpy as np
from skimage.metrics import structural_similarity as ssim
from app.core.multilevel import multilevel_vectorize, generate_svg
import cairosvg
from io import BytesIO
from PIL import Image


def render(svg_str, w, h):
    png = cairosvg.svg2png(bytestring=svg_str.encode(), output_width=w, output_height=h)
    return cv2.cvtColor(np.array(Image.open(BytesIO(png)).convert("RGB")), cv2.COLOR_RGB2BGR)


def meas(src, rend, sigma=1.5):
    g1, g2 = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY), cv2.cvtColor(rend, cv2.COLOR_BGR2GRAY)
    if sigma > 0:
        k = int(sigma * 6) | 1
        g1 = cv2.GaussianBlur(g1, (k, k), sigma)
        g2 = cv2.GaussianBlur(g2, (k, k), sigma)
    return ssim(g1, g2)


ref = cv2.imread("/Users/alexwaldmann/Desktop/SVG-gen/Ref.png")
crop = ref[50:460, 486:1050]
mahal = cv2.imread("/tmp/mahal_right.png")

for name, img in [("crop", crop), ("mahal", mahal)]:
    if img is None: continue
    h, w = img.shape[:2]
    t0 = time.time()
    result = multilevel_vectorize(img)
    svg = generate_svg(result, remove_background=False)
    elapsed = time.time() - t0
    rend = render(svg, w, h)
    blur_ssim = meas(img, rend, 1.5)
    raw_ssim = meas(img, rend, 0)
    mae = np.mean(np.abs(img.astype(float) - rend.astype(float)))
    kb = len(svg.encode()) / 1024
    print(f"{name}: blur={blur_ssim:.4f} raw={raw_ssim:.4f} MAE={mae:.2f} "
          f"paths={result.path_count} nodes={result.node_count} "
          f"KB={kb:.0f} time={elapsed:.1f}s")
