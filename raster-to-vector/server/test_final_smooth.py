"""Quick validation of refined smoothing (sigma=0.5 with length-adaptive + corners)."""
import sys, os, cv2, numpy as np, time
from pathlib import Path
from io import BytesIO
from PIL import Image
import cairosvg
from skimage.metrics import structural_similarity as ssim

sys.path.insert(0, str(Path(__file__).parent))
from app.core.multilevel import multilevel_vectorize, generate_svg

ref = cv2.imread(os.path.join(os.path.dirname(__file__), "../../Ref.png"))
crop = ref[50:460, 486:1050]
mahal = cv2.imread("/tmp/mahal_right.png")

def render_svg(s, w, h):
    png = cairosvg.svg2png(bytestring=s.encode(), output_width=w, output_height=h)
    return cv2.cvtColor(np.array(Image.open(BytesIO(png)).convert("RGB")), cv2.COLOR_RGB2BGR)

def meas(src, rend, bs=1.5):
    g1 = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(rend, cv2.COLOR_BGR2GRAY)
    if bs > 0:
        k = int(bs * 6) | 1
        g1 = cv2.GaussianBlur(g1, (k, k), bs)
        g2 = cv2.GaussianBlur(g2, (k, k), bs)
    return ssim(g1, g2)

print("=== smooth_sigma=0.5 (refined: length-adaptive + corner preservation) ===")
for name, img in [("crop", crop), ("mahal", mahal)]:
    h, w = img.shape[:2]
    blurs, raws = [], []
    for i in range(3):
        r = multilevel_vectorize(img)
        svg = generate_svg(r, remove_background=False)
        rend = render_svg(svg, w, h)
        b = meas(img, rend, 1.5)
        raw = meas(img, rend, 0)
        blurs.append(b)
        raws.append(raw)
        print(f"  {name} run{i+1}: blur={b:.4f} raw={raw:.4f} paths={r.path_count} nodes={r.node_count}")
    print(f"  {name} AVG:  blur={np.mean(blurs):.4f} raw={np.mean(raws):.4f}")

# Save SVGs for visual inspection
np.random.seed(42)
r = multilevel_vectorize(crop)
svg = generate_svg(r, remove_background=False)
with open("/tmp/svg_final_crop.svg", "w") as f:
    f.write(svg)
print(f"\nSaved /tmp/svg_final_crop.svg ({r.node_count} nodes)")

np.random.seed(42)
r = multilevel_vectorize(mahal)
svg = generate_svg(r, remove_background=False)
with open("/tmp/svg_final_mahal.svg", "w") as f:
    f.write(svg)
print(f"Saved /tmp/svg_final_mahal.svg ({r.node_count} nodes)")
