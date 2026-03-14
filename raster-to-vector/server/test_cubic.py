"""Test INTER_CUBIC upscale vs INTER_LINEAR for smoother contours."""
import sys, os, cv2, numpy as np, time
from pathlib import Path
from io import BytesIO
from PIL import Image
import cairosvg
from skimage.metrics import structural_similarity as ssim

sys.path.insert(0, str(Path(__file__).parent))
import app.core.multilevel as ml
from app.core.multilevel import generate_svg

ref = cv2.imread(os.path.join(os.path.dirname(__file__), "../../Ref.png"))
crop = ref[50:460, 486:1050]

def render_svg(s, w, h):
    png = cairosvg.svg2png(bytestring=s.encode(), output_width=w, output_height=h)
    return cv2.cvtColor(np.array(Image.open(BytesIO(png)).convert("RGB")), cv2.COLOR_RGB2BGR)

def meas(src, rend, bs=1.5):
    g1, g2 = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY), cv2.cvtColor(rend, cv2.COLOR_BGR2GRAY)
    if bs > 0:
        k = int(bs*6)|1; g1 = cv2.GaussianBlur(g1,(k,k),bs); g2 = cv2.GaussianBlur(g2,(k,k),bs)
    return ssim(g1, g2)

h, w = crop.shape[:2]

# Test INTER_CUBIC by monkey-patching
orig_resize = cv2.resize

def cubic_resize(src, dsize, **kwargs):
    if kwargs.get('interpolation') == cv2.INTER_LINEAR and dsize[0] > src.shape[1]:
        kwargs['interpolation'] = cv2.INTER_CUBIC
    return orig_resize(src, dsize, **kwargs)

print("--- INTER_LINEAR (current) + sigma=0.5 ---")
for i in range(3):
    r = ml.multilevel_vectorize(crop, smooth_sigma=0.5)
    svg = generate_svg(r, remove_background=False)
    rend = render_svg(svg, w, h)
    print(f"  run{i+1}: blur={meas(crop,rend,1.5):.4f} raw={meas(crop,rend,0):.4f} nodes={r.node_count}")

print("\n--- INTER_CUBIC (patched) + sigma=0.5 ---")
cv2.resize = cubic_resize
for i in range(3):
    r = ml.multilevel_vectorize(crop, smooth_sigma=0.5)
    svg = generate_svg(r, remove_background=False)
    rend = render_svg(svg, w, h)
    print(f"  run{i+1}: blur={meas(crop,rend,1.5):.4f} raw={meas(crop,rend,0):.4f} nodes={r.node_count}")

# Save cubic SVG
np.random.seed(42)
r = ml.multilevel_vectorize(crop, smooth_sigma=0.5)
svg = generate_svg(r, remove_background=False)
with open("/tmp/svg_cubic_crop.svg", "w") as f:
    f.write(svg)
print(f"\nSaved /tmp/svg_cubic_crop.svg ({r.node_count} nodes)")

cv2.resize = orig_resize
