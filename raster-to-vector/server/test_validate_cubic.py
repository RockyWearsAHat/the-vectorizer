"""Validate: INTER_CUBIC + smooth_sigma=0.5 (the combined approach)."""
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
    g1, g2 = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY), cv2.cvtColor(rend, cv2.COLOR_BGR2GRAY)
    if bs > 0:
        k = int(bs*6)|1; g1 = cv2.GaussianBlur(g1,(k,k),bs); g2 = cv2.GaussianBlur(g2,(k,k),bs)
    return ssim(g1, g2)

print("=" * 70)
print("VALIDATION: INTER_CUBIC + smooth_sigma=0.5 (5 runs)")
print("=" * 70)

for name, img in [("crop", crop), ("mahal", mahal)]:
    h, w = img.shape[:2]
    print(f"\n--- {name} ({w}x{h}) ---")
    blurs, raws = [], []
    for i in range(5):
        t0 = time.time()
        r = multilevel_vectorize(img)
        svg = generate_svg(r, remove_background=False)
        dt = time.time() - t0
        rend = render_svg(svg, w, h)
        b = meas(img, rend, 1.5)
        raw = meas(img, rend, 0)
        blurs.append(b)
        raws.append(raw)
        print(f"  run{i+1}: blur={b:.4f} raw={raw:.4f} paths={r.path_count} nodes={r.node_count} {dt:.1f}s")
    print(f"  AVG: blur={np.mean(blurs):.4f}+/-{np.std(blurs):.4f}  raw={np.mean(raws):.4f}+/-{np.std(raws):.4f}")

print("\nComparison:")
print("  Before smoothing:     crop blur=0.9950 raw=0.9724 | mahal blur=0.9933 raw=0.9765")
print("  LINEAR + sigma=0.5:   crop blur=0.9937 raw=0.9695 | mahal blur=0.9929 raw=0.9758")
print("  CUBIC + sigma=0.5:    (see above)")
