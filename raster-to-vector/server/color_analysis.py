import cv2, sys, io, numpy as np
from PIL import Image
sys.path.insert(0, ".")
from app.core.multilevel import multilevel_vectorize, generate_svg
import cairosvg

img = cv2.imread("../../Ref.png")
result = multilevel_vectorize(img)
svg = generate_svg(result)
h, w = img.shape[:2]

# Rasterize SVG to color
png = cairosvg.svg2png(bytestring=svg.encode(), output_width=w, output_height=h, background_color="white")
svg_bgr = cv2.cvtColor(np.array(Image.open(io.BytesIO(png)).convert("RGB")), cv2.COLOR_RGB2BGR)

# Only compare at dark/foreground pixels (where the letter is)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
fg_mask = gray < 128

# Color difference at foreground pixels
for ch, name in [(0,"B"),(1,"G"),(2,"R")]:
    src_ch = img[:,:,ch][fg_mask].astype(float)
    svg_ch = svg_bgr[:,:,ch][fg_mask].astype(float)
    diff = svg_ch - src_ch
    print(f"{name}: src_mean={src_ch.mean():.1f}  svg_mean={svg_ch.mean():.1f}  bias={diff.mean():+.2f}  MAE={np.abs(diff).mean():.2f}")

# Check cluster centers vs actual pixel colors
print(f"\nLayers:")
for i, layer in enumerate(result.layers):
    print(f"  {i}: color={layer.color}  opacities={layer.opacities}")

print(f"\nBackground: {result.background_color}")

# Edge vs interior analysis
edges = cv2.Canny(gray, 50, 150)
edge_dilated = cv2.dilate(edges, np.ones((5,5), np.uint8))
edge_zone = (edge_dilated > 0) & fg_mask
interior_zone = fg_mask & ~edge_zone

for zone, label in [(edge_zone, "edge"), (interior_zone, "interior")]:
    if zone.sum() == 0:
        continue
    for ch, name in [(0,"B"),(1,"G"),(2,"R")]:
        diff = svg_bgr[:,:,ch][zone].astype(float) - img[:,:,ch][zone].astype(float)
        print(f"  {label} {name}: bias={diff.mean():+.2f}  MAE={np.abs(diff).mean():.2f}")
