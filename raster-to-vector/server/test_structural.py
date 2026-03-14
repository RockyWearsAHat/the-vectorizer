"""Test structural improvements: straight-line detection + sweep winners."""
import cv2
import time
from app.core.multilevel import multilevel_vectorize, generate_svg
from app.core.comparison import compare

images = {
    "crop": None,
    "ref": cv2.imread("/Users/alexwaldmann/Desktop/SVG-gen/Ref.png"),
    "mahal": cv2.imread("/tmp/mahal_right.png"),
}

# Create crop
ref = images["ref"]
h, w = ref.shape[:2]
crop_h, crop_w = min(410, h), min(564, w)
x_start = max(0, (w - crop_w) // 2)
images["crop"] = ref[0:crop_h, x_start:x_start + crop_w]

for name, img in images.items():
    if img is None:
        print(f"  {name}: IMAGE NOT FOUND")
        continue
    t0 = time.time()
    result = multilevel_vectorize(img, num_levels=24)
    svg = generate_svg(result, remove_background=False)
    comp = compare(img, svg)
    dt = time.time() - t0

    # Count L vs C commands
    l_count = svg.count(" L") + svg.count('"L')  # L after space or quote
    c_count = svg.count(" C") + svg.count('"C')

    print(f"  {name:6s} {img.shape[1]}x{img.shape[0]}: "
          f"SSIM={comp.ssim_score:.4f} MAE={comp.mae:.2f} "
          f"layers={len(result.layers)} paths={result.path_count} "
          f"L={l_count} C={c_count} "
          f"time={dt:.1f}s")

    # Save SVGs for visual inspection
    with open(f"/tmp/structural_{name}.svg", "w") as f:
        f.write(svg)

print("\nSaved SVGs to /tmp/structural_*.svg")
