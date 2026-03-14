"""Test vectorization on the REAL reference image."""
import cv2
import numpy as np
from app.core.multilevel import multilevel_vectorize, generate_svg
from app.core.comparison import compare

img = cv2.imread("/Users/alexwaldmann/Desktop/SVG-gen/Ref.png")
if img is None:
    print("ERROR: Could not load Ref.png")
    exit(1)

h, w = img.shape[:2]
print(f"Image size: {w}x{h}")
print(f"Channels: {img.shape[2] if len(img.shape) == 3 else 1}")

# Analyze the image colors
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(f"Gray range: {gray.min()} - {gray.max()}")
print(f"Gray mean: {gray.mean():.1f}, std: {gray.std():.1f}")

# Histogram of gray values
hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
print("\nTop 10 gray values by pixel count:")
top = np.argsort(hist)[::-1][:10]
for g in top:
    print(f"  gray={g:3d}: {int(hist[g]):7d} px ({hist[g]/gray.size*100:.1f}%)")

# Vectorize with different K values
for K in [16, 24, 32, 48]:
    result = multilevel_vectorize(img, num_levels=K)
    svg = generate_svg(result, remove_background=False)
    comp = compare(img, svg)
    print(f"\nK={K}: layers={len(result.layers)}, paths={result.path_count}, "
          f"nodes={result.node_count}, SSIM={comp.ssim_score:.4f}, MAE={comp.mae:.2f}")
    
    # Save best SVG
    if K == 24:
        with open("/tmp/ref_vectorized.svg", "w") as f:
            f.write(svg)
        svg_nobg = generate_svg(result, remove_background=True)
        with open("/tmp/ref_vectorized_nobg.svg", "w") as f:
            f.write(svg_nobg)
