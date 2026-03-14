"""Diagnostic script: save a test image that mimics the Mahal Blooms image
  and dump cluster info + rendered SVG for inspection."""

import cv2
import numpy as np
from app.core.multilevel import multilevel_vectorize, generate_svg

# Create a synthetic image that mimics the Mahal Blooms problem:
# - White background (~245)
# - Very light pink/gray florals (~215-235)
# - Dark text (~20-40)
h, w = 400, 520
img = np.full((h, w, 3), 245, dtype=np.uint8)  # near-white bg

# Floral shapes: light pink-gray, roughly 20-30 gray levels below white
# Draw some petal-like ellipses 
cv2.ellipse(img, (180, 120), (100, 60), -30, 0, 360, (230, 220, 225), -1)
cv2.ellipse(img, (250, 100), (80, 50), 20, 0, 360, (225, 215, 220), -1)
cv2.ellipse(img, (340, 130), (90, 55), -10, 0, 360, (235, 225, 230), -1)
cv2.ellipse(img, (200, 180), (70, 45), 40, 0, 360, (228, 218, 223), -1)
cv2.ellipse(img, (350, 200), (85, 50), -20, 0, 360, (232, 222, 227), -1)
# Smaller flower elements
cv2.ellipse(img, (420, 260), (50, 35), 15, 0, 360, (230, 220, 225), -1)
cv2.ellipse(img, (150, 250), (60, 40), -25, 0, 360, (227, 217, 222), -1)

# Dark text: "MB" approximation
cv2.putText(img, "M", (170, 340), cv2.FONT_HERSHEY_SIMPLEX, 4, (30, 30, 30), 8)
cv2.putText(img, "B", (300, 370), cv2.FONT_HERSHEY_SIMPLEX, 4.5, (30, 30, 30), 8)

cv2.imwrite("/tmp/mahal_synthetic.png", img)
print(f"Saved synthetic test image: {w}x{h}")

# Run vectorization
result = multilevel_vectorize(img, num_levels=24)

print(f"\n--- Vectorization result ---")
print(f"Background color: {result.background_color}")
print(f"Layers: {len(result.layers)}")
print(f"Total paths: {result.path_count}")
print(f"Total nodes: {result.node_count}")

for i, layer in enumerate(result.layers):
    print(f"  Layer {i}: color={layer.color}, paths={len(layer.paths)}, "
          f"d_len={sum(len(d) for d in layer.paths)}")

# Generate and save SVG
svg_bg = generate_svg(result, remove_background=False)
with open("/tmp/mahal_synthetic.svg", "w") as f:
    f.write(svg_bg)

# Compare
from app.core.comparison import compare
comp = compare(img, svg_bg)
print(f"\n--- Comparison ---")
print(f"SSIM: {comp.ssim_score:.4f}")
print(f"MAE: {comp.mae:.4f}")
print(f"Diff pixels: {comp.pixel_diff_ratio:.4f}")

# Now let's debug K-means directly
print(f"\n--- K-means debug ---")
pixels = img.reshape(-1, 3).astype(np.float32)
K = 24
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
_, labels, centers = cv2.kmeans(pixels, K, None, criteria, 6, cv2.KMEANS_PP_CENTERS)

labels_flat = labels.flatten()
for k in range(K):
    count = np.sum(labels_flat == k)
    c = centers[k].astype(int)
    gray = int(cv2.cvtColor(centers[k].reshape(1,1,3).astype(np.uint8), cv2.COLOR_BGR2GRAY)[0,0])
    pct = count / len(labels_flat) * 100
    print(f"  Cluster {k:2d}: BGR=({c[0]:3d},{c[1]:3d},{c[2]:3d}) gray={gray:3d} pixels={count:6d} ({pct:5.1f}%)")
