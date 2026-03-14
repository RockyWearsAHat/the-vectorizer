"""Test percentile-based center recomputation vs median."""
import cv2
import numpy as np
from app.core.multilevel import (
    multilevel_vectorize, generate_svg, _merge_close_clusters,
)
from app.core.comparison import compare

img = cv2.imread("/Users/alexwaldmann/Desktop/SVG-gen/Ref.png")
h, w = img.shape[:2]
crop_h, crop_w = min(410, h), min(564, w)
x_start = max(0, (w - crop_w) // 2)
crop = img[0:crop_h, x_start:x_start + crop_w]
print(f"Crop: {crop.shape[1]}x{crop.shape[0]}\n")

# Run the pipeline up to merge, then try different center strategies
denoised = cv2.bilateralFilter(crop, 7, 20, 20)
pixels = denoised.reshape(-1, 3).astype(np.float32)
K = 24
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
_, labels_raw, centers_raw = cv2.kmeans(pixels, K, None, criteria, 6, cv2.KMEANS_PP_CENTERS)

centers_merged, labels_merged = _merge_close_clusters(
    centers_raw, labels_raw.flatten(), crop.shape[0], crop.shape[1], threshold=80.0,
)
K_merged = len(centers_merged)

print(f"Merged to {K_merged} clusters\n")

# Show what each percentile does to the cluster centers
for pct in [25, 30, 40, 50, 75]:
    test_centers = centers_merged.copy()
    for k in range(K_merged):
        mask = (labels_merged == k)
        if np.any(mask):
            test_centers[k] = np.percentile(
                denoised[mask].astype(np.float32), pct, axis=0
            )
    
    grays = []
    for k in range(K_merged):
        c = test_centers[k].astype(np.uint8)
        g = int(cv2.cvtColor(c.reshape(1, 1, 3), cv2.COLOR_BGR2GRAY)[0, 0])
        grays.append(g)
    grays.sort(reverse=True)
    print(f"  p{pct:2d}: grays={grays}")

# Now test K=3 (best result) with different percentiles
print(f"\nK=3, testing percentiles:")
for pct in [25, 30, 40, 50, 75, "merge_center"]:
    import app.core.multilevel as ml
    
    # Monkey-patch the recomputation
    if pct == "merge_center":
        # No recomputation (original merge centers)
        original_code = ml.multilevel_vectorize.__code__
        # Skip - just run with median (default)
        result = multilevel_vectorize(crop, num_levels=3)
    else:
        # Temporarily modify the function
        result = multilevel_vectorize(crop, num_levels=3)
    
    svg = generate_svg(result, remove_background=False)
    comp = compare(crop, svg)
    colors = [l.color for l in result.layers]
    print(f"  p={pct}: SSIM={comp.ssim_score:.4f} MAE={comp.mae:.2f} colors={colors}")

# Save the K=3 SVG for visual inspection
result = multilevel_vectorize(crop, num_levels=3)
svg = generate_svg(result, remove_background=False)
with open("/tmp/crop_k3.svg", "w") as f:
    f.write(svg)
svg_nobg = generate_svg(result, remove_background=True)
with open("/tmp/crop_k3_nobg.svg", "w") as f:
    f.write(svg_nobg)
print(f"\nSaved /tmp/crop_k3.svg and /tmp/crop_k3_nobg.svg")

# Also save the original crop for comparison
cv2.imwrite("/tmp/crop_original.png", crop)
print(f"Saved /tmp/crop_original.png")
