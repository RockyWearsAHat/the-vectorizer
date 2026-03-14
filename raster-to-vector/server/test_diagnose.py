"""Diagnose which clusters produce the gray ghosting."""
import cv2, numpy as np

img = cv2.imread("/Users/alexwaldmann/Desktop/SVG-gen/Ref.png")
crop = img[50:460, 486:1050]
h, w = crop.shape[:2]

# Replicate the pipeline's preprocessing
denoised_km = cv2.bilateralFilter(crop, 15, 12, 30)
pixels = denoised_km.reshape(-1, 3).astype(np.float32)
K = 24
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
_, labels, centers = cv2.kmeans(pixels, K, None, criteria, 6, cv2.KMEANS_PP_CENTERS)

# Show pre-merge clusters
print("=== Pre-merge clusters ===")
centers_u = centers.astype(np.uint8)
for k in range(K):
    c = centers_u[k]
    gray = int(cv2.cvtColor(c.reshape(1,1,3), cv2.COLOR_BGR2GRAY)[0,0])
    count = np.count_nonzero(labels.flatten() == k)
    pct = 100.0 * count / (h * w)
    print(f"  Cluster {k:2d}: BGR=({c[0]:3d},{c[1]:3d},{c[2]:3d}) gray={gray:3d} pixels={count:6d} ({pct:.1f}%)")

# Now merge
from app.core.multilevel import _merge_close_clusters, detect_background
centers_merged, labels_merged = _merge_close_clusters(centers, labels.flatten(), h, w, threshold=60.0)
K2 = len(centers_merged)

bg_color, bg_gray = detect_background(crop)
print(f"\n=== Post-merge clusters (K={K2}) ===")
print(f"Background: BGR=({bg_color[0]},{bg_color[1]},{bg_color[2]}) gray={bg_gray}")
centers_u2 = centers_merged.astype(np.uint8)

bg_dists = np.array([
    np.linalg.norm(centers_merged[k].astype(np.float32) - bg_color.astype(np.float32))
    for k in range(K2)
])
bg_cluster = int(np.argmin(bg_dists)) if bg_dists.min() < 40.0 else -1

grays = []
for k in range(K2):
    c = centers_u2[k]
    gray = int(cv2.cvtColor(c.reshape(1,1,3), cv2.COLOR_BGR2GRAY)[0,0])
    grays.append(gray)
    count = np.count_nonzero(labels_merged == k)
    pct = 100.0 * count / (h * w)
    is_bg = " [BG]" if k == bg_cluster else ""
    print(f"  Cluster {k:2d}: BGR=({c[0]:3d},{c[1]:3d},{c[2]:3d}) gray={gray:3d} pixels={count:6d} ({pct:.1f}%){is_bg}")

# Identify intermediate gray clusters (not bg, not darkest)
grays = np.array(grays)
order = np.argsort(-grays)
print(f"\nRender order (lightest first): {order}")
print(f"Background cluster: {bg_cluster}")
print(f"\nIntermediate clusters (gray 50-200, not bg):")
for k in order:
    if k == bg_cluster:
        continue
    if 50 < grays[k] < 200:
        c = centers_u2[k]
        count = np.count_nonzero(labels_merged == k)
        pct = 100.0 * count / (h * w)
        print(f"  Cluster {k}: gray={grays[k]} ({pct:.1f}% of pixels) - GRAY GHOST SOURCE")
