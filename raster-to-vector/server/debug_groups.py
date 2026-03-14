"""Debug: show luminance groups for mahal_right."""
import cv2
import numpy as np
from app.core.multilevel import _group_clusters_by_luminance, _merge_close_clusters

img = cv2.imread('/tmp/mahal_right.png')
h, w = img.shape[:2]
denoised = cv2.bilateralFilter(img, 7, 20, 20)
pixels = denoised.reshape(-1, 3).astype(np.float32)
K = 24
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
_, labels, centers = cv2.kmeans(pixels, K, None, criteria, 6, cv2.KMEANS_PP_CENTERS)
centers, labels = _merge_close_clusters(centers, labels.flatten(), h, w, threshold=12.0)
K = len(centers)

centers_f = centers.astype(np.float32)
groups = _group_clusters_by_luminance(centers_f, min_gap=30.0)

print(f"K={K}, {len(groups)} luminance groups:")
for gi, group in enumerate(groups):
    grays = []
    for ci in group:
        c = centers[ci].astype(np.uint8)
        g = int(cv2.cvtColor(c.reshape(1, 1, 3), cv2.COLOR_BGR2GRAY)[0, 0])
        grays.append((g, ci))
    grays.sort()
    gray_str = ", ".join(f"gray={g}(#{ci})" for g, ci in grays)
    print(f"  Group {gi}: [{gray_str}]")
