"""Debug: check significant clusters for each image."""
import cv2
import numpy as np
from app.core.multilevel import _find_significant_clusters, _merge_close_clusters

for path, name in [("/tmp/mahal_right.png", "mahal"),
                     ("/Users/alexwaldmann/Desktop/SVG-gen/Ref.png", "ref")]:
    img = cv2.imread(path)
    if img is None:
        continue
    h, w = img.shape[:2]
    denoised = cv2.bilateralFilter(img, 7, 20, 20)
    pixels = denoised.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(pixels, 24, None, criteria, 6, cv2.KMEANS_PP_CENTERS)
    centers, labels = _merge_close_clusters(centers, labels.flatten(), h, w, threshold=40.0)
    K = len(centers)
    centers_f = centers.astype(np.float32)
    sig = _find_significant_clusters(centers_f, labels)

    labels_flat = labels.flatten()
    total = len(labels_flat)
    print(f"\n{name} ({w}x{h}), K={K}, significant={int(np.sum(sig))}:")
    for k in range(K):
        c = centers[k].astype(np.uint8)
        g = int(cv2.cvtColor(c.reshape(1, 1, 3), cv2.COLOR_BGR2GRAY)[0, 0])
        count = int(np.sum(labels_flat == k))
        pct = count / total * 100
        mark = " *** SIG" if sig[k] else ""
        print(f"  #{k:2d} gray={g:3d} {count:7d}px ({pct:5.2f}%){mark}")
