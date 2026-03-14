import cv2, numpy as np
from app.core.multilevel import _merge_close_clusters

img = cv2.imread('/Users/alexwaldmann/Desktop/SVG-gen/Ref.png')
crop = img[50:460, 486:1050]
mahal = cv2.imread('/tmp/mahal_right.png')

for name, src in [('crop', crop), ('mahal', mahal)]:
    h, w = src.shape[:2]
    denoised = cv2.bilateralFilter(src, 15, 12, 30)
    pixels = denoised.reshape(-1, 3).astype(np.float32)
    K = 24
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(pixels, K, None, criteria, 6, cv2.KMEANS_PP_CENTERS)
    print(f'{name}: K-means={K} centers')
    centers2, labels2 = _merge_close_clusters(centers, labels.flatten(), h, w, threshold=60.0)
    print(f'{name}: After merge (thresh=60): K={len(centers2)}')
    centers3, labels3 = _merge_close_clusters(centers, labels.flatten(), h, w, threshold=40.0)
    print(f'{name}: After merge (thresh=40): K={len(centers3)}')
    centers4, labels4 = _merge_close_clusters(centers, labels.flatten(), h, w, threshold=30.0)
    print(f'{name}: After merge (thresh=30): K={len(centers4)}')
