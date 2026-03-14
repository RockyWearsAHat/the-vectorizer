"""Check what clusters the LineArt image produces."""
import cv2, numpy as np, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from app.core.multilevel import detect_background, _merge_close_clusters

img2 = np.ones((400, 400, 3), dtype=np.uint8) * 250
cv2.circle(img2, (150, 150), 80, (200, 200, 200), -1, cv2.LINE_AA)
cv2.circle(img2, (280, 200), 60, (180, 180, 180), -1, cv2.LINE_AA)
pts = np.array([[100, 300], [200, 250], [300, 320], [250, 380], [120, 370]], np.int32)
cv2.fillPoly(img2, [pts], (160, 160, 160), cv2.LINE_AA)
cv2.circle(img2, (150, 150), 80, (30, 30, 30), 3, cv2.LINE_AA)
cv2.circle(img2, (280, 200), 60, (30, 30, 30), 3, cv2.LINE_AA)
cv2.polylines(img2, [pts], True, (30, 30, 30), 3, cv2.LINE_AA)
cv2.line(img2, (50, 50), (350, 50), (60, 60, 60), 2, cv2.LINE_AA)
cv2.line(img2, (50, 50), (50, 350), (60, 60, 60), 2, cv2.LINE_AA)

bg_color, bg_gray = detect_background(img2)
print(f"bg_color={bg_color}, bg_gray={bg_gray}")

denoised = cv2.bilateralFilter(img2, 7, 10, 10)
pixels = denoised.reshape(-1, 3).astype(np.float32)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
_, labels, centers = cv2.kmeans(pixels, 24, None, criteria, 6, cv2.KMEANS_PP_CENTERS)
centers, labels = _merge_close_clusters(centers, labels.flatten(), 400, 400, threshold=80.0)
print(f"K={len(centers)}")
for i, c in enumerate(centers):
    d = np.linalg.norm(c.astype(np.float32) - bg_color.astype(np.float32))
    print(f"  {i}: BGR={c.astype(int)} dist_to_bg={d:.1f}")
