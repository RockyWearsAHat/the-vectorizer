"""Quick quality check."""
import cv2, numpy as np, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from app.core.multilevel import multilevel_vectorize, generate_svg
from app.core.comparison import compare

img = np.ones((400, 400, 3), dtype=np.uint8) * 245
for y in range(400):
    for x in range(400):
        d = ((x - 200) ** 2 + (y - 150) ** 2) ** 0.5
        if d < 120:
            val = int(50 + d * 1.4)
            img[y, x] = [val, val, val]
cv2.putText(img, "M", (100, 350), cv2.FONT_HERSHEY_SIMPLEX, 4, (120, 120, 120), 8, cv2.LINE_AA)
cv2.putText(img, "B", (220, 350), cv2.FONT_HERSHEY_SIMPLEX, 3, (160, 160, 160), 5, cv2.LINE_AA)
r = multilevel_vectorize(img)
svg = generate_svg(r, remove_background=False)
c = compare(img, svg)
print(f"Tonal: K_eff={len(r.layers)+1} paths={r.path_count} SSIM={c.ssim_score:.4f}")

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
r2 = multilevel_vectorize(img2)
svg2 = generate_svg(r2, remove_background=False)
c2 = compare(img2, svg2)
print(f"LineArt: K_eff={len(r2.layers)+1} paths={r2.path_count} SSIM={c2.ssim_score:.4f}")
