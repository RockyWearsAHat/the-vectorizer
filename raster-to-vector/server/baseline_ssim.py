"""Check: what SSIM would a blank white image score against a
typical subtle-detail image?  This tells us our "baseline"."""

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# Simulate reading the real image: mostly white bg, light florals, dark text
h, w = 400, 520
img = np.full((h, w, 3), 245, dtype=np.uint8)

# Florals
cv2.ellipse(img, (180, 120), (100, 60), -30, 0, 360, (230, 220, 225), -1)
cv2.ellipse(img, (250, 100), (80, 50), 20, 0, 360, (225, 215, 220), -1)
cv2.ellipse(img, (340, 130), (90, 55), -10, 0, 360, (235, 225, 230), -1)
cv2.ellipse(img, (200, 180), (70, 45), 40, 0, 360, (228, 218, 223), -1)
cv2.ellipse(img, (350, 200), (85, 50), -20, 0, 360, (232, 222, 227), -1)

# Dark text
cv2.putText(img, "M", (170, 340), cv2.FONT_HERSHEY_SIMPLEX, 4, (30, 30, 30), 8)
cv2.putText(img, "B", (300, 370), cv2.FONT_HERSHEY_SIMPLEX, 4.5, (30, 30, 30), 8)

src_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Blank white
white = np.full_like(src_gray, 255)
s = ssim(src_gray, white)
print(f"SSIM vs plain white: {s:.4f}")

# Blank at bg color (245)
bg = np.full_like(src_gray, 245)
s2 = ssim(src_gray, bg)
print(f"SSIM vs bg gray (245): {s2:.4f}")
