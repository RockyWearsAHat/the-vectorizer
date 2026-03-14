"""Test with anti-aliased text (like the real image would have).
This simulates a realistic logo more closely."""
import cv2
import numpy as np
from app.core.multilevel import multilevel_vectorize, generate_svg
from app.core.comparison import compare

h, w = 400, 520
img = np.full((h, w, 3), 245, dtype=np.uint8)

# Florals with soft edges (anti-aliased)
cv2.ellipse(img, (180, 120), (100, 60), -30, 0, 360, (230, 220, 225), -1, cv2.LINE_AA)
cv2.ellipse(img, (250, 100), (80, 50), 20, 0, 360, (225, 215, 220), -1, cv2.LINE_AA)
cv2.ellipse(img, (340, 130), (90, 55), -10, 0, 360, (235, 225, 230), -1, cv2.LINE_AA)
cv2.ellipse(img, (200, 180), (70, 45), 40, 0, 360, (228, 218, 223), -1, cv2.LINE_AA)
cv2.ellipse(img, (350, 200), (85, 50), -20, 0, 360, (232, 222, 227), -1, cv2.LINE_AA)

# Dark text with anti-aliasing
cv2.putText(img, "M", (170, 340), cv2.FONT_HERSHEY_SIMPLEX, 4, (30, 30, 30), 8, cv2.LINE_AA)
cv2.putText(img, "B", (300, 370), cv2.FONT_HERSHEY_SIMPLEX, 4.5, (30, 30, 30), 8, cv2.LINE_AA)

# Add simulated AI noise (Gaussian noise, sigma=5 per channel)
np.random.seed(42)
noise = np.random.normal(0, 5, img.shape).astype(np.float32)
img_noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

cv2.imwrite("/tmp/test_noisy_logo.png", img_noisy)

print("Testing with clean image:")
result = multilevel_vectorize(img, num_levels=24)
svg = generate_svg(result, remove_background=False)
comp = compare(img, svg)
print(f"  Layers: {len(result.layers)}, Paths: {result.path_count}, SSIM: {comp.ssim_score:.4f}")

print("\nTesting with noisy image (simulated AI):")
result2 = multilevel_vectorize(img_noisy, num_levels=24)
svg2 = generate_svg(result2, remove_background=False)
comp2 = compare(img_noisy, svg2)
print(f"  Layers: {len(result2.layers)}, Paths: {result2.path_count}, SSIM: {comp2.ssim_score:.4f}")

# Save SVGs for inspection
with open("/tmp/test_clean.svg", "w") as f: f.write(svg)
with open("/tmp/test_noisy.svg", "w") as f: f.write(svg2)

# Now test with higher noise to simulate more aggressive SD noise
print("\nTesting with higher noise (sigma=10):")
noise2 = np.random.normal(0, 10, img.shape).astype(np.float32)
img_noisy2 = np.clip(img.astype(np.float32) + noise2, 0, 255).astype(np.uint8)
result3 = multilevel_vectorize(img_noisy2, num_levels=24)
svg3 = generate_svg(result3, remove_background=False)
comp3 = compare(img_noisy2, svg3)
print(f"  Layers: {len(result3.layers)}, Paths: {result3.path_count}, SSIM: {comp3.ssim_score:.4f}")
