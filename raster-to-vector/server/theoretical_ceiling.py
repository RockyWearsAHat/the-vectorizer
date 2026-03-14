"""Theoretical analysis: what's the SSIM ceiling?

The user says 0.5% loss is the structural minimum from antialiasing.
Let's measure: if we had PERFECT color matching (just rasterize
source colors into flat SVG regions), what SSIM would we get?
That tells us the antialiasing-only floor.

Then measure: how much of the current gap is from color error vs shape error.
"""
import cv2
import numpy as np
from skimage.metrics import structural_similarity
from app.core.multilevel import multilevel_vectorize, generate_svg
from app.core.comparison import compare
import cairosvg
from io import BytesIO
from PIL import Image


def render_svg(svg_str, width, height):
    png = cairosvg.svg2png(bytestring=svg_str.encode(), output_width=width, output_height=height)
    arr = np.array(Image.open(BytesIO(png)))
    return cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2BGR)


ref = cv2.imread("/Users/alexwaldmann/Desktop/SVG-gen/Ref.png")
h, w = ref.shape[:2]
crop = ref[0:min(410, h), max(0, (w-564)//2):max(0, (w-564)//2)+564]

result = multilevel_vectorize(crop, num_levels=24)
svg = generate_svg(result, remove_background=False)
rendered = render_svg(svg, crop.shape[1], crop.shape[0])

src_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
ren_gray = cv2.cvtColor(rendered, cv2.COLOR_BGR2GRAY).astype(np.float32)

# Current SSIM
src_b = cv2.GaussianBlur(src_gray, (0,0), 1.5)
ren_b = cv2.GaussianBlur(ren_gray, (0,0), 1.5)
current_ssim = structural_similarity(src_b, ren_b, data_range=255)
print(f"Current SSIM (blur): {current_ssim:.4f}")

# Test 1: What if the source itself was Gaussian-blurred?
# This simulates "perfect vector with antialiasing"
# The SSIM of source vs blurred source is the theoretical ceiling
# for vector representation
for sigma in [0.5, 1.0, 1.5, 2.0, 3.0]:
    src_blur = cv2.GaussianBlur(src_gray, (0,0), sigma)
    # SSIM of original vs blurred original
    ssim = structural_similarity(src_gray, src_blur, data_range=255)
    print(f"  Source vs Blur(σ={sigma}): SSIM = {ssim:.4f}")

print()

# Test 2: If we could replace each SVG region's color with the
# actual mean color of the source pixels in that region, what SSIM?
# This isolates "shape error" from "color error"
print("=== Color correction experiment ===")

# For each pixel, find its nearest SVG color
svg_colors = []
for layer in result.layers:
    b, g, r = int(layer.color[1:3], 16), int(layer.color[3:5], 16), int(layer.color[5:7], 16)
    svg_colors.append(np.array([b, g, r], dtype=np.float32))
bg_b, bg_g, bg_r = int(result.background_color[1:3], 16), int(result.background_color[3:5], 16), int(result.background_color[5:7], 16)
svg_colors.append(np.array([bg_b, bg_g, bg_r], dtype=np.float32))

# Assign each pixel in rendered image to nearest SVG color
ren_f = rendered.astype(np.float32).reshape(-1, 3)
all_colors = np.array(svg_colors)  # shape (N, 3)

# For each SVG color region, compute mean source color
corrected = rendered.copy().astype(np.float32)
for i, sc in enumerate(svg_colors):
    # Mask: pixels that are close to this SVG color (within 2 units)
    diff = np.abs(rendered.astype(np.float32) - sc.reshape(1, 1, 3))
    close = np.all(diff < 3, axis=2)  # strict match
    if close.sum() == 0:
        continue
    # Mean source color in this region
    mean_src = crop[close].astype(np.float32).mean(axis=0)
    corrected[close] = mean_src
    pct = close.sum() / close.size * 100
    color_hex = f"#{int(sc[2]):02x}{int(sc[1]):02x}{int(sc[0]):02x}"
    mean_hex = f"#{int(mean_src[2]):02x}{int(mean_src[1]):02x}{int(mean_src[0]):02x}"
    print(f"  Region {color_hex}: {pct:.1f}% of pixels, SVG={color_hex} actual_mean={mean_hex} "
          f"delta={mean_src - sc}")

corrected = np.clip(corrected, 0, 255).astype(np.uint8)
corr_gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY).astype(np.float32)
corr_b = cv2.GaussianBlur(corr_gray, (0,0), 1.5)
corrected_ssim = structural_similarity(src_b, corr_b, data_range=255)
print(f"\n  Color-corrected SSIM: {corrected_ssim:.4f}")
print(f"  Current SSIM:        {current_ssim:.4f}")
print(f"  Improvement:         {corrected_ssim - current_ssim:.4f}")

print()

# Test 3: What if we just quantize the source image to the same
# number of colors (perfect quantization, no shape changed)?
# This is the "color-only loss"
print("=== Quantization-only loss ===")
for n_colors in [5, 8, 12, 16, 24, 32, 48]:
    pixels = crop.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(pixels, n_colors, None, criteria, 6, cv2.KMEANS_PP_CENTERS)
    quantized = centers[labels.flatten()].reshape(crop.shape).astype(np.uint8)
    q_gray = cv2.cvtColor(quantized, cv2.COLOR_BGR2GRAY).astype(np.float32)
    q_b = cv2.GaussianBlur(q_gray, (0,0), 1.5)
    q_ssim = structural_similarity(src_b, q_b, data_range=255)
    print(f"  {n_colors:3d} colors (pixel-perfect): SSIM = {q_ssim:.4f}")

print()

# Test 4: Measure how much of the error is purely from the
# Gaussian blur used in SSIM comparison itself
print("=== SSIM comparison bias ===")
# Perfect reproduction = 1.0, but our blur introduces a floor
ssim_perfect = structural_similarity(src_b, src_b, data_range=255)
print(f"  Source vs Source (through blur): {ssim_perfect:.6f}")
# Source vs source with 1-pixel shift
shifted = np.roll(src_gray, 1, axis=1)
shifted_b = cv2.GaussianBlur(shifted, (0,0), 1.5)
ssim_shift = structural_similarity(src_b, shifted_b, data_range=255)
print(f"  Source vs 1px-shifted source:    {ssim_shift:.6f}")
