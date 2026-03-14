"""Deep diagnosis: where exactly is the 1.8% SSIM loss?

Break down error spatially, by color region, and by source.
"""
import cv2
import numpy as np
from app.core.multilevel import multilevel_vectorize, generate_svg
from app.core.comparison import compare
from skimage.metrics import structural_similarity
import cairosvg
from io import BytesIO
from PIL import Image

def render_svg(svg_str, width, height):
    png = cairosvg.svg2png(bytestring=svg_str.encode(), output_width=width, output_height=height)
    arr = np.array(Image.open(BytesIO(png)))
    return cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2BGR)

# Load
ref = cv2.imread("/Users/alexwaldmann/Desktop/SVG-gen/Ref.png")
h, w = ref.shape[:2]
crop = ref[0:min(410, h), max(0, (w-564)//2):max(0, (w-564)//2)+564]

# Run pipeline
result = multilevel_vectorize(crop, num_levels=24)
svg = generate_svg(result, remove_background=False)
rendered = render_svg(svg, crop.shape[1], crop.shape[0])

print(f"Clusters surviving: {len(result.layers)} layers")
print(f"Paths: {result.path_count}, Nodes: {result.node_count}")
print(f"Colors: {[l.color for l in result.layers]}")
print()

# --- Per-pixel error analysis ---
src_f = crop.astype(np.float32)
ren_f = rendered.astype(np.float32)
diff = src_f - ren_f  # signed difference per channel
abs_diff = np.abs(diff)
per_pixel_err = np.mean(abs_diff, axis=2)  # average across BGR channels

print("=== Per-pixel error stats ===")
print(f"  Mean absolute error: {per_pixel_err.mean():.2f}")
print(f"  Median error: {np.median(per_pixel_err):.2f}")
print(f"  90th percentile: {np.percentile(per_pixel_err, 90):.2f}")
print(f"  95th percentile: {np.percentile(per_pixel_err, 95):.2f}")
print(f"  99th percentile: {np.percentile(per_pixel_err, 99):.2f}")
print(f"  Max error: {per_pixel_err.max():.2f}")
print()

# Fraction of pixels with various error levels
for thresh in [0, 1, 2, 3, 5, 8, 10, 15, 20, 30, 50]:
    frac = np.mean(per_pixel_err > thresh) * 100
    print(f"  Pixels with error > {thresh:2d}: {frac:5.1f}%")
print()

# --- SSIM with and without blur ---
src_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
ren_gray = cv2.cvtColor(rendered, cv2.COLOR_BGR2GRAY)

ssim_raw = structural_similarity(src_gray, ren_gray, data_range=255)
src_blur = cv2.GaussianBlur(src_gray, (0,0), 1.5)
ren_blur = cv2.GaussianBlur(ren_gray, (0,0), 1.5)
ssim_blur = structural_similarity(src_blur, ren_blur, data_range=255)

print(f"=== SSIM ===")
print(f"  Raw (no blur):     {ssim_raw:.4f}")
print(f"  With blur σ=1.5:   {ssim_blur:.4f}")
print(f"  Gap from 0.995:    {0.995 - ssim_blur:.4f}")
print()

# --- SSIM map: where is the error? ---
_, ssim_map = structural_similarity(src_blur, ren_blur, data_range=255, full=True)
ssim_map_f = ssim_map.astype(np.float32)

# Find worst regions
low_ssim = ssim_map_f < 0.95  # regions with SSIM < 0.95
low_ssim_pct = np.mean(low_ssim) * 100
print(f"=== Spatial error distribution ===")
print(f"  Pixels with local SSIM < 0.95: {low_ssim_pct:.1f}%")
print(f"  Pixels with local SSIM < 0.90: {np.mean(ssim_map_f < 0.90)*100:.1f}%")
print(f"  Pixels with local SSIM < 0.80: {np.mean(ssim_map_f < 0.80)*100:.1f}%")
print()

# --- Error by brightness band ---
print("=== Error by source brightness ===")
src_lum = src_gray.astype(np.float32)
for lo, hi in [(0, 50), (50, 100), (100, 150), (150, 200), (200, 255)]:
    mask = (src_lum >= lo) & (src_lum < hi)
    if mask.sum() == 0:
        continue
    band_err = per_pixel_err[mask].mean()
    band_pct = mask.sum() / mask.size * 100
    band_ssim = ssim_map_f[mask].mean()
    print(f"  Lum [{lo:3d}-{hi:3d}]: {band_pct:5.1f}% of pixels, "
          f"MAE={band_err:.1f}, local SSIM={band_ssim:.4f}")
print()

# --- Error by signed direction: over-bright vs over-dark ---
signed_err = np.mean(diff, axis=2)  # positive = SVG brighter than source
over_bright = signed_err > 3
over_dark = signed_err < -3
print(f"=== Color direction ===")
print(f"  SVG too bright (>3): {np.mean(over_bright)*100:.1f}% of pixels, mean overshoot: {signed_err[over_bright].mean():.1f}" if over_bright.any() else "  SVG too bright: 0%")
print(f"  SVG too dark (<-3):  {np.mean(over_dark)*100:.1f}% of pixels, mean undershoot: {signed_err[over_dark].mean():.1f}" if over_dark.any() else "  SVG too dark: 0%")
print()

# --- What colors are NOT covered? ---
# For pixels with high error, what's the source color?
high_err_mask = per_pixel_err > 10
if high_err_mask.any():
    he_colors = crop[high_err_mask]
    he_rendered = rendered[high_err_mask]
    print(f"=== High-error pixels (MAE > 10): {np.mean(high_err_mask)*100:.1f}% ===")
    print(f"  Source colors: mean BGR = {he_colors.mean(axis=0).astype(int)}")
    print(f"  SVG colors:   mean BGR = {he_rendered.mean(axis=0).astype(int)}")
    print(f"  Delta:         {(he_colors.mean(axis=0) - he_rendered.mean(axis=0)).astype(int)}")
    
    # Cluster the high-error pixels to see what colors are missing
    he_f = he_colors.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, _, he_centers = cv2.kmeans(he_f, min(5, len(he_f)), None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    he_centers_u = he_centers.astype(np.uint8)
    print(f"  Top error source colors (BGR):")
    for i, c in enumerate(he_centers_u):
        hex_c = f"#{c[2]:02x}{c[1]:02x}{c[0]:02x}"
        print(f"    {i}: BGR={c} = {hex_c}")
    
    print(f"\n  SVG layer colors: {[l.color for l in result.layers]}")
    print(f"  Background: {result.background_color}")
print()

# Save error visualization
err_vis = np.clip(per_pixel_err * 5, 0, 255).astype(np.uint8)  # 5x amplified
cv2.imwrite("/tmp/error_map.png", err_vis)

ssim_err_vis = np.clip((1 - ssim_map_f) * 500, 0, 255).astype(np.uint8)
cv2.imwrite("/tmp/ssim_error_map.png", ssim_err_vis)

print("Saved /tmp/error_map.png (5x amplified MAE)")
print("Saved /tmp/ssim_error_map.png (500x amplified 1-SSIM)")
