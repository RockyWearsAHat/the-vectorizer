"""Diagnose remaining artifacts — where exactly is the error?

Outputs:
  1. Per-region error breakdown (bg, light fill, mid/edge, dark)
  2. Error heatmap saved to /tmp/ for visual inspection
  3. Top error pixels analysis
"""
import sys, os, cv2, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from app.core.multilevel import multilevel_vectorize, generate_svg
from app.core.comparison import compare

ref = cv2.imread("/Users/alexwaldmann/Desktop/SVG-gen/Ref.png")
crop = ref[200:610, 400:964]

result = multilevel_vectorize(crop)
svg = generate_svg(result, remove_background=False)
comp = compare(crop, svg)

print(f"Overall: SSIM={comp.ssim_score:.4f} MAE={comp.mae:.2f} diff={comp.pixel_diff_ratio*100:.1f}%")
print(f"Layers: {len(result.layers)}, Paths: {result.path_count}, Nodes: {result.node_count}")
print()

# Get the rasterized SVG and source in grayscale
src_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
svg_gray = comp.svg_raster
diff = np.abs(src_gray.astype(np.float32) - svg_gray.astype(np.float32))

# Region analysis
regions = {
    "background (>240)":  src_gray > 240,
    "light fill (200-240)": (src_gray >= 200) & (src_gray <= 240),
    "mid/edge (100-200)":   (src_gray >= 100) & (src_gray < 200),
    "dark (<100)":          src_gray < 100,
}

print(f"{'Region':<25} {'% pixels':>8} {'mean_err':>8} {'max_err':>8} {'% of total err':>14}")
print("-" * 70)
total_err = diff.sum()
for name, mask in regions.items():
    npix = mask.sum()
    pct = npix / mask.size * 100
    mean_e = diff[mask].mean() if npix > 0 else 0
    max_e = diff[mask].max() if npix > 0 else 0
    err_pct = diff[mask].sum() / total_err * 100 if total_err > 0 else 0
    print(f"{name:<25} {pct:>7.1f}% {mean_e:>8.2f} {max_e:>8.0f} {err_pct:>13.1f}%")

# Save error heatmap
print()
# Find the worst error spots
high_error = diff > 30
coords = np.argwhere(high_error)
if len(coords) > 0:
    print(f"Pixels with error > 30: {len(coords)} ({len(coords)/diff.size*100:.2f}%)")

    # Cluster the high-error pixels to find artifact regions
    from scipy import ndimage
    labeled, n_features = ndimage.label(high_error)
    print(f"Artifact clusters: {n_features}")

    # Top 10 largest artifact regions
    cluster_sizes = []
    for i in range(1, n_features + 1):
        size = (labeled == i).sum()
        cluster_sizes.append((size, i))
    cluster_sizes.sort(reverse=True)

    print(f"\nTop artifact regions:")
    for size, idx in cluster_sizes[:10]:
        ys, xs = np.where(labeled == idx)
        mean_src = src_gray[labeled == idx].mean()
        mean_svg = svg_gray[labeled == idx].mean()
        mean_diff = diff[labeled == idx].mean()
        print(f"  Region {idx}: {size}px at ({xs.min()}-{xs.max()}, {ys.min()}-{ys.max()}) "
              f"src_gray={mean_src:.0f} svg_gray={mean_svg:.0f} err={mean_diff:.1f}")

# Save visualizations
heatmap = cv2.applyColorMap((diff * 3).clip(0, 255).astype(np.uint8), cv2.COLORMAP_JET)
cv2.imwrite("/tmp/error_heatmap.png", heatmap)

# Save side-by-side
side = np.hstack([
    cv2.cvtColor(src_gray, cv2.COLOR_GRAY2BGR),
    cv2.cvtColor(svg_gray, cv2.COLOR_GRAY2BGR),
    heatmap
])
cv2.imwrite("/tmp/error_sidebyside.png", side)
print("\nSaved /tmp/error_heatmap.png and /tmp/error_sidebyside.png")

# Analyze error distribution
print(f"\nError distribution:")
for t in [5, 10, 15, 20, 30, 50]:
    pct = (diff > t).sum() / diff.size * 100
    print(f"  > {t}: {pct:.2f}%")

# Check: is the SVG generally darker or lighter than source?
overall_bias = (svg_gray.astype(float) - src_gray.astype(float)).mean()
print(f"\nOverall brightness bias: {overall_bias:+.2f} ({'SVG brighter' if overall_bias > 0 else 'SVG darker'})")

# Per-region bias
for name, mask in regions.items():
    if mask.sum() > 0:
        bias = (svg_gray[mask].astype(float) - src_gray[mask].astype(float)).mean()
        print(f"  {name}: {bias:+.2f}")
