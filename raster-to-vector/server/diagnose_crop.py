"""Diagnose why the cropped 564x410 region gets poor SSIM."""
import cv2
import numpy as np
from app.core.multilevel import multilevel_vectorize, generate_svg, detect_background
from app.core.comparison import compare


def analyze(img, name):
    h, w = img.shape[:2]
    print(f"\n{'='*60}")
    print(f"{name}: {w}x{h}")

    # Show what K-means + merge produces
    denoised = cv2.bilateralFilter(img, 7, 20, 20)
    pixels = denoised.reshape(-1, 3).astype(np.float32)
    K = 24
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(pixels, K, None, criteria, 6, cv2.KMEANS_PP_CENTERS)

    # Show pre-merge cluster centers + sizes
    labels_flat = labels.flatten()
    grays = []
    for k in range(K):
        c = centers[k].astype(np.uint8)
        g = int(cv2.cvtColor(c.reshape(1, 1, 3), cv2.COLOR_BGR2GRAY)[0, 0])
        count = np.sum(labels_flat == k)
        pct = 100.0 * count / len(labels_flat)
        grays.append((g, pct, k))
    grays.sort()
    print(f"\nPre-merge clusters (K={K}):")
    for g, pct, k in grays:
        bar = '#' * int(pct * 2)
        print(f"  cluster {k:2d}: gray={g:3d}  {pct:5.1f}%  {bar}")

    # Now run full pipeline
    result = multilevel_vectorize(img, num_levels=24)
    svg = generate_svg(result, remove_background=False)
    comp = compare(img, svg)

    print(f"\nPost-merge: {len(result.layers)} layers, {result.path_count} paths")
    for i, layer in enumerate(result.layers):
        print(f"  layer {i}: color={layer.color}")
    print(f"SSIM={comp.ssim_score:.4f}  MAE={comp.mae:.2f}  diff={comp.pixel_diff_ratio:.4f}")


# Load the image the user appears to be uploading (Ref.png = Mahal Blooms logo)
img = cv2.imread("/Users/alexwaldmann/Desktop/SVG-gen/Ref.png")
if img is None:
    print("Ref.png not found, trying mahal_right.png")
    img = cv2.imread("/tmp/mahal_right.png")

if img is not None:
    h, w = img.shape[:2]

    # Full image
    analyze(img, "Full image")

    # Simulate the user's crop: top 564x410 region (the floral + MB part)
    # The user cropped the top portion, excluding "MAHAL BLOOMS" text
    crop_h = min(410, h)
    crop_w = min(564, w)
    # Center the crop horizontally
    x_start = max(0, (w - crop_w) // 2)
    cropped = img[0:crop_h, x_start:x_start + crop_w]
    analyze(cropped, f"Cropped {crop_w}x{crop_h}")

    # Also try with fewer initial clusters
    print(f"\n{'='*60}")
    print("Testing K=6 on crop:")
    result6 = multilevel_vectorize(cropped, num_levels=6)
    svg6 = generate_svg(result6, remove_background=False)
    comp6 = compare(cropped, svg6)
    print(f"  {len(result6.layers)} layers, SSIM={comp6.ssim_score:.4f}")
    for i, layer in enumerate(result6.layers):
        print(f"  layer {i}: color={layer.color}")
