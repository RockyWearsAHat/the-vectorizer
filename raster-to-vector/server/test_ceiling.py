"""Measure theoretical SSIM ceiling from color quantization."""
import cv2, numpy as np, io
from app.core.multilevel import _merge_close_clusters
from skimage.metrics import structural_similarity as ssim
from app.core.comparison import compare
import cairosvg
from PIL import Image

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
    centers2, labels2 = _merge_close_clusters(centers, labels.flatten(), h, w, threshold=60.0)
    K2 = len(centers2)

    # Reconstruct image using quantized colors (pixel-perfect placement)
    recon = centers2[labels2].reshape(h, w, 3).astype(np.uint8)
    src_rgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    recon_rgb = cv2.cvtColor(recon, cv2.COLOR_BGR2RGB)
    raw = ssim(src_rgb, recon_rgb, channel_axis=2)

    # Also blur both before SSIM
    src_blur = cv2.GaussianBlur(src_rgb.astype(np.float32), (0,0), sigmaX=1.5)
    recon_blur = cv2.GaussianBlur(recon_rgb.astype(np.float32), (0,0), sigmaX=1.5)
    blur = ssim(src_blur, recon_blur, channel_axis=2, data_range=255.0)

    print(f'{name}: K={K2} quantized_raw={raw:.4f} quantized_blur={blur:.4f}')

    # Also try with threshold=40 (more clusters)
    centers3, labels3 = _merge_close_clusters(centers, labels.flatten(), h, w, threshold=40.0)
    K3 = len(centers3)
    recon3 = centers3[labels3].reshape(h, w, 3).astype(np.uint8)
    recon3_rgb = cv2.cvtColor(recon3, cv2.COLOR_BGR2RGB)
    raw3 = ssim(src_rgb, recon3_rgb, channel_axis=2)
    recon3_blur = cv2.GaussianBlur(recon3_rgb.astype(np.float32), (0,0), sigmaX=1.5)
    blur3 = ssim(src_blur, recon3_blur, channel_axis=2, data_range=255.0)
    print(f'{name}: K={K3} quantized_raw={raw3:.4f} quantized_blur={blur3:.4f}')

    # No merge (full 24 clusters)
    recon_full = centers[labels.flatten()].reshape(h, w, 3).astype(np.uint8)
    recon_full_rgb = cv2.cvtColor(recon_full, cv2.COLOR_BGR2RGB)
    raw_full = ssim(src_rgb, recon_full_rgb, channel_axis=2)
    recon_full_blur = cv2.GaussianBlur(recon_full_rgb.astype(np.float32), (0,0), sigmaX=1.5)
    blur_full = ssim(src_blur, recon_full_blur, channel_axis=2, data_range=255.0)
    print(f'{name}: K={K} quantized_raw={raw_full:.4f} quantized_blur={blur_full:.4f}')
