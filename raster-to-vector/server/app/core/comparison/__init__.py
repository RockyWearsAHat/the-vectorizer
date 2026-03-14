"""Fidelity comparison module.

Stage 7: Rasterize generated SVG, align with source, compute difference metrics,
and generate visual comparison outputs.
"""

import io
import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from dataclasses import dataclass

import cairosvg


@dataclass
class ComparisonResult:
    mae: float                  # mean absolute error
    ssim_score: float           # structural similarity index
    pixel_diff_ratio: float     # fraction of pixels that differ
    diff_map: np.ndarray        # absolute difference image (grayscale)
    heatmap: np.ndarray         # colored heatmap of differences (BGR)
    overlay: np.ndarray         # semi-transparent overlay (BGR)
    svg_raster: np.ndarray      # rasterized SVG (grayscale)


def compare(
    source_image: np.ndarray,
    svg_string: str,
    *,
    threshold: int = 15,
) -> ComparisonResult:
    """Compare source raster with rasterized SVG.

    Args:
        source_image: Original raster (BGR or grayscale).
        svg_string: Generated SVG string.
        threshold: Pixel difference threshold for binary diff.

    Returns:
        ComparisonResult with metrics and visual outputs.
    """
    h, w = source_image.shape[:2]

    # Convert source to grayscale
    if len(source_image.shape) == 3:
        src_gray = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
        src_bgr = source_image.copy()
    else:
        src_gray = source_image.copy()
        src_bgr = cv2.cvtColor(source_image, cv2.COLOR_GRAY2BGR)

    # Rasterize SVG
    svg_raster = _rasterize_svg(svg_string, w, h)

    # Denoise both images before SSIM so we measure structural fidelity
    # rather than penalising the SVG's flat fills against per-pixel noise.
    # Gaussian blur at sigma=1.5 smooths noise without hiding real detail.
    src_smooth = cv2.GaussianBlur(src_gray, (0, 0), sigmaX=1.5)
    svg_smooth = cv2.GaussianBlur(svg_raster, (0, 0), sigmaX=1.5)

    # Compute metrics on smoothed images
    mae = float(np.mean(np.abs(src_smooth.astype(float) - svg_smooth.astype(float))))

    ssim_score = float(ssim(src_smooth, svg_smooth))

    diff = np.abs(src_smooth.astype(np.int16) - svg_smooth.astype(np.int16)).astype(np.uint8)
    pixel_diff_count = np.sum(diff > threshold)
    total_pixels = h * w
    pixel_diff_ratio = float(pixel_diff_count / total_pixels)

    # Colored heatmap
    heatmap = cv2.applyColorMap(diff, cv2.COLORMAP_JET)

    # Overlay: blend source with SVG raster
    svg_bgr = cv2.cvtColor(svg_raster, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(src_bgr, 0.5, svg_bgr, 0.5, 0)

    return ComparisonResult(
        mae=mae,
        ssim_score=ssim_score,
        pixel_diff_ratio=pixel_diff_ratio,
        diff_map=diff,
        heatmap=heatmap,
        overlay=overlay,
        svg_raster=svg_raster,
    )


def _rasterize_svg(svg_string: str, width: int, height: int) -> np.ndarray:
    """Rasterize an SVG string to a grayscale numpy array at the given dimensions."""
    # Render with white background
    png_bytes = cairosvg.svg2png(
        bytestring=svg_string.encode("utf-8"),
        output_width=width,
        output_height=height,
        background_color="white",
    )
    img = Image.open(io.BytesIO(png_bytes)).convert("L")
    return np.array(img)
