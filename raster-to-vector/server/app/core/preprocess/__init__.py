"""Image preprocessing module.

Stage 1: Convert to grayscale, normalize, threshold, denoise, preserve thin lines.
"""

import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class PreprocessResult:
    binary: np.ndarray          # thresholded binary image (ink=255, bg=0)
    grayscale: np.ndarray       # normalized grayscale
    original_size: tuple[int, int]  # (height, width)


def preprocess(
    image: np.ndarray,
    *,
    blur_kernel: int = 3,
    threshold_block_size: int = 25,
    threshold_c: int = 10,
    min_component_area: int = 8,
    invert: bool = True,
) -> PreprocessResult:
    """Full preprocessing pipeline for a raster image or cropped region.

    Args:
        image: BGR or grayscale uint8 image.
        blur_kernel: Gaussian blur kernel size (odd number).
        threshold_block_size: Adaptive threshold neighbourhood.
        threshold_c: Adaptive threshold constant subtracted from mean.
        min_component_area: Connected components smaller than this are removed.
        invert: If True, assume dark-on-light artwork (ink is dark).

    Returns:
        PreprocessResult with binary mask and grayscale.
    """
    h, w = image.shape[:2]

    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Normalize intensity
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    # Light denoise preserving edges
    denoised = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)

    # Adaptive thresholding – works well on scanned drawings
    binary = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY,
        threshold_block_size,
        threshold_c,
    )

    # Remove small noise components
    binary = _remove_small_components(binary, min_component_area)

    return PreprocessResult(binary=binary, grayscale=gray, original_size=(h, w))


def _remove_small_components(binary: np.ndarray, min_area: int) -> np.ndarray:
    """Remove connected components smaller than min_area pixels."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)
    for i in range(1, num_labels):  # skip background
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255
    return cleaned
