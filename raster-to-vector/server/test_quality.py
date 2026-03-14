"""Quality profiling script for the multi-level pipeline."""

import cv2
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from app.core.multilevel import multilevel_vectorize, generate_svg
from app.core.comparison import compare


def create_tonal_test_image():
    """Create a test image with gradients, strokes, and gray fills."""
    img = np.ones((400, 400, 3), dtype=np.uint8) * 245

    # Gradient-filled circle
    for y in range(400):
        for x in range(400):
            d = np.sqrt((x - 200) ** 2 + (y - 150) ** 2)
            if d < 120:
                val = int(50 + d * 1.4)
                img[y, x] = [val, val, val]

    # Gray letters
    cv2.putText(img, "M", (100, 350), cv2.FONT_HERSHEY_SIMPLEX, 4, (120, 120, 120), 8, cv2.LINE_AA)
    cv2.putText(img, "B", (220, 350), cv2.FONT_HERSHEY_SIMPLEX, 3, (160, 160, 160), 5, cv2.LINE_AA)

    # Thin strokes
    pts1 = np.array([[50, 200], [100, 100], [180, 80], [200, 150], [250, 70], [300, 100], [280, 200]], np.int32)
    cv2.polylines(img, [pts1], False, (30, 30, 30), 2, cv2.LINE_AA)
    pts2 = np.array([[150, 180], [170, 120], [210, 100], [250, 120], [230, 180]], np.int32)
    cv2.polylines(img, [pts2], False, (60, 60, 60), 1, cv2.LINE_AA)

    return img


def create_lineart_test_image():
    """Create a test image with flat fills and clean strokes - like real line art."""
    img = np.ones((400, 400, 3), dtype=np.uint8) * 250

    # Flat filled regions
    cv2.circle(img, (150, 150), 80, (200, 200, 200), -1, cv2.LINE_AA)
    cv2.circle(img, (280, 200), 60, (180, 180, 180), -1, cv2.LINE_AA)
    pts = np.array([[100, 300], [200, 250], [300, 320], [250, 380], [120, 370]], np.int32)
    cv2.fillPoly(img, [pts], (160, 160, 160), cv2.LINE_AA)

    # Clean black strokes
    cv2.circle(img, (150, 150), 80, (30, 30, 30), 3, cv2.LINE_AA)
    cv2.circle(img, (280, 200), 60, (30, 30, 30), 3, cv2.LINE_AA)
    cv2.polylines(img, [pts], True, (30, 30, 30), 3, cv2.LINE_AA)

    # Detail lines
    cv2.line(img, (50, 50), (350, 50), (60, 60, 60), 2, cv2.LINE_AA)
    cv2.line(img, (50, 50), (50, 350), (60, 60, 60), 2, cv2.LINE_AA)

    return img


def test_quality(img, label=""):
    """Test with various parameter combinations."""
    print(f"\n=== {label} ({img.shape[1]}x{img.shape[0]}) ===")
    for levels in [8, 12, 16, 24, 32, 48]:
        for eps in [0.2, 0.3, 0.5, 0.8]:
            for me in [0.3, 0.5]:
                result = multilevel_vectorize(
                    img, num_levels=levels,
                    simplify_epsilon=eps, max_error=me,
                )
                svg = generate_svg(result, remove_background=False)
                comp = compare(img, svg)
                print(
                    f"  lvl={levels:3d} eps={eps:.1f} me={me:.1f} -> "
                    f"SSIM={comp.ssim_score:.4f} MAE={comp.mae:.2f} "
                    f"paths={result.path_count:4d} nodes={result.node_count:4d}"
                )


if __name__ == "__main__":
    tonal_img = create_tonal_test_image()
    test_quality(tonal_img, "Tonal illustration")

    lineart_img = create_lineart_test_image()
    test_quality(lineart_img, "Line art")
