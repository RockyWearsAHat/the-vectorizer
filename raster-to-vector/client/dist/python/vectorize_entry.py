"""Entry point for browser-based vectorization via Pyodide."""

import json
import numpy as np
import cv2
from multilevel import multilevel_vectorize, generate_svg


def run(image_bytes, *, crop_x=0, crop_y=0, crop_w=0, crop_h=0,
        remove_bg=True, num_levels=24):
    """Vectorize an image from raw bytes. Returns JSON string."""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image")

    h, w = img.shape[:2]

    # Apply crop if specified
    if crop_w > 0 and crop_h > 0:
        x1 = max(0, min(int(crop_x), w))
        y1 = max(0, min(int(crop_y), h))
        x2 = min(x1 + int(crop_w), w)
        y2 = min(y1 + int(crop_h), h)
        if x2 > x1 and y2 > y1:
            img = img[y1:y2, x1:x2]

    result = multilevel_vectorize(img, num_levels=num_levels)
    svg = generate_svg(result, remove_background=remove_bg)

    return json.dumps({
        "svg": svg,
        "width": result.width,
        "height": result.height,
        "path_count": result.path_count,
        "node_count": result.node_count,
    })
