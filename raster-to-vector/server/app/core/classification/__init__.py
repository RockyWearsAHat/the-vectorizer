"""Geometry classification module.

Stage 2: Classify connected components as stroke-like or fill-like.
"""

import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class ClassifiedComponents:
    stroke_mask: np.ndarray     # binary mask of stroke-like regions
    fill_mask: np.ndarray       # binary mask of fill-like regions
    stroke_labels: list[int]
    fill_labels: list[int]
    labels: np.ndarray          # full label image
    stats: np.ndarray           # CC stats


def classify(
    binary: np.ndarray,
    *,
    solidity_threshold: float = 0.6,
    stroke_width_ratio: float = 0.15,
    min_area: int = 20,
) -> ClassifiedComponents:
    """Classify each connected component as stroke-like or fill-like.

    Heuristics:
        - solidity < threshold → stroke-like (thin, elongated)
        - stroke-width consistency via distance transform
        - area vs bounding-box aspect ratio
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    stroke_labels: list[int] = []
    fill_labels: list[int] = []

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        component_mask = (labels == i).astype(np.uint8) * 255

        # Solidity = area / convex-hull area
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        hull = cv2.convexHull(contours[0])
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 1.0

        # Bounding box aspect ratio
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        aspect = min(bw, bh) / max(bw, bh) if max(bw, bh) > 0 else 1.0

        # Stroke-width consistency via distance transform
        dist = cv2.distanceTransform(component_mask, cv2.DIST_L2, 5)
        max_dist = dist.max()
        mean_dist = dist[dist > 0].mean() if np.any(dist > 0) else 0

        # Classification logic
        is_stroke = False
        if solidity < solidity_threshold:
            is_stroke = True
        elif max_dist > 0 and (mean_dist / max_dist) > 0.5 and aspect < stroke_width_ratio * 3:
            is_stroke = True
        elif bw > 0 and bh > 0 and (area / (bw * bh)) < 0.3:
            is_stroke = True

        if is_stroke:
            stroke_labels.append(i)
        else:
            fill_labels.append(i)

    stroke_mask = np.isin(labels, stroke_labels).astype(np.uint8) * 255 if stroke_labels else np.zeros_like(binary)
    fill_mask = np.isin(labels, fill_labels).astype(np.uint8) * 255 if fill_labels else np.zeros_like(binary)

    return ClassifiedComponents(
        stroke_mask=stroke_mask,
        fill_mask=fill_mask,
        stroke_labels=stroke_labels,
        fill_labels=fill_labels,
        labels=labels,
        stats=stats,
    )
