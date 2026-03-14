"""Stroke reconstruction module.

Stage 3: Skeletonize stroke-like regions, extract medial axis paths,
estimate stroke width, and produce path data for curve fitting.
"""

import cv2
import numpy as np
from skimage.morphology import skeletonize, thin
from scipy import ndimage
from dataclasses import dataclass


@dataclass
class StrokePath:
    points: np.ndarray          # Nx2 ordered path points
    widths: np.ndarray          # N estimated widths at each point
    is_closed: bool


@dataclass
class StrokeResult:
    paths: list[StrokePath]
    skeleton: np.ndarray        # skeleton visualization


def reconstruct_strokes(
    stroke_mask: np.ndarray,
    labels: np.ndarray,
    stroke_labels: list[int],
    *,
    prune_length: int = 5,
    simplify_epsilon: float = 1.0,
) -> StrokeResult:
    """Reconstruct stroke geometry from stroke-classified components.

    Process per component:
      1. Skeletonize
      2. Compute distance transform for width estimation
      3. Trace skeleton paths
      4. Prune short branches
      5. Simplify paths
    """
    full_skeleton = np.zeros_like(stroke_mask, dtype=np.uint8)
    all_paths: list[StrokePath] = []

    for label_id in stroke_labels:
        component = (labels == label_id).astype(np.uint8)

        # Distance transform for width
        dist = cv2.distanceTransform(component * 255, cv2.DIST_L2, 5)

        # Skeletonize
        skel = skeletonize(component > 0).astype(np.uint8)
        full_skeleton[skel > 0] = 255

        # Prune short branches
        skel = _prune_skeleton(skel, prune_length)

        # Trace paths through skeleton
        paths = _trace_skeleton_paths(skel)

        for pts in paths:
            if len(pts) < 2:
                continue

            # Estimate widths from distance transform
            widths = np.array([dist[int(p[1]), int(p[0])] * 2 for p in pts])

            # Simplify path (Ramer-Douglas-Peucker)
            pts_cv = pts.reshape(-1, 1, 2).astype(np.float32)
            simplified = cv2.approxPolyDP(pts_cv, simplify_epsilon, closed=False)
            simplified_pts = simplified.reshape(-1, 2)

            # Re-interpolate widths for simplified points
            if len(simplified_pts) < len(pts):
                indices = _find_closest_indices(pts, simplified_pts)
                widths = widths[indices]

            is_closed = np.linalg.norm(simplified_pts[0] - simplified_pts[-1]) < 3.0

            all_paths.append(StrokePath(
                points=simplified_pts,
                widths=widths,
                is_closed=is_closed,
            ))

    return StrokeResult(paths=all_paths, skeleton=full_skeleton)


def _prune_skeleton(skel: np.ndarray, min_branch_length: int) -> np.ndarray:
    """Remove skeleton branches shorter than min_branch_length."""
    # Find endpoints (pixels with exactly 1 neighbour)
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    neighbours = cv2.filter2D(skel, -1, kernel)
    endpoints = (skel > 0) & (neighbours == 1)

    pruned = skel.copy()
    for _ in range(min_branch_length):
        endpoint_mask = np.zeros_like(skel)
        neighbours = cv2.filter2D(pruned, -1, kernel)
        endpoint_mask[(pruned > 0) & (neighbours <= 1)] = 1
        pruned[endpoint_mask > 0] = 0

    return pruned


def _trace_skeleton_paths(skel: np.ndarray) -> list[np.ndarray]:
    """Trace connected paths through a skeleton image.

    Returns list of Nx2 arrays of (x, y) points in order.
    """
    points = np.column_stack(np.where(skel > 0))  # (y, x)
    if len(points) == 0:
        return []

    # Label connected skeleton segments
    labeled, num = ndimage.label(skel, structure=np.ones((3, 3)))
    paths = []

    for seg_id in range(1, num + 1):
        seg_points = np.column_stack(np.where(labeled == seg_id))  # (y, x)
        if len(seg_points) < 2:
            continue

        # Order points by tracing from an endpoint
        ordered = _order_points(seg_points, skel)
        # Convert (y,x) to (x,y)
        ordered_xy = ordered[:, ::-1].copy()
        paths.append(ordered_xy)

    return paths


def _order_points(points: np.ndarray, skel: np.ndarray) -> np.ndarray:
    """Order skeleton points by greedy nearest-neighbour walk from an endpoint."""
    from scipy.spatial import KDTree

    if len(points) <= 2:
        return points

    tree = KDTree(points)

    # Find endpoint: pixel with fewest skeleton neighbours
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    neighbours = cv2.filter2D(skel, -1, kernel)

    min_n = 8
    start_idx = 0
    for i, (y, x) in enumerate(points):
        n = neighbours[y, x]
        if n < min_n:
            min_n = n
            start_idx = i

    visited = np.zeros(len(points), dtype=bool)
    ordered = [points[start_idx]]
    visited[start_idx] = True

    current = points[start_idx]
    for _ in range(len(points) - 1):
        dists, idxs = tree.query(current, k=min(8, len(points)))
        found = False
        for d, idx in zip(dists, idxs):
            if not visited[idx]:
                visited[idx] = True
                ordered.append(points[idx])
                current = points[idx]
                found = True
                break
        if not found:
            break

    return np.array(ordered)


def _find_closest_indices(original: np.ndarray, simplified: np.ndarray) -> np.ndarray:
    """For each simplified point, find the closest original point index."""
    from scipy.spatial import KDTree
    tree = KDTree(original)
    _, indices = tree.query(simplified)
    return indices
