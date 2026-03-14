"""Fill reconstruction module.

Stage 4: Detect contours for fill-like regions, preserve holes,
simplify contours, and prepare closed paths for curve fitting.
"""

import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class FillPath:
    outer: np.ndarray           # Nx2 outer contour points
    holes: list[np.ndarray]     # list of Mx2 hole contour points
    color: tuple[int, int, int] # fill color (default black)


@dataclass
class FillResult:
    paths: list[FillPath]


def reconstruct_fills(
    fill_mask: np.ndarray,
    labels: np.ndarray,
    fill_labels: list[int],
    *,
    simplify_epsilon: float = 1.5,
    min_contour_area: int = 10,
) -> FillResult:
    """Reconstruct filled vector shapes from fill-classified components.

    For each component:
      1. Find contour hierarchy (outer + holes)
      2. Simplify contours
      3. Package as FillPath
    """
    all_paths: list[FillPath] = []

    for label_id in fill_labels:
        component = (labels == label_id).astype(np.uint8) * 255

        contours, hierarchy = cv2.findContours(
            component, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours or hierarchy is None:
            continue

        hierarchy = hierarchy[0]  # shape (N, 4)

        # Process top-level contours (those with no parent)
        i = 0
        while i >= 0:
            if hierarchy[i][3] == -1:  # no parent → outer contour
                outer_contour = contours[i].reshape(-1, 2).astype(np.float64)
                if cv2.contourArea(outer_contour.reshape(-1, 1, 2).astype(np.float32)) < min_contour_area:
                    i = hierarchy[i][0]  # next sibling
                    continue

                # Simplify outer
                outer_simplified = cv2.approxPolyDP(
                    outer_contour.reshape(-1, 1, 2).astype(np.float32),
                    simplify_epsilon,
                    closed=True,
                ).reshape(-1, 2)

                # Collect holes (children)
                holes = []
                child = hierarchy[i][2]  # first child
                while child >= 0:
                    hole_contour = contours[child].reshape(-1, 2).astype(np.float64)
                    if cv2.contourArea(hole_contour.reshape(-1, 1, 2).astype(np.float32)) >= min_contour_area:
                        hole_simplified = cv2.approxPolyDP(
                            hole_contour.reshape(-1, 1, 2).astype(np.float32),
                            simplify_epsilon,
                            closed=True,
                        ).reshape(-1, 2)
                        holes.append(hole_simplified)
                    child = hierarchy[child][0]  # next sibling of child

                all_paths.append(FillPath(
                    outer=outer_simplified,
                    holes=holes,
                    color=(0, 0, 0),
                ))

            i = hierarchy[i][0]  # next sibling

    return FillResult(paths=all_paths)
