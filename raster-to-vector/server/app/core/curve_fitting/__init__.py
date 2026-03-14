"""Curve fitting module.

Stage 5: Fit smooth cubic Bézier curves to polyline paths.
Minimizes geometric error while reducing node count and preserving sharp corners.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class BezierSegment:
    p0: np.ndarray  # start point
    p1: np.ndarray  # control point 1
    p2: np.ndarray  # control point 2
    p3: np.ndarray  # end point
    is_line: bool = False  # True when this segment is a straight line


@dataclass
class FittedCurve:
    segments: list[BezierSegment]
    is_closed: bool


def fit_bezier_path(
    points: np.ndarray,
    *,
    max_error: float = 2.0,
    corner_threshold: float = 45.0,
    is_closed: bool = False,
    line_tolerance: float | None = None,
) -> FittedCurve:
    """Fit a sequence of cubic Bézier curves to a polyline.

    Uses an adaptive algorithm that splits at corners and recursively
    fits Bézier segments to each smooth section.

    Args:
        points: Nx2 array of polyline points.
        max_error: Maximum allowed fitting error in pixels.
        corner_threshold: Angle threshold in degrees for corner detection.
        is_closed: Whether the path should be closed.
        line_tolerance: Collinearity tolerance for straight-line detection.
            Defaults to max_error when None.

    Returns:
        FittedCurve with list of cubic Bézier segments.
    """
    if line_tolerance is None:
        line_tolerance = max_error

    if len(points) < 2:
        return FittedCurve(segments=[], is_closed=is_closed)

    if len(points) == 2:
        seg = _line_to_bezier(points[0], points[1])
        return FittedCurve(segments=[seg], is_closed=is_closed)

    # Detect corners to split the path
    corners = _detect_corners(points, corner_threshold)
    corners = [0] + sorted(set(corners)) + [len(points) - 1]

    all_segments: list[BezierSegment] = []

    for i in range(len(corners) - 1):
        start = corners[i]
        end = corners[i + 1]
        section = points[start:end + 1]

        if len(section) < 2:
            continue

        if len(section) == 2:
            all_segments.append(_line_to_bezier(section[0], section[1]))
        else:
            segments = _fit_cubic_bezier(section, max_error, line_tolerance)
            all_segments.extend(segments)

    return FittedCurve(segments=all_segments, is_closed=is_closed)


def fit_closed_bezier(
    points: np.ndarray,
    *,
    max_error: float = 2.0,
    corner_threshold: float = 45.0,
    line_tolerance: float | None = None,
) -> FittedCurve:
    """Fit Bézier curves to a closed contour."""
    # Close the loop
    if not np.allclose(points[0], points[-1]):
        points = np.vstack([points, points[0:1]])
    return fit_bezier_path(
        points, max_error=max_error, corner_threshold=corner_threshold,
        is_closed=True, line_tolerance=line_tolerance,
    )


def _detect_corners(points: np.ndarray, threshold_deg: float) -> list[int]:
    """Detect corner points where the path changes direction sharply."""
    corners = []
    threshold_rad = np.radians(threshold_deg)

    for i in range(1, len(points) - 1):
        v1 = points[i] - points[i - 1]
        v2 = points[i + 1] - points[i]
        len1 = np.linalg.norm(v1)
        len2 = np.linalg.norm(v2)

        if len1 == 0 or len2 == 0:
            continue

        cos_angle = np.clip(np.dot(v1, v2) / (len1 * len2), -1.0, 1.0)
        angle = np.arccos(cos_angle)

        if angle > threshold_rad:
            corners.append(i)

    return corners


def _fit_cubic_bezier(
    points: np.ndarray, max_error: float, line_tolerance: float,
    _depth: int = 0,
) -> list[BezierSegment]:
    """Recursively fit cubic Bézier segments to a section of points.

    Uses Newton-Raphson reparameterization to improve fit quality before
    splitting, producing smoother curves with fewer segments.
    """
    if len(points) <= 2:
        return [_line_to_bezier(points[0], points[-1])]

    # Recursion guard: fall back to line segments
    if _depth > 24 or len(points) < 3:
        return [_line_to_bezier(points[0], points[-1])]

    # --- Collinearity check ---
    # If all points lie within line_tolerance of the chord, emit a
    # single straight line.  line_tolerance is intentionally more
    # generous than max_error so pixel-staircase artifacts along
    # straight edges collapse into clean L commands.
    if _max_perpendicular_distance(points) <= line_tolerance:
        return [_line_to_bezier(points[0], points[-1])]

    # Estimate tangent directions
    t_hat1 = _compute_tangent(points, 0, forward=True)
    t_hat2 = _compute_tangent(points, len(points) - 1, forward=False)

    # Parameterize by chord length
    t_params = _chord_length_parameterize(points)

    # Fit single Bézier
    seg = _fit_single_bezier(points, t_params, t_hat1, t_hat2)
    error, split_idx = _compute_max_error(points, seg, t_params)

    if error <= max_error:
        # Post-fit check: convert near-straight Bézier to true line
        if _is_near_straight(seg, line_tolerance):
            return [_line_to_bezier(seg.p0, seg.p3)]
        return [seg]

    # Newton-Raphson reparameterization: refine t_params to reduce error
    # before resorting to splitting
    if error <= max_error * 4:
        for _ in range(3):
            t_params = _reparameterize(points, seg, t_params)
            seg = _fit_single_bezier(points, t_params, t_hat1, t_hat2)
            error, split_idx = _compute_max_error(points, seg, t_params)
            if error <= max_error:
                if _is_near_straight(seg, line_tolerance):
                    return [_line_to_bezier(seg.p0, seg.p3)]
                return [seg]

    # Ensure valid split index
    split_idx = max(1, min(split_idx, len(points) - 2))

    # If we can't make a meaningful split, accept current fit
    if len(points[:split_idx + 1]) < 2 or len(points[split_idx:]) < 2:
        return [seg]

    left = _fit_cubic_bezier(points[:split_idx + 1], max_error, line_tolerance, _depth + 1)
    right = _fit_cubic_bezier(points[split_idx:], max_error, line_tolerance, _depth + 1)
    return left + right


def _fit_single_bezier(
    points: np.ndarray,
    t_params: np.ndarray,
    t_hat1: np.ndarray,
    t_hat2: np.ndarray,
) -> BezierSegment:
    """Fit a single cubic Bézier to points using least-squares."""
    p0 = points[0].copy()
    p3 = points[-1].copy()

    # Estimate alpha (control point distances)
    a = np.zeros((len(points), 2, 2))
    for i, t in enumerate(t_params):
        b1 = _bernstein(1, t) * t_hat1
        b2 = _bernstein(2, t) * t_hat2
        a[i][0] = b1
        a[i][1] = b2

    c = np.zeros((2, 2))
    x = np.zeros(2)

    for i in range(len(points)):
        c[0][0] += np.dot(a[i][0], a[i][0])
        c[0][1] += np.dot(a[i][0], a[i][1])
        c[1][0] = c[0][1]
        c[1][1] += np.dot(a[i][1], a[i][1])

        t = t_params[i]
        tmp = (
            points[i]
            - p0 * _bernstein(0, t)
            - p0 * _bernstein(1, t)
            - p3 * _bernstein(2, t)
            - p3 * _bernstein(3, t)
        )
        x[0] += np.dot(a[i][0], tmp)
        x[1] += np.dot(a[i][1], tmp)

    det = c[0][0] * c[1][1] - c[0][1] * c[1][0]
    if abs(det) < 1e-12:
        alpha1 = alpha2 = np.linalg.norm(p3 - p0) / 3.0
    else:
        alpha1 = (x[0] * c[1][1] - x[1] * c[0][1]) / det
        alpha2 = (c[0][0] * x[1] - c[1][0] * x[0]) / det

    seg_len = np.linalg.norm(p3 - p0)
    epsilon = 1e-6 * seg_len

    if alpha1 < epsilon or alpha2 < epsilon:
        alpha1 = alpha2 = seg_len / 3.0

    p1 = p0 + t_hat1 * alpha1
    p2 = p3 + t_hat2 * alpha2

    return BezierSegment(p0=p0, p1=p1, p2=p2, p3=p3)


def _compute_tangent(points: np.ndarray, idx: int, forward: bool) -> np.ndarray:
    """Compute unit tangent at a point."""
    if forward:
        diff = points[min(idx + 1, len(points) - 1)] - points[idx]
    else:
        diff = points[max(idx - 1, 0)] - points[idx]
    norm = np.linalg.norm(diff)
    return diff / norm if norm > 0 else np.array([1.0, 0.0])


def _chord_length_parameterize(points: np.ndarray) -> np.ndarray:
    """Assign parameter values by cumulative chord length."""
    dists = np.zeros(len(points))
    for i in range(1, len(points)):
        dists[i] = dists[i - 1] + np.linalg.norm(points[i] - points[i - 1])
    total = dists[-1]
    if total > 0:
        dists /= total
    return dists


def _bernstein(i: int, t: float) -> float:
    """Evaluate the i-th cubic Bernstein basis polynomial at t."""
    if i == 0:
        return (1 - t) ** 3
    elif i == 1:
        return 3 * (1 - t) ** 2 * t
    elif i == 2:
        return 3 * (1 - t) * t ** 2
    else:
        return t ** 3


def _evaluate_bezier(seg: BezierSegment, t: float) -> np.ndarray:
    """Evaluate a cubic Bézier at parameter t."""
    return (
        seg.p0 * _bernstein(0, t)
        + seg.p1 * _bernstein(1, t)
        + seg.p2 * _bernstein(2, t)
        + seg.p3 * _bernstein(3, t)
    )


def _compute_max_error(
    points: np.ndarray, seg: BezierSegment, t_params: np.ndarray
) -> tuple[float, int]:
    """Compute max distance between points and Bézier curve. Return error and split index."""
    max_err = 0.0
    split_idx = len(points) // 2

    for i in range(len(points)):
        pt = _evaluate_bezier(seg, t_params[i])
        err = np.linalg.norm(points[i] - pt)
        if err > max_err:
            max_err = err
            split_idx = i

    return max_err, split_idx


def _is_near_straight(seg: BezierSegment, tol: float) -> bool:
    """Check if a cubic Bézier's control points are within tol of the chord."""
    d = seg.p3 - seg.p0
    chord_len = np.linalg.norm(d)
    if chord_len < 1e-10:
        return True
    n = np.array([-d[1], d[0]]) / chord_len
    d1 = abs(float(np.dot(seg.p1 - seg.p0, n)))
    d2 = abs(float(np.dot(seg.p2 - seg.p0, n)))
    return max(d1, d2) <= tol


def _line_to_bezier(p0: np.ndarray, p1: np.ndarray) -> BezierSegment:
    """Convert a line segment to a degenerate cubic Bézier."""
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    return BezierSegment(
        p0=p0,
        p1=p0 + (p1 - p0) / 3.0,
        p2=p0 + (p1 - p0) * 2.0 / 3.0,
        p3=p1,
        is_line=True,
    )


def _max_perpendicular_distance(points: np.ndarray) -> float:
    """Max perpendicular distance of interior points from the chord p0→p[-1]."""
    p0 = points[0]
    p1 = points[-1]
    d = p1 - p0
    seg_len = np.linalg.norm(d)
    if seg_len < 1e-10:
        # Degenerate: all points at same location
        return float(np.max(np.linalg.norm(points - p0, axis=1)))
    # Unit normal perpendicular to chord
    n = np.array([-d[1], d[0]]) / seg_len
    # Signed perpendicular distance of each interior point
    diffs = points[1:-1] - p0
    perp = np.abs(diffs @ n)
    return float(np.max(perp)) if len(perp) > 0 else 0.0


def _reparameterize(
    points: np.ndarray, seg: BezierSegment, t_params: np.ndarray
) -> np.ndarray:
    """Newton-Raphson reparameterization: adjust t values to minimise distance."""
    new_t = t_params.copy()
    for i in range(len(points)):
        t = new_t[i]
        pt = _evaluate_bezier(seg, t)
        d1 = _evaluate_bezier_derivative(seg, t)
        d2 = _evaluate_bezier_second_derivative(seg, t)

        diff = pt - points[i]
        numerator = np.dot(diff, d1)
        denominator = np.dot(d1, d1) + np.dot(diff, d2)

        if abs(denominator) > 1e-12:
            new_t[i] = t - numerator / denominator
            new_t[i] = max(0.0, min(1.0, new_t[i]))

    return new_t


def _evaluate_bezier_derivative(seg: BezierSegment, t: float) -> np.ndarray:
    """First derivative of cubic Bézier at parameter t."""
    return (
        3 * (1 - t) ** 2 * (seg.p1 - seg.p0)
        + 6 * (1 - t) * t * (seg.p2 - seg.p1)
        + 3 * t ** 2 * (seg.p3 - seg.p2)
    )


def _evaluate_bezier_second_derivative(seg: BezierSegment, t: float) -> np.ndarray:
    """Second derivative of cubic Bézier at parameter t."""
    return (
        6 * (1 - t) * (seg.p2 - 2 * seg.p1 + seg.p0)
        + 6 * t * (seg.p3 - 2 * seg.p2 + seg.p1)
    )
