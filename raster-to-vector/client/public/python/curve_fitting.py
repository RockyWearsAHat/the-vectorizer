"""Curve fitting module — browser build (Pyodide).

Identical to server/app/core/curve_fitting/__init__.py.
Stage 5: Fit smooth cubic Bézier curves to polyline paths.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class BezierSegment:
    p0: np.ndarray
    p1: np.ndarray
    p2: np.ndarray
    p3: np.ndarray
    is_line: bool = False


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
    if line_tolerance is None:
        line_tolerance = max_error

    if len(points) < 2:
        return FittedCurve(segments=[], is_closed=is_closed)

    if len(points) == 2:
        seg = _line_to_bezier(points[0], points[1])
        return FittedCurve(segments=[seg], is_closed=is_closed)

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
    if not np.allclose(points[0], points[-1]):
        points = np.vstack([points, points[0:1]])
    return fit_bezier_path(
        points, max_error=max_error, corner_threshold=corner_threshold,
        is_closed=True, line_tolerance=line_tolerance,
    )


def _detect_corners(points: np.ndarray, threshold_deg: float) -> list[int]:
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
    if len(points) <= 2:
        return [_line_to_bezier(points[0], points[-1])]

    if _depth > 24 or len(points) < 3:
        return [_line_to_bezier(points[0], points[-1])]

    if _max_perpendicular_distance(points) <= line_tolerance:
        return [_line_to_bezier(points[0], points[-1])]

    t_hat1 = _compute_tangent(points, 0, forward=True)
    t_hat2 = _compute_tangent(points, len(points) - 1, forward=False)

    t_params = _chord_length_parameterize(points)

    seg = _fit_single_bezier(points, t_params, t_hat1, t_hat2)
    error, split_idx = _compute_max_error(points, seg, t_params)

    if error <= max_error:
        return [seg]

    if error <= max_error * 4:
        for _ in range(3):
            t_params = _reparameterize(points, seg, t_params)
            seg = _fit_single_bezier(points, t_params, t_hat1, t_hat2)
            error, split_idx = _compute_max_error(points, seg, t_params)
            if error <= max_error:
                return [seg]

    split_idx = max(1, min(split_idx, len(points) - 2))

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
    p0 = points[0].copy()
    p3 = points[-1].copy()

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
    if forward:
        diff = points[min(idx + 1, len(points) - 1)] - points[idx]
    else:
        diff = points[max(idx - 1, 0)] - points[idx]
    norm = np.linalg.norm(diff)
    return diff / norm if norm > 0 else np.array([1.0, 0.0])


def _chord_length_parameterize(points: np.ndarray) -> np.ndarray:
    dists = np.zeros(len(points))
    for i in range(1, len(points)):
        dists[i] = dists[i - 1] + np.linalg.norm(points[i] - points[i - 1])
    total = dists[-1]
    if total > 0:
        dists /= total
    return dists


def _bernstein(i: int, t: float) -> float:
    if i == 0:
        return (1 - t) ** 3
    elif i == 1:
        return 3 * (1 - t) ** 2 * t
    elif i == 2:
        return 3 * (1 - t) * t ** 2
    else:
        return t ** 3


def _evaluate_bezier(seg: BezierSegment, t: float) -> np.ndarray:
    return (
        seg.p0 * _bernstein(0, t)
        + seg.p1 * _bernstein(1, t)
        + seg.p2 * _bernstein(2, t)
        + seg.p3 * _bernstein(3, t)
    )


def _compute_max_error(
    points: np.ndarray, seg: BezierSegment, t_params: np.ndarray
) -> tuple[float, int]:
    max_err = 0.0
    split_idx = len(points) // 2

    for i in range(len(points)):
        pt = _evaluate_bezier(seg, t_params[i])
        err = np.linalg.norm(points[i] - pt)
        if err > max_err:
            max_err = err
            split_idx = i

    return max_err, split_idx


def _line_to_bezier(p0: np.ndarray, p1: np.ndarray) -> BezierSegment:
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
    p0 = points[0]
    p1 = points[-1]
    d = p1 - p0
    seg_len = np.linalg.norm(d)
    if seg_len < 1e-10:
        return float(np.max(np.linalg.norm(points - p0, axis=1)))
    n = np.array([-d[1], d[0]]) / seg_len
    diffs = points[1:-1] - p0
    perp = np.abs(diffs @ n)
    return float(np.max(perp)) if len(perp) > 0 else 0.0


def _reparameterize(
    points: np.ndarray, seg: BezierSegment, t_params: np.ndarray
) -> np.ndarray:
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
    return (
        3 * (1 - t) ** 2 * (seg.p1 - seg.p0)
        + 6 * (1 - t) * t * (seg.p2 - seg.p1)
        + 3 * t ** 2 * (seg.p3 - seg.p2)
    )


def _evaluate_bezier_second_derivative(seg: BezierSegment, t: float) -> np.ndarray:
    return (
        6 * (1 - t) * (seg.p2 - 2 * seg.p1 + seg.p0)
        + 6 * t * (seg.p3 - 2 * seg.p2 + seg.p1)
    )
