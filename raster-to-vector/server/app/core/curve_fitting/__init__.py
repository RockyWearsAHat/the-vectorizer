"""Curve fitting module.

Stage 5: Fit smooth cubic Bézier curves to polyline paths.
Minimizes geometric error while reducing node count and preserving sharp corners.
"""

import math
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
    """Fit Bézier curves to a closed contour.

    Uses a fast O(n) algorithm for large contours: Catmull-Rom tangent
    estimation at each vertex produces smooth cubic Bézier segments
    with no recursion.  Falls back to the full recursive fitter for
    small contours (≤40 vertices) where the quality difference matters.
    """
    if not np.allclose(points[0], points[-1]):
        points = np.vstack([points, points[0:1]])

    if line_tolerance is None:
        line_tolerance = max_error

    # Small-to-medium contours: use the full recursive fitter for best quality.
    # With CHAIN_APPROX_SIMPLE + RDP simplification, most contours are
    # well under this threshold and get smooth Bézier fits.
    if len(points) <= 200:
        return fit_bezier_path(
            points, max_error=max_error, corner_threshold=corner_threshold,
            is_closed=True, line_tolerance=line_tolerance,
        )

    # Large contours: O(n) direct fitting using polygon edge tangents
    return _fit_closed_direct(points, max_error, corner_threshold, line_tolerance)


def _fit_closed_direct(
    points: np.ndarray,
    max_error: float,
    corner_threshold: float,
    line_tolerance: float,
) -> FittedCurve:
    """O(n) closed-contour fitting with corner-split section fitting.

    1. Detect corners (vectorised angle check).
    2. Split the contour at corners into smooth sections.
    3. Fit each section as a single (or few) cubic Bézier(s) using
       least-squares fitting, producing smoother curves with fewer nodes
       than per-edge Catmull-Rom.
    """
    n = len(points)  # includes closing duplicate
    if n < 3:
        return FittedCurve(segments=[], is_closed=True)

    # Vectorised corner detection
    corners = _detect_corners(points, corner_threshold)

    # If no corners, fit the entire contour as one smooth loop
    if not corners:
        segs = _fit_cubic_bezier(points, max_error, line_tolerance)
        return FittedCurve(segments=segs, is_closed=True)

    # Split at corners and fit each smooth section
    segments: list[BezierSegment] = []
    corner_list = sorted(set(corners))

    # Build sections: corner[i] → corner[i+1]
    for idx in range(len(corner_list)):
        i_start = corner_list[idx]
        i_end = corner_list[(idx + 1) % len(corner_list)]

        # Extract section points (wrapping around if needed)
        if i_end > i_start:
            section = points[i_start:i_end + 1]
        else:
            # Wrap around: from i_start to end, then start to i_end
            section = np.concatenate([points[i_start:n - 1], points[:i_end + 1]])

        if len(section) < 2:
            continue

        if len(section) == 2:
            segments.append(_line_to_bezier(section[0], section[-1]))
            continue

        # Collinearity check: if all points are near-straight, emit a line
        if _max_perpendicular_distance(section) <= line_tolerance:
            segments.append(_line_to_bezier(section[0], section[-1]))
            continue

        # Fit this smooth section with recursive Bézier fitter
        section_segs = _fit_cubic_bezier(section, max_error, line_tolerance)
        segments.extend(section_segs)

    return FittedCurve(segments=segments, is_closed=True)


def _detect_corners(points: np.ndarray, threshold_deg: float) -> list[int]:
    """Detect corner points where the path changes direction sharply."""
    if len(points) < 3:
        return []
    threshold_rad = np.radians(threshold_deg)
    v1 = points[1:-1] - points[:-2]
    v2 = points[2:] - points[1:-1]
    len1 = np.linalg.norm(v1, axis=1)
    len2 = np.linalg.norm(v2, axis=1)
    valid = (len1 > 0) & (len2 > 0)
    cos_angle = np.full(len(v1), 1.0)
    cos_angle[valid] = np.clip(
        np.sum(v1[valid] * v2[valid], axis=1) / (len1[valid] * len2[valid]),
        -1.0, 1.0,
    )
    angles = np.arccos(cos_angle)
    corner_mask = valid & (angles > threshold_rad)
    return (np.where(corner_mask)[0] + 1).tolist()


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

    # Recursion guard: fall back to line segments to prevent
    # exponential blowup on noisy contours
    if _depth > 8 or len(points) < 3:
        return [_line_to_bezier(points[0], points[-1])]

    # --- Collinearity check ---
    # If all points lie within line_tolerance of the chord, emit a
    # single straight line.  line_tolerance is intentionally more
    # generous than max_error so pixel-staircase artifacts along
    # straight edges collapse into clean L commands.
    if _max_perpendicular_distance(points) <= line_tolerance:
        return [_line_to_bezier(points[0], points[-1])]

    # For large sections, skip the expensive fit+error cycle and split
    # at the midpoint immediately.  A single Bézier can't fit >60
    # points well anyway, so we avoid O(n) parameterize + fit + error
    # at each recursion level, turning O(n log n) into O(n).
    if len(points) > 60:
        mid = len(points) // 2
        left = _fit_cubic_bezier(points[:mid + 1], max_error, line_tolerance, _depth + 1)
        right = _fit_cubic_bezier(points[mid:], max_error, line_tolerance, _depth + 1)
        return left + right

    # Estimate tangent directions
    t_hat1 = _compute_tangent(points, 0, forward=True)
    t_hat2 = _compute_tangent(points, len(points) - 1, forward=False)

    # Parameterize by chord length
    t_params = _chord_length_parameterize(points)

    # Fit single Bézier
    seg, error, split_idx = _fit_single_bezier_with_error(points, t_params, t_hat1, t_hat2)

    if error <= max_error:
        # Post-fit check: convert near-straight Bézier to true line
        if _is_near_straight(seg, line_tolerance):
            return [_line_to_bezier(seg.p0, seg.p3)]
        return [seg]

    # Newton-Raphson reparameterization: refine t_params to reduce error
    # before resorting to splitting
    if error <= max_error * 4:
        for _ in range(5):
            t_params = _reparameterize(points, seg, t_params)
            seg, error, split_idx = _fit_single_bezier_with_error(points, t_params, t_hat1, t_hat2)
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

    t = t_params
    # Bernstein basis values — vectorised over all points
    b1 = 3.0 * (1.0 - t) ** 2 * t          # (n,)
    b2 = 3.0 * (1.0 - t) * t ** 2           # (n,)

    # a vectors: a0[i] = b1[i]*t_hat1,  a1[i] = b2[i]*t_hat2
    a0 = b1[:, None] * t_hat1               # (n, 2)
    a1 = b2[:, None] * t_hat2               # (n, 2)

    c00 = np.dot(a0.ravel(), a0.ravel())
    c01 = np.dot(a0.ravel(), a1.ravel())
    c11 = np.dot(a1.ravel(), a1.ravel())

    # tmp = points - (b0+b1)*p0 - (b2+b3)*p3
    b0 = (1.0 - t) ** 3
    b3 = t ** 3
    tmp = points - (b0 + b1)[:, None] * p0 - (b2 + b3)[:, None] * p3

    x0 = np.dot(a0.ravel(), tmp.ravel())
    x1 = np.dot(a1.ravel(), tmp.ravel())

    det = c00 * c11 - c01 * c01
    if abs(det) < 1e-12:
        alpha1 = alpha2 = math.hypot(*(p3 - p0)) / 3.0
    else:
        alpha1 = (x0 * c11 - x1 * c01) / det
        alpha2 = (c00 * x1 - c01 * x0) / det

    seg_len = math.hypot(*(p3 - p0))
    epsilon = 1e-6 * seg_len

    if alpha1 < epsilon or alpha2 < epsilon:
        alpha1 = alpha2 = seg_len / 3.0

    p1 = p0 + t_hat1 * alpha1
    p2 = p3 + t_hat2 * alpha2

    return BezierSegment(p0=p0, p1=p1, p2=p2, p3=p3)


def _fit_single_bezier_with_error(
    points: np.ndarray,
    t_params: np.ndarray,
    t_hat1: np.ndarray,
    t_hat2: np.ndarray,
) -> tuple[BezierSegment, float, int]:
    """Fit a single cubic Bézier AND compute max error in one pass.

    Shares the Bernstein basis computation between fitting and error
    evaluation, avoiding redundant work.
    """
    p0 = points[0].copy()
    p3 = points[-1].copy()
    t = t_params
    omt = 1.0 - t

    # Bernstein basis values — computed ONCE, used for both fit and error
    b0 = omt ** 3
    b1 = 3.0 * omt ** 2 * t
    b2 = 3.0 * omt * t ** 2
    b3 = t ** 3

    # --- Least-squares fit ---
    a0 = b1[:, None] * t_hat1
    a1 = b2[:, None] * t_hat2

    c00 = np.dot(a0.ravel(), a0.ravel())
    c01 = np.dot(a0.ravel(), a1.ravel())
    c11 = np.dot(a1.ravel(), a1.ravel())

    tmp = points - (b0 + b1)[:, None] * p0 - (b2 + b3)[:, None] * p3

    x0 = np.dot(a0.ravel(), tmp.ravel())
    x1 = np.dot(a1.ravel(), tmp.ravel())

    det = c00 * c11 - c01 * c01
    if abs(det) < 1e-12:
        alpha1 = alpha2 = math.hypot(*(p3 - p0)) / 3.0
    else:
        alpha1 = (x0 * c11 - x1 * c01) / det
        alpha2 = (c00 * x1 - c01 * x0) / det

    seg_len = math.hypot(*(p3 - p0))
    epsilon = 1e-6 * seg_len

    if alpha1 < epsilon or alpha2 < epsilon:
        alpha1 = alpha2 = seg_len / 3.0

    p1 = p0 + t_hat1 * alpha1
    p2 = p3 + t_hat2 * alpha2
    seg = BezierSegment(p0=p0, p1=p1, p2=p2, p3=p3)

    # --- Max error (reuses b0, b1, b2, b3) ---
    pts = (b0[:, None] * p0 + b1[:, None] * p1
           + b2[:, None] * p2 + b3[:, None] * p3)
    diff = points - pts
    sq_err = np.sum(diff * diff, axis=1)
    split_idx = int(np.argmax(sq_err))
    max_err = float(np.sqrt(sq_err[split_idx]))

    return seg, max_err, split_idx


def _compute_tangent(points: np.ndarray, idx: int, forward: bool) -> np.ndarray:
    """Compute unit tangent at a point using multi-neighbour average."""
    span = min(8, len(points) - 1 - idx) if forward else min(8, idx)
    if span <= 0:
        return np.array([1.0, 0.0])
    if forward:
        diff = points[idx + span] - points[idx]
    else:
        diff = points[idx - span] - points[idx]
    norm = math.hypot(diff[0], diff[1])
    return diff / norm if norm > 0 else np.array([1.0, 0.0])


def _chord_length_parameterize(points: np.ndarray) -> np.ndarray:
    """Assign parameter values by cumulative chord length."""
    diffs = np.diff(points, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    cum = np.empty(len(points))
    cum[0] = 0.0
    np.cumsum(seg_lens, out=cum[1:])
    total = cum[-1]
    if total > 0:
        cum /= total
    return cum


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
    t = t_params
    b0 = (1.0 - t) ** 3
    b1 = 3.0 * (1.0 - t) ** 2 * t
    b2 = 3.0 * (1.0 - t) * t ** 2
    b3 = t ** 3
    pts = (b0[:, None] * seg.p0 + b1[:, None] * seg.p1
           + b2[:, None] * seg.p2 + b3[:, None] * seg.p3)
    diff = points - pts
    sq_err = np.sum(diff * diff, axis=1)
    split_idx = int(np.argmax(sq_err))
    max_err = float(np.sqrt(sq_err[split_idx]))
    return max_err, split_idx


def _is_near_straight(seg: BezierSegment, tol: float) -> bool:
    """Check if a cubic Bézier's control points are within tol of the chord."""
    d = seg.p3 - seg.p0
    chord_len = math.hypot(d[0], d[1])
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
    seg_len = math.hypot(d[0], d[1])
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
    t = t_params.copy()
    omt = 1.0 - t  # one-minus-t

    # Bézier evaluation (vectorised)
    b0 = omt ** 3
    b1 = 3.0 * omt ** 2 * t
    b2 = 3.0 * omt * t ** 2
    b3 = t ** 3
    pts = (b0[:, None] * seg.p0 + b1[:, None] * seg.p1
           + b2[:, None] * seg.p2 + b3[:, None] * seg.p3)

    # First derivative
    d10 = seg.p1 - seg.p0
    d21 = seg.p2 - seg.p1
    d32 = seg.p3 - seg.p2
    d1 = (3.0 * omt ** 2)[:, None] * d10 + (6.0 * omt * t)[:, None] * d21 + (3.0 * t ** 2)[:, None] * d32

    # Second derivative
    dd0 = seg.p2 - 2.0 * seg.p1 + seg.p0
    dd1 = seg.p3 - 2.0 * seg.p2 + seg.p1
    d2 = (6.0 * omt)[:, None] * dd0 + (6.0 * t)[:, None] * dd1

    diff = pts - points
    numerator = np.sum(diff * d1, axis=1)
    denominator = np.sum(d1 * d1, axis=1) + np.sum(diff * d2, axis=1)

    valid = np.abs(denominator) > 1e-12
    t[valid] -= numerator[valid] / denominator[valid]
    return np.clip(t, 0.0, 1.0)


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


def reduce_nodes(curve: FittedCurve, max_error: float = 0.5) -> FittedCurve:
    """Post-fit pass: merge consecutive line segments into cubic Bézier curves.

    Collects runs of consecutive line segments and attempts to re-fit them
    as a single (or fewer) cubic Bézier curve(s).  This reduces node count
    significantly for contours that have many short collinear runs.
    """
    if not curve.segments:
        return curve

    new_segments: list[BezierSegment] = []
    run: list[BezierSegment] = []

    def _flush_run() -> None:
        """Try to merge the current run of line segments into fewer curves."""
        if not run:
            return
        if len(run) <= 2:
            new_segments.extend(run)
            return
        # Collect all endpoints of the line-segment run
        pts = [run[0].p0]
        for seg in run:
            pts.append(seg.p3)
        pts_arr = np.array(pts, dtype=np.float64)
        # Re-fit as Bézier curves with generous tolerance
        merged = _fit_cubic_bezier(pts_arr, max_error, max_error)
        new_segments.extend(merged)

    for seg in curve.segments:
        if seg.is_line:
            run.append(seg)
        else:
            _flush_run()
            run = []
            new_segments.append(seg)

    _flush_run()

    return FittedCurve(segments=new_segments, is_closed=curve.is_closed)


def enforce_g1_continuity(curve: FittedCurve) -> FittedCurve:
    """Enforce G1 tangent continuity at all Bézier segment joins.

    At each join point, projects the outgoing and incoming control points
    onto a shared average tangent direction.  This eliminates visible kinks
    at segment boundaries — the #1 visual indicator of computer-traced
    (vs hand-crafted) vector art.

    Preserves control arm lengths (curvature magnitude) while aligning
    tangent directions.  Skips corners where the angle change is
    intentionally sharp (> 70°).
    """
    segs = curve.segments
    if len(segs) < 2:
        return curve

    new_segs = [BezierSegment(p0=s.p0.copy(), p1=s.p1.copy(),
                              p2=s.p2.copy(), p3=s.p3.copy(),
                              is_line=s.is_line) for s in segs]

    n = len(new_segs)
    loop = curve.is_closed
    count = n if loop else n - 1

    for i in range(count):
        j = (i + 1) % n
        seg_a = new_segs[i]
        seg_b = new_segs[j]

        # Skip if either segment is a line — lines have fixed direction
        if seg_a.is_line and seg_b.is_line:
            continue

        join = seg_a.p3  # = seg_b.p0

        t_out = seg_a.p3 - seg_a.p2  # outgoing tangent of seg A
        t_in = seg_b.p1 - seg_b.p0   # incoming tangent of seg B

        len_out = math.hypot(t_out[0], t_out[1])
        len_in = math.hypot(t_in[0], t_in[1])

        if len_out < 1e-10 or len_in < 1e-10:
            continue

        dir_out = t_out / len_out
        dir_in = t_in / len_in

        # Check if this is an intentional corner (angle > 80°)
        dot = float(np.dot(dir_out, dir_in))
        if dot < 0.17:  # cos(80°) ≈ 0.17 → sharp corner, don't smooth
            continue

        # Weighted average tangent — weight by control arm length so
        # longer segments have more influence on the shared direction.
        avg_dir = dir_out * len_out + dir_in * len_in
        avg_len = math.hypot(avg_dir[0], avg_dir[1])
        if avg_len < 1e-10:
            continue
        avg_dir = avg_dir / avg_len

        # Project control points onto average tangent
        if not seg_a.is_line:
            new_segs[i].p2 = join - avg_dir * len_out
        if not seg_b.is_line:
            new_segs[j].p1 = join + avg_dir * len_in

    return FittedCurve(segments=new_segs, is_closed=curve.is_closed)


def merge_segments_artistic(curve: FittedCurve, tolerance: float = 0.5) -> FittedCurve:
    """Merge adjacent segments into fewer, smoother curves.

    Inspired by Potrace's optiCurve: attempts to merge consecutive
    segment groups into single cubic Béziers.  Uses area-preserving
    constraints to maintain shape fidelity while dramatically reducing
    node count.

    This is the key transform from "computer traced" to "hand-crafted":
    a gentle curve that was split into 5 segments by the error-threshold
    fitter gets consolidated into 1-2 smooth segments.
    """
    segs = curve.segments
    if len(segs) <= 2:
        return curve

    n = len(segs)
    # Collect all join points
    pts = [segs[0].p0]
    for s in segs:
        pts.append(s.p3)
    pts = np.array(pts)

    # DP: cost[j] = min segments to represent segs[0..j]
    INF = float('inf')
    cost = [INF] * (n + 1)
    cost[0] = 0
    parent = [-1] * (n + 1)
    MAX_MERGE = min(10, n)  # max segments to merge at once

    for j in range(1, n + 1):
        for i in range(max(0, j - MAX_MERGE), j):
            # Try merging segs[i..j-1] into one curve
            if j - i == 1:
                # Single segment — no merge needed, cost = 1
                new_cost = cost[i] + 1
                if new_cost < cost[j]:
                    cost[j] = new_cost
                    parent[j] = i
                continue

            # Check all intermediate segments — skip if any is a corner
            has_corner = False
            for k in range(i + 1, j):
                # Corner = sharp angle between adjacent segments
                t_before = segs[k - 1].p3 - segs[k - 1].p2
                t_after = segs[k].p1 - segs[k].p0
                lb = math.hypot(t_before[0], t_before[1])
                la = math.hypot(t_after[0], t_after[1])
                if lb > 1e-10 and la > 1e-10:
                    dot = float(np.dot(t_before / lb, t_after / la))
                    if dot < 0.5:  # > 60° angle change → corner
                        has_corner = True
                        break
            if has_corner:
                continue

            # Collect intermediate points
            merge_pts = [pts[i]]
            for k in range(i, j):
                merge_pts.append(pts[k + 1])
            merge_pts = np.array(merge_pts)

            # Fit single Bézier to the merged points
            p0 = merge_pts[0]
            p3 = merge_pts[-1]
            chord = math.hypot(*(p3 - p0))
            if chord < 1e-10:
                continue

            # Use tangent from first and last segment for direction hints
            t_hat1 = _compute_tangent(merge_pts, 0, forward=True)
            t_hat2 = _compute_tangent(merge_pts, len(merge_pts) - 1, forward=False)
            t_params = _chord_length_parameterize(merge_pts)
            seg_try, err, _ = _fit_single_bezier_with_error(merge_pts, t_params, t_hat1, t_hat2)

            if err <= tolerance:
                new_cost = cost[i] + 1
                if new_cost < cost[j]:
                    cost[j] = new_cost
                    parent[j] = i

    # Backtrace to find optimal grouping
    groups = []
    j = n
    while j > 0:
        i = parent[j]
        if i < 0:
            # Fallback: single segments
            for k in range(j - 1, -1, -1):
                groups.append((k, k + 1))
            break
        groups.append((i, j))
        j = i
    groups.reverse()

    # Build merged segments
    new_segs = []
    for (i, j) in groups:
        if j - i == 1:
            new_segs.append(segs[i])
        else:
            merge_pts = [pts[i]]
            for k in range(i, j):
                merge_pts.append(pts[k + 1])
            merge_pts = np.array(merge_pts)
            t_hat1 = _compute_tangent(merge_pts, 0, forward=True)
            t_hat2 = _compute_tangent(merge_pts, len(merge_pts) - 1, forward=False)
            t_params = _chord_length_parameterize(merge_pts)
            seg_merged = _fit_single_bezier(merge_pts, t_params, t_hat1, t_hat2)
            if _is_near_straight(seg_merged, tolerance):
                seg_merged = _line_to_bezier(seg_merged.p0, seg_merged.p3)
            new_segs.append(seg_merged)

    return FittedCurve(segments=new_segs, is_closed=curve.is_closed)
