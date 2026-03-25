"""Verify tangent fix: circle should produce curves now."""
import numpy as np
import sys, math
sys.path.insert(0, 'raster-to-vector/server')
from app.core.curve_fitting import (
    fit_closed_bezier, _fit_cubic_bezier,
    _max_perpendicular_distance, _detect_corners,
)
import cv2

# Full circle, 60 points, radius 50
t = np.linspace(0, 2*np.pi, 60, endpoint=False)
circle = np.column_stack([np.cos(t)*50 + 100, np.sin(t)*50 + 100])

# RDP-simplified
circle_cv = circle.reshape(-1, 1, 2).astype(np.float32)
simplified = cv2.approxPolyDP(circle_cv, 1.5, closed=True).reshape(-1, 2)
print(f"RDP circle: {len(simplified)} pts from 60")

# fit_closed_bezier on raw circle
for ct, lt in [(55.0, 1.2), (90.0, 1.2), (55.0, 0.3)]:
    curve = fit_closed_bezier(circle, max_error=2.0, corner_threshold=ct, line_tolerance=lt)
    n_l = sum(1 for s in curve.segments if s.is_line)
    n_c = sum(1 for s in curve.segments if not s.is_line)
    print(f"  corner={ct}, ltol={lt}: {len(curve.segments)} segs ({n_l} L, {n_c} C)")

# fit_closed_bezier on RDP circle  
curve = fit_closed_bezier(simplified.astype(np.float64), max_error=2.0, corner_threshold=55.0, line_tolerance=1.2)
n_l = sum(1 for s in curve.segments if s.is_line)
n_c = sum(1 for s in curve.segments if not s.is_line)
print(f"  RDP circle: {len(curve.segments)} segs ({n_l} L, {n_c} C)")

# Quarter arc test
section = circle[:20]
segs = _fit_cubic_bezier(section, 2.0, 1.2)
n_l = sum(1 for s in segs if s.is_line)
n_c = sum(1 for s in segs if not s.is_line)
print(f"\nQuarter arc 20pts: {len(segs)} segs ({n_l} L, {n_c} C)")

