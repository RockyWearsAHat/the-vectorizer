"""SVG export module.

Stage 6: Convert fitted Bézier curves into clean SVG output with proper
<path> elements, viewBox, transparent background, and logical grouping.
"""

import io
import svgwrite
import numpy as np
from dataclasses import dataclass

from ..stroke_reconstruction import StrokePath, StrokeResult
from ..fill_reconstruction import FillPath, FillResult
from ..curve_fitting import FittedCurve, BezierSegment, fit_bezier_path, fit_closed_bezier


@dataclass
class SVGOutput:
    svg_string: str
    width: int
    height: int
    path_count: int
    node_count: int


def export_svg(
    stroke_result: StrokeResult,
    fill_result: FillResult,
    image_size: tuple[int, int],
    *,
    stroke_color: str = "#000000",
    fill_color: str = "#000000",
    max_error: float = 2.0,
    corner_threshold: float = 45.0,
) -> SVGOutput:
    """Generate SVG from stroke and fill reconstruction results.

    Args:
        stroke_result: Reconstructed stroke paths.
        fill_result: Reconstructed fill paths.
        image_size: (height, width) of source image.
        stroke_color: Color for stroke paths.
        fill_color: Color for fill paths.
        max_error: Bézier fitting error tolerance.
        corner_threshold: Corner detection angle.

    Returns:
        SVGOutput with SVG string and metadata.
    """
    h, w = image_size
    dwg = svgwrite.Drawing(size=(f"{w}px", f"{h}px"))
    dwg.viewbox(0, 0, w, h)

    path_count = 0
    node_count = 0

    # Group for fill shapes
    fill_group = dwg.g(id="fills")
    for fp in fill_result.paths:
        curve = fit_closed_bezier(
            fp.outer, max_error=max_error, corner_threshold=corner_threshold
        )
        d = _curve_to_path_d(curve)

        # Add holes
        for hole in fp.holes:
            hole_curve = fit_closed_bezier(
                hole, max_error=max_error, corner_threshold=corner_threshold
            )
            d += " " + _curve_to_path_d(hole_curve)

        if d.strip():
            fill_group.add(dwg.path(
                d=d,
                fill=fill_color,
                stroke="none",
                fill_rule="evenodd",
            ))
            path_count += 1
            node_count += d.count("C") + d.count("L") + d.count("M")

    dwg.add(fill_group)

    # Group for stroke paths
    stroke_group = dwg.g(id="strokes")
    for sp in stroke_result.paths:
        curve = fit_bezier_path(
            sp.points,
            max_error=max_error,
            corner_threshold=corner_threshold,
            is_closed=sp.is_closed,
        )
        d = _curve_to_path_d(curve)

        if d.strip():
            avg_width = float(np.mean(sp.widths)) if len(sp.widths) > 0 else 2.0
            avg_width = max(0.5, avg_width)

            stroke_group.add(dwg.path(
                d=d,
                fill="none",
                stroke=stroke_color,
                stroke_width=f"{avg_width:.1f}",
                stroke_linecap="round",
                stroke_linejoin="round",
            ))
            path_count += 1
            node_count += d.count("C") + d.count("L") + d.count("M")

    dwg.add(stroke_group)

    svg_string = dwg.tostring()

    return SVGOutput(
        svg_string=svg_string,
        width=w,
        height=h,
        path_count=path_count,
        node_count=node_count,
    )


def _curve_to_path_d(curve: FittedCurve) -> str:
    """Convert a FittedCurve to an SVG path `d` attribute string."""
    if not curve.segments:
        return ""

    parts = []
    first = curve.segments[0]
    parts.append(f"M {first.p0[0]:.2f},{first.p0[1]:.2f}")

    for seg in curve.segments:
        parts.append(
            f"C {seg.p1[0]:.2f},{seg.p1[1]:.2f} "
            f"{seg.p2[0]:.2f},{seg.p2[1]:.2f} "
            f"{seg.p3[0]:.2f},{seg.p3[1]:.2f}"
        )

    if curve.is_closed:
        parts.append("Z")

    return " ".join(parts)
