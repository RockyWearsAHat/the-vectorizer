"""Deep diagnostic: find exactly WHERE and WHY the SVG differs from the source.

Goes far beyond SSIM — breaks down:
  1. Per-patch error hotspots (64×64 tiles ranked by MAE)
  2. Edge fidelity: edge-only comparison (Canny on source vs SVG)
  3. Color accuracy: per-cluster ΔE in Lab space
  4. Straight-line wobble detection on the actual SVG path data
  5. Junction quality: gaps/overlaps at contour intersections
  6. Thin feature preservation: skeleton vs SVG coverage
  7. Halo quality: anti-aliasing gradient smoothness check

Outputs a ranked list of actionable findings with locations.
"""

import cv2
import io
import sys
import re
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, ".")
from app.core.multilevel import multilevel_vectorize, generate_svg
from app.core.comparison import _rasterize_svg

import cairosvg


# ---------------------------------------------------------------------------
# 0. Load and vectorize
# ---------------------------------------------------------------------------

def load_and_vectorize(img_path: str):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(img_path)
    result = multilevel_vectorize(img)
    svg = generate_svg(result)
    return img, result, svg


# ---------------------------------------------------------------------------
# 1. Per-patch error hotspots
# ---------------------------------------------------------------------------

@dataclass
class PatchError:
    row: int
    col: int
    y0: int
    x0: int
    y1: int
    x1: int
    mae: float
    ssim_val: float
    max_pixel_err: float
    dominant_error_type: str  # "edge_shift", "color_mismatch", "missing_detail", "halo_artifact"


def patch_analysis(src_gray, svg_gray, tile=64):
    """Tile the image and rank patches by error."""
    h, w = src_gray.shape
    patches = []
    for r in range(0, h, tile):
        for c in range(0, w, tile):
            y1, x1 = min(r + tile, h), min(c + tile, w)
            sp = src_gray[r:y1, c:x1]
            vp = svg_gray[r:y1, c:x1]
            if sp.shape[0] < 8 or sp.shape[1] < 8:
                continue
            diff = np.abs(sp.astype(float) - vp.astype(float))
            mae = float(diff.mean())
            max_err = float(diff.max())

            # SSIM per patch
            try:
                s = float(ssim(sp, vp))
            except Exception:
                s = 1.0

            # Classify dominant error type
            etype = _classify_patch_error(sp, vp, diff)

            patches.append(PatchError(
                row=r // tile, col=c // tile,
                y0=r, x0=c, y1=y1, x1=x1,
                mae=mae, ssim_val=s, max_pixel_err=max_err,
                dominant_error_type=etype,
            ))
    patches.sort(key=lambda p: p.mae, reverse=True)
    return patches


def _classify_patch_error(src, svg, diff):
    """Classify what kind of error dominates a patch."""
    # Edge detection on both
    src_edges = cv2.Canny(src, 50, 150)
    svg_edges = cv2.Canny(svg, 50, 150)

    # Error concentrated near edges?
    edge_mask = cv2.dilate(src_edges | svg_edges, np.ones((5, 5), np.uint8))
    edge_area = edge_mask > 0
    flat_area = ~edge_area

    if edge_area.sum() > 0 and flat_area.sum() > 0:
        edge_err = diff[edge_area].mean()
        flat_err = diff[flat_area].mean()
        if edge_err > flat_err * 2.0:
            return "edge_shift"
        elif flat_err > edge_err * 1.5:
            return "color_mismatch"

    # Check for missing fine detail (high-freq diff)
    high_freq = cv2.Laplacian(diff.astype(np.float32), cv2.CV_32F)
    if np.abs(high_freq).mean() > 3.0:
        return "missing_detail"

    return "halo_artifact"


# ---------------------------------------------------------------------------
# 2. Edge fidelity analysis
# ---------------------------------------------------------------------------

@dataclass
class EdgeMetrics:
    precision: float       # fraction of SVG edges that match source edges
    recall: float          # fraction of source edges captured in SVG
    f1: float
    avg_edge_displacement: float  # average pixel distance of nearest edge
    max_edge_displacement: float
    displaced_regions: list  # regions where edges are most shifted


def edge_fidelity(src_gray, svg_gray):
    """Compare edge maps: source vs rendered SVG."""
    src_edges = cv2.Canny(src_gray, 40, 120)
    svg_edges = cv2.Canny(svg_gray, 40, 120)

    # Distance transform from source edges
    inv_src = (src_edges == 0).astype(np.uint8)
    dt_src = distance_transform_edt(inv_src)

    # Distance of each SVG edge pixel to nearest source edge
    svg_edge_pts = np.argwhere(svg_edges > 0)
    if len(svg_edge_pts) == 0:
        return EdgeMetrics(0, 0, 0, 999, 999, [])

    svg_dists = dt_src[svg_edge_pts[:, 0], svg_edge_pts[:, 1]]

    # Precision: SVG edges within 2px of a source edge
    precision = float((svg_dists <= 2.0).sum() / len(svg_dists))

    # Recall: source edges within 2px of an SVG edge
    inv_svg = (svg_edges == 0).astype(np.uint8)
    dt_svg = distance_transform_edt(inv_svg)
    src_edge_pts = np.argwhere(src_edges > 0)
    if len(src_edge_pts) > 0:
        src_dists = dt_svg[src_edge_pts[:, 0], src_edge_pts[:, 1]]
        recall = float((src_dists <= 2.0).sum() / len(src_dists))
    else:
        recall = 1.0

    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    avg_disp = float(svg_dists.mean())
    max_disp = float(svg_dists.max())

    # Find worst displacement regions (top 5 connected components)
    bad_mask = np.zeros_like(svg_edges)
    for i, pt in enumerate(svg_edge_pts):
        if svg_dists[i] > 3.0:
            bad_mask[pt[0], pt[1]] = 255

    n_comp, labels_cc = cv2.connectedComponents(bad_mask)
    regions = []
    for c in range(1, n_comp):
        pts = np.argwhere(labels_cc == c)
        if len(pts) < 5:
            continue
        y0, x0 = pts.min(axis=0)
        y1, x1 = pts.max(axis=0)
        avg_d = float(dt_src[labels_cc == c].mean())
        regions.append({
            "bbox": [int(y0), int(x0), int(y1), int(x1)],
            "pixel_count": len(pts),
            "avg_displacement": round(avg_d, 2),
        })
    regions.sort(key=lambda r: r["avg_displacement"], reverse=True)

    return EdgeMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        avg_edge_displacement=avg_disp,
        max_edge_displacement=max_disp,
        displaced_regions=regions[:10],
    )


# ---------------------------------------------------------------------------
# 3. Color accuracy per cluster
# ---------------------------------------------------------------------------

@dataclass
class ClusterColorError:
    cluster_idx: int
    target_bgr: tuple
    actual_mean_bgr: tuple
    delta_e: float   # CIE ΔE in Lab
    pixel_count: int
    coverage_error: float  # fraction of cluster pixels that differ >10


def color_accuracy(src_bgr, svg_string, result):
    """Measure per-cluster color fidelity in Lab space."""
    h, w = src_bgr.shape[:2]

    # Rasterize SVG to color
    png_bytes = cairosvg.svg2png(
        bytestring=svg_string.encode("utf-8"),
        output_width=w, output_height=h,
        background_color="white",
    )
    svg_pil = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    svg_bgr = cv2.cvtColor(np.array(svg_pil), cv2.COLOR_RGB2BGR)

    # Convert both to Lab
    src_lab = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2Lab).astype(np.float64)
    svg_lab = cv2.cvtColor(svg_bgr, cv2.COLOR_BGR2Lab).astype(np.float64)

    errors = []
    for li, layer in enumerate(result.layers):
        # Parse the hex color
        hex_color = layer.color
        b = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        r = int(hex_color[5:7], 16)
        target_bgr = np.array([[[b, g, r]]], dtype=np.uint8)
        target_lab = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2Lab).astype(np.float64)[0, 0]

        # Find where SVG color matches this layer (within tolerance)
        svg_lab_flat = svg_lab.reshape(-1, 3)
        # Pixels close to this cluster's color in rendered SVG
        d = np.sqrt(np.sum((svg_lab_flat - target_lab) ** 2, axis=1))
        mask = d < 15.0  # within ΔE=15 of target
        if mask.sum() < 10:
            continue

        # Corresponding source pixels
        src_lab_flat = src_lab.reshape(-1, 3)
        src_at_cluster = src_lab_flat[mask]
        svg_at_cluster = svg_lab_flat[mask]

        # Mean ΔE between source and SVG at cluster locations
        per_pixel_de = np.sqrt(np.sum((src_at_cluster - svg_at_cluster) ** 2, axis=1))
        mean_de = float(per_pixel_de.mean())

        # Coverage: how many pixels differ by more than perceptible ΔE
        src_mean = src_at_cluster.mean(axis=0)
        actual_bgr = cv2.cvtColor(
            np.array([[src_mean]], dtype=np.float64).astype(np.uint8),
            cv2.COLOR_Lab2BGR
        )[0, 0]

        coverage_err = float((per_pixel_de > 10.0).sum() / len(per_pixel_de))

        errors.append(ClusterColorError(
            cluster_idx=li,
            target_bgr=(int(b), int(g), int(r)),
            actual_mean_bgr=tuple(int(x) for x in actual_bgr),
            delta_e=mean_de,
            pixel_count=int(mask.sum()),
            coverage_error=coverage_err,
        ))

    errors.sort(key=lambda e: e.delta_e, reverse=True)
    return errors


# ---------------------------------------------------------------------------
# 4. SVG path wobble detection
# ---------------------------------------------------------------------------

@dataclass
class WobbleReport:
    total_line_segments: int
    total_curve_segments: int
    wobble_candidates: list  # segments that should be straight but aren't
    max_wobble_deviation: float


def detect_wobble(svg_string):
    """Parse SVG paths and find curve segments that should be straight lines.

    A Bézier curve is "nearly straight" if its control points are close
    to the chord — these are wobble candidates that the fitter should
    have emitted as L commands instead.
    """
    # Extract all path d attributes
    paths = re.findall(r'd="([^"]+)"', svg_string)
    total_lines = 0
    total_curves = 0
    wobble_candidates = []

    for path_d in paths:
        segments = _parse_svg_path(path_d)
        for seg in segments:
            if seg["type"] == "L":
                total_lines += 1
            elif seg["type"] == "C":
                total_curves += 1
                # Check if this curve is nearly straight
                p0 = np.array(seg["p0"])
                p1 = np.array(seg["p1"])
                p2 = np.array(seg["p2"])
                p3 = np.array(seg["p3"])
                chord_len = np.linalg.norm(p3 - p0)
                if chord_len < 0.5:
                    continue
                # Max distance of control points from chord
                d = p3 - p0
                n = np.array([-d[1], d[0]]) / chord_len
                d1 = abs(float(np.dot(p1 - p0, n)))
                d2 = abs(float(np.dot(p2 - p0, n)))
                max_dev = max(d1, d2)
                # If control points are within 0.3px of chord but it's still
                # a C command, that's a wobble candidate
                if max_dev < 0.5 and chord_len > 2.0:
                    wobble_candidates.append({
                        "p0": [round(p0[0], 2), round(p0[1], 2)],
                        "p3": [round(p3[0], 2), round(p3[1], 2)],
                        "chord_length": round(chord_len, 2),
                        "max_control_deviation": round(max_dev, 3),
                    })

    max_dev = max((w["max_control_deviation"] for w in wobble_candidates), default=0)
    return WobbleReport(
        total_line_segments=total_lines,
        total_curve_segments=total_curves,
        wobble_candidates=wobble_candidates,
        max_wobble_deviation=max_dev,
    )


def _parse_svg_path(d_str):
    """Minimal SVG path parser — extracts M, L, C, Z commands."""
    segments = []
    tokens = re.findall(r'[MLCZ]|[-+]?\d*\.?\d+', d_str)
    i = 0
    cur = np.array([0.0, 0.0])
    while i < len(tokens):
        cmd = tokens[i]
        if cmd == "M":
            cur = np.array([float(tokens[i+1]), float(tokens[i+2])])
            i += 3
        elif cmd == "L":
            p = np.array([float(tokens[i+1]), float(tokens[i+2])])
            segments.append({"type": "L", "p0": cur.tolist(), "p3": p.tolist()})
            cur = p
            i += 3
        elif cmd == "C":
            p1 = np.array([float(tokens[i+1]), float(tokens[i+2])])
            p2 = np.array([float(tokens[i+3]), float(tokens[i+4])])
            p3 = np.array([float(tokens[i+5]), float(tokens[i+6])])
            segments.append({
                "type": "C", "p0": cur.tolist(),
                "p1": p1.tolist(), "p2": p2.tolist(), "p3": p3.tolist()
            })
            cur = p3
            i += 7
        elif cmd == "Z":
            i += 1
        else:
            i += 1
    return segments


# ---------------------------------------------------------------------------
# 5. Anti-aliasing gradient quality
# ---------------------------------------------------------------------------

@dataclass
class AAMetrics:
    mean_transition_width: float    # average edge transition width in pixels
    src_transition_width: float     # same measurement on source
    transition_ratio: float         # SVG/source — ideally ~1.0
    too_sharp_fraction: float       # edges sharper than source (aliased)
    too_soft_fraction: float        # edges softer than source (overshoot halo)


def aa_quality(src_gray, svg_gray):
    """Measure anti-aliasing transition quality at edges."""
    # Compute gradient magnitude for both
    src_gx = cv2.Sobel(src_gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    src_gy = cv2.Sobel(src_gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    src_grad = np.sqrt(src_gx**2 + src_gy**2)

    svg_gx = cv2.Sobel(svg_gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    svg_gy = cv2.Sobel(svg_gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    svg_grad = np.sqrt(svg_gx**2 + svg_gy**2)

    # Only compare at significant edges
    edge_mask = src_grad > 10.0

    if edge_mask.sum() < 100:
        return AAMetrics(0, 0, 1.0, 0, 0)

    # Transition width ∝ 1/gradient_magnitude
    src_at_edges = src_grad[edge_mask]
    svg_at_edges = svg_grad[edge_mask]

    # Avoid division by zero
    src_tw = 1.0 / np.clip(src_at_edges, 1.0, None)
    svg_tw = 1.0 / np.clip(svg_at_edges, 1.0, None)

    mean_svg_tw = float(svg_tw.mean())
    mean_src_tw = float(src_tw.mean())
    ratio = mean_svg_tw / mean_src_tw if mean_src_tw > 0 else 1.0

    # Classify: too sharp (SVG gradient > source) vs too soft
    too_sharp = float((svg_at_edges > src_at_edges * 1.5).sum() / len(src_at_edges))
    too_soft = float((svg_at_edges < src_at_edges * 0.5).sum() / len(src_at_edges))

    return AAMetrics(
        mean_transition_width=mean_svg_tw,
        src_transition_width=mean_src_tw,
        transition_ratio=ratio,
        too_sharp_fraction=too_sharp,
        too_soft_fraction=too_soft,
    )


# ---------------------------------------------------------------------------
# 6. Thin feature preservation
# ---------------------------------------------------------------------------

@dataclass
class ThinFeatureMetrics:
    src_skeleton_pixels: int
    svg_coverage_of_skeleton: float  # fraction of src skeleton covered in SVG
    missed_skeleton_regions: list    # bounding boxes of breaks


def thin_features(src_gray, svg_gray):
    """Check if thin strokes/lines are preserved."""
    # Binary threshold both
    _, src_bin = cv2.threshold(src_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, svg_bin = cv2.threshold(svg_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Skeletonize source foreground
    src_skel = skeletonize(src_bin > 0).astype(np.uint8) * 255

    skel_pixels = int((src_skel > 0).sum())
    if skel_pixels == 0:
        return ThinFeatureMetrics(0, 1.0, [])

    # Dilate SVG foreground slightly for tolerance
    svg_dilated = cv2.dilate(svg_bin, np.ones((3, 3), np.uint8))

    # Coverage: skeleton pixels that are foreground in SVG
    covered = (src_skel > 0) & (svg_dilated > 0)
    coverage = float(covered.sum() / skel_pixels)

    # Find missed regions
    missed = (src_skel > 0) & (svg_dilated == 0)
    missed_u8 = missed.astype(np.uint8) * 255
    n_comp, labels_cc = cv2.connectedComponents(missed_u8)
    regions = []
    for c in range(1, n_comp):
        pts = np.argwhere(labels_cc == c)
        if len(pts) < 3:
            continue
        y0, x0 = pts.min(axis=0)
        y1, x1 = pts.max(axis=0)
        regions.append({
            "bbox": [int(y0), int(x0), int(y1), int(x1)],
            "pixel_count": len(pts),
        })
    regions.sort(key=lambda r: r["pixel_count"], reverse=True)

    return ThinFeatureMetrics(
        src_skeleton_pixels=skel_pixels,
        svg_coverage_of_skeleton=coverage,
        missed_skeleton_regions=regions[:10],
    )


# ---------------------------------------------------------------------------
# 7. Per-channel color error analysis (find if B, G, or R is off)
# ---------------------------------------------------------------------------

@dataclass
class ChannelError:
    channel: str
    mean_signed_error: float  # +ve = SVG brighter, -ve = SVG darker
    mean_abs_error: float
    std_error: float


def channel_analysis(src_bgr, svg_string):
    """Compare each color channel: find systematic biases."""
    h, w = src_bgr.shape[:2]
    png_bytes = cairosvg.svg2png(
        bytestring=svg_string.encode("utf-8"),
        output_width=w, output_height=h,
        background_color="white",
    )
    svg_pil = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    svg_bgr = cv2.cvtColor(np.array(svg_pil), cv2.COLOR_RGB2BGR)

    results = []
    for ch, name in [(0, "Blue"), (1, "Green"), (2, "Red")]:
        diff = svg_bgr[:, :, ch].astype(float) - src_bgr[:, :, ch].astype(float)
        results.append(ChannelError(
            channel=name,
            mean_signed_error=float(diff.mean()),
            mean_abs_error=float(np.abs(diff).mean()),
            std_error=float(diff.std()),
        ))
    return results


# ---------------------------------------------------------------------------
# 8. Spatial error distribution
# ---------------------------------------------------------------------------

def spatial_error_profile(src_gray, svg_gray):
    """Where do errors concentrate: center vs border, horizontal vs vertical?"""
    diff = np.abs(src_gray.astype(float) - svg_gray.astype(float))
    h, w = diff.shape
    margin = min(h, w) // 8

    border_err = np.concatenate([
        diff[:margin, :].ravel(), diff[-margin:, :].ravel(),
        diff[margin:-margin, :margin].ravel(), diff[margin:-margin, -margin:].ravel(),
    ]).mean()
    center_err = diff[margin:-margin, margin:-margin].mean()

    # Top vs bottom, left vs right
    top_err = diff[:h//2, :].mean()
    bot_err = diff[h//2:, :].mean()
    left_err = diff[:, :w//2].mean()
    right_err = diff[:, w//2:].mean()

    return {
        "border_mae": round(float(border_err), 3),
        "center_mae": round(float(center_err), 3),
        "top_mae": round(float(top_err), 3),
        "bottom_mae": round(float(bot_err), 3),
        "left_mae": round(float(left_err), 3),
        "right_mae": round(float(right_err), 3),
        "border_vs_center": round(float(border_err / max(center_err, 0.01)), 3),
    }


# ===========================================================================
# MAIN
# ===========================================================================

def run_full_diagnostic(img_path: str):
    print(f"Loading and vectorizing {img_path}...")
    img, result, svg = load_and_vectorize(img_path)
    h, w = img.shape[:2]
    print(f"  Image: {w}×{h}, {result.path_count} paths, {result.node_count} nodes")
    print(f"  Layers: {len(result.layers)}, background: {result.background_color}")

    # Rasterize for comparison
    src_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    svg_gray = _rasterize_svg(svg, w, h)

    # Global metrics
    src_smooth = cv2.GaussianBlur(src_gray, (0, 0), sigmaX=1.5)
    svg_smooth = cv2.GaussianBlur(svg_gray, (0, 0), sigmaX=1.5)
    global_ssim = float(ssim(src_smooth, svg_smooth))
    global_mae = float(np.mean(np.abs(src_smooth.astype(float) - svg_smooth.astype(float))))
    print(f"\n{'='*60}")
    print(f"GLOBAL METRICS: SSIM={global_ssim:.4f}  MAE={global_mae:.2f}")
    print(f"{'='*60}")

    # --- 1. Patch hotspots ---
    print(f"\n--- 1. PATCH ERROR HOTSPOTS (64×64 tiles, top 15) ---")
    patches = patch_analysis(src_gray, svg_gray, tile=64)
    error_type_counts = {}
    for p in patches:
        error_type_counts[p.dominant_error_type] = error_type_counts.get(p.dominant_error_type, 0) + 1

    for i, p in enumerate(patches[:15]):
        print(f"  #{i+1:2d}  y={p.y0:4d}-{p.y1:4d} x={p.x0:4d}-{p.x1:4d}  "
              f"MAE={p.mae:5.1f}  SSIM={p.ssim_val:.3f}  "
              f"maxErr={p.max_pixel_err:3.0f}  type={p.dominant_error_type}")

    print(f"\n  Error type distribution (all {len(patches)} patches):")
    for et, count in sorted(error_type_counts.items(), key=lambda x: -x[1]):
        frac = count / len(patches)
        print(f"    {et:20s}: {count:4d} ({frac:.0%})")

    # --- 2. Edge fidelity ---
    print(f"\n--- 2. EDGE FIDELITY ---")
    em = edge_fidelity(src_gray, svg_gray)
    print(f"  Precision (SVG edges match source):   {em.precision:.3f}")
    print(f"  Recall (source edges in SVG):         {em.recall:.3f}")
    print(f"  F1 score:                             {em.f1:.3f}")
    print(f"  Average edge displacement:            {em.avg_edge_displacement:.2f} px")
    print(f"  Max edge displacement:                {em.max_edge_displacement:.1f} px")
    if em.displaced_regions:
        print(f"  Top displaced regions:")
        for r in em.displaced_regions[:5]:
            print(f"    bbox={r['bbox']}  pixels={r['pixel_count']}  avg_shift={r['avg_displacement']}px")

    # --- 3. Color accuracy ---
    print(f"\n--- 3. COLOR ACCURACY (Lab ΔE per cluster) ---")
    cerrs = color_accuracy(img, svg, result)
    for ce in cerrs[:10]:
        print(f"  Cluster {ce.cluster_idx}: target_bgr={ce.target_bgr}  "
              f"ΔE={ce.delta_e:.1f}  pixels={ce.pixel_count}  "
              f"coverage_err={ce.coverage_error:.0%}")

    # --- 4. SVG path wobble ---
    print(f"\n--- 4. SVG PATH WOBBLE DETECTION ---")
    wobble = detect_wobble(svg)
    print(f"  Total line segments (L):     {wobble.total_line_segments}")
    print(f"  Total curve segments (C):    {wobble.total_curve_segments}")
    print(f"  L/(L+C) ratio:              {wobble.total_line_segments / max(1, wobble.total_line_segments + wobble.total_curve_segments):.1%}")
    print(f"  Near-straight C candidates:  {len(wobble.wobble_candidates)}")
    print(f"  Max wobble deviation:        {wobble.max_wobble_deviation:.3f} px")
    if wobble.wobble_candidates:
        print(f"  Sample wobble segments (control pt deviation < 0.5px from chord):")
        for w in wobble.wobble_candidates[:5]:
            print(f"    p0={w['p0']} → p3={w['p3']}  "
                  f"chord={w['chord_length']:.1f}px  dev={w['max_control_deviation']:.3f}px")

    # --- 5. Anti-aliasing quality ---
    print(f"\n--- 5. ANTI-ALIASING QUALITY ---")
    aa = aa_quality(src_gray, svg_gray)
    print(f"  Source avg transition width:  {aa.src_transition_width:.4f}")
    print(f"  SVG avg transition width:     {aa.mean_transition_width:.4f}")
    print(f"  Ratio (SVG/source):           {aa.transition_ratio:.3f}  (1.0=perfect)")
    print(f"  Too sharp (aliased) edges:    {aa.too_sharp_fraction:.1%}")
    print(f"  Too soft (over-blurred) edges:{aa.too_soft_fraction:.1%}")

    # --- 6. Thin feature preservation ---
    print(f"\n--- 6. THIN FEATURE PRESERVATION ---")
    tf = thin_features(src_gray, svg_gray)
    print(f"  Source skeleton pixels:       {tf.src_skeleton_pixels}")
    print(f"  SVG coverage of skeleton:     {tf.svg_coverage_of_skeleton:.1%}")
    if tf.missed_skeleton_regions:
        print(f"  Missed regions (breaks):")
        for r in tf.missed_skeleton_regions[:5]:
            print(f"    bbox={r['bbox']}  pixels={r['pixel_count']}")

    # --- 7. Per-channel color analysis ---
    print(f"\n--- 7. PER-CHANNEL COLOR ERROR ---")
    channels = channel_analysis(img, svg)
    for ch in channels:
        bias = "brighter" if ch.mean_signed_error > 0 else "darker"
        print(f"  {ch.channel:5s}: signed={ch.mean_signed_error:+.2f} ({bias})  "
              f"MAE={ch.mean_abs_error:.2f}  std={ch.std_error:.2f}")

    # --- 8. Spatial distribution ---
    print(f"\n--- 8. SPATIAL ERROR DISTRIBUTION ---")
    spatial = spatial_error_profile(src_gray, svg_gray)
    for k, v in spatial.items():
        print(f"  {k:20s}: {v}")

    # --- ACTIONABLE SUMMARY ---
    print(f"\n{'='*60}")
    print("ACTIONABLE FINDINGS (ranked by impact)")
    print(f"{'='*60}")

    findings = []

    # Edge analysis findings
    if em.f1 < 0.90:
        findings.append((1.0 - em.f1, "EDGE_FIDELITY",
            f"Edge F1={em.f1:.3f} — {(1-em.recall)*100:.0f}% of source edges missing in SVG. "
            f"Avg displacement={em.avg_edge_displacement:.1f}px. "
            f"Check iso-threshold and contour smoothing."))
    elif em.avg_edge_displacement > 1.5:
        findings.append((em.avg_edge_displacement / 5.0, "EDGE_SHIFT",
            f"Edges present but shifted avg {em.avg_edge_displacement:.1f}px. "
            f"Likely inner_iso/outer_iso too aggressive or smooth_sigma too high."))

    # Color findings
    if cerrs:
        worst_de = cerrs[0].delta_e
        if worst_de > 10:
            findings.append((worst_de / 20.0, "COLOR_ERROR",
                f"Worst cluster ΔE={worst_de:.1f} (perceptible ≥3, bad ≥10). "
                f"Cluster {cerrs[0].cluster_idx} with {cerrs[0].pixel_count} pixels."))

    # Wobble findings
    ratio_L = wobble.total_line_segments / max(1, wobble.total_line_segments + wobble.total_curve_segments)
    if len(wobble.wobble_candidates) > 20:
        findings.append((len(wobble.wobble_candidates) / 100.0, "WOBBLE",
            f"{len(wobble.wobble_candidates)} near-straight curves NOT emitted as lines. "
            f"L/(L+C) ratio={ratio_L:.0%}. line_tolerance may be too tight."))

    # AA findings
    if aa.too_soft_fraction > 0.15:
        findings.append((aa.too_soft_fraction, "OVER_ANTIALIASED",
            f"{aa.too_soft_fraction:.0%} of edges softer than source. "
            f"Halo opacity or smooth_sigma may be too high."))
    if aa.too_sharp_fraction > 0.15:
        findings.append((aa.too_sharp_fraction, "UNDER_ANTIALIASED",
            f"{aa.too_sharp_fraction:.0%} of edges sharper than source (aliased). "
            f"Halo may be too faint or missing on some clusters."))

    # Thin feature findings
    if tf.svg_coverage_of_skeleton < 0.95:
        findings.append((1.0 - tf.svg_coverage_of_skeleton, "THIN_FEATURE_BREAK",
            f"Only {tf.svg_coverage_of_skeleton:.0%} of thin strokes preserved. "
            f"{len(tf.missed_skeleton_regions)} break regions found."))

    # Spatial findings  
    if spatial["border_vs_center"] > 2.0:
        findings.append((spatial["border_vs_center"] / 5.0, "BORDER_ERROR",
            f"Border error {spatial['border_vs_center']:.1f}× center error. "
            f"Background detection or border handling issue."))

    # Channel findings
    for ch in channels:
        if abs(ch.mean_signed_error) > 2.0:
            findings.append((abs(ch.mean_signed_error) / 10.0, "CHANNEL_BIAS",
                f"{ch.channel} channel bias: {ch.mean_signed_error:+.1f} — "
                f"SVG systematically {'brighter' if ch.mean_signed_error > 0 else 'darker'}. "
                f"Check cluster centre computation."))

    # Patch type analysis
    total_patches = len(patches)
    for et, count in error_type_counts.items():
        frac = count / total_patches
        if frac > 0.3 and et != "halo_artifact":
            findings.append((frac, f"DOMINANT_{et.upper()}",
                f"{frac:.0%} of patches have {et} as primary error. "
                f"This is the dominant failure mode."))

    findings.sort(key=lambda f: f[0], reverse=True)
    if not findings:
        print("  No significant issues found — output is near-optimal!")
    else:
        for rank, (score, category, desc) in enumerate(findings, 1):
            print(f"\n  {rank}. [{category}] (impact: {score:.2f})")
            print(f"     {desc}")

    print(f"\n{'='*60}")
    print("DIAGNOSTIC COMPLETE")
    print(f"{'='*60}")

    return {
        "global_ssim": global_ssim,
        "global_mae": global_mae,
        "patches": patches,
        "edge_metrics": em,
        "color_errors": cerrs,
        "wobble": wobble,
        "aa": aa,
        "thin_features": tf,
        "channels": channels,
        "spatial": spatial,
        "findings": findings,
    }


if __name__ == "__main__":
    import os
    img_path = sys.argv[1] if len(sys.argv) > 1 else "../../Ref.png"
    os.environ.setdefault("DYLD_LIBRARY_PATH", "/opt/homebrew/opt/cairo/lib")
    run_full_diagnostic(img_path)
