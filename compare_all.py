"""Batch vectorize all test images and produce side-by-side comparison PNGs.

Generates:
  _comparisons/{name}_comparison.png  — original | SVG render | error map
  _comparisons/{name}_output.svg      — the SVG
  _comparisons/{name}_metrics.txt     — structural metrics
  _comparisons/summary.txt            — aggregate table
"""
import sys, os, time, glob, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "raster-to-vector", "server"))
import cv2
import numpy as np
import cairosvg
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from app.core.multilevel import multilevel_vectorize, generate_svg, optimize_svg_colors

parser = argparse.ArgumentParser(description="Batch vectorize test images and produce comparisons.")
group = parser.add_mutually_exclusive_group()
group.add_argument("--fast", action="store_true", default=True,
                   help="(default) Skip test1.jpg and optimize_svg_colors for speed")
group.add_argument("--full", action="store_true",
                   help="Run full suite: all images + color optimization")
args = parser.parse_args()

if args.full:
    args.fast = False

FAST_MODE = args.fast

OUT_DIR = "_comparisons"
os.makedirs(OUT_DIR, exist_ok=True)

# Collect test images
IMAGE_FILES = sorted(glob.glob("Ref.png") + glob.glob("test[0-9]*.jpg") + glob.glob("test[0-9]*.png"))
if not IMAGE_FILES:
    print("No test images found (Ref.png, test1.jpg, etc.)")
    sys.exit(1)

if FAST_MODE:
    skipped = [f for f in IMAGE_FILES if os.path.basename(f).startswith("test1")]
    IMAGE_FILES = [f for f in IMAGE_FILES if not os.path.basename(f).startswith("test1")]
    print(f"FAST MODE active:")
    print(f"  - Skipping: {', '.join(skipped) if skipped else '(none)'}")
    print(f"  - Skipping optimize_svg_colors (saves ~10-15 min)")

print(f"Found {len(IMAGE_FILES)} images: {', '.join(IMAGE_FILES)}")


def structural_metrics(ref_gray, svg_gray, dark_thresh=None):
    """Compute structural comparison metrics between ref and SVG grayscale images.

    Auto-detects dark threshold from the image content.
    """
    ref_g = ref_gray.astype(float)
    svg_g = svg_gray.astype(float)

    # Auto dark threshold: midpoint between darkest and lightest cluster
    if dark_thresh is None:
        # Find the two dominant peaks (dark features vs background)
        hist, bins = np.histogram(ref_g.ravel(), bins=256, range=(0, 256))
        # Smooth histogram
        from scipy.ndimage import gaussian_filter1d
        hist_s = gaussian_filter1d(hist.astype(float), sigma=5)
        # Find peaks
        peaks = []
        for i in range(1, 254):
            if hist_s[i] > hist_s[i-1] and hist_s[i] > hist_s[i+1] and hist_s[i] > hist_s.max() * 0.01:
                peaks.append(i)
        if len(peaks) >= 2:
            # Use midpoint between first and last major peak
            dark_thresh = (peaks[0] + peaks[-1]) / 2
        else:
            dark_thresh = 128

    ref_bin = (ref_g < dark_thresh).astype(np.uint8)
    svg_bin = (svg_g < dark_thresh).astype(np.uint8)

    ref_dark = int(ref_bin.sum())
    svg_dark = int(svg_bin.sum())

    # Missing and extra
    missing = int(((ref_bin == 1) & (svg_bin == 0)).sum())
    extra = int(((ref_bin == 0) & (svg_bin == 1)).sum())

    # Skeleton-based width analysis
    metrics = {
        "dark_threshold": dark_thresh,
        "ref_dark_px": ref_dark,
        "svg_dark_px": svg_dark,
        "missing_px": missing,
        "extra_px": extra,
        "total_error_px": missing + extra,
        "missing_pct": 100 * missing / max(ref_dark, 1),
        "extra_pct": 100 * extra / max(ref_dark, 1),
    }

    # Skeleton analysis — only if there are enough dark features
    if ref_dark > 100:
        try:
            ref_skel = skeletonize(ref_bin > 0)
            ref_dt = distance_transform_edt(ref_bin)
            svg_dt = distance_transform_edt(svg_bin)

            skel_pts = ref_skel.sum()
            if skel_pts > 0:
                ref_widths = ref_dt[ref_skel] * 2.0
                svg_widths_at_ref = svg_dt[ref_skel] * 2.0
                svg_dark_at_skel = int((svg_g[ref_skel] < dark_thresh).sum())

                metrics["skeleton_length"] = int(skel_pts)
                metrics["feature_presence_pct"] = 100 * svg_dark_at_skel / skel_pts
                metrics["mean_width_ref"] = float(ref_widths.mean())
                metrics["mean_width_svg"] = float(svg_widths_at_ref.mean())
                metrics["width_diff_mean"] = float((svg_widths_at_ref - ref_widths).mean())
                metrics["width_diff_median"] = float(np.median(svg_widths_at_ref - ref_widths))
        except Exception as e:
            metrics["skeleton_error"] = str(e)

    # Pixel-level diff stats
    diff = np.abs(ref_g - svg_g)
    metrics["mean_pixel_diff"] = float(diff.mean())
    metrics["diff_gt10"] = int((diff > 10).sum())
    metrics["diff_gt20"] = int((diff > 20).sum())
    metrics["diff_gt30"] = int((diff > 30).sum())

    return metrics


def make_comparison_image(ref_bgr, svg_bgr, name):
    """Create a side-by-side comparison: original | SVG | error map."""
    h, w = ref_bgr.shape[:2]

    ref_g = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY).astype(float)
    svg_g = cv2.cvtColor(svg_bgr, cv2.COLOR_BGR2GRAY).astype(float)

    # Error map: scale difference to visible range
    diff = np.abs(ref_g - svg_g)
    # Red channel = error intensity, green = 0, blue shows original
    error_map = np.zeros((h, w, 3), dtype=np.uint8)
    error_scaled = np.clip(diff * 4, 0, 255).astype(np.uint8)  # 4x amplification
    error_map[:, :, 2] = error_scaled  # red in BGR
    error_map[:, :, 0] = np.clip(255 - diff * 2, 0, 255).astype(np.uint8)  # blue background

    # Labels
    label_h = 40
    panels = []
    for img, label in [(ref_bgr, f"{name} (original)"), (svg_bgr, "SVG render"), (error_map, "Error map (4x)")]:
        panel = np.zeros((h + label_h, w, 3), dtype=np.uint8)
        panel[:label_h] = 30  # dark gray bar
        panel[label_h:] = img
        # Put text
        cv2.putText(panel, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        panels.append(panel)

    # Stack horizontally with thin divider
    divider = np.ones((h + label_h, 3, 3), dtype=np.uint8) * 128
    combined = np.hstack([panels[0], divider, panels[1], divider, panels[2]])
    return combined


summary_lines = []
summary_lines.append(f"{'Image':<20s} {'Feat%':>6s} {'Miss%':>6s} {'Xtra%':>6s} {'WdErr':>6s} {'MnDif':>6s} {'Time':>5s} {'Nodes':>7s} {'SVG_KB':>7s}")
summary_lines.append("-" * 90)

for img_path in IMAGE_FILES:
    name = os.path.splitext(os.path.basename(img_path))[0]
    print(f"\n{'='*60}")
    print(f"Processing: {img_path}")

    ref = cv2.imread(img_path)
    if ref is None:
        print(f"  ERROR: Could not read {img_path}")
        continue

    h, w = ref.shape[:2]

    # Full-resolution processing — no downscaling.  Pipeline optimizations
    # (vectorised Bézier fitting, O(n) direct fitter, adaptive
    # superresolution budget) keep processing tractable at any size.

    t0 = time.time()
    r = multilevel_vectorize(ref, mediator_threshold=0.3)
    svg = generate_svg(r, remove_background=False)
    if not FAST_MODE:
        svg = optimize_svg_colors(svg, ref)
    elapsed = time.time() - t0

    # Save SVG
    svg_path = os.path.join(OUT_DIR, f"{name}_output.svg")
    with open(svg_path, "w") as f:
        f.write(svg)
    svg_kb = len(svg) / 1024

    # Render SVG back to image for comparison
    png = cairosvg.svg2png(bytestring=svg.encode(), output_width=w, output_height=h)
    svg_img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)

    ref_g = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    svg_g = cv2.cvtColor(svg_img, cv2.COLOR_BGR2GRAY)

    # Metrics
    m = structural_metrics(ref_g, svg_g)

    # Comparison image
    comp = make_comparison_image(ref, svg_img, name)
    comp_path = os.path.join(OUT_DIR, f"{name}_comparison.png")
    cv2.imwrite(comp_path, comp)

    # Metrics file
    met_path = os.path.join(OUT_DIR, f"{name}_metrics.txt")
    with open(met_path, "w") as f:
        f.write(f"Image: {img_path} ({w}x{h})\n")
        f.write(f"Time: {elapsed:.1f}s\n")
        f.write(f"SVG: {svg_kb:.0f} KB, {r.path_count} paths, {r.node_count:,} nodes\n")
        f.write(f"\n--- Structural Metrics ---\n")
        for k, v in m.items():
            f.write(f"  {k}: {v}\n")

    feat_pct = m.get("feature_presence_pct", -1)
    miss_pct = m.get("missing_pct", -1)
    xtra_pct = m.get("extra_pct", -1)
    wd_err = m.get("width_diff_mean", 0)
    mn_diff = m.get("mean_pixel_diff", 0)

    print(f"  {elapsed:.1f}s | {r.node_count:,} nodes | {svg_kb:.0f}KB | img {IMAGE_FILES.index(img_path)+1}/{len(IMAGE_FILES)}")
    print(f"  Feature presence: {feat_pct:.1f}% | Missing: {miss_pct:.1f}% | Extra: {xtra_pct:.1f}%")
    print(f"  Width error: {wd_err:+.2f}px | Mean pixel diff: {mn_diff:.2f}")

    summary_lines.append(f"{name:<20s} {feat_pct:6.1f} {miss_pct:6.1f} {xtra_pct:6.1f} {wd_err:+6.2f} {mn_diff:6.2f} {elapsed:5.1f} {r.node_count:7,} {svg_kb:7.0f}")

# Write summary
summary = "\n".join(summary_lines)
print(f"\n{'='*60}")
print("SUMMARY")
print(summary)

summary_path = os.path.join(OUT_DIR, "summary.txt")
with open(summary_path, "w") as f:
    f.write(summary + "\n")

print(f"\nResults saved to {OUT_DIR}/")
