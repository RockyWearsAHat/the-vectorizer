"""Sweep merge threshold + halo params to reduce ghosting."""
import cv2, numpy as np, time, io, sys
from app.core.multilevel import (
    multilevel_vectorize, generate_svg, _merge_close_clusters,
    detect_background,
)
from app.core.comparison import compare
from skimage.metrics import structural_similarity as ssim
import cairosvg
from PIL import Image


def measure(img, label):
    """Run vectorize + measure quality, return dict."""
    t0 = time.time()
    result = multilevel_vectorize(img, num_levels=24)
    svg = generate_svg(result, remove_background=False)
    dt = time.time() - t0
    comp = compare(img, svg)
    png = cairosvg.svg2png(
        bytestring=svg.encode(),
        output_width=img.shape[1], output_height=img.shape[0],
    )
    svg_arr = np.array(Image.open(io.BytesIO(png)).convert("RGB"))
    src_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    raw = ssim(src_rgb, svg_arr, channel_axis=2)
    return {
        "blur": comp.ssim_score, "raw": raw, "mae": comp.mae,
        "paths": result.path_count, "nodes": result.node_count,
        "kb": len(svg) // 1024, "time": dt,
    }


# Load images
ref = cv2.imread("/Users/alexwaldmann/Desktop/SVG-gen/Ref.png")
crop = ref[50:460, 486:1050]
mahal = cv2.imread("/tmp/mahal_right.png")

# Baseline (current production = threshold 60, halo 0.22/0.45, min_area 3)
print("=== Baseline (current) ===")
r = measure(crop, "crop")
print(f"  crop:  blur={r['blur']:.4f} raw={r['raw']:.4f} mae={r['mae']:.2f} paths={r['paths']} nodes={r['nodes']}")
sys.stdout.flush()
r2 = measure(mahal, "mahal")
print(f"  mahal: blur={r2['blur']:.4f} raw={r2['raw']:.4f} mae={r2['mae']:.2f} paths={r2['paths']} nodes={r2['nodes']}")
sys.stdout.flush()

# Now test variations by monkey-patching the module
import app.core.multilevel as ml

orig_func = ml.multilevel_vectorize

def test_config(name, merge_thresh, halo_iso, halo_op, min_area):
    """Patch and test a config."""
    # We need to modify the source... use a wrapper approach
    # Save originals
    old_code = ml.multilevel_vectorize.__code__
    # Can't easily patch constants. Let's just modify the module-level
    # defaults by re-reading and exec'ing with different constants.
    # Simpler: write a wrapper that modifies the relevant parameters.
    pass

# Actually, let's just test a few key merge thresholds by calling
# _merge_close_clusters directly and seeing cluster counts.
print("\n=== Merge threshold sweep ===")
denoised_km = cv2.bilateralFilter(crop, 15, 12, 30)
pixels = denoised_km.reshape(-1, 3).astype(np.float32)
h, w = crop.shape[:2]
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
_, labels, centers = cv2.kmeans(pixels, 24, None, criteria, 6, cv2.KMEANS_PP_CENTERS)

for thresh in [60, 70, 80, 90, 100, 120]:
    c, l = _merge_close_clusters(centers, labels.flatten(), h, w, threshold=float(thresh))
    bg_color, _ = detect_background(crop)
    bg_dists = np.array([np.linalg.norm(c[k] - bg_color.astype(np.float32)) for k in range(len(c))])
    bg_idx = int(np.argmin(bg_dists))
    non_bg = len(c) - 1
    grays = [int(cv2.cvtColor(c[k:k+1].astype(np.uint8).reshape(1,1,3), cv2.COLOR_BGR2GRAY)[0,0]) for k in range(len(c))]
    print(f"  threshold={thresh}: K={len(c)} non-bg={non_bg} grays={sorted(grays)}")
