"""Deep analysis of the remaining ~2% SSIM gap.

Examines: where errors concentrate, what params contribute most,
and which knobs have the most room to improve.
"""
import sys, os, cv2, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from app.core.comparison import compare
from sweep_crisp import vectorize_custom
from app.core.multilevel import generate_svg, _bgr_to_hex, detect_background
from skimage.metrics import structural_similarity as ssim

ref = cv2.imread("/Users/alexwaldmann/Desktop/SVG-gen/Ref.png")
crop = ref[200:610, 400:964]
mahal = cv2.imread("/tmp/mahal_right.png")

# --- Part 1: Error spatial analysis on crop ---
print("=" * 60)
print("PART 1: Where does the remaining error live? (crop)")
print("=" * 60)

result = vectorize_custom(crop, 1.0, [0.20, 0.50], [0.55, 1.00])
svg = generate_svg(result, remove_background=False)
m = compare(crop, svg)

# Get the diff map
diff = m.diff_map  # grayscale absolute diff
h, w = crop.shape[:2]

# Analyze by region
src_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
dark_mask = src_gray < 80
mid_mask = (src_gray >= 80) & (src_gray < 200)
light_mask = src_gray >= 200

for name, mask in [("dark(<80)", dark_mask), ("mid(80-200)", mid_mask), ("light(>200)", light_mask)]:
    n_pixels = mask.sum()
    pct = 100 * n_pixels / (h * w)
    mean_err = diff[mask].mean() if n_pixels > 0 else 0
    max_err = diff[mask].max() if n_pixels > 0 else 0
    err_contribution = diff[mask].sum() / diff.sum() * 100 if n_pixels > 0 else 0
    print(f"  {name:>14}: {pct:5.1f}% pixels, mean_err={mean_err:.1f}, max={max_err}, contribution={err_contribution:.1f}%")

# Save error map for inspection
cv2.imwrite("/tmp/crop_error_map.png", diff)
print(f"\n  Overall: SSIM={m.ssim_score:.4f}, MAE={m.mae:.2f}")

# --- Part 2: Parameter sensitivity sweep ---
print("\n" + "=" * 60)
print("PART 2: Parameter sensitivity (crop only, faster)")
print("=" * 60)

# Test each param independently
from app.core.multilevel import (
    detect_background, _bgr_to_hex, _merge_close_clusters,
    _fit_contour, _polygon_area, VectorLayer, MultilevelResult,
)
from skimage.measure import find_contours


def vectorize_full(image_bgr, sigma, iso_levels, iso_opacities,
                   bilateral_d=7, bilateral_sc=20, bilateral_ss=20,
                   merge_thresh=80.0, min_area=30, K=24,
                   simplify_eps=0.3, max_err=0.3, corner_thresh=60.0):
    h, w = image_bgr.shape[:2]
    if len(image_bgr.shape) == 2:
        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    bg_color, _ = detect_background(image_bgr)
    bg_hex = _bgr_to_hex(bg_color)

    denoised = cv2.bilateralFilter(image_bgr, bilateral_d, bilateral_sc, bilateral_ss)
    pixels = denoised.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(pixels, K, None, criteria, 6, cv2.KMEANS_PP_CENTERS)
    centers, labels = _merge_close_clusters(centers, labels.flatten(), h, w, threshold=merge_thresh)
    Keff = len(centers)
    centers_u = centers.astype(np.uint8)
    centers_f = centers.astype(np.float32)

    pixels_3d = denoised.astype(np.float32)
    dist_map = np.empty((h, w, Keff), dtype=np.float32)
    for k in range(Keff):
        diff = pixels_3d - centers_f[k]
        dist_map[:, :, k] = np.sqrt(np.sum(diff * diff, axis=2))

    grays = np.array([
        int(cv2.cvtColor(c.reshape(1,1,3), cv2.COLOR_BGR2GRAY)[0,0])
        for c in centers_u
    ])
    order = np.argsort(-grays)

    layers = []
    total_paths = 0
    total_nodes = 0

    for cluster_idx in order:
        color_hex = _bgr_to_hex(centers_u[cluster_idx])
        d_k = dist_map[:, :, cluster_idx]
        other_mask = np.ones(Keff, dtype=bool)
        other_mask[cluster_idx] = False
        d_other = np.min(dist_map[:, :, other_mask], axis=2)
        denom = d_k + d_other
        denom = np.where(denom < 1e-10, 1e-10, denom)
        soft = d_other / denom
        soft = cv2.GaussianBlur(soft, (0, 0), sigmaX=sigma)

        layer_paths = []
        layer_opacities = []
        for iso, opacity in zip(iso_levels, iso_opacities):
            contour_list = find_contours(soft, iso)
            iso_parts = []
            for contour in contour_list:
                if len(contour) < 4:
                    continue
                xy = contour[:, ::-1].astype(np.float64)
                area = abs(_polygon_area(xy))
                if area < min_area:
                    continue
                d = _fit_contour(xy, simplify_eps, max_err, corner_thresh)
                if d:
                    iso_parts.append(d)
            if iso_parts:
                combined = " ".join(iso_parts)
                layer_paths.append(combined)
                layer_opacities.append(opacity)
                total_paths += 1
                total_nodes += combined.count("C") + combined.count("M")

        if layer_paths:
            layers.append(VectorLayer(paths=layer_paths, opacities=layer_opacities, color=color_hex))

    return MultilevelResult(layers=layers, width=w, height=h, background_color=bg_hex,
                            path_count=total_paths, node_count=total_nodes), Keff


def test_config(img, label, **kw):
    sigma = kw.pop("sigma", 1.0)
    iso_l = kw.pop("iso_levels", [0.20, 0.50])
    iso_o = kw.pop("iso_opacities", [0.55, 1.00])
    result, keff = vectorize_full(img, sigma, iso_l, iso_o, **kw)
    svg = generate_svg(result, remove_background=False)
    m = compare(img, svg)
    return m.ssim_score, m.mae, keff, result.path_count

# Baseline
s, mae, keff, paths = test_config(crop, "baseline")
print(f"  {'baseline':<28} SSIM={s:.4f} MAE={mae:.2f} K={keff} paths={paths}")

# More clusters (lower merge threshold)
for mt in [60, 50, 40, 35]:
    s, mae, keff, paths = test_config(crop, f"merge_thresh={mt}", merge_thresh=mt)
    print(f"  {'merge_thresh='+str(mt):<28} SSIM={s:.4f} MAE={mae:.2f} K={keff} paths={paths}")

# More initial K
for k in [32, 48, 64]:
    s, mae, keff, paths = test_config(crop, f"K={k}", K=k)
    print(f"  {'K='+str(k):<28} SSIM={s:.4f} MAE={mae:.2f} K={keff} paths={paths}")

# Less bilateral (preserve more tones)
for sc in [15, 10, 5]:
    s, mae, keff, paths = test_config(crop, f"bilateral_sc={sc}", bilateral_sc=sc)
    print(f"  {'bilateral_sc='+str(sc):<28} SSIM={s:.4f} MAE={mae:.2f} K={keff} paths={paths}")

# Finer curve fitting
for me in [0.2, 0.15, 0.1]:
    s, mae, keff, paths = test_config(crop, f"max_err={me}", max_err=me)
    print(f"  {'max_err='+str(me):<28} SSIM={s:.4f} MAE={mae:.2f} K={keff} paths={paths}")

# Smaller min contour area
for ma in [20, 15, 10, 5]:
    s, mae, keff, paths = test_config(crop, f"min_area={ma}", min_area=ma)
    print(f"  {'min_area='+str(ma):<28} SSIM={s:.4f} MAE={mae:.2f} K={keff} paths={paths}")

# Simplify epsilon
for se in [0.2, 0.15, 0.1]:
    s, mae, keff, paths = test_config(crop, f"simplify_eps={se}", simplify_eps=se)
    print(f"  {'simplify_eps='+str(se):<28} SSIM={s:.4f} MAE={mae:.2f} K={keff} paths={paths}")

# --- Part 3: Best combo ---
print("\n" + "=" * 60)
print("PART 3: Combined best params")
print("=" * 60)

combos = [
    ("A: mt60+area15",          dict(merge_thresh=60, min_area=15)),
    ("B: mt50+area15",          dict(merge_thresh=50, min_area=15)),
    ("C: mt60+area15+me0.2",    dict(merge_thresh=60, min_area=15, max_err=0.2)),
    ("D: mt50+area10+me0.2",    dict(merge_thresh=50, min_area=10, max_err=0.2)),
    ("E: mt60+bilat10+area15",  dict(merge_thresh=60, bilateral_sc=10, min_area=15)),
    ("F: mt50+bilat10+area10",  dict(merge_thresh=50, bilateral_sc=10, min_area=10)),
    ("G: K48+mt50+area10",      dict(K=48, merge_thresh=50, min_area=10)),
    ("H: K48+mt40+area10+me0.2",dict(K=48, merge_thresh=40, min_area=10, max_err=0.2)),
]

for label, kw in combos:
    results = []
    for img_name, img in [("crop", crop), ("ref", ref), ("mahal", mahal)]:
        s, mae, keff, paths = test_config(img, label, **kw.copy())
        results.append((s, mae, keff, paths))
    avg_ssim = np.mean([r[0] for r in results])
    print(f"  {label:<32} crop={results[0][0]:.4f} ref={results[1][0]:.4f} mahal={results[2][0]:.4f} avg={avg_ssim:.4f} K={results[0][2]}/{results[1][2]}/{results[2][2]}")
