"""Diagnose thin-feature behavior in the pipeline.

For each cluster, measure:
 - pixel fraction
 - mean local thickness (via distance transform)
 - how wide the soft-field blob actually becomes after smoothing
This tells us exactly which clusters are thin lines and how much
the soft field spreads them.
"""
import cv2, numpy as np
from scipy.ndimage import distance_transform_edt
from app.core.multilevel import detect_background, _compute_edge_weight, _merge_close_clusters

img = cv2.imread("/Users/alexwaldmann/Desktop/SVG-gen/Ref.png")
crop = img[50:460, 486:1050]

h, w = crop.shape[:2]
bg_color, bg_gray = detect_background(crop)

denoised_km = cv2.bilateralFilter(crop, 15, 12, 30)
denoised_dist = cv2.bilateralFilter(crop, 7, 5, 20)

pixels = denoised_km.reshape(-1, 3).astype(np.float32)
K = 24
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
_, labels, centers = cv2.kmeans(pixels, K, None, criteria, 6, cv2.KMEANS_PP_CENTERS)

centers, labels = _merge_close_clusters(centers, labels.flatten(), h, w, threshold=60.0)
K = len(centers)
centers_u = centers.astype(np.uint8)
centers_f = centers.astype(np.float32)

bg_dists = np.array([np.linalg.norm(centers_f[k] - bg_color.astype(np.float32)) for k in range(K)])
bg_cluster_idx = int(np.argmin(bg_dists))
bg_cluster = bg_cluster_idx if bg_dists[bg_cluster_idx] < 40.0 else -1

grays = np.array([int(cv2.cvtColor(c.reshape(1,1,3), cv2.COLOR_BGR2GRAY)[0,0]) for c in centers_u])
non_bg_grays = grays[[i for i in range(K) if i != bg_cluster]]
darkest_gray = int(non_bg_grays.min()) if len(non_bg_grays) else 0
bg_gray_val = int(grays[bg_cluster]) if bg_cluster >= 0 else 255
gray_span = max(bg_gray_val - darkest_gray, 1)

print(f"Image: {w}x{h}, K={K}, bg_cluster={bg_cluster}")
print(f"gray range: {darkest_gray}-{bg_gray_val}, span={gray_span}")
print()
print(f"{'idx':>3} {'gray':>4} {'light':>5} {'pix%':>5} {'mean_thick':>10} {'max_thick':>9} {'color_bgr':>20}")

for k in range(K):
    if k == bg_cluster:
        continue
    mask = (labels == k)
    pix_frac = np.count_nonzero(mask) / (h * w)
    lightness = (grays[k] - darkest_gray) / gray_span

    # Distance transform: for each foreground pixel, distance to nearest bg
    dt = distance_transform_edt(mask)
    fg_dts = dt[mask]
    if len(fg_dts) > 0:
        mean_thick = float(fg_dts.mean())
        max_thick = float(fg_dts.max())
    else:
        mean_thick = max_thick = 0

    color = centers_u[k]
    print(f"{k:3d} {grays[k]:4d} {lightness:5.2f} {pix_frac*100:5.1f} "
          f"{mean_thick:10.1f} {max_thick:9.1f}   BGR({color[0]:3d},{color[1]:3d},{color[2]:3d})")

# Also measure soft-field spread for the darkest cluster
print("\n--- Soft field spread analysis (darkest non-bg cluster) ---")
darkest_k = np.argmin(grays[[i for i in range(K) if i != bg_cluster]])
# Map back to original index
non_bg = [i for i in range(K) if i != bg_cluster]
dk = non_bg[darkest_k]

pixels_3d = denoised_dist.astype(np.float32)
dist_map = np.empty((h, w, K), dtype=np.float32)
for k in range(K):
    diff = pixels_3d - centers_f[k]
    dist_map[:, :, k] = np.sqrt(np.sum(diff * diff, axis=2))

d_k = dist_map[:, :, dk]
other_mask = np.ones(K, dtype=bool)
other_mask[dk] = False
d_other = np.min(dist_map[:, :, other_mask], axis=2)
denom = d_k + d_other
denom = np.where(denom < 1e-10, 1e-10, denom)
soft_raw = d_other / denom

# Where the cluster's binary mask is 1px wide, how wide is soft > 0.1?
binary_mask = (labels == dk)
dt_mask = distance_transform_edt(binary_mask)

# Find thin segments (thickness <= 2)
thin_mask = binary_mask & (dt_mask <= 2.0)
thin_ys, thin_xs = np.where(thin_mask)

if len(thin_ys) > 0:
    # Sample some thin points and measure soft field width around them
    np.random.seed(42)
    sample_n = min(50, len(thin_ys))
    sample_idx = np.random.choice(len(thin_ys), sample_n, replace=False)
    
    widths = []
    for si in sample_idx:
        y, x = thin_ys[si], thin_xs[si]
        # Measure: in a 21px window centered on this pixel,
        # how many pixels have soft_raw > 0.1?
        r = 10
        y0, y1 = max(0, y-r), min(h, y+r+1)
        x0, x1 = max(0, x-r), min(w, x+r+1)
        patch = soft_raw[y0:y1, x0:x1]
        spread = np.count_nonzero(patch > 0.10)
        widths.append(spread)
    
    print(f"Darkest cluster {dk}: gray={grays[dk]}, thin pixels={np.count_nonzero(thin_mask)}")
    print(f"At thin points, soft>0.10 covers avg {np.mean(widths):.0f} / {(2*r+1)**2} pixels in 21x21 window")
    print(f"  min={np.min(widths)} max={np.max(widths)} median={np.median(widths):.0f}")
    
    # After smoothing
    S = 4
    soft_up = cv2.resize(soft_raw, (w*S, h*S), interpolation=cv2.INTER_LINEAR)
    soft_crisp = cv2.GaussianBlur(soft_up, (0,0), sigmaX=0.6*S)
    soft_smooth = cv2.GaussianBlur(soft_up, (0,0), sigmaX=1.5*S)
    ew = _compute_edge_weight(crop)
    ew_up = cv2.resize(ew, (w*S, h*S), interpolation=cv2.INTER_LINEAR)
    soft = ew_up * soft_crisp + (1.0 - ew_up) * soft_smooth
    
    widths_smoothed = []
    for si in sample_idx:
        y, x = thin_ys[si]*S, thin_xs[si]*S
        r = 10*S
        y0, y1 = max(0, y-r), min(h*S, y+r+1)
        x0, x1 = max(0, x-r), min(w*S, x+r+1)
        patch = soft[y0:y1, x0:x1]
        # How many 1x-equivalent pixels have soft > 0.22 (halo threshold)?
        count_halo = np.count_nonzero(patch > 0.22) / (S*S)
        widths_smoothed.append(count_halo)
    
    print(f"\nAfter 4x upscale + dual-sigma smoothing:")
    print(f"  soft>0.22 (halo iso) covers avg {np.mean(widths_smoothed):.1f} equiv-pixels in window")
    print(f"  min={np.min(widths_smoothed):.1f} max={np.max(widths_smoothed):.1f} median={np.median(widths_smoothed):.1f}")
else:
    print("No thin segments found for darkest cluster")
