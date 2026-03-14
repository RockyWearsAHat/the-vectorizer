"""Diagnose per-cluster boundary contrast.

For each cluster, look at its border pixels and measure the color distance
between the cluster and its neighbors on the other side of the border.
High contrast = edge mediator (AA artifact at sharp boundaries).
Low contrast = gradient participant (real fill or smooth transition).
"""
import cv2, numpy as np
from scipy.ndimage import distance_transform_edt
from app.core.multilevel import detect_background, _compute_edge_weight, _merge_close_clusters

img = cv2.imread("/Users/alexwaldmann/Desktop/SVG-gen/Ref.png")
crop = img[50:460, 486:1050]

h, w = crop.shape[:2]
bg_color, bg_gray = detect_background(crop)

denoised_km = cv2.bilateralFilter(crop, 15, 12, 30)
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

# --- Boundary contrast for each cluster ---
# For each cluster, dilate its mask by 1px, find neighbor cluster IDs,
# compute mean color distance to those neighbors.
print(f"Image: {w}x{h}, K={K}, bg_cluster={bg_cluster}")
print()
print(f"{'idx':>3} {'gray':>4} {'pix%':>5} {'mean_thick':>10} {'n_neighbors':>11} {'mean_bdist':>10} {'max_bdist':>9} {'is_mediator':>11}")

kernel = np.ones((3,3), np.uint8)

for k in range(K):
    if k == bg_cluster:
        continue
    
    mask_k = (labels == k).astype(np.uint8)
    pix_frac = np.count_nonzero(mask_k) / (h * w)
    
    # Mean thickness
    dt = distance_transform_edt(mask_k)
    fg_dts = dt[mask_k > 0]
    mean_thick = float(fg_dts.mean()) if len(fg_dts) > 0 else 0
    
    # Dilate to find border zone
    dilated = cv2.dilate(mask_k, kernel, iterations=1)
    border_zone = (dilated > 0) & (mask_k == 0)
    
    # What clusters are adjacent?
    neighbor_labels = labels[border_zone]
    unique_neighbors = set(int(n) for n in np.unique(neighbor_labels)) - {k}
    
    # Color distance to each neighbor
    neighbor_dists = []
    for n in unique_neighbors:
        d = float(np.linalg.norm(centers_f[k] - centers_f[n]))
        neighbor_dists.append(d)
    
    mean_bdist = np.mean(neighbor_dists) if neighbor_dists else 0
    max_bdist = max(neighbor_dists) if neighbor_dists else 0
    
    # A cluster is an "edge mediator" if:
    # 1. It's thin (mean thickness <= 2)
    # 2. Its neighbors are far apart from EACH OTHER (it bridges a gap)
    # Find max inter-neighbor distance
    neighbor_list = sorted(unique_neighbors)
    max_inter = 0
    for i in range(len(neighbor_list)):
        for j in range(i+1, len(neighbor_list)):
            d = float(np.linalg.norm(centers_f[neighbor_list[i]] - centers_f[neighbor_list[j]]))
            if d > max_inter:
                max_inter = d
    
    is_mediator = mean_thick <= 2.0 and max_inter > 80
    
    color = centers_u[k]
    print(f"{k:3d} {grays[k]:4d} {pix_frac*100:5.1f} {mean_thick:10.1f} "
          f"{len(unique_neighbors):11d} {mean_bdist:10.1f} {max_bdist:9.1f} "
          f"{'YES' if is_mediator else 'no':>11}  "
          f"BGR({color[0]:3d},{color[1]:3d},{color[2]:3d})  "
          f"inter_max={max_inter:.0f}")

print("\n--- What does boundary contrast look like for the primary (dark) cluster? ---")
dk = [k for k in range(K) if k != bg_cluster]
dk_sorted = sorted(dk, key=lambda k: grays[k])
primary = dk_sorted[0]
mask_p = (labels == primary).astype(np.uint8)
dilated_p = cv2.dilate(mask_p, kernel, iterations=1)
border_p = (dilated_p > 0) & (mask_p == 0)
neighbor_ids = labels[border_p]
unique_p = set(int(n) for n in np.unique(neighbor_ids)) - {primary}
print(f"Primary cluster {primary} (gray={grays[primary]}): neighbors = {unique_p}")
for n in unique_p:
    d = float(np.linalg.norm(centers_f[primary] - centers_f[n]))
    print(f"  -> cluster {n} (gray={grays[n]}): dist={d:.1f}")
