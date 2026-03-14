"""Boundary contrast v2: detect if a cluster's color is an interpolation
between its neighbors (it sits ON the line segment between two neighbor
colors in BGR space) vs. being a distinct endpoint color.

Also: measure what fraction of the cluster's area is "interior" (>2px
from any other cluster) - real content has interiors, AA mediators don't.
"""
import cv2, numpy as np
from scipy.ndimage import distance_transform_edt
from app.core.multilevel import detect_background, _merge_close_clusters

img = cv2.imread("/Users/alexwaldmann/Desktop/SVG-gen/Ref.png")
crop = img[50:460, 486:1050]

for name, src in [("crop", crop), ("mahal", cv2.imread("/tmp/mahal_right.png"))]:
    if src is None:
        continue
    print(f"\n{'='*80}")
    print(f"  {name}: {src.shape[1]}x{src.shape[0]}")
    print(f"{'='*80}")
    _run(src)

h, w = crop.shape[:2]
bg_color, _ = detect_background(crop)

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

def point_to_segment_distance(p, a, b):
    """Distance from point p to line segment a-b in N-d space."""
    ab = b - a
    ab_len_sq = np.dot(ab, ab)
    if ab_len_sq < 1e-10:
        return np.linalg.norm(p - a)
    t = np.clip(np.dot(p - a, ab) / ab_len_sq, 0.0, 1.0)
    proj = a + t * ab
    return np.linalg.norm(p - proj)

kernel = np.ones((3,3), np.uint8)

def _run(image_bgr):
    h, w = image_bgr.shape[:2]
    bg_color, _ = detect_background(image_bgr)
    if k == bg_cluster:
        continue
    
    mask_k = (labels == k).astype(np.uint8)
    pix_count = np.count_nonzero(mask_k)
    pix_frac = pix_count / (h * w)
    
    # Mean thickness via distance transform
    dt = distance_transform_edt(mask_k)
    fg_dts = dt[mask_k > 0]
    mean_thick = float(fg_dts.mean()) if len(fg_dts) > 0 else 0
    
    # Interior fraction: pixels > 2px from any border
    interior_count = np.count_nonzero(fg_dts > 2.0) if len(fg_dts) > 0 else 0
    interior_frac = interior_count / max(pix_count, 1)
    
    # Find neighbors
    dilated = cv2.dilate(mask_k, kernel, iterations=1)
    border_zone = (dilated > 0) & (mask_k == 0)
    neighbor_labels = labels[border_zone]
    unique_neighbors = sorted(set(int(n) for n in np.unique(neighbor_labels)) - {k})
    
    # Check if this cluster's color is an interpolation between any
    # pair of its neighbors (it sits on the line segment between them)
    c = centers_f[k]
    min_interp_dist = float('inf')
    nearest_pair = (-1, -1)
    for i in range(len(unique_neighbors)):
        for j in range(i+1, len(unique_neighbors)):
            ni, nj = unique_neighbors[i], unique_neighbors[j]
            d = point_to_segment_distance(c, centers_f[ni], centers_f[nj])
            if d < min_interp_dist:
                min_interp_dist = d
                nearest_pair = (ni, nj)
    
    if len(unique_neighbors) < 2:
        min_interp_dist = 999  # Can't be interpolation with < 2 neighbors
    
    # Classification:
    # - "edge mediator" if thin AND color is close to interpolation
    #   between two neighbors (< 20 in BGR space)
    # - "real fill" otherwise
    is_mediator = mean_thick <= 2.0 and min_interp_dist < 20.0 and interior_frac < 0.15
    role = "MEDIATOR" if is_mediator else "FILL"
    
    color = centers_u[k]
    seg_desc = f"({nearest_pair[0]},{nearest_pair[1]})" if nearest_pair[0] >= 0 else "n/a"
    print(f"{k:3d} {grays[k]:4d} {pix_frac*100:5.1f} {mean_thick:5.1f} "
          f"{interior_frac*100:8.1f}% {min_interp_dist:11.1f} "
          f"{seg_desc:>20} {role:>12}  "
          f"BGR({color[0]:3d},{color[1]:3d},{color[2]:3d})")
