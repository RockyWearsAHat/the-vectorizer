"""Boundary contrast v2: interpolation + interior detection for crop and mahal."""
import cv2, numpy as np
from scipy.ndimage import distance_transform_edt
from app.core.multilevel import detect_background, _merge_close_clusters

def point_to_segment_distance(p, a, b):
    ab = b - a
    ab_len_sq = np.dot(ab, ab)
    if ab_len_sq < 1e-10:
        return np.linalg.norm(p - a)
    t = np.clip(np.dot(p - a, ab) / ab_len_sq, 0.0, 1.0)
    return np.linalg.norm(p - (a + t * ab))

kernel = np.ones((3,3), np.uint8)

def analyze(image_bgr, name):
    h, w = image_bgr.shape[:2]
    bg_color, _ = detect_background(image_bgr)
    denoised_km = cv2.bilateralFilter(image_bgr, 15, 12, 30)
    pixels = denoised_km.reshape(-1, 3).astype(np.float32)
    K = 24
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(pixels, K, None, criteria, 6, cv2.KMEANS_PP_CENTERS)
    centers, labels = _merge_close_clusters(centers, labels.flatten(), h, w, threshold=60.0)
    K = len(centers)
    centers_u = centers.astype(np.uint8)
    centers_f = centers.astype(np.float32)

    bg_dists = np.array([np.linalg.norm(centers_f[k] - bg_color.astype(np.float32)) for k in range(K)])
    bg_idx = int(np.argmin(bg_dists))
    bg_cluster = bg_idx if bg_dists[bg_idx] < 40.0 else -1

    grays = np.array([int(cv2.cvtColor(c.reshape(1,1,3), cv2.COLOR_BGR2GRAY)[0,0]) for c in centers_u])

    print(f"\n{'='*90}")
    print(f"  {name}: {w}x{h}, K={K}, bg_cluster={bg_cluster}")
    print(f"{'='*90}")
    print(f"{'idx':>3} {'gray':>4} {'pix%':>5} {'thick':>5} {'int%':>5} "
          f"{'interp_d':>8} {'seg':>8} {'role':>10}  color")

    for k in range(K):
        if k == bg_cluster:
            continue
        mask_k = (labels == k).astype(np.uint8)
        pix_count = np.count_nonzero(mask_k)
        pix_frac = pix_count / (h * w)

        dt = distance_transform_edt(mask_k)
        fg_dts = dt[mask_k > 0]
        mean_thick = float(fg_dts.mean()) if len(fg_dts) > 0 else 0
        interior_count = np.count_nonzero(fg_dts > 2.0) if len(fg_dts) > 0 else 0
        interior_frac = interior_count / max(pix_count, 1)

        dilated = cv2.dilate(mask_k, kernel, iterations=1)
        border_zone = (dilated > 0) & (mask_k == 0)
        unique_neighbors = sorted(set(int(n) for n in np.unique(labels[border_zone])) - {k})

        c = centers_f[k]
        min_interp = float('inf')
        nearest_pair = (-1, -1)
        for i in range(len(unique_neighbors)):
            for j in range(i+1, len(unique_neighbors)):
                ni, nj = unique_neighbors[i], unique_neighbors[j]
                d = point_to_segment_distance(c, centers_f[ni], centers_f[nj])
                if d < min_interp:
                    min_interp = d
                    nearest_pair = (ni, nj)
        if len(unique_neighbors) < 2:
            min_interp = 999

        is_mediator = mean_thick <= 2.0 and min_interp < 20.0 and interior_frac < 0.15
        role = "MEDIATOR" if is_mediator else "FILL"
        seg = f"({nearest_pair[0]},{nearest_pair[1]})" if nearest_pair[0] >= 0 else "n/a"
        b, g, r = centers_u[k]
        print(f"{k:3d} {grays[k]:4d} {pix_frac*100:5.1f} {mean_thick:5.1f} "
              f"{interior_frac*100:5.1f} {min_interp:8.1f} {seg:>8} {role:>10}  "
              f"({r:3d},{g:3d},{b:3d})")

img = cv2.imread("/Users/alexwaldmann/Desktop/SVG-gen/Ref.png")
crop = img[50:460, 486:1050]
analyze(crop, "crop")

mahal = cv2.imread("/tmp/mahal_right.png")
if mahal is not None:
    analyze(mahal, "mahal")
