"""Sweep v3: refine the dual-denoise approach around the winner."""
import cv2
import numpy as np
import time
from sweep_denoise2 import run_pipeline_denoise_both, images

configs = [
    # (name, km_d, km_sc, km_ss, dist_d, dist_sc, dist_ss)
    # Baseline winner
    ("WINNER: 15/12/30 + 7/5/20", 15, 12, 30, 7, 5, 20),
    # Tighter distance color gate
    ("15/12/30 + 7/3/20", 15, 12, 30, 7, 3, 20),
    ("15/12/30 + 7/4/15", 15, 12, 30, 7, 4, 15),
    # Wider distance spatial with tight color
    ("15/12/30 + 11/5/25", 15, 12, 30, 11, 5, 25),
    ("15/12/30 + 15/5/30", 15, 12, 30, 15, 5, 30),
    # Stronger K-means denoise
    ("21/15/40 + 7/5/20", 21, 15, 40, 7, 5, 20),
    ("21/12/40 + 7/5/20", 21, 12, 40, 7, 5, 20),
    # Lighter K-means, same distance
    ("9/10/20 + 7/5/20", 9, 10, 20, 7, 5, 20),
    # Same denoise for both (just mild color-gated)
    ("7/5/20 + 7/5/20 (same)", 7, 5, 20, 7, 5, 20),
    # Medium K-means, slightly wider distance
    ("11/10/25 + 9/5/20", 11, 10, 25, 9, 5, 20),
    # Distance with larger kernel, very tight
    ("15/12/30 + 15/4/30", 15, 12, 30, 15, 4, 30),
    # sc=6 for distance (slightly more tolerant but still tight)
    ("15/12/30 + 7/6/20", 15, 12, 30, 7, 6, 20),
    ("15/12/30 + 9/6/25", 15, 12, 30, 9, 6, 25),
    # Original for distance (sc=0 = no blur)
    ("15/12/30 + ORIG", 15, 12, 30, 0, 0, 0),
]

print(f"{'Config':>35s}  {'crop':>6s}  {'mahal':>6s}  {'avg':>6s}  {'time':>5s}")
print("-" * 65)

for entry in configs:
    name = entry[0]
    km_d, km_sc, km_ss = entry[1], entry[2], entry[3]
    dt_d, dt_sc, dt_ss = entry[4], entry[5], entry[6]

    ssims = {}
    t0 = time.time()
    for img_name, img in images.items():
        if img is None:
            continue
        dn_km = cv2.bilateralFilter(img, km_d, km_sc, km_ss)
        if dt_d == 0:
            dn_dist = img  # raw original
        else:
            dn_dist = cv2.bilateralFilter(img, dt_d, dt_sc, dt_ss)
        ssim, _, _, _ = run_pipeline_denoise_both(img, dn_km, dn_dist)
        ssims[img_name] = ssim

    dt = time.time() - t0
    avg = np.mean(list(ssims.values()))
    print(f"  {name:>33s}  {ssims.get('crop', 0):.4f}  {ssims.get('mahal', 0):.4f}  {avg:.4f}  {dt:.1f}s")
