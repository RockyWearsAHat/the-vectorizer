"""Test vectorization on both real images with luminance grouping."""
import cv2
import numpy as np
from app.core.multilevel import multilevel_vectorize, generate_svg
from app.core.comparison import compare

def test_image(path, name):
    img = cv2.imread(path)
    if img is None:
        print(f"ERROR: Could not load {path}")
        return
    h, w = img.shape[:2]
    
    for K in [12, 24]:
        result = multilevel_vectorize(img, num_levels=K)
        svg = generate_svg(result, remove_background=False)
        comp = compare(img, svg)
        print(f"{name} K={K:2d}: layers={len(result.layers):2d} "
              f"paths={result.path_count:2d} nodes={result.node_count:5d} "
              f"SSIM={comp.ssim_score:.4f} MAE={comp.mae:.2f}")
        
        if K == 24:
            out = f"/tmp/{name}_vec.svg"
            with open(out, "w") as f:
                f.write(svg)
            print(f"  Saved: {out}")

# Test on both real images
for path, name in [
    ("/Users/alexwaldmann/Desktop/SVG-gen/Ref.png", "ref"),
    ("/tmp/mahal_right.png", "mahal"),
]:
    import os
    if os.path.exists(path):
        test_image(path, name)
    else:
        print(f"Skipping {path} (not found)")
