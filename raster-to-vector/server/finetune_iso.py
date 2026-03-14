"""Fine-tune the iso configuration."""
import cv2, numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

# Re-use the test_config and gen_svg from sweep_iso_config
exec(open("sweep_iso_config.py").read().split("print(f\"{'Config'")[0])

configs2 = [
    ("4-level",   [0.10, 0.25, 0.40, 0.55],
                  [0.20, 0.40, 0.65, 1.00]),
    
    ("4b",        [0.10, 0.25, 0.40, 0.60],
                  [0.20, 0.40, 0.70, 1.00]),
    
    ("4c",        [0.12, 0.25, 0.40, 0.55],
                  [0.25, 0.45, 0.70, 1.00]),
    
    ("4d",        [0.10, 0.22, 0.38, 0.55],
                  [0.22, 0.42, 0.65, 1.00]),
    
    ("4e-wide",   [0.08, 0.20, 0.38, 0.55],
                  [0.18, 0.38, 0.65, 1.00]),

    ("3-level",   [0.12, 0.30, 0.55],
                  [0.30, 0.60, 1.00]),

    ("3b",        [0.10, 0.28, 0.50],
                  [0.30, 0.55, 1.00]),

    ("2-level",   [0.20, 0.50],
                  [0.55, 1.00]),

    ("2b-wider",  [0.15, 0.50],
                  [0.45, 1.00]),

    ("2c-tight",  [0.25, 0.50],
                  [0.55, 1.00]),
]

print(f"{'Config':<12} {'Crop SSIM':>9} {'Crop MAE':>9} {'Mahal SSIM':>10} {'Mahal MAE':>10} {'Ref SSIM':>9}")
print("-" * 65)

ref = cv2.imread("/Users/alexwaldmann/Desktop/SVG-gen/Ref.png")

for name, isos, opacs in configs2:
    r_crop = test_config(crop, isos, opacs)
    svg_crop = gen_svg(r_crop, remove_bg=False)
    c_crop = compare(crop, svg_crop)
    
    r_mahal = test_config(mahal, isos, opacs)
    svg_mahal = gen_svg(r_mahal, remove_bg=False)
    c_mahal = compare(mahal, svg_mahal)

    r_ref = test_config(ref, isos, opacs)
    svg_ref = gen_svg(r_ref, remove_bg=False)
    c_ref = compare(ref, svg_ref)
    
    print(f"{name:<12} {c_crop.ssim_score:9.4f} {c_crop.mae:9.2f} {c_mahal.ssim_score:10.4f} {c_mahal.mae:10.2f} {c_ref.ssim_score:9.4f}")
