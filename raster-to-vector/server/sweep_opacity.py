"""Sweep halo opacity and iso-level for optimal feathered edges."""
import cv2
from app.core.multilevel import multilevel_vectorize, generate_svg, VectorLayer
from app.core.comparison import compare

img = cv2.imread("/Users/alexwaldmann/Desktop/SVG-gen/Ref.png")
h, w = img.shape[:2]
crop_h, crop_w = min(410, h), min(564, w)
x_start = max(0, (w - crop_w) // 2)
crop = img[0:crop_h, x_start:x_start+crop_w]

# First get the result (paths don't change with opacity)
result = multilevel_vectorize(crop, num_levels=24)

def gen_with_opacity(result, halo_opacity, remove_bg=False):
    w, h = result.width, result.height
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {w} {h}" width="{w}" height="{h}">',
    ]
    if not remove_bg:
        parts.append(f'<rect width="{w}" height="{h}" fill="{result.background_color}"/>')
    for layer in result.layers:
        halo_d = layer.paths[0] if len(layer.paths) > 0 else ""
        core_d = layer.paths[1] if len(layer.paths) > 1 else ""
        if halo_d:
            parts.append(
                f'<path d="{halo_d}" fill="{layer.color}"'
                f' fill-rule="evenodd" opacity="{halo_opacity}"/>'
            )
        if core_d:
            parts.append(
                f'<path d="{core_d}" fill="{layer.color}"'
                f' fill-rule="evenodd"/>'
            )
        elif halo_d:
            parts.append(
                f'<path d="{halo_d}" fill="{layer.color}"'
                f' fill-rule="evenodd"/>'
            )
    parts.append("</svg>")
    return "\n".join(parts)

print(f"Crop {crop.shape[1]}x{crop.shape[0]}")
print(f"{'Opacity':>8} {'SSIM':>7} {'MAE':>6} {'Diff%':>6}")
print("-" * 35)
for opacity in [0.0, 0.15, 0.25, 0.35, 0.45, 0.55, 0.7, 1.0]:
    if opacity == 0.0:
        # No halo — just core paths
        svg = gen_with_opacity(result, 0.0, remove_bg=False)
    else:
        svg = gen_with_opacity(result, opacity, remove_bg=False)
    comp = compare(crop, svg)
    print(f"{opacity:8.2f} {comp.ssim_score:7.4f} {comp.mae:6.2f} {comp.pixel_diff_ratio*100:5.1f}%")
