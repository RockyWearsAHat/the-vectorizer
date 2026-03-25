"""Render SVGs at native size and extract close-up crops for quality inspection."""
import cairosvg
from PIL import Image
import io
import os

Image.MAX_IMAGE_PIXELS = None  # allow large images
os.makedirs("_comparisons", exist_ok=True)

for name in ["Ref", "test3", "test2", "test4", "test5"]:
    svg_path = f"_comparisons/{name}_output.svg"
    orig_candidates = [
        f"{name}.png",
        f"{name}.jpg",
        f"raster-to-vector/shared/sample-images/{name}.png",
        f"raster-to-vector/shared/sample-images/{name}.jpg",
    ]
    orig_path = None
    for c in orig_candidates:
        if os.path.exists(c):
            orig_path = c
            break
    if not os.path.exists(svg_path):
        print(f"SKIP {name}: no SVG")
        continue

    with open(svg_path) as f:
        svg_data = f.read()

    # Render at native scale
    png_data = cairosvg.svg2png(bytestring=svg_data.encode(), scale=1)
    img = Image.open(io.BytesIO(png_data))
    w, h = img.size

    # Save render at reasonable size (max 2000px wide for inspection)
    if w > 2000:
        ratio = 2000 / w
        img_sm = img.resize((2000, int(h * ratio)), Image.LANCZOS)
    else:
        img_sm = img
    img_sm.save(f"_comparisons/{name}_svg_render.png")

    # Also load the original at matching size for side-by-side
    if orig_path:
        orig = Image.open(orig_path)
        orig_match = orig.resize((w, h), Image.LANCZOS)
    else:
        orig_match = None

    # Crop center detail (25% of image at native resolution)
    cx, cy = w // 2, h // 2
    crop_w, crop_h = w // 4, h // 4
    crop = img.crop((cx - crop_w // 2, cy - crop_h // 2,
                     cx + crop_w // 2, cy + crop_h // 2))
    crop.save(f"_comparisons/{name}_closeup_center.png")

    # Crop top-left detail area
    tl_w, tl_h = w // 3, h // 3
    crop_tl = img.crop((0, 0, tl_w, tl_h))
    crop_tl.save(f"_comparisons/{name}_closeup_topleft.png")

    # Side-by-side comparison crops if original available
    if orig_match:
        orig_crop = orig_match.crop(
            (cx - crop_w // 2, cy - crop_h // 2,
             cx + crop_w // 2, cy + crop_h // 2)
        )
        # Side by side: original | SVG
        combined = Image.new("RGB", (crop_w * 2 + 4, crop_h), (128, 128, 128))
        combined.paste(orig_crop, (0, 0))
        combined.paste(crop, (crop_w + 4, 0))
        combined.save(f"_comparisons/{name}_sidebyside_center.png")

        orig_tl = orig_match.crop((0, 0, tl_w, tl_h))
        combined_tl = Image.new("RGB", (tl_w * 2 + 4, tl_h), (128, 128, 128))
        combined_tl.paste(orig_tl, (0, 0))
        combined_tl.paste(crop_tl, (tl_w + 4, 0))
        combined_tl.save(f"_comparisons/{name}_sidebyside_topleft.png")

    print(f"{name}: {w}x{h}, saved renders and crops")

print("Done")
