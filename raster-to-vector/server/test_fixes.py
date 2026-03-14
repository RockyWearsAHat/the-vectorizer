"""Quick validation of thin-line + halo fixes."""
import cv2, numpy as np, time, io
from app.core.multilevel import multilevel_vectorize, generate_svg
from app.core.comparison import compare
from skimage.metrics import structural_similarity as ssim
import cairosvg
from PIL import Image

img = cv2.imread("/Users/alexwaldmann/Desktop/SVG-gen/Ref.png")
crop = img[50:460, 486:1050]

t0 = time.time()
result = multilevel_vectorize(crop, num_levels=24)
svg = generate_svg(result, remove_background=False)
dt = time.time() - t0

comp = compare(crop, svg)
print(f"crop: blur={comp.ssim_score:.4f} mae={comp.mae:.2f} "
      f"paths={result.path_count} nodes={result.node_count} "
      f"KB={len(svg)//1024} time={dt:.1f}s")

png = cairosvg.svg2png(
    bytestring=svg.encode(),
    output_width=crop.shape[1], output_height=crop.shape[0],
)
svg_arr = np.array(Image.open(io.BytesIO(png)).convert("RGB"))
src_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
raw = ssim(src_rgb, svg_arr, channel_axis=2)
print(f"crop: raw={raw:.4f}")

# Also test mahal
mahal = cv2.imread("/tmp/mahal_right.png")
if mahal is not None:
    t0 = time.time()
    result2 = multilevel_vectorize(mahal, num_levels=24)
    svg2 = generate_svg(result2, remove_background=False)
    dt2 = time.time() - t0
    comp2 = compare(mahal, svg2)
    png2 = cairosvg.svg2png(
        bytestring=svg2.encode(),
        output_width=mahal.shape[1], output_height=mahal.shape[0],
    )
    svg_arr2 = np.array(Image.open(io.BytesIO(png2)).convert("RGB"))
    src_rgb2 = cv2.cvtColor(mahal, cv2.COLOR_BGR2RGB)
    raw2 = ssim(src_rgb2, svg_arr2, channel_axis=2)
    print(f"mahal: blur={comp2.ssim_score:.4f} raw={raw2:.4f} "
          f"mae={comp2.mae:.2f} paths={result2.path_count} "
          f"nodes={result2.node_count} KB={len(svg2)//1024} time={dt2:.1f}s")
