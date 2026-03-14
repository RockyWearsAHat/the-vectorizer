"""SVG quality analysis: line vs curve ratio, SSIM, node count."""
import cv2, numpy as np, io
from app.core.multilevel import multilevel_vectorize, generate_svg
from app.core.comparison import compare
from skimage.metrics import structural_similarity as ssim
import cairosvg
from PIL import Image

img = cv2.imread('/Users/alexwaldmann/Desktop/SVG-gen/Ref.png')
crop = img[50:460, 486:1050]
mahal = cv2.imread('/tmp/mahal_right.png')

for name, src in [('crop', crop), ('mahal', mahal)]:
    scores_blur = []
    scores_raw = []
    for _ in range(5):
        result = multilevel_vectorize(src, num_levels=24)
        svg = generate_svg(result, remove_background=False)
        comp = compare(src, svg)
        scores_blur.append(comp.ssim_score)

        png = cairosvg.svg2png(
            bytestring=svg.encode(),
            output_width=src.shape[1], output_height=src.shape[0],
        )
        svg_arr = np.array(Image.open(io.BytesIO(png)).convert("RGB"))
        src_rgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        raw = ssim(src_rgb, svg_arr, channel_axis=2)
        scores_raw.append(raw)

    avg_b = np.mean(scores_blur)
    avg_r = np.mean(scores_raw)
    print(f'{name}: blur={avg_b:.4f} raw={avg_r:.4f} '
          f'nodes={result.node_count} KB={len(svg)//1024}')

    # Count L vs C commands in last SVG
    l_count = svg.count('L')
    c_count = svg.count('C')
    print(f'  L={l_count} C={c_count} ratio={l_count/(l_count+c_count)*100:.1f}% lines')
