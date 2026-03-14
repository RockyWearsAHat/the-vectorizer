"""Quick production validation after adaptive iso change."""
import cv2, time, sys
sys.path.insert(0, ".")
from app.core.multilevel import multilevel_vectorize, generate_svg
from app.core.comparison import compare

ref = cv2.imread("/Users/alexwaldmann/Desktop/SVG-gen/Ref.png")
crop = ref[50:460, 486:1050]

t0 = time.time()
result = multilevel_vectorize(crop)
svg = generate_svg(result, remove_background=False)
elapsed = time.time() - t0
metrics = compare(crop, svg)
print(f"crop: blur={metrics.ssim_score:.4f} paths={result.path_count} nodes={result.node_count} time={elapsed:.1f}s")

mahal = cv2.imread("/tmp/mahal_right.png")
if mahal is not None:
    t0 = time.time()
    r2 = multilevel_vectorize(mahal)
    s2 = generate_svg(r2, remove_background=False)
    e2 = time.time() - t0
    m2 = compare(mahal, s2)
    print(f"mahal: blur={m2.ssim_score:.4f} paths={r2.path_count} nodes={r2.node_count} time={e2:.1f}s")
