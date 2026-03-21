"""Generate fresh SVG from an input image with current pipeline settings."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "raster-to-vector", "server"))

import cv2

from app.core.multilevel import multilevel_vectorize, generate_svg, optimize_svg_colors


def main() -> int:
    input_path = sys.argv[1] if len(sys.argv) > 1 else "Ref.png"
    image = cv2.imread(input_path)
    if image is None:
        print(f"Could not read input image: {input_path}", file=sys.stderr)
        return 1

    result = multilevel_vectorize(image, mediator_threshold=0.3)
    svg = generate_svg(result, remove_background=False)
    svg = optimize_svg_colors(svg, image)

    output_name = f"{os.path.splitext(os.path.basename(input_path))[0]}_output.svg"
    with open(output_name, "w") as f:
        f.write(svg)

    print(
        f"{output_name}  {len(svg):,} bytes  {result.path_count} paths  {result.node_count:,} nodes"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
