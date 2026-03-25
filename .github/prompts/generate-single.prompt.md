---
description: "Generate SVG for a single image and show metrics."
agent: "subagent-developer"
argument-hint: "Image to generate SVG for (e.g. test2.jpg, Ref.png)"
---

Image: ${input:image:Which image to generate SVG for? (e.g. test2.jpg, Ref.png)}

Run `generate.py` on the specified image. Report the output SVG path and key metrics (feature*presence*%, mean_pixel_diff, node_count).
