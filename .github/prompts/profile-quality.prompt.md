---
description: "Profile where quality is lost in the pipeline for a specific image. Shows SSIM at each stage to identify the bottleneck step."
agent: "subagent-developer"
argument-hint: "Image to profile (e.g. test2.jpg, Ref.png)"
---

Image: ${input:image:Which image to profile? (e.g. test2.jpg, Ref.png)}

Profile the quality-loss waterfall for the specified image:

1. Load the `subagent-dev-methodology` and `svg-pipeline-knowledge` skills.
2. Add temporary debug hooks to `multilevel_vectorize()` to capture intermediate outputs:
   - After K-means quantization: render label map as flat cluster colors
   - After agglomerative merge: render merged label map
   - After soft field computation: render soft-field blended colors
   - After iso thresholding: render binary masks as cluster colors
   - After contour extraction + Bézier fitting: render final SVG
3. Measure SSIM between each intermediate rendering and the original image.
4. Print the waterfall showing SSIM drop at each stage.
5. Identify which step causes the largest quality delta.
6. Remove any debug hooks (don't leave them in production code).
7. Update `kb-per-image.md` with the finding: "Biggest quality loss at [step] (SSIM Δ = [value])".
