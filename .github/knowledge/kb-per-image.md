# Per-Image Diagnosis & Known Issues

## Ref.png — 1536×1024, floral logo (line art)

- **Path**: Line art fast path (bypasses normal pipeline)
- **Feat%**: 90.6 | **Extra%**: 27.6 | **WdErr**: +0.04 | **MnDif**: 5.22 | **Time**: 30s
- **Root cause of prior high Extra%**: The old DT erosion (`> 0.3`) was a no-op (min DT at native res is always ≥ 0.96). All AA fringe pixels were rendered as solid black.
- **Fix applied (March 2026)**: Intensity-based strict/fringe split. Strict mask (≤ min(otsu\*0.90, 155)) → full fill. Hysteresis additions (strict..core) → 0.55 opacity fringe. This reduced Extra% from 46.8→27.6 and WdErr from +0.98→+0.04.
- Visual validation: PASS — hand-traced quality, appropriate stroke weights, no missing features.
- 95.4% of dark pixels survive pipeline — the remaining over-expansion (27.6 Extra%) comes from the fill geometry itself.
- Uses hysteresis thresholding: strict=min(otsu\*0.90, 155), lenient=min(otsu, 165)
- crispEdges shape-rendering on SVG paths
- New diagnostic: with the current line-art fast path, the computed fringe mask is effectively empty on `Ref` (`fringe count = 0` in direct mask inspection). The 52% Extra is therefore not coming from a colored AA fringe layer; it is coming from the surviving fill geometry itself. Large fill components dominate (`max DT` up to ~24 px), while stroke pixels are a small minority.
- Tried a structurally different per-component adaptive erosion pass on those thick fill components. It cut `Extra` sharply (**52.0 -> 26.8**) and improved `MnDif` (**6.21 -> 5.91**), but catastrophically collapsed feature coverage (**85.7 -> 53.5**). Conclusion: Ref does need a different line-art construction, but not another erosion-based tweak layered on the current fill extraction.
- New failed direction: globally loosening the hysteresis core cap (`165 -> 175`) does recover some right-edge structure, but only by swelling the whole drawing again (`81.4 / 12.2 / 47.4 / 6.21 -> 82.8 / 11.1 / 58.0 / 6.61`). Any Ref fix for the far-right missing strip has to be localized, not a global core-threshold lift.
- New failed direction: broad-fill opacity gating on Ref can reduce the center-lower heavy block, but both the global and selective variants were visually rejected. They either turn dense blacks into gray underfills or shift the dominant error to missing right-side structure. Tone-only fixes are not enough; the next Ref move has to recover edge structure without globally lightening the heavy motifs.
- New accepted direction (March 2026): tighten line-art contour fitting after the 1.5px stroke-cap restoration. Changing line-art fill fitting from `epsilon*0.20 / max_error*0.30` to `epsilon*0.14 / max_error*0.22` and lowering the contour area floor from `2*S*S` to `max(S*S, 8)` materially improved Ref structure in blind visual review. Default-mode metrics moved from roughly **81.4 / 12.2 / 47.4 / +0.76 / 6.21** to **89.0 / 7.1 / 49.9 / +0.92 / 8.75**; targeted low-saturation color-optimized validation reached **89.0 / 7.1 / 49.7 / +0.92 / 6.05**. Net: much better feature recovery and less obvious auto-trace simplification, but broad floral contours still read slightly too heavy.

## test2.jpg — 4016×2256, McLaren car

- **Feat%**: 85.3 | **Miss%**: 3.4 | **WdErr**: -6.49 | **MnDif**: 12.36 | **Time**: 19.2s
- Width error negative = SVG slightly thinner than original
- Previously had 98.7% Feature with different params — investigate regression source
- Good structure overall, main challenge is color fidelity on automotive paint

## test3.jpg — 6124×4082, botanical ink stems

- **Feat%**: 89.8 | **Xtra%**: 25.3 | **WdErr**: +17.67 | **MnDif**: 1.99 | **Time**: 26.4s
- Lowest MnDif of all images — good overall pixel accuracy
- High Extra% and width error suggest over-expansion of ink strokes
- Large image (25MP) — S is capped low, limiting contour detail
- Chrominance-aware iso helped +1.4% Feature
- Major diagnostic finding: the apparent 44.6% catastrophic regression under `--optimize-colors` was not a geometry failure. K-means, merge state, and soft-field keep ratios were all intact; the loss came from the universal `optimize_svg_colors` pass. Gating that pass off for high-saturation images restored test3 to **87.0 | 6.6 | 16.7 | +13.92 | 1.87** in the same validation mode.
- New debug finding (March 2026): `test3` is not behaving like a low-saturation ink-only case in the current cluster space. With `SVG_CLUSTER_DEBUG=1`, the surviving non-background clusters are all high-saturation red/purple families, all classified as non-thin, and all keep essentially all of their own label pixels at the current iso (`keep_ratio` ~`0.94-1.00`). That means the remaining width/Extra problem is not coming from late smoothing or obvious thin-cluster misclassification; it is coming from how a small set of broad saturated clusters overlap and expand during extraction.

## test4.jpg — 3310×2481, aerial forest

- **Feat%**: 90.9 | **Xtra%**: 2.3 | **MnDif**: 14.86 | **Time**: 23.1s | **Nodes**: 88,577
- **PRIMARY BOTTLENECK: Node count 123K — 5× all other images.** Root cause: dense texture generates many small K-means fragments that survive into the contour pipeline. K-means on dense natural texture produces ~2418 fragments; region merging reduces to ~151 (He et al. 2024).
- Warm color loss: amber cluster center LAB(140,135,179) is bright; shadowed autumn pixels (L≈100) closer to green/neutral competitors in weighted LAB distance and some large warm fills were being misclassified as thin features, tightening iso too far.
- Verified on debug pass: a large warm cluster was running at thin iso `0.439` with only `8,395 / 20,849` dark warm pixels surviving; after excluding large saturated fills from thin handling on yellow-dominant images and relaxing their iso by `0.008`, the same family of warm clusters kept materially more coverage and `test4` improved to **Feat 90.2 | Miss 3.8 | Extra 2.5 | MnDif 15.99**.
- Current accepted direction: gate the warm-fill thin/iso relaxation to yellow-dominant images only. Ungated application regressed mixed-color photos.
- K=14→11 after gradient merge

## test5.jpg — 3888×2592, street mural

- **Feat%**: 86.3 | **Xtra%**: 12.8 | **WdErr**: +26.27 | **MnDif**: 12.25 | **Time**: 29.7s
- Yellow panel: ACTUALLY FINE — 99.4% of warm pixels preserved (delta-E 4.2). Not a color issue.
- Real issue is detail/texture loss, limited by S=2 resolution and group/cluster cap
- K=14→14 after gradient merge (zero merges for this image)
- gradient_aware_merge runs only 1 iteration with no merges
- Latest targeted `test5` loop after region/path matching changes: **Feat% 82.7 | Miss% 3.1 | Extra% 8.3 | MnDif 10.44 | Time 43.2s** with 3 surviving SVG gradients.
- Diagnostic outcome: the hand-like warm secondary components are now surfaced, but they fail `_fit_gradient_region` on the linear spatial-correlation gate (`corr 0.02` / `0.13` on warm secondary components). This is now a model-fit limitation, not just attachment or size filtering.
- Broad warm-gradient leakage into cool wall geometry was reduced by replacing the old largest-group fallback with scored region-to-path matching.
- New accepted iteration: radial fallback for small warm secondary regions plus scored path assignment. Latest targeted `test5` run: **Feat% 82.7 | Miss% 3.1 | Extra% 8.3 | MnDif 10.50 | Time 41.9s**. Blind visual check: mural hand/warm area improved and less posterized, though still simplified.
- Accepted follow-up: conservative large-group detail overlays inside very large high-variance saturated contour groups. This did not move the broad structure metrics beyond **82.2 / 3.2 / 9.0**, but it improved `MnDif` from **10.59 -> 10.46** and the residual mismatch now reads more like refinement inside warm mural structures than a wholesale region failure.
- Accepted gradient regions now include a hand-sized warm radial region at approx bbox `(762,1523)-(1506,1899)`. A second warm radial region at approx `(2297,1207)-(3887,1435)` also appears and may still need refinement.
- Stability finding: the pipeline had major nondeterminism on `test2` until OpenCV threads and per-cluster parallelism were disabled; repeated runs swung between ~83% and ~96% Feature with identical inputs.
- After stabilizing execution, current single-image `test5` sits at **Feat% 81.3 | Miss% 3.4 | Extra% 8.5 | MnDif 10.72 | Time 55.5s**. This is a trustworthy but worse baseline than some earlier lucky runs, so future tuning must improve from the stable path, not from nondeterministic highs.

## test1.jpg — 4719×2303, antique map (full mode only)

- **Feat%**: 92.4 | **MnDif**: 11.61 | **Time**: 987.6s | **Nodes**: 59,986
- Extremely slow — needs investigation for why it's 30× slower than others
- Included in --full mode only, not default batch

## UPDATE RULES

- After investigating an image-specific issue, add findings here
- Include metric evidence, not just visual observations
