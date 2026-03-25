# Current Pipeline Parameters & Rationale

Last verified: March 2026 (matches multilevel/**init**.py line 311-323)

## multilevel_vectorize() defaults

| Parameter          | Value | Rationale                                                       |
| ------------------ | ----- | --------------------------------------------------------------- |
| simplify_epsilon   | 1.0   | RDP tolerance (was 1.5; reduced June 2025 after fitter fix)     |
| max_error          | 1.5   | Bézier fitting max deviation (was 2.0; reduced June 2025)       |
| line_tolerance     | 0.5   | Straight-line threshold (was 1.2; reduced to force more curves) |
| corner_threshold   | 55.0° | Corner angle for section splitting                              |
| min_contour_area   | 12    | Min area in real px to keep a contour                           |
| contour_scale      | 4     | Max superresolution factor (actual S adapts)                    |
| smooth_sigma       | 0.50  | Base Gaussian sigma for contour smoothing                       |
| mediator_threshold | 0.3   | Mediator cluster absorption threshold                           |

## Adaptive parameters (computed at runtime)

| Parameter           | Logic                                                           | Why                                                           |
| ------------------- | --------------------------------------------------------------- | ------------------------------------------------------------- |
| K (clusters)        | auto via \_estimate_initial_k(), max 16 for >4MP                | More K → halo interference, but 16 needed for color diversity |
| S (upscale)         | budget 500M px, min S=1 (>8MP), max S=2 (>4MP), up to 4 (small) | Balances quality vs memory/speed                              |
| iso (threshold)     | thin=0.50 (sq-dist 0.382), non-thin=0.44 (sq-dist 0.440)        | 0.37 kills test5, 0.45 gaps, 0.44 sweet spot                  |
| chrominance iso adj | -0.015 for chroma>20 clusters                                   | Helps saturated color regions (test3 +1.4%)                   |
| smoothing sigma     | (0.6+t*0.9)*S where t=width ratio                               | Width-adaptive: thin features get less smoothing              |
| K-means iters       | 5/>4MP, 8/>2MP, 10/else                                         | Quality vs speed tradeoff                                     |
| K-means attempts    | 6/>4MP, 8/else                                                  | 3→6 can cause 20pp Feature regression                         |

## Bézier artistic pipeline (per contour)

1. RDP simplify (epsilon=simplify_epsilon)
2. 4-point interpolatory subdivision (ω=1/16, skip corners)
3. fit_closed_bezier (corner-split, recursion max 8)
4. reduce_nodes (tolerance=max_error × 2.5, with curve promotion)
5. merge_segments_artistic (tolerance=max_error × 2.0, MAX_MERGE=12)
6. enforce_g1_continuity (cos(80°)=0.17 corner skip) — **DEPRECATED: do not re-enable**
7. merge_segments_artistic (2nd pass, skip if >200 segments)
8. enforce_g1_continuity (final) — **DEPRECATED: do not re-enable**

- 2-second time budget per cluster → polygon fallback (4s for large images)

## CHROMA_WEIGHT in LAB space = 2.0

- Was 1.5, increased to help separate chromatic colors (yellow especially)
- 3.0 REGRESSES test5 (-1.5% Feature)

## Post-threshold cleanup

- morph close 3×3 for gap bridging
- morph open 3×3 ellipse for spike removal

## Contour extraction

- cv2.findContours(RETR_CCOMP, CHAIN_APPROX_SIMPLE)
- MAX_GROUPS: 100 (K≥8), 200 (K<8) per cluster, sorted by area
- Micro-fragment filter: skip < 6 points

## UPDATE RULES

- When changing any parameter, update this file with old→new value and measured effect
- ALWAYS verify against actual code defaults — do not trust stale values
