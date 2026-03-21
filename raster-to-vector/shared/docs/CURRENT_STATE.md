# Current State

## Active pipeline entry points

- Batch comparison harness: `compare_all.py`
- Single-image generator: `generate.py`
- Close-up inspection helper: `_inspect_closeup.py`
- Main pipeline implementation: `raster-to-vector/server/app/core/multilevel/__init__.py`
- Bezier fitting and node reduction: `raster-to-vector/server/app/core/curve_fitting/__init__.py`

## Latest canonical metrics

Source: `_comparisons/summary.txt`

| Image | Feat% | Miss% | Xtra% |   WdErr | MnDif | Time |  Nodes | SVG_KB |
| ----- | ----: | ----: | ----: | ------: | ----: | ---: | -----: | -----: |
| Ref   |  92.5 |   4.6 |  53.6 |   +1.13 |  6.62 |  6.3 |  7,034 |    101 |
| test2 |  99.0 |   0.1 |   2.8 | +216.89 | 10.90 |  7.1 |  8,708 |    133 |
| test3 |  84.7 |   9.4 |  18.8 |  +12.67 |  2.02 | 12.2 |  8,629 |    149 |
| test4 |  97.6 |   0.3 |   4.0 | +256.07 | 13.39 |  9.3 | 45,514 |    764 |
| test5 |  67.9 |   7.1 |   2.5 |   -1.45 | 13.73 |  9.5 | 19,318 |    305 |

## Main quality blockers

- Width expansion is the most urgent defect: `test2` and `test4` are dramatically over-thickened.
- `Ref` still carries heavy extra ink (`Xtra% 53.6`), so line-art edges are still too bold.
- `test5` feature capture remains weak (`Feat% 67.9`), indicating mural detail loss.
- Node count on `test4` is high enough to signal contour over-complexity even when recall is good.

## Immediate next target

Root-cause width expansion before any further broad experimentation. The next iteration should identify where the pipeline is inflating boundaries, then validate the fix against `test2` and `test4` before tuning anything else.

## Canonical outputs

The canonical generated outputs are the current batch files in `_comparisons/`:

- `summary.txt`
- `*_metrics.txt`
- primary `*_comparison.png`
- primary `*_output.svg`

Older debug renders, small comparison variants, baseline snapshots, and cache folders are intentionally noncanonical and may be removed during cleanup.
