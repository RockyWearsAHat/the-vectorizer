---
name: "subagent-dev-methodology"
description: "Methodology for autonomous subagent development: iteration pattern, visual-first validation, performance constraints, test execution, and close-up inspection."
---

# Subagent Development Methodology

Load this skill when acting as a subagent via `runSubagent`. It defines how to iterate, validate, prevent bias, and report results.

## Core Quality Philosophy

**The goal is an SVG that looks like it was hand-traced by an artist in a vector program (e.g. Adobe Illustrator), NOT a pixel-perfect raster reproduction.** An artist tracing an image simplifies, cleans edges, and interprets edges logically and visually — the output should have that quality. A slight amount of numeric error vs. the raster original is inherent and expected because raster ≠ vector.

- **Primary validation:** Visual quality via the vision tool. Does the SVG look hand-crafted? Are curves clean? Are shapes faithful to artistic intent?
- **Secondary validation:** Numeric metrics from `compare_all.py`. These catch obvious regressions quickly (saving vision quota) but are NOT the final arbiter.

## Development Iteration Pattern

Each improvement cycle is one implementer pass, optionally followed by one validator pass:

### Implementer Pass

1. Read relevant KB files via memory tool (baselines, what-failed, etc.)
2. Read only the relevant code and prior findings
3. Apply the batched change set for one clear hypothesis
4. Run `compare_all.py` — capture numeric metrics as a **quick regression screen**
5. If metrics show a clear regression (e.g. Feature% dropped >5%), investigate before proceeding
6. Generate close-ups (`_inspect_closeup.py`) and run **visual comparison** via the vision tool (follow vision-safety-guard skill rules)
7. Make the accept/reject decision based on **visual quality**: does the SVG look more like hand-traced artist work?
8. Update KB files with findings (baselines, works/failed)
9. Return ONLY: changed files list, before→after metric table, visual assessment summary, recommendation (accept/reject/iterate)
10. Keep response under 40 lines

### Optional Validator Pass

- Only if the implementer touched risky architecture or the visual assessment is borderline
- Run independent visual comparison + metrics, compare to implementer's findings
- Return ONLY: confirmed metric table, visual assessment, regressions found, accept/reject

## Anti-Confirmation-Bias Rules

**The #1 failure mode is:** agent invents a defect from bad visual analysis → "fixes" it → asks vision tool if the invented defect is gone → vision tool says yes → agent declares success in a self-confirming loop.

To prevent this:

1. EVERY iteration must start with a clear HYPOTHESIS (e.g., "make test5 stroke edges cleaner and more artist-like" or "reduce test4 node bloat without losing detail")
2. EVERY visual assessment prompt must be OPEN-ENDED — never ask "is X fixed?" Always ask "describe all visible differences"
3. You must NEVER use the vision tool to confirm your own work passed in the same prompt that describes what you changed — the vision tool must judge blindly
4. Cross-check visual findings against metrics: if vision says "looks better" but metrics show a massive regression (e.g., Feature% dropped 10+%), investigate the discrepancy before accepting
5. If metrics are flat or slightly worse but the SVG visually looks more like clean hand-traced artwork, that CAN still be a PASS — artistic quality is the goal, not metric optimization
6. Conversely, if metrics improved but the SVG looks more machine-generated or pixelated, that is a FAIL

## Performance & Testing

See `pipeline-edits.instructions.md` for performance constraints, environment setup, and per-image timing targets.
See `validation-testing.instructions.md` for test commands, output structure, and quality metrics.

Choose the smallest validation that fits the change: single-image quick test → fast `compare_all.py` → full mode → close-up + vision comparison (the **primary quality gate** for non-trivial changes).

## Close-up Quality Inspection (core validation step)

Generate close-ups as part of the standard validation flow, not just for debugging.

```bash
DYLD_LIBRARY_PATH="/opt/homebrew/opt/cairo/lib" python _inspect_closeup.py
```

Then use the vision MCP tool following the vision-safety-guard skill rules exactly. The key question is always: **does this look like an artist hand-traced it in a vector program?**
