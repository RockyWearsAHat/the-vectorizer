---
name: "vision-safety-guard"
description: "Rules for vision tool usage. Ensures reliable visual-first validation with structured prompts and error-map awareness."
---

# Vision Safety Guard

**The vision tool is the PRIMARY quality gate for this project.** The goal is SVGs that look like an artist hand-traced the image in a vector program — that is a visual judgment, not a numeric one. Metrics are a useful secondary screen for catching regressions, but visual quality is what ultimately matters.

That said, the vision tool CAN hallucinate, misidentify error maps as real content, and confirm whatever it's led to believe. These rules prevent those failure modes while keeping vision front and center.

## Who Calls the Vision Tool

ONLY subagents. The main agent must NEVER call `mcp_aioserver-vis_analyze_images` directly.

## Rule 1: NEVER Ask Leading Questions

- **BAD:** "Is the red triangle artifact gone?" "Check if the gap slivers are fixed."
- **GOOD:** "Describe all visible differences between Image 1 and Image 2."
- The prompt must be OPEN-ENDED. Describe what the images are, not what you expect to find.

## Rule 2: ALWAYS Label Every Image Explicitly

- Every image passed to the vision tool must be labeled in the prompt with its role.
- Comparison PNGs from compare_all.py contain THREE panels: original | SVG render | error map.
- The error map uses RED to show pixel differences — this is NOT actual red color in the image.
- If sending a comparison PNG, the prompt MUST say: "This is a 3-panel comparison. Left=original raster. Middle=SVG render. Right=ERROR MAP where red intensity shows pixel difference magnitude, NOT actual colors."

## Rule 3: Vision IS the Pass/Fail Gate (with metric sanity check)

- Pass/fail is determined by VISUAL QUALITY: does the SVG look like clean, hand-traced artist work?
- Numeric metrics from `compare_all.py` are a secondary sanity check — if metrics show a massive regression (e.g., Feature% drops 10+%), investigate before accepting.
- If vision says "looks better, cleaner curves, more artistic" but metrics are flat or slightly worse, that CAN be a **PASS** — some numeric error is inherent in raster-to-vector conversion.
- If metrics improved but the SVG looks more machine-generated, jaggy, or pixelated, that is a **FAIL**.
- The key question is always: "Would an artist looking at this SVG believe it was hand-traced in Illustrator?"
- If metrics show a massive regression (>10% Feature% drop) while vision says "looks good," investigate the discrepancy before accepting — one of the two signals is wrong.

## Rule 4: NEVER Invent Defects From Vision Analysis

If the vision tool reports something (e.g., "I see red spots"), you must:

1. Cross-check against the numeric metrics — is there actually a measurable problem?
2. Determine if the observation is from the error map vs the actual image
3. Only report it as a real defect if metrics confirm it

## Rule 5: Prompts Must Request Structured Output

### Single comparison analysis:

```python
mcp_aioserver-vis_analyze_images(
    images=["/path/to/comparison.png"],
    prompt="""This is a 3-panel comparison image.
    Left panel: original raster image.
    Middle panel: SVG vectorization render.
    Right panel: ERROR MAP — red intensity = pixel difference magnitude (NOT actual image colors).

    Analyze ONLY the left and middle panels for visual differences.
    List each difference with: location, type (missing detail / added artifact / color shift / edge quality), severity (minor/moderate/major).
    Do NOT interpret red in the right panel as an image defect."""
)
```

### Before/after comparison:

```python
mcp_aioserver-vis_analyze_images(
    images=["/path/to/before_comparison.png", "/path/to/after_comparison.png"],
    prompt="""Two 3-panel comparison images (left=original, middle=SVG, right=error map where red=difference NOT actual color).
    Image 1: BEFORE the change. Image 2: AFTER the change.
    Compare the MIDDLE panels (SVG renders) between the two images.
    List what improved, what regressed, and what stayed the same.
    Do NOT interpret error map red as actual image content."""
)
```

## Close-up Inspection Protocol

Use as a core part of validation for all non-trivial pipeline changes:

```bash
DYLD_LIBRARY_PATH="/opt/homebrew/opt/cairo/lib" python _inspect_closeup.py
```

Then when using vision tool:

1. ALWAYS label images: "Left=original, Middle=SVG, Right=ERROR MAP (red=difference, NOT real color)"
2. NEVER ask leading questions like "is X fixed?" or "do you see Y artifact?"
3. Ask ONLY: "Describe all visible differences between the original (left) and SVG (middle) panels. Does the SVG look like it was hand-traced by an artist in a vector program?"
4. Cross-reference visual observations against numeric metrics to catch discrepancies
5. If the vision tool mentions colors in the error map (right panel), IGNORE those

## Reporting After Vision Analysis

- Visual quality assessment (primary — does it look hand-traced by an artist?)
- Numeric metrics (secondary — sanity check for regressions)
- Visual observations cross-referenced with metrics where relevant
- If vision and metrics disagree, explain the discrepancy and make the call based on visual quality
