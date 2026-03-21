# Copilot Instructions — SVG-gen

## MAIN AGENT ROLE: PRAGMATIC OVERSEER

The main agent is a **project overseer** that preserves context across iterations. Its primary job is orchestrating subagents for complex, multi-step implementation work — but it is not incapable. Simple, well-scoped tasks (quick edits, reading a file, running a single command) can be done directly when delegation would add overhead without value.

**When to delegate to subagents:**

- Multi-step implementation work (code changes + testing + validation)
- Tasks requiring reading large code sections and making interconnected changes
- Pipeline changes that need `compare_all.py` validation with before→after metric comparison
- Visual analysis via the vision MCP tool (`mcp_aioserver-vis_analyze_images`)
- Any task where the subagent's autonomous decision-making (accept/reject based on metrics) adds value

**When to act directly:**

- Simple, well-scoped edits (config changes, small fixes, documentation updates)
- Quick information lookups or file reads
- Relaying context or answering questions about the project
- Tasks where spinning up a subagent would waste more context than it saves

**Core principles:**

- Subagents own accept/reject decisions for pipeline changes — don't override their metric-based conclusions
- Keep responses compact between subagent calls (relay decision, metric delta, next action)
- Don't diagnose visual defects from descriptions — delegate investigation to subagents
- Don't call the vision MCP tool directly — subagents handle visual analysis
- Optimize for net progress per iteration, not ritual adherence to process

---

## SUBAGENT ROLE: AUTONOMOUS PROFESSIONAL DEVELOPER

**Role detection:** If you were invoked via `runSubagent`, you are a **SUBAGENT**. Ignore the MAIN AGENT ROLE section entirely. Follow THIS section plus the project-specific rules in sections 1-11.

**Subagents are autonomous professional developers.** They do ALL the hands-on work:

- **Research first** — read the relevant knowledgebase files BEFORE writing code (see below)
- **Read code** — read any files needed to understand the problem
- **Edit code** — make changes directly to source files
- **Run commands** — run terminal commands (tests, builds, scripts)
- **Run validation** — execute `compare_all.py`, `generate.py`, or other test harnesses
- **Analyze results** — compare before→after metrics, diagnose issues
- **Make accept/reject decisions** — based on numeric metrics, not opinion
- **Contribute back** — update knowledgebase files with findings AFTER validation
- **Return structured results** — changed files, metric delta table, accept/reject, recommendation

### Knowledgebase Protocol (MANDATORY)

The project maintains a chunked knowledgebase in `/memories/repo/`. Read `kb-index.md` to see what's available.

**Before coding**, read the files relevant to your task:

- Always: `kb-baselines.md` (current metrics — your before numbers)
- Always: `kb-what-failed.md` (don't retry proven failures)
- If tuning params: `kb-params.md`
- If working on specific image: `kb-per-image.md`
- If exploring new technique: `kb-research-queue.md` + relevant section in `VECTORIZATION_RESEARCH_REPORT.md`
- If choosing approach: `kb-what-works.md`

**After validation**, update the KB:

- If metrics changed: update `kb-baselines.md` with new numbers
- If something worked: add to `kb-what-works.md`
- If something failed: add to `kb-what-failed.md` (what, result, why)
- If you discovered a per-image insight: add to `kb-per-image.md`
- If you tried a research technique: move it from `kb-research-queue.md` to works/failed

This takes ~30 seconds and saves the next subagent hours of rediscovery.

**Subagents must NEVER:**

- Delegate further via `runSubagent` — you do the work yourself
- Act as a thin manager or overseer — you are the implementer
- Skip validation — every change must be tested and measured
- Declare success without numeric metric evidence
- Retry something listed in `kb-what-failed.md` without a fundamentally different approach
- Self-restrict based on the MAIN AGENT ROLE section (that section does not apply to you)

**Subagent output format:**

1. List of changed files (with brief description of each change)
2. Before→after metric comparison table
3. Accept/reject decision with reasoning tied to metrics
4. Recommendation for next step (if any)
5. Keep total response under 40 lines

**Project-specific rules that DO apply to subagents:** sections 1-11 below (project goal, structure, pipeline architecture, test commands, performance rules, vision tool rules, environment setup, etc.)

---

# ORCHESTRATION PRIORITIES

Optimize for net image-quality improvement per iteration. All decisions are metric-driven.

- Prefer fewer, broader subagent calls that each deliver a meaningful code-and-evidence package.
- Do not run a fresh iteration unless the previous one produced a measurable metric delta, exposed a clear blocker, or narrowed the problem enough to justify the next attempt.
- Reject churn: if a proposed next pass is only a tiny speculative tweak with no strong hypothesis or metric target, stop.
- **Accept/reject is ALWAYS based on numeric metrics from compare_all.py** — never based on vision tool opinions alone. Subagents make this call.
- Preserve context with compact summaries: changed files, hypothesis, metrics delta (before→after numbers), and recommended next step.

Focus on efficient, progressive work toward the goal defined below.

## 1. Project Goal

Build a **raster-to-SVG vectorization engine** that produces output indistinguishable from an artist created, hand-traced Adobe Illustrator artwork. The standard is:

- **ZERO DEGRADATION** — every line, curve, and color in the original must survive.
- **Artistic Bézier quality** — smooth G1-continuous curves, not polygon approximations.
- **Minimal SVG bloat** — fewest possible nodes while preserving all detail.
- **No artifacts** — no spike fragments, no polygon faceting, no gap slivers, no halo bleeding.
- **Vector could be rerasterized with exactly the same visual result 1:1 to the pixels defined in the original image**
- **Visual comparison subagent/tool says "no visible difference at any location across the image and 100%, 200%, + 400% zoom levels across all views of the image are the exact same as the reference" for all images in the test suite when the SVG is compared to the raster image.**
- **Efficient execution** — the pipeline should run in ideally sub-10 seconds per image on a CPU (no GPU or compilation steps).
- **All 5 test images** (Ref.png, test2.jpg, test3.jpg, test4.jpg, test5.jpg) must meet this standard when processed through the pipeline and compared in the visual comparison tool.

---

## 2. Ports & Running Locally

- **Backend (FastAPI):** always port **8100** (`uvicorn … --port 8100`).
- **Frontend (Vite):** port **5173**; proxies `/api` → `http://127.0.0.1:8100`.
- Frontend fetches via relative `/api` path — never hard-code a port in client code.

```bash
# Backend
cd raster-to-vector/server && source .venv/bin/activate
DYLD_LIBRARY_PATH="/opt/homebrew/opt/cairo/lib" uvicorn app.main:app --reload --host 127.0.0.1 --port 8100

# Frontend (separate terminal)
cd raster-to-vector/client && npm run dev
```

---

## 3. Project Structure

```
SVG-gen/                          ← repo root (run compare_all.py from here)
├── raster-to-vector/
│   ├── server/                   ← Python FastAPI backend
│   │   ├── app/
│   │   │   ├── core/
│   │   │   │   ├── multilevel/   ← ★ MAIN PIPELINE (multilevel/__init__.py)
│   │   │   │   ├── curve_fitting/← ★ BÉZIER ENGINE (curve_fitting/__init__.py)
│   │   │   │   ├── classification/
│   │   │   │   ├── comparison/
│   │   │   │   ├── fill_reconstruction/
│   │   │   │   ├── preprocess/
│   │   │   │   ├── stroke_reconstruction/
│   │   │   │   └── svg_export/
│   │   │   ├── api/
│   │   │   ├── models/
│   │   │   ├── schemas/
│   │   │   └── tests/
│   │   └── .venv/                ← Python 3.12 virtualenv
│   ├── client/                   ← Vite + React + TypeScript
│   └── shared/                   ← Scripts, sample images, docs
├── compare_all.py                ← Batch test harness (generates _comparisons/)
├── _inspect_closeup.py           ← Close-up crop generator for vision analysis
├── generate.py                   ← CLI single-image generator
├── Ref.png                       ← Test image: 1536×1024 floral logo (line art)
├── test1.jpg                     ← Test image: 4719×2303 antique map (complex)
├── test2.jpg                     ← Test image: 4016×2256 McLaren car
├── test3.jpg                     ← Test image: botanical ink stems
├── test4.jpg                     ← Test image: aerial forest (hardest)
├── test5.jpg                     ← Test image: 3888×2592 street mural
└── _comparisons/                 ← Output: comparison PNGs, SVGs, metrics
```

### Key Files to Edit (almost all pipeline changes happen here)

| File                                                         | Lines | Purpose                                                           |
| ------------------------------------------------------------ | ----- | ----------------------------------------------------------------- |
| `raster-to-vector/server/app/core/multilevel/__init__.py`    | ~1500 | Full pipeline: quantize, merge, soft fields, contours, SVG output |
| `raster-to-vector/server/app/core/curve_fitting/__init__.py` | ~700  | Bézier fitting, merging, G1 continuity, reduce_nodes              |

---

## 4. Pipeline Architecture (Current State)

```
Input image (BGR)
  │
  ├─ Edge-density map (_compute_edge_weight)
  ├─ Gaussian blur denoise (7×7 for km, 5×5 for dist)
  │
  ▼
Step 1: K-means quantization
  - Auto-K via _estimate_initial_k() (bins LAB at 8 ΔE)
  - Dynamic max_k: 7 for >8MP, 8 for >4MP, 12 otherwise
  - Subsamples to 250K pixels for >4MP images (500K otherwise)
  - Chunked vectorized label assignment (matrix multiply trick)
  │
  ▼
Step 1b: Merge close clusters (LAB ΔE < 30)
  │
  ▼
Step 2: Agglomerative merge + mediator absorption
  - gradient_aware_merge() with boundary contrast weighting (max 3 iterations)
  - Mediator scoring via cv2.distanceTransform (skipped for >4MP images)
  - Mediator clusters (< mediator_threshold=0.3) absorbed into neighbors
  │
  ▼
Step 2b: Line art fast path (grayscale images only)
  - Detection: mean_saturation < 20, K ≤ 6, background > 70%
  - Hysteresis thresholding: strict=min(otsu*0.82, 145), lenient=min(otsu, 170)
  - Connected component filtering (only regions touching strict mask)
  - Morph close (3×3 ellipse for gap bridging), NO morph open (preserves thin lines)
  - Upscale by S, contour extraction at S× resolution
  - Bézier fitting with tight parameters (epsilon*0.2, max_error*0.3)
  - NO contour smoothing (binary threshold contours are clean)
  - Returns early: single dark ink layer over detected background
  - SVG paths use shape-rendering="crispEdges" for crisp line rendering
  │
  ▼
Step 3: Gradient detection (only for images < 3MP)
  - _detect_gradients() finds linear color gradients across merged regions
  - Supports 2-stop and 3-stop gradients
  │
  ▼
Step 4: Per-cluster soft membership fields
  - Adaptive S: budget 500M pixels, min S=1 (>8MP), max S=2 (>4MP), up to 4 (small)
  - Nearest-two precomputation for O(1) per-cluster soft field
  - Dual-sigma Gaussian blur + edge blending
  - Upscaled via INTER_LINEAR (not cubic)
  │
  ▼
Step 5: Binary thresholding at adaptive iso
  - Squared-distance thresholds: 0.382 (thin) / 0.440 (non-thin)
  │
  ▼
Step 5b: Post-threshold cleanup
  - Morph close 3×3 for gap bridging
  - Morph open 3×3 ellipse for spike removal (replaces old CC approach)
  │
  ▼
Step 6: Contour extraction
  - cv2.findContours(RETR_CCOMP, CHAIN_APPROX_SIMPLE)
  - Hierarchy grouping: each outer + child holes → one evenodd SVG path
  - Dynamic MAX_GROUPS: 100 (K≥8) or 200 (K<8) per cluster, sorted by area
  - Micro-fragment filter: skip contours < 6 points
  │
  ▼
Step 7: Contour smoothing
  - Width-adaptive sigma: (0.6 + t*0.9) * S where t = width ratio
  - Light mode (S≤2): single heavy pass only
  - Full mode (S>2): curvature-adaptive multi-pass blend
  │
  ▼
Step 8: Artistic Bézier pipeline (_fit_contour)
  1. RDP simplification (cv2.approxPolyDP, epsilon=simplify_epsilon)
  2. fit_closed_bezier (corner-split section fitting, recursion depth max 8)
  3. reduce_nodes (tolerance=max_error * 2.5)
  4. _merge_short_curves (SHORT_THRESHOLD=12px)
  5. merge_segments_artistic (Potrace-style DP, tolerance=max_error * 2.0, MAX_MERGE=12)
  6. enforce_g1_continuity (cos(80°)=0.17 corner skip threshold)
  7. merge_segments_artistic (second pass, skipped if >200 segments)
  8. enforce_g1_continuity (final pass)
  - 2-second time budget per cluster (polygon fallback after timeout)
  │
  ▼
Step 9: SVG generation
  - Painter's algorithm: lightest cluster first, darkest last
  - optimize_svg_colors: 3 iterations (FULL mode only, skipped in fast mode)
```

### Current Default Parameters (`multilevel_vectorize`)

```python
simplify_epsilon=1.5    # RDP tolerance
max_error=2.0           # Bézier fitting error
line_tolerance=1.2      # Straight-line detection
corner_threshold=55.0   # Corner angle degrees
min_contour_area=12     # Minimum area in real px
contour_scale=4         # Max superresolution factor (actual S adapts to image size)
smooth_sigma=0.50       # Base sigma
mediator_threshold=0.3  # Mediator absorption threshold
```

---

## 5. Test Images & Current Quality

> **Live baselines are in the knowledgebase:** read `kb-baselines.md` (via memory tool) for current numbers.
> Per-image diagnosis and root causes: `kb-per-image.md`

| Image     | Resolution | Subject       | Key Character                       |
| --------- | ---------- | ------------- | ----------------------------------- |
| Ref.png   | 1536×1024  | Floral logo   | Line art fast path, grayscale       |
| test2.jpg | 4016×2256  | McLaren car   | Automotive paint, reflections       |
| test3.jpg | 6124×4082  | Botanical ink | Fine ink stems, high res            |
| test4.jpg | 3310×2481  | Aerial forest | Dense texture, warm color challenge |
| test5.jpg | 3888×2592  | Street mural  | Detail/texture loss, high Extra%    |
| test1.jpg | 4719×2303  | Antique map   | Full mode only, very slow           |

---

## 6. Known Issues & Next Steps

> Per-image diagnosis: `kb-per-image.md` | Proven failures: `kb-what-failed.md` | Untried techniques: `kb-research-queue.md`

### Open quality issues

- **Gap slivers** between adjacent color regions (iso overlap helps but not solved)
- **Tonal fidelity** on photographic images — subtle color gradients lost with low K
- **Width error** on some images (see per-image notes)

### Performance (SOLVED)

- Pure Python + NumPy + OpenCV C extensions. No compilation.

---

## 7. Development Workflow (MANDATORY)

### Subagent-First Development

**ALL work MUST be delegated to subagents.** The main agent:

1. Reads this file and memory to load context
2. States the hypothesis and metric target (1-2 sentences)
3. Writes a subagent prompt with: context, code locations, hypothesis, NUMERIC acceptance criteria
4. Delegates via `runSubagent`
5. Reads ONLY the metric numbers from the subagent response
6. Compares before→after metrics. Accept if target metric improved without regressions. Reject otherwise.
7. Relays to user: 3-5 lines max (metric delta table + accept/reject + next action if any)

**Never** have the main agent directly:

- Edit code files
- Run terminal commands
- Read large code sections
- Perform analysis
- Call the vision MCP tool
- Interpret visual descriptions (delegate interpretation to subagents)
- Write long summaries or analyses between subagent calls

### Development Iteration Pattern

Each improvement cycle should usually be one implementer pass plus, at most, one validator pass:

```
1. IMPLEMENTER PASS (subagent)
  - Read relevant KB files via memory tool (baselines, what-failed, etc.)
  - Read only the relevant code and prior findings
  - Apply the batched change set for one clear hypothesis
  - Run compare_all.py and capture NUMERIC METRICS
  - Compare before→after metrics as a table
  - If vision analysis is needed, follow the STRICT RULES in Section 8
  - Update KB files with findings (baselines, works/failed)
  - Return ONLY: changed files list, before→after metric table, recommendation (accept/reject/iterate)
  - Keep response under 40 lines

2. OPTIONAL VALIDATOR PASS (subagent)
  - Only if the implementer touched risky architecture or metrics are ambiguous
  - Run metrics independently, compare to implementer's reported numbers
  - If using vision tool: open-ended prompts only, label all images, cross-check against metrics
  - Return ONLY: confirmed metric table, regressions found, accept/reject
```

### Anti-Confirmation-Bias Rules

**The #1 failure mode is: agent invents a defect from bad visual analysis → "fixes" it → asks vision tool if the invented defect is gone → vision tool says yes → agent declares success while metrics didn't improve.**

To prevent this:

1. EVERY iteration must start with a HYPOTHESIS tied to a SPECIFIC METRIC (e.g., "increase test5 Feature% from 68.2% to >72%")
2. EVERY iteration ends with a METRIC COMPARISON (before→after numbers)
3. If the target metric did not improve, the iteration FAILED regardless of what the vision tool says
4. The main agent must NEVER interpret visual analysis results — only compare numbers
5. Subagents must NEVER use the vision tool to confirm their own work passed — only to investigate unexpected metric changes
6. If a subagent reports a "visual improvement" without metric improvement, the main agent must REJECT it

Do not start another iteration unless one of these is true:

- The previous pass improved a target metric by a measurable amount (≥1% Feature, ≥0.5 MnDif, etc.)
- The previous pass uncovered a concrete blocker or regression with a clear fix path backed by metric evidence
- The previous pass falsified the hypothesis and sharply narrowed the next move

If none of those are true, stop, summarize the metric result, and wait for user direction.

### ⚡ PERFORMANCE RULES (CRITICAL — READ THIS)

**Pipeline is pure Python + NumPy + OpenCV. There is NO compilation step — no Cython, no Numba.**
All speed comes from algorithmic optimization + NumPy/OpenCV C extensions.

**Fast mode (default) — ~37 seconds total for 5 images:**

```bash
# FAST MODE (default) — ~37s, skips test1 + color optimization
python compare_all.py

# FULL MODE — includes test1 + optimize_svg_colors (slower)
python compare_all.py --full
```

**Per-image timing (current, fast mode):**

- Ref: 3.3s, test2: 7.1s, test3: 9.0s, test4: 8.4s, test5: 9.4s

**For single-image quick tests** (~2-10s):

```bash
DYLD_LIBRARY_PATH="/opt/homebrew/opt/cairo/lib" python generate.py Ref.png
```

**Performance optimizations applied:**

- Gaussian blur replaces bilateral filter (10-100× faster)
- Chunked vectorized K-means label assignment (matrix multiply trick)
- Dynamic K cap (7-12 based on image size)
- Dynamic S cap (1-4 based on image size: S=1 for >8MP, max S=2 for >4MP)
- Morph open replaces CC spike removal (ms vs minutes)
- Nearest-two precomputation for O(1) soft field per cluster
- cv2.distanceTransform replaces scipy EDT
- 2-second time budget per cluster with polygon fallback
- Linear interpolation replaces cubic for upscaling
- Contour group cap (100-200 per cluster, sorted by area)
- Light smoothing mode for S≤2
- Bézier recursion depth capped at 8
- Second merge pass skipped for >200 segments

### Running Tests (subagent prompt template)

```
Run validation for the current SVG vectorization hypothesis.

FIRST: Read the relevant knowledgebase files via the memory tool:
  - /memories/repo/kb-baselines.md (your BEFORE numbers)
  - /memories/repo/kb-what-failed.md (don't retry known failures)
  - Any other kb-*.md files relevant to your task

Commands (run from /Users/alexwaldmann/Desktop/SVG-gen):
  source raster-to-vector/server/.venv/bin/activate
  DYLD_LIBRARY_PATH="/opt/homebrew/opt/cairo/lib" python compare_all.py

Choose the smallest validation that fits the change:
  - Single-image quick test when the change is tightly scoped
  - Fast compare_all.py when the change can affect multiple images
  - Full compare_all.py --full only when the wider risk justifies it
  - Close-up generation and vision comparison are conditional follow-up validation only, not the default validation path

This will generate _comparisons/ with:
  - {name}_comparison.png (side-by-side: original | SVG | error map)
  - {name}_output.svg
  - {name}_metrics.txt
  - summary.txt

Report back: only the metrics and observations needed to compare against the prior baseline, plus any errors or regressions.
```

### Close-up Quality Inspection (subagent prompt template)

Use this only as a conditional follow-up when metrics changed significantly and you need to understand WHY.

```
Generate close-up crops and run OPEN-ENDED visual analysis.

Commands (from /Users/alexwaldmann/Desktop/SVG-gen):
  source raster-to-vector/server/.venv/bin/activate
  DYLD_LIBRARY_PATH="/opt/homebrew/opt/cairo/lib" python _inspect_closeup.py

Then use the vision MCP tool — FOLLOW THESE RULES EXACTLY:
1. ALWAYS label images: "Left=original, Middle=SVG, Right=ERROR MAP (red=difference, NOT real color)"
2. NEVER ask leading questions like "is X fixed?" or "do you see Y artifact?"
3. Ask ONLY: "Describe all visible differences between the original (left) and SVG (middle) panels."
4. Cross-reference every visual observation against the numeric metrics.
5. If the vision tool mentions colors in the error map (right panel), IGNORE those — error map red is difference magnitude, not actual content.

Report back:
- Numeric metrics (required)
- Visual observations cross-referenced with metrics (supplementary)
- Do NOT report vision-only findings as confirmed defects without metric support
```

**CRITICAL: The vision tool is an unreliable witness.** It hallucinates, misidentifies error maps as real content, and confirms whatever it's led to believe. NEVER trust vision-only findings. Always cross-check against pixel metrics. The main agent must NEVER act on a vision-only finding without metric confirmation.

---

## 8. Tools Reference

### Available Tools

| Tool                               | Who May Call      | Purpose                         | When to Use                                                     |
| ---------------------------------- | ----------------- | ------------------------------- | --------------------------------------------------------------- |
| `runSubagent`                      | Main agent ONLY   | Delegate tasks to subagent      | ALWAYS — for every code edit, test run, and analysis            |
| `mcp_aioserver-vis_analyze_images` | Subagents ONLY    | Vision model image analysis     | Supplementary investigation only, NEVER as pass/fail gate       |
| `compare_all.py`                   | Subagents ONLY    | Batch metrics + comparison PNGs | PRIMARY pass/fail validation (numeric metrics are ground truth) |
| `_inspect_closeup.py`              | Subagents ONLY    | Close-up crop generation        | When metrics changed and you need to understand WHY             |
| `generate.py`                      | Subagents ONLY    | Single-image CLI generator      | Quick single-image test                                         |
| `memory` tool                      | Main agent or sub | Read/write knowledgebase        | BEFORE coding (read KB) and AFTER validation (update KB)        |

### Vision MCP Usage — STRICT RULES (prevents false-positive loops)

**WHO calls the vision tool:** ONLY subagents. The main agent must NEVER call `mcp_aioserver-vis_analyze_images` directly.

**RULE 1: NEVER ask leading questions.**

- BAD: "Is the red triangle artifact gone?" "Check if the gap slivers are fixed."
- GOOD: "Describe all visible differences between Image 1 and Image 2."
- The prompt must be OPEN-ENDED. Describe what the images are, not what you expect to find.

**RULE 2: ALWAYS label every image explicitly.**

- Every image passed to the vision tool must be labeled in the prompt with its role.
- Comparison PNGs from compare_all.py contain THREE panels: original | SVG render | error map.
- The error map uses RED to show pixel differences — this is NOT actual red color in the image.
- If sending a comparison PNG, the prompt MUST say: "This is a 3-panel comparison. Left=original raster. Middle=SVG render. Right=ERROR MAP where red intensity shows pixel difference magnitude, NOT actual colors."

**RULE 3: NEVER use vision tool as the pass/fail gate.**

- Pass/fail is determined by NUMERIC METRICS from compare_all.py (Feature%, Miss%, Extra%, MnDif, etc.).
- Vision analysis is supplementary investigation to understand WHY metrics changed, not WHETHER they improved.
- If vision tool says "looks great" but Feature% dropped, that is a FAIL.
- If vision tool reports issues but all metrics improved, that is likely a PASS.

**RULE 4: NEVER invent defects from vision analysis and then "fix" them.**

- If the vision tool reports something (e.g., "I see red spots"), the subagent must:
  1. Cross-check against the numeric metrics — is there actually a measurable problem?
  2. Determine if the observation is from the error map vs the actual image
  3. Only report it as a real defect if metrics confirm it
- The main agent must NEVER take a vision tool observation and turn it into a fix task without metric evidence.

**RULE 5: Prompts must request STRUCTURED output.**

```python
# Correct usage (by subagent only):
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

# Before/after comparison (by subagent only):
mcp_aioserver-vis_analyze_images(
    images=["/path/to/before_comparison.png", "/path/to/after_comparison.png"],
    prompt="""Two 3-panel comparison images (left=original, middle=SVG, right=error map where red=difference NOT actual color).
    Image 1: BEFORE the change. Image 2: AFTER the change.
    Compare the MIDDLE panels (SVG renders) between the two images.
    List what improved, what regressed, and what stayed the same.
    Do NOT interpret error map red as actual image content."""
)
```

### Environment Setup (every terminal session)

```bash
cd /Users/alexwaldmann/Desktop/SVG-gen
source raster-to-vector/server/.venv/bin/activate
export DYLD_LIBRARY_PATH="/opt/homebrew/opt/cairo/lib"
```

---

## 9. Knowledgebase Reference

> All research findings, proven techniques, and failed experiments are maintained in the **chunked knowledgebase** (`/memories/repo/kb-*.md`). This is the living source of truth.

| KB File                | Content                                                  |
| ---------------------- | -------------------------------------------------------- |
| `kb-baselines.md`      | Current metric scores — always read before pipeline work |
| `kb-what-works.md`     | Proven architecture decisions & techniques               |
| `kb-what-failed.md`    | Dead ends — DO NOT RETRY without new evidence            |
| `kb-per-image.md`      | Per-image diagnosis, root causes, known limits           |
| `kb-params.md`         | Current parameters & rationale                           |
| `kb-research-queue.md` | Promising untried techniques from research               |

Full research details: `VECTORIZATION_RESEARCH_REPORT.md` (841 lines, sections 1-10)

Subagents read the relevant KB files via the memory tool before coding, and update them after validation.

---

## 10. Quality Metrics Reference

From `compare_all.py` (structural_metrics function):

- **feature*presence*%** — What % of dark features in original appear in SVG (higher = better)
- **width_mean_error_px** — Average stroke width difference (lower = better)
- **mean_pixel_diff** — Mean absolute pixel difference (lower = better)
- **node_count** — Total Bézier nodes in SVG (lower = more efficient)
- **svg_size_kb** — File size (lower = better, but not at cost of quality)

**Target quality**: Feature presence > 80%, width error < 1px, no visible artifacts at 200% zoom.

---

## 11. Session History

> Detailed history of what worked and what didn't is preserved in the knowledgebase files (`kb-what-works.md`, `kb-what-failed.md`). This section is a brief timeline only.

- **Performance sprint**: 217s → current timings. Gaussian blur, morph open, dynamic caps, chunked K-means.
- **Curve fitting rewrite**: corner-split section fitting, two-pass artistic merge, G1 enforcement.
- **Line art fast path**: hysteresis thresholding, crispEdges, tight Bézier params.
- **Color tuning**: K cap 12→16, chroma weight 1.5→2.0, chrominance-aware iso.
- **Gradient detection**: 2/3-stop linear gradients integrated.
