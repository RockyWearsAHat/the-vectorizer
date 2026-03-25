---
description: "Improve SVG quality for a specific test image."
agent: "main-orchestrator"
---

Improve SVG vectorization quality for: **${input:image:Which image? (e.g. test4.jpg, Ref.png, test2.jpg)}**

1. Read `kb-baselines.md` for current metrics and `kb-what-failed.md` to avoid retrying dead ends.
2. Analyze the current output visually and numerically.
3. Form a hypothesis targeting a specific quality improvement.
4. Implement the change, run validation, and accept/reject based on visual quality (primary) cross-checked with metrics (secondary).
5. Update KB files with findings.
