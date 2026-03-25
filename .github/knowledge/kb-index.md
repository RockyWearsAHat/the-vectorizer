# SVG-gen Knowledgebase Index

Subagents: read the files relevant to your task BEFORE coding. Update them with findings AFTER.

**Location:** All KB files live in `.github/knowledge/` (git-tracked). Read via `read_file` tool.

## KB Files

| File                            | What's in it                                                 | Read when…                          |
| ------------------------------- | ------------------------------------------------------------ | ----------------------------------- |
| `kb-baselines.md`               | Current metric scores (the ground truth)                     | Always — before any pipeline change |
| `kb-what-works.md`              | Proven architecture decisions & techniques                   | Planning an approach                |
| `kb-what-failed.md`             | Things tried that regressed — DO NOT RETRY                   | Planning an approach                |
| `kb-per-image.md`               | Per-image diagnosis, root causes, known limits               | Working on a specific image         |
| `kb-params.md`                  | Current pipeline parameters & their rationale                | Tuning any parameter                |
| `kb-research-queue.md`          | **Prioritized hypotheses** with status (READY/DONE/BLOCKED)  | Picking next improvement            |
| `svg-vectorization-research.md` | Domain research: best SVG generators, algorithms, techniques | Before implementing a new algorithm |

## Skills (read via readFile from .github/skills/)

| Skill                       | When to load                                                           |
| --------------------------- | ---------------------------------------------------------------------- |
| `svg-pipeline-knowledge`    | Editing pipeline code                                                  |
| `math-implementation-guide` | Implementing a NEW algorithm (translates research math → NumPy/OpenCV) |
| `subagent-dev-methodology`  | Any implementation work (iteration rules, anti-bias)                   |
| `vision-safety-guard`       | Before any vision tool call                                            |

## Tools

| Tool                  | Purpose                                                                |
| --------------------- | ---------------------------------------------------------------------- |
| `check_regression.py` | Auto-diffs compare_all.py output vs baselines. Run after every change. |
| `compare_all.py`      | Full batch validation (generates metrics + comparison PNGs)            |
| `generate.py`         | Single-image quick test                                                |
| `_inspect_closeup.py` | Close-up crops for vision validation                                   |

## Reading KB Files

```python
# Example: read baselines before any change
# Use read_file tool on: .github/knowledge/kb-baselines.md
```

All KB files are plain markdown in `.github/knowledge/`. No special tool needed — use the standard `read_file` / `readFile` tool.
