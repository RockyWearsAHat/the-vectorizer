---
description: "Run a hypothesis-driven experiment: implement one pipeline change, validate with metrics + regression check, accept/reject."
agent: "main-orchestrator"
---

Run one hypothesis-driven experiment cycle. Read `kb-research-queue.md` and pick the top READY hypothesis, cross-check `kb-what-failed.md`, implement via subagent, validate with `compare_all.py` + `check_regression.py`, then update the KB with the result.
