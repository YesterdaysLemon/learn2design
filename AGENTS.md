# Agent guide

This repository values reproducible evidence over speculative feature work.

## Start here

1. Read [`README.md`](README.md) for project status.
2. Read [`docs/CURRENT_HANDOFF.md`](docs/CURRENT_HANDOFF.md) for the next decision.
3. Read the README and dated research record for the area you will change.

The current gate is a frozen A100 comparison. Do not add another optimizer or
ML model until that experiment resolves the existing `semantic_prior` versus
`no_prior` question.

## Working rules

- Treat checked-in code, current competition rules, and raw artifacts as truth.
- Keep claims narrower than the evidence; CPU and low-memory diagnostics are not
  competition-performance results.
- Do not commit the official dataset or generated artifacts.
- Do not start paid or long-running compute without explicit owner approval.
- Preserve paired seeds, topology identity, budgets, provenance, and feasibility
  when changing evaluation code.
- Put implementation in `submission/` or `experiments/`, durable reasoning in
  `research/`, and short operating guides in `docs/`.
- Work on a focused branch and keep unrelated changes out of the PR.

## Minimum checks

```bash
uv sync --frozen --group dev --group integration
uv run --frozen --group dev --group integration pytest -q
python tools/build_submission.py
```

Run the relevant integration or artifact-integrity checks for any changed
experiment path. Update `docs/CURRENT_HANDOFF.md` only when the live decision or
next gate actually changes.
