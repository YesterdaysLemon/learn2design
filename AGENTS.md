# Agent guide

This repository values reproducible evidence over speculative feature work.

## Start here

1. Read [`README.md`](README.md) for project status.
2. Read [`docs/CURRENT_HANDOFF.md`](docs/CURRENT_HANDOFF.md) for the next decision.
3. Read the README and dated research record for the area you will change.

The terminal H100 `coverage-triage-screen-v1` completed and failed its frozen
7-of-8 topology-win rule with 5 wins, so the submitted random-start candidate
is retained and Stage B is closed. Do not rerun or top up Stage A, launch the
older 12-topology coverage profile, or spend on the proposed Stage B. Further
work must be unpaid and mechanism-first, or wait for public-leaderboard
feedback. Any materially different paid study needs a fresh rationale,
untouched panel, frozen rule and cost envelope, plus separate owner approval.

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
