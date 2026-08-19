# Learn2Design 2026 experiment lab

This repository is our open, reproducible attempt at the [Learn2Design 2026 competition](https://www.learn2design2026.com/): building an optimizer for differentiable gravitational-wave detector design.

The competition asks for an algorithm—not one fixed detector. On each hidden topology, the algorithm must tune roughly 200 continuous parameters within a four-hour budget. Lower average best loss is better.

## Current status

**Baseline stage.** A first submission candidate and deterministic artifact builder are in place. The optimizer has not yet earned a leaderboard or multi-topology performance claim.

## Build the current candidate

Requires Python 3.11–3.13. The contract tests are intentionally light and do not run the expensive physics simulator.

```bash
python -m pip install pytest
pytest
python tools/build_submission.py
```

The builder writes `artifacts/generated/submission.zip` and a SHA-256 manifest. CI runs the same checks and publishes both files as a workflow artifact.

For a real API smoke test on the smaller constrained Voyager problem:

```bash
uv sync --group dev --group integration
uv run --group integration python tools/smoke_candidate.py --max-time 30
```

CPU compilation can take much longer than the timed search. The resulting JSON records the device, seed, budget, and Objective summary; it is a mechanics check, not evidence of UIFO performance.

## Where things live

- `submission/` — the exact files placed at the root of the competition ZIP
- `tests/` — fast correctness and packaging checks
- `research/` — literature notes, experiment records, and longer technical reports
- `artifacts/` — generated submission bundles and evaluation summaries (not hand-edited)
- `docs/` — short project plans and contributor-facing guides

The repository root is intentionally small. Durable evidence goes in the folders above rather than in a growing root-level ledger.

## Reproducibility rule

Every reported result should identify the code revision, configuration, topology split, random seeds, budget, hardware, and raw output location. Public and hidden leaderboard results must be labeled separately.

## Official resources

- [Competition page](https://www.learn2design2026.com/)
- [Official starter kit](https://github.com/artificial-scientist-lab/Learn2Design-2026)
- [Submission portal](https://submit.learn2design2026.com/)

## License

Our original code and documentation are released under the [MIT License](LICENSE). Upstream datasets, simulator code, and competition materials retain their own licenses and terms.
