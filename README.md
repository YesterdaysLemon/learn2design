# Learn2Design 2026 experiment lab

This repository is our open, reproducible attempt at the [Learn2Design 2026 competition](https://www.learn2design2026.com/): building an optimizer for differentiable gravitational-wave detector design.

The competition asks for an algorithm—not one fixed detector. On each hidden topology, the algorithm must tune roughly 200 continuous parameters within a four-hour budget. Lower average best loss is better.

## Current status

**Validated patience-600/no-prior submission-review stage.** The frozen
`development-v2` A100 screen retained no-prior initialization. The subsequent
bounded restart mechanics gate passed, but the eight-topology patience-200
screen failed its predeclared promotion rule: 4 p200 wins, 0 ties, 4 losses;
mean p200-minus-p600 difference `-0.016933403182594786`; median difference
`+0.05236019711778772`; frozen action `retain_patience_600`. Production replay,
an independent history-based reference calculation, and the archived summary
agree. The separately frozen `submission-like-screen-v1` then completed 20/20
runs and passed all five operational evidence criteria at ten topology units.
Its frozen action is `candidate_evidence_complete_for_submission_review`.
Do not run `confirmation-v1`.

The source results remain private and outside Git. The aggregate evidence,
outcome-blind integrity workflow, exploratory limitations, and next-gate
recommendation are in
[`research/2026-08-21-development-v2-a100-results.md`](research/2026-08-21-development-v2-a100-results.md).
The packaged candidate now defaults to no-prior initialization, matching the
frozen action. The semantic-prior material remains available only for explicit
historical experiment replay; the paired harness supplies that flag per arm.

The current submission remains patience 600. The cost-bounded, one-arm
submission-like screen is now complete and independently validated, but it is
not an official-budget performance claim or evidence of leaderboard
competitiveness. The unpaid history-only diagnostic is also complete: the
later-sweep loss gap remains after matching evaluation counts, so simple
throughput drift is unlikely to explain it. Seed and sweep order remain
confounded, making this a search-robustness lead rather than a causal result.
The next research gate is to freeze one small initial-population robustness
change before testing it on a new disjoint panel. See the aggregate results and
limitations in
[`research/2026-08-23-submission-like-screen-a100-results.md`](research/2026-08-23-submission-like-screen-a100-results.md).
The final package, private-data provenance, and provider-cleanup ledger is in
[`research/2026-08-23-final-submission-review.md`](research/2026-08-23-final-submission-review.md).
The optimizer still has no hidden-leaderboard or official-budget multi-topology
performance claim. No new paid comparison should start until that narrow change,
budget, panel, and decision rule are frozen.

The owner reports that the Round-1 ZIP was uploaded on 2026-08-24, before the
optional first public-leaderboard deadline on 2026-08-26 AoE. The next gate is
now the cheaper pre-result `coverage-triage-screen-v1`: the same opt-in
midpoint Latin-hypercube transform on 8 newly generated, archive- and
prior-panel-disjoint topologies with paired seeds 37/41. It freezes 32 serial
600-second runs, requires at least 7/8 topology wins plus 13 other guards, and
can only produce a review request for a precommitted Stage-B design. That
proposed profile is not registered or executable. Its
exact H100/CUDA-13 contract, 169-member validator, detached summary commitment,
no-import replay, raw-transform checks, terminal-attempt ledger, and post-cleanup
billing receipt are implemented locally. The provider stop is 8 hours:
`$26.32` maximum GPU charge and `$30` maximum total charge at `$3.29/hour`.
No paid H100 has been created; clean launch-revision packaging, the encrypted
evidence backup, a cold provider smoke, current price/stock review, and a new
explicit approval of the `$30` cap remain required. See
[`research/2026-08-24-h100-coverage-triage-plan.md`](research/2026-08-24-h100-coverage-triage-plan.md)
and [`docs/H100_COVERAGE_TRIAGE_RUNBOOK.md`](docs/H100_COVERAGE_TRIAGE_RUNBOOK.md).

New contributors should begin with [`docs/CURRENT_HANDOFF.md`](docs/CURRENT_HANDOFF.md) and [`CONTRIBUTING.md`](CONTRIBUTING.md). Coding agents should also follow [`AGENTS.md`](AGENTS.md); these are compact entrypoints into the detailed experiment and research records, not separate ledgers.

## Build the current candidate

Requires Python 3.11–3.13. The contract tests are intentionally light and do not run the expensive physics simulator.

```bash
python -m pip install "pytest>=8.4,<9"
pytest -m "not integration"
python tools/build_submission.py
```

The builder writes `artifacts/generated/submission.zip` and a SHA-256 manifest. CI runs the same checks and publishes both files as a workflow artifact.

For a real API smoke test on the smaller constrained Voyager problem:

```bash
uv sync --group dev --group integration
uv run --group integration python tools/smoke_candidate.py --max-time 30
```

CPU compilation can take much longer than the timed search. The resulting JSON records the device, seed, budget, and Objective summary; it is a mechanics check, not evidence of UIFO performance.

To profile a local copy of the official design archive without copying the 75 MB dataset into this repository:

```bash
uv run --group integration python tools/profile_archive.py /path/to/dataset.h5
```

The JSON profile records the input checksum, topology reuse, lineage grouping, parameter dimensions, and loss quantiles. It is descriptive only; stored losses are not substitutes for live Objective evaluations.

For paired size-3 UIFO evaluation on a JAX-compatible accelerator, use the resumable isolated-process harness:

```bash
uv run --group integration python tools/run_uifo_paired.py \
  --topology-seeds 1001 1002 \
  --optimizer-seeds 7 11 \
  --max-time 600 \
  --dry-run
```

See `experiments/uifo_paired/` for the artifact schema, execution command, audited-panel format, and evidence boundaries. CPU execution is rejected unless explicitly requested as a non-representative mechanics run; confirmation claims additionally require an archive-exclusion audit and accelerator/device provenance.

Memory-limited candidate tails can be investigated with `tools/run_uifo_candidate_probe.py`. That six-worker forward/reverse diagnostic requires an otherwise idle GPU and makes no optimizer-performance claim.

## Where things live

- `submission/` — the exact files placed at the root of the competition ZIP
- `experiments/` — bounded model and optimizer comparisons with explicit gates
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
