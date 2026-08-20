# Learn2Design 2026 experiment lab

This repository is our open, reproducible attempt at the [Learn2Design 2026 competition](https://www.learn2design2026.com/): building an optimizer for differentiable gravitational-wave detector design.

The competition asks for an algorithm—not one fixed detector. On each hidden topology, the algorithm must tune roughly 200 continuous parameters within a four-hour budget. Lower average best loss is better.

## Current status

**Accelerator-gated candidate stage.** The repository now contains a deterministic submission candidate, a frozen-topology paired UIFO harness, and audited development/confirmation panels. An offline topology-conditioned neural initializer failed its control gate; the smaller semantic archive prior remains only a live-test candidate. A clean RTX 4060 diagnostic found no gross candidate-specific latency tail, but the 8 GB device cannot run the required full-population UIFO comparison. The optimizer has not yet earned a leaderboard or multi-topology performance claim.

The next decisive experiment is a predeclared three-arm development screen on one otherwise idle A100 80 GB accelerator. The causal comparison remains `semantic_prior` versus `no_prior`; organizer-style Adam is an orientation arm. The canonical paid-machine command, deployment ladder, recovery procedure, and artifact evacuation checklist live in [`docs/A100_RENTAL_RUNBOOK.md`](docs/A100_RENTAL_RUNBOOK.md).

This plan contains 96 serial runs and 16 scored Objective-hours. Persistent JAX compilation caching is forcibly disabled because compilation occurs inside the scored clock; cross-process cache reuse would bias later arms. The semantic prior is promoted only if it survives the frozen confirmation rule in `research/2026-08-19-accelerator-evaluation-plan.md`; otherwise the no-prior arm remains the honest candidate.

New contributors should begin with [`docs/CURRENT_HANDOFF.md`](docs/CURRENT_HANDOFF.md) and [`CONTRIBUTING.md`](CONTRIBUTING.md). Coding agents should also follow [`AGENTS.md`](AGENTS.md); these are compact entrypoints into the detailed experiment and research records, not separate ledgers.

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
