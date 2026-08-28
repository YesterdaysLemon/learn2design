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
That initial-population gate is now resolved: the frozen coverage-balanced
treatment missed its topology-win criterion, so the random-start baseline is
retained and no confirmation was unlocked. See the preceding aggregate results
and limitations in
[`research/2026-08-23-submission-like-screen-a100-results.md`](research/2026-08-23-submission-like-screen-a100-results.md).
The final package, private-data provenance, and provider-cleanup ledger is in
[`research/2026-08-23-final-submission-review.md`](research/2026-08-23-final-submission-review.md).
The optimizer still has no hidden-leaderboard or official-budget multi-topology
performance claim. The owner reports that the Round-1 ZIP was uploaded on
2026-08-24, before the optional first public-leaderboard deadline on 2026-08-26
AoE.

The subsequently frozen `coverage-triage-screen-v1` H100 Stage A is complete
and independently validated: 32/32 runs, 16/16 seed pairs, and 8/8 topology
blocks completed with finite feasible results. Coverage-balanced initialization
recorded 5 wins, 0 ties, and 3 losses versus the submitted random-start control;
its topology macro mean and median differences were
`-0.17491992648617732` and `-0.2051665182256992`, respectively. It passed 13 of
14 guards but missed the precommitted requirement of at least 7/8 topology
wins. Production replay, an independent no-import history-first calculation,
and the detached summary agree. The frozen action is
`retain_random_start_candidate`: the submitted package remains unchanged and
Stage B is not unlocked or authorized. The all-in Runpod charge was
`$23.44330380158499`, below the `$30` cap; post-cleanup inventory contained no
pod, network volume, endpoint, or template.
See
[`research/2026-08-25-h100-coverage-triage-results.md`](research/2026-08-25-h100-coverage-triage-results.md),
the frozen
[`research/2026-08-24-h100-coverage-triage-plan.md`](research/2026-08-24-h100-coverage-triage-plan.md),
and [`docs/H100_COVERAGE_TRIAGE_RUNBOOK.md`](docs/H100_COVERAGE_TRIAGE_RUNBOOK.md).

The guarded autonomous local laboratory has completed nine unpaid CPU
checkpoints. `anchor-lane-stability-v1` confirmed its narrow lane-local/shared-
boundary mechanics; `feasible-progress-clock-v1` confirmed that finite
infeasible loss improvements reset the current member clock; and
`infeasible-prefix-indistinguishability-v1` confirmed the abstract
scalar/Boolean information obstruction for a deterministic target-lane rule.
`public-signal-surface-v1` then confirmed that the authenticated UIFO aux
contract exposes richer current constraint diagnostics and that the protected
population helper preserves them, while the optimizer currently consumes only
the Boolean. `full-surface-prefix-indistinguishability-v1` then confirmed a
finite synthetic twin across the complete allowed snapshot through its frozen
bound, with the next primary difference confined to `aux.is_feasible`. These
were followed by `normal-path-jax-boundary-v1`, which mapped the protected
host/device boundary and exactly matched one frozen no-restart batch with an
experiment-only pure-JAX transition across eager, JIT, and explicit compiled
execution. These are software or analytic boundary results, not
competition-performance evidence, and they do not change the submitted
package. `supervised-toy-signal-v1` then established the first learning
contract: a deterministic ridge surrogate recovered one deliberately exposed
signal on held-out generator regimes, while frozen label-shuffle and
signal-ablation controls stayed at chance. This validates the toy harness, not
an optimizer policy or competition gain. `contextual-bandit-toy-signal-v1`
then learned the deliberately exposed online sign signal from chosen-action
reward, kept its train state frozen across held-out regimes, and returned to
chance under context-shuffle and signal-ablation controls. This validates only
the immediate-reward bandit harness, not delayed credit, production RL, or a
candidate treatment. `two-step-delayed-credit-v1` then learned the frozen
two-action terminal-return mapping, scored `1.0` on all held-out regimes, and
returned to `0.0`, `0.5`, and `0.5` under transition-shuffle,
reward-origin-misalignment, and signal-ablation controls respectively. This
validates only that fixed synthetic delayed-credit harness. See the sanitized
[`anchor result`](research/2026-08-26-anchor-lane-stability-results.md),
[`clock result`](research/2026-08-27-feasible-progress-clock-results.md),
[`information-boundary result`](research/2026-08-27-infeasible-prefix-indistinguishability-results.md),
[`signal-surface result`](research/2026-08-27-public-signal-surface-results.md),
[`full-surface result`](research/2026-08-27-full-surface-prefix-results.md),
[`JAX-boundary result`](research/2026-08-27-normal-path-jax-boundary-results.md),
[`supervised toy-signal result`](research/2026-08-27-supervised-toy-signal-results.md),
[`contextual-bandit result`](research/2026-08-27-contextual-bandit-toy-signal-results.md),
[`two-step delayed-credit result`](research/2026-08-28-two-step-delayed-credit-results.md),
and [`laboratory protocol`](docs/AUTONOMOUS_LAB.md). The controller is back in
`awaiting_study` with no approved study pending. The next admissible learning
rung is a newly frozen bootstrapped multi-step value-propagation question, not
meta-RL or candidate integration. The first proposed fixture,
`multistep-td-propagation-v1`, was rejected before terminal execution because
its public state leaked prior target agreement and its complete-family,
held-out, scoring, timing, and negative-control proofs were incomplete. A
successor must use a fresh ID and target-independent state; see the sanitized
[`preflight rejection`](research/2026-08-28-multistep-td-propagation-preflight-rejection.md).
No terminal fixture will be repeated.

New contributors should begin with
[`docs/CURRENT_HANDOFF.md`](docs/CURRENT_HANDOFF.md) and
[`CONTRIBUTING.md`](CONTRIBUTING.md). Coding agents should also follow
[`AGENTS.md`](AGENTS.md). Recurring unpaid mechanics work is governed by the
[`autonomous local laboratory protocol`](docs/AUTONOMOUS_LAB.md). These are
compact entrypoints into the detailed experiment and research records, not
separate evidence ledgers.

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
