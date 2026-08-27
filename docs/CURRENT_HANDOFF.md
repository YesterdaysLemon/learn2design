# Current handoff

Updated: 2026-08-27

This is the authority for the project's next action. Dated files in `research/`
are the evidence record, not a competing task list.

## Current state

- The owner reports that the Round-1 candidate was uploaded on 2026-08-24. The
  repository cannot independently verify the portal receipt. The submitted
  baseline remains revision `5ce3cdb2ddf4c505622a0aeef805936a4ea607d7`
  and ZIP SHA-256
  `4cc0dbc65a3e61ca5358c18655c432caf478fbdfc07f10512553781f8822924b`.

- The submission candidate is deterministic and packages successfully.
- The topology-conditioned neural initializer failed its offline control gate.
- The frozen `development-v2` A100 screen completed 64/64 runs and validated at
  all 16 topology inference units.
- Production replay, an independent history-based calculation, and the archived
  summary agree.
- The semantic prior failed the frozen policy: 7/0/9 topology wins/ties/losses,
  mean difference `+0.029737199613027163`, action
  `retain_no_prior_candidate`.
- The packaged submission now defaults to no-prior initialization, matching
  the frozen `retain_no_prior_candidate` action. The historical harness still
  selects each comparison arm explicitly.
- Authenticated no-prior trajectories show that the configured patience of 600
  never triggers inside the 600-second development screen.
- The loss-blind restart mechanics predecessor passed and the complete
  `restart-screen-v1` evidence validated at 32 runs, 16 seed pairs, and 8
  topology inference units.
- Production replay, the independent history-only evaluator, and the archived
  restart summary agree: p200 wins/ties/losses 4/0/4, mean difference
  `-0.016933403182594786`, median `+0.05236019711778772`, frozen action
  `retain_patience_600`.
- The separately frozen `submission-like-screen-v1` completed 20/20 runs and
  all 10 two-seed topology blocks on the untouched panel. All runs were
  physically and finite feasible.
- Production replay, a deliberately independent history-first evaluator, and
  the archived summary agree at `1e-12` absolute and relative tolerance. The
  frozen operational action is
  `candidate_evidence_complete_for_submission_review`.
- The private history-only trajectory diagnostic is complete. At the last
  common 6,312-evaluation checkpoint, seed-31/second-sweep minus
  seed-29/first-sweep mean loss was `+0.35777353288148966`; the corresponding
  matched-wall-time contrast was `+0.2955358367854244`. The direction agreed
  on five of six comparable progress fractions. This weakens a
  throughput-only explanation but cannot separate seed from sweep/session
  order.
- The deterministic builder now pins the ZIP creator platform. Windows and
  Linux builds reproduce evaluated candidate SHA-256
  `4cc0dbc65a3e61ca5358c18655c432caf478fbdfc07f10512553781f8822924b`.
- Private source and generated artifacts remain outside Git. The historical
  evidence bundle received a hash-verified encrypted second copy before the new
  paid attempt. The completed Stage-A archive and replay outputs are also held
  on private local storage.
- The opt-in midpoint Latin-hypercube transform of the seven random non-anchor
  starts was tested as `coverage_balanced`. The packaged default remained the
  submitted random initializer throughout.
- `coverage-triage-v1` contains 8 newly generated topologies with zero exact
  overlap against the official archive and every earlier generated panel. Its
  SHA-256 is
  `f400cdc3a947cd076ce9bd9f48a2dafcb98dfd3f9f938a74ceb11ca88c360972`.
  All four older generated panels remained byte-identical. The untouched
  12-topology `coverage-robustness-v1` panel remains unobserved.
- The single terminal `coverage-triage-screen-v1` attempt completed 32/32
  serial 600-second runs, 16/16 paired seed comparisons, and 8/8 topology
  inference blocks on one exact secure H100/CUDA-13 stack. All runs were
  physically and finite feasible, with zero worker errors or interruptions.
- Coverage balancing recorded 5 wins, 0 ties, and 3 losses. Its topology macro
  mean and median differences versus random start were
  `-0.17491992648617732` and `-0.2051665182256992`. It passed 13/14 frozen
  criteria but missed the required minimum of 7/8 topology wins. The
  descriptive 95% topology-bootstrap mean interval was
  `[-0.40161518941401997, 0.05658467766835468]`.
- The exact 169-member archive, six-file source lock, terminal-attempt receipt,
  external plan, and post-cleanup billing receipt authenticated. Production
  replay, the no-import history-first replay, and the detached summary matched
  across all runs, topology values, and frozen criteria. The summary remained
  unopened until both independent replays agreed.
- The frozen status/action is
  `failed / retain_random_start_candidate`. This does not unlock the proposed
  Stage B or the older robustness profile.
- Settled Runpod billing was `7.11735002356822` equivalent GPU-hours,
  `$23.416081577539444` GPU charge, and `$23.44330380158499` all-in, below the
  `$30` cap. After local hash verification, the pod and network volume were
  deleted. Live inventory is zero pods, endpoints, templates, and volumes.
- The first autonomous local mechanics checkpoint,
  `anchor-lane-stability-v1`, passed all six frozen CPU cases at the submitted
  population of eight. Exact and fresh-process traces matched; lane zero stayed
  invariant to seven changed suffix members until the forced shared-incumbent
  restart boundary; exceptional arithmetic and the `8,8,2` partial tail had no
  sanitizer or state violations. The private result SHA-256 is
  `95c1b979e70c6bc29c25901beb528a4ce184e06b4e6add842c7285f47bbab68e`.
  This is software-mechanics evidence only.
- The second autonomous checkpoint, `feasible-progress-clock-v1`, passed all
  seven frozen CPU cases. Across 32 finite-but-infeasible descending
  observations, all 32 reset their member clocks and no restart occurred. The
  plateau and delayed-plateau controls restarted every member exactly at their
  declared boundaries; the mixed case restarted exactly members 1, 3, 5, and
  7. The private immutable result SHA-256 is
  `d256572c0b67ff107ffca253f25e720175806912b3fe94c87f54478a6c98956e`.
  This confirms clock semantics only, not an optimizer improvement.
- The third autonomous checkpoint,
  `infeasible-prefix-indistinguishability-v1`, passed all six frozen CPU
  cases. The forever-infeasible and late-crossing paths were identical through
  `B = 13`; all 8,192 action vectors were exhausted, with 8,191 satisfying
  bounded restart only, one satisfying late-crossing preservation only, and
  zero satisfying both. The same zero-joint partition held at all six frozen
  bounds, while the extra-signal control satisfied both obligations only after
  breaking prefix identity. The private immutable result SHA-256 is
  `8aaf61bbcf21ea14e938f99f63f1c6e93f31b8d44307c79c9215ef84208b4ee5`.
  This is an abstract deterministic one-lane information boundary, not a claim
  about the full submitted optimizer or its performance.
- The fourth autonomous checkpoint, `public-signal-surface-v1`, passed all
  eight frozen CPU cases. Under the authenticated `dfbench==0.3.3` wheel and
  source identities, the UIFO aux contract exposes `sensitivity_loss`,
  `penalty`, `violations`, and raw `power_values.{hard,soft,detector}` beyond
  `is_feasible`. Scalar/batched calls and the protected population helper's
  full/1/2/4 chunk modes preserved the complete aux projection exactly. The
  current optimizer receives that tree but directly consumes only
  `is_feasible`. The private immutable result SHA-256 is
  `1548c94a5b46a0fca3054d252f8a96d38717c881b22f05036c868d6409d905cc`.
  This confirms current interface visibility only, not predictive value or a
  restart improvement.

## Next decision

Retain the packaged patience-600/no-prior random-start candidate. Do not run
`confirmation-v1`, rerun or top up `coverage-triage-screen-v1`, launch the
proposed `coverage-confirmation-screen-v1`, or launch the older
`coverage-robustness-screen-v1`. Do not tune a rescue rule or subgroup on the
observed triage panel. The terminal Stage-A failure is resolved and Stage B is
closed.

No new paid experiment is currently authorized or needed. Further research
should be local and mechanism-first, or wait for official public-leaderboard
feedback. A materially different initializer or optimizer change needs a new
rationale, implementation audit, untouched panel, frozen decision rule and
cost envelope, then separate owner approval before provisioning. The favorable
aggregate direction in Stage A is exploratory and cannot override the frozen
5/8-win failure.

Unpaid mechanism work now follows the guarded
[`AUTONOMOUS_LAB.md`](AUTONOMOUS_LAB.md) protocol. Its first five frozen
checkpoints passed. The third resolves the scalar/Boolean online-information
question; the fourth closes the public aux inventory and confirms that richer
current constraint diagnostics exist. It does not establish that any such
diagnostic predicts a future feasibility crossing. The fifth closes the
full-surface universal-certificate question: its two protected synthetic paths
were byte-identical through `B = 8` across the exact allowed typed snapshot and
differed at the next primary feasibility leaf. This shows only that finite
current observations are not a universal certificate without added
assumptions; it does not show that they lack distributional value or that any
policy improves performance.

The next admissible systems-mechanics question is now frozen as
`normal-path-jax-boundary-v1` in
[`2026-08-27-normal-path-jax-boundary-plan.md`](../research/2026-08-27-normal-path-jax-boundary-plan.md).
It inventories compilation, dispatch, host conversion, host-device
synchronization, callback, RNG, budget, and timing boundaries for one synthetic
normal-path batch, then asks whether an experiment-only pure-JAX transition can
reproduce the same typed public observations without changing the protected
submission. It is queued but has no controller result yet. Local CPU timings
remain diagnostic mechanics only, not accelerator or competition-performance
evidence.

After that checkpoint, a learning lane must first freeze the topology,
observation, action, reward, trajectory, split, and leakage contract and pass
toy controls. Test a supervised or surrogate baseline before a contextual
bandit or meta-RL controller. A native rewrite, official-data training,
candidate integration, accelerator benchmark, or paid run remains a separate
owner gate. The private controller is `awaiting_study` with the new versioned
systems checkpoint pending and will not rerun any terminal study.

## Public deadlines

The currently published [official competition timeline](https://github.com/artificial-scientist-lab/Learn2Design-2026/blob/main/README.md#timeline)
lists optional public-leaderboard deadlines on 2026-08-26, 2026-09-12, and
2026-09-29 Anywhere on Earth, followed by the prize-determining final deadline
on 2026-10-15 AoE. Recheck it before any schedule-critical launch; the
owner-reported 2026-08-24 upload means a baseline is already on file before the
first public deadline. The completed Stage-A failure does not affect that
submission's eligibility.

## What can contribute now

The exact private-artifact reproduction command and generated-bundle layout are
in [`DEVELOPMENT_V2_RESULTS_HANDOFF.md`](DEVELOPMENT_V2_RESULTS_HANDOFF.md). The
durable aggregate result and exploratory evidence boundaries are in
[`2026-08-21-development-v2-a100-results.md`](../research/2026-08-21-development-v2-a100-results.md).

Useful unpaid work now includes reviewing public-leaderboard feedback when it
arrives, profiling the retained candidate locally, and developing a genuinely
new mechanism without selecting against the observed Stage-A panel. The
validated full-surface information-boundary result and next systems rung are in
[`2026-08-27-full-surface-prefix-results.md`](../research/2026-08-27-full-surface-prefix-results.md).
The validated public signal-surface inventory and claim boundary are in
[`2026-08-27-public-signal-surface-results.md`](../research/2026-08-27-public-signal-surface-results.md).
The validated scalar/Boolean information-boundary result is in
[`2026-08-27-infeasible-prefix-indistinguishability-results.md`](../research/2026-08-27-infeasible-prefix-indistinguishability-results.md).
The finite-infeasible clock result that motivated it is in
[`2026-08-27-feasible-progress-clock-results.md`](../research/2026-08-27-feasible-progress-clock-results.md).
The validated anchor-lane mechanics result and its narrow claim boundary are in
[`2026-08-26-anchor-lane-stability-results.md`](../research/2026-08-26-anchor-lane-stability-results.md). The
validated Stage-A aggregate result, cost boundary, and exact terminal action
are in
[`2026-08-25-h100-coverage-triage-results.md`](../research/2026-08-25-h100-coverage-triage-results.md).
The validated restart result and exact private reproduction command are in
[`2026-08-21-patience-200-a100-results.md`](../research/2026-08-21-patience-200-a100-results.md).
Do not claim an order-of-magnitude gain: no restart target supplied a complete
eight-topology comparison.

The exact pre-result profile is in
[`2026-08-22-submission-like-screen-plan.md`](../research/2026-08-22-submission-like-screen-plan.md).
The validated aggregate result and limitations are in
[`2026-08-23-submission-like-screen-a100-results.md`](../research/2026-08-23-submission-like-screen-a100-results.md).
The package, private-data provenance, and provider-cleanup ledger is in
[`2026-08-23-final-submission-review.md`](../research/2026-08-23-final-submission-review.md).
The sealed operating and replay commands are in
[`SUBMISSION_LIKE_SCREEN_RUNBOOK.md`](SUBMISSION_LIKE_SCREEN_RUNBOOK.md).
