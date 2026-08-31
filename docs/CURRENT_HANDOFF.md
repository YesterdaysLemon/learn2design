# Current handoff

Updated: 2026-08-31

This is the authority for the project's next action. Dated files in `research/`
are the evidence record, not a competing task list.

## Current state

- The owner reports that the Round-1 candidate was uploaded on 2026-08-24. The
  repository cannot independently verify the portal receipt. The submitted
  baseline remains revision `5ce3cdb2ddf4c505622a0aeef805936a4ea607d7`
  and ZIP SHA-256
  `4cc0dbc65a3e61ca5358c18655c432caf478fbdfc07f10512553781f8822924b`.

- On 2026-08-30 the owner supplied the organizing team's Round-1 result email:
  owner-reported score `0.444293`, best reported score `0.019674`, placement
  14th of 43 evaluated participants, and an invitation to Round 2. The
  repository cannot independently authenticate the email or hidden result.
  The sanitized intake and evidence boundary are frozen in
  [`2026-08-30-round1-feedback-and-round2-program.md`](../research/2026-08-30-round1-feedback-and-round2-program.md).

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
- The eighth autonomous checkpoint, `contextual-bandit-toy-signal-v1`, passed
  all nine frozen CPU cases. The deterministic two-bin, two-action policy
  learned the exposed context sign from chosen-action reward alone: train mean
  reward was `0.75` with regret `256`, validation and test macro reward were
  `1.0`, and the constant, shuffled-context, and both signal-ablation controls
  scored `0.5`. The private immutable result SHA-256 is
  `3396ec8c8203c86659b21d2beaa4c5dfee91bbcdfdd9b1245d3057086aa0fc9b`.
  This validates immediate-reward bandit mechanics only, not delayed credit,
  optimizer value, or competition performance.
- The ninth autonomous checkpoint, `two-step-delayed-credit-v1`, passed all
  eleven frozen CPU cases.
  Its fixed empirical value table reached `1.0` on untouched held-out regimes
  and lost the signal under its transition, reward-origin, and signal-ablation
  controls. This validates only that synthetic two-step harness.
- The proposed `multistep-td-propagation-v1` checkpoint was rejected during
  pre-result audit and never invoked through the guarded controller. Its public
  successor state exposed prior target agreement, its family commitment did
  not cover every legal successor/outcome row, and several promised isolation,
  scoring, timing, and control-balance sentinels were incomplete. It is not a
  terminal study and contributes no evidence; see
  [`2026-08-28-multistep-td-propagation-preflight-rejection.md`](../research/2026-08-28-multistep-td-propagation-preflight-rejection.md).
- Its fresh successor, `multistep-td-action-prefix-v2`, was also rejected
  before terminal execution. Development diagnostics passed, but independent
  audits found that the target-swap proof was tautological, the myopic baseline
  read evaluator truth, several held-out and timing sentinels were not connected
  to the exercised paths, and the family, trace, control-difference, and
  all-boundary dependency proofs were incomplete. The controller was never
  invoked, v2 is absent from the registry and allowlist, and it contributes no
  evidence; see
  [`2026-08-28-multistep-td-action-prefix-v2-preflight-rejection.md`](../research/2026-08-28-multistep-td-action-prefix-v2-preflight-rejection.md).
- The tenth autonomous checkpoint, `multistep-td-action-prefix-v3`, passed all
  nineteen frozen CPU cases in its single guarded invocation at revision
  `9d8c64887c730043d2da7c313ac9240fd3f3e85c`. Its four-sweep synchronous
  tabular TD learner propagated the toy terminal scalar across three bootstrap
  boundaries and returned `1.0` on train, validation, test, and every minimum
  held-out regime without held-out updates. Constants, feedback-only myopic,
  and no-bootstrap returned `0.5`; seeded-random returned `0.0703125`; and the
  transition-target, reward-origin, and complete signal-ablation controls
  returned `0.5`. All complete-family, target-swap, split-disjointness, typed
  trajectory, physical timing, trace authentication, train-only source,
  synchronous-order, comparator replay, all-boundary dependency,
  control-difference, sanitizer, and fresh-process cases passed. The private
  immutable result SHA-256 is
  `c6e7cecd8d6e9fa7e12aee116f141522321f400a9286940b38a2023e54f5d86f`.
  Its public signed signal deliberately encodes evaluator truth, so this
  validates only the fixed synthetic synchronous-TD harness and toy
  propagation mechanics. See the
  [`V3 terminal result`](../research/2026-08-29-multistep-td-action-prefix-v3-results.md).
- The frozen `constraint-aware-progress-toy-v1` contract has an exact
  pre-result implementation at commit
  `269698a3974cc12f9871e0e8a3580fbc7230cce9`. Its synthetic fixture,
  independently replaying worker, twelve-case registry contract, source
  approvals, dedicated controller validation, and Windows process boundary
  passed three final hostile read-only audits. Focused tests, the complete
  local-lab suite, and the single full repository pass were green. Its first
  guarded launch at revision `4098ce1ba25a163e96ac3cf735b7cd7e419bc64c`
  entered `cycle_started` and parked about two seconds later because the worker
  emitted forbidden stderr. No terminal result was authenticated and the ID
  was not added to `completed_studies`, so this is infrastructure evidence,
  not a scientific failure. V1 is policy-quarantined and must never be retried;
  removal from the runnable queue or an explicit controller refusal is still
  pending; see the
  [`pre-result implementation record`](../research/2026-08-30-constraint-aware-progress-toy-v1-pre-result-implementation.md).

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
[`AUTONOMOUS_LAB.md`](AUTONOMOUS_LAB.md) protocol. Its first ten terminal
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

The sixth checkpoint completed the systems-mechanics inventory. On the frozen
x64-enabled CPU runtime, an experiment-only pure-JAX no-restart transition
exactly matched 46 typed observation/state/telemetry leaves across eager, JIT,
and explicitly compiled execution. The protected route still has Objective
logging, scalar, budget, clock, callback, conditional-RNG, and Python-loop
boundaries. The result is not a runtime, accelerator, kernel-count, UIFO, or
candidate-improvement claim; see
[`2026-08-27-normal-path-jax-boundary-results.md`](../research/2026-08-27-normal-path-jax-boundary-results.md).

The seventh checkpoint established that first learning contract.
`supervised-toy-signal-v1` passed all eight frozen CPU cases: its deterministic
ridge surrogate scored `1.0` on every train, validation, test, and minimum
held-out regime macro; the constant baseline scored `0.5`, the frozen random
test baseline scored `0.54296875`, and both the within-block label-shuffle and
signal-ablation controls scored `0.5`. Its split, leakage, attribution, and
fresh-process controls also passed. This validates only a deliberately easy
synthetic supervised harness; see
[`2026-08-27-supervised-toy-signal-results.md`](../research/2026-08-27-supervised-toy-signal-results.md).

The eighth checkpoint established the immediate-reward bandit contract.
`contextual-bandit-toy-signal-v1` passed all nine frozen CPU cases: its fixed
empirical-reward table achieved the analytic `0.75` train reward and `256`
regret, then scored `1.0` on every held-out regime without updating. Both
constant policies scored `0.5`, the seeded-random test policy scored
`0.51171875`, and the shuffled-context and two signal-ablation variants scored
`0.5`. Its update-order, keyed-scoring, split, leakage, attribution, and
fresh-process controls also passed. This validates only a deliberately easy
synthetic contextual-bandit harness; see
[`2026-08-27-contextual-bandit-toy-signal-results.md`](../research/2026-08-27-contextual-bandit-toy-signal-results.md).

The ninth checkpoint established the two-step delayed-credit contract.
`two-step-delayed-credit-v1` passed all eleven frozen CPU cases. Its fixed
four-state empirical-return table had the analytic `0.25` behavior return and
`96` regret, then scored `1.0` on train and every untouched held-out regime.
Constants and the myopic policy scored `0.5`; the seeded-random validation/test
policy scored `0.203125 / 0.265625`. The transition-shuffle control scored
`0.0`, while reward-origin misalignment and both signal-ablation variants
scored `0.5`. Exact update order, keyed scoring, split isolation, leakage,
attribution, delayed-queue, and fresh-process controls passed; see
[`2026-08-28-two-step-delayed-credit-results.md`](../research/2026-08-28-two-step-delayed-credit-results.md).

The target-independent v2 plan was frozen before implementation, but its
implementation was rejected during pre-result audit. Its source, worker, plan,
generator commitment, thresholds, regimes, seeds, permutations, cases, and
development diagnostics are quarantined. Do not repair, register, execute, or
reuse `multistep-td-action-prefix-v2`; its full disposition and successor
requirements are in the
[`v2 preflight rejection`](../research/2026-08-28-multistep-td-action-prefix-v2-preflight-rejection.md).

The fresh [`multistep-td-action-prefix-v3` plan](../research/2026-08-28-multistep-td-action-prefix-v3-plan.md)
was frozen in a clean plan-only commit before implementation. Its exact
topology-independent fixture, dedicated network-disabled worker, 19-case
registry contract, source approvals, and controller allowlist were committed
at a separate clean pre-result boundary. The controller then invoked it exactly
once on local CPU. All nineteen cases passed and the controller returned to
`awaiting_study` with no approved study pending, no active cycle, no lease, and
failure streak zero. The durable aggregate evidence and exact claim boundary
are in the
[`V3 terminal result`](../research/2026-08-29-multistep-td-action-prefix-v3-results.md).

The fresh
[`online-sarsa-latched-choice-v1` plan](../research/2026-08-29-online-sarsa-latched-choice-v1-plan.md)
was rejected during hostile pre-result audit and never invoked through the
guarded controller. Its frozen schema requires several `_rejected` fields to be
exact integer counts while also requiring every `_rejected` field to be an
exact JSON Boolean, so no result can satisfy the complete contract. The
incomplete task-owned skeleton was removed; no worker, registry entry,
controller allowlist, source approval, private result, sidecar, state
transition, lease, terminal event, or evidentiary claim exists for this ID.
See the
[`v1 preflight rejection`](../research/2026-08-29-online-sarsa-latched-choice-v1-preflight-rejection.md).

The v1 plan, family, schedule, regimes, seeds, thresholds, cases, discarded
skeleton, and any development diagnostics are quarantined. Do not repair,
register, allowlist, execute, import, reuse, or select a successor against them.
The private controller remains `awaiting_study`, and the ten earlier terminal
studies remain the complete approved local evidence set.

The fresh
[`online-sarsa-latched-choice-v2` plan](../research/2026-08-29-online-sarsa-latched-choice-v2-plan.md)
was rejected during hostile pre-result audit and never invoked. The proposed
implementation contained asserted rather than exercised family, timing,
capability, trace, source-isolation, comparator, control, sanitizer, and
process-isolation proofs; its beacon-ablation control was also unreachable.
The entire uncommitted implementation and registry/controller wiring were
removed. There is no V2 fixture, worker, registry entry, allowlist, source
approval, learner execution, private result, sidecar, controller transition,
lease, terminal event, or evidence. See the
[`v2 preflight rejection`](../research/2026-08-30-online-sarsa-latched-choice-v2-preflight-rejection.md).

V1 and V2 remain rejected and quarantined. The fresh
[`online-sarsa-latched-choice-v3` plan](../research/2026-08-30-online-sarsa-latched-choice-v3-plan.md)
was also rejected during hostile pre-result audit and never implemented or
invoked. Its frozen isolation profile commits the complete loaded Python PE
module closure, including Windows KnownDLLs, while requiring every DLL and
parent to carry an exact controller-owned DACL. The actual core System32 DLLs
are TrustedInstaller-owned and cannot be made to satisfy that rule without an
unsafe destructive system mutation. No V3 fixture, worker, registry entry,
allowlist, source approval, learner execution, private result, sidecar,
controller transition, lease, terminal event, or evidence exists. See the
[`v3 preflight rejection`](../research/2026-08-30-online-sarsa-latched-choice-v3-preflight-rejection.md).

V1, V2, and V3 remain rejected and quarantined. The private controller remains
`awaiting_study`, and the ten earlier terminal studies remain the complete
approved local evidence set.

The README now reflects the Round-2 priority. This handoff remains the
authoritative live gate; the frozen V4 record is retained as historical work,
not a competing task list.

The
[`online-sarsa-latched-choice-v4` plan](../research/2026-08-30-online-sarsa-latched-choice-v4-plan.md)
remains frozen and intact after its three hostile read-only audits and
host-feasibility probes. It has no fixture, worker, registry entry, allowlist,
source approval, learner execution, private result, sidecar, controller
transition, lease, terminal event, or evidentiary claim. Public Round-1
feedback changed the research priority, so V4 is paused rather than failed,
amended, or quarantined. Do not implement or invoke it while the Round-2 gate
below is live.

The owner authorized a local Round-2 research pivot and a two-hour laboratory
cadence on 2026-08-30, then explicitly authorized infrastructure recovery and
resumption of local research on 2026-08-31. The controller remains `parked`
with no active cycle, stop marker, or lease. Do not clear it yet: V1 remains in
the current registry and a naive unpark could rerun the forbidden ID.

The live checkpoint is the frozen
[`constraint-progress-startup-forensics-v1`](../research/2026-08-31-constraint-progress-startup-forensics-v1-plan.md)
non-result diagnostic. Implement and hostile-audit only its standard-library
Windows Job/pipe probe, then execute it at most once outside the controller.
It must not import the V1 fixture or worker, observe optimizer metrics, or touch
the private lab root. A pass permits only removal of V1 from the runnable queue
or an explicit controller refusal plus a fresh V2 plan; V2 implementation and
result-bearing execution remain separate gates.

Keep `submission/`, every terminal study, all rejected fixtures, V4, and the
untouched topology panels unchanged. Candidate packaging, official data,
private outcome panels, accelerator benchmarking, paid compute, PR merge, and
portal interaction remain separate owner gates. No terminal, rejected, or
retired study will be rerun.

## Public deadlines

The currently published [official competition timeline](https://github.com/artificial-scientist-lab/Learn2Design-2026/blob/main/README.md#timeline)
lists optional public-leaderboard deadlines on 2026-08-26, 2026-09-12, and
2026-09-29 Anywhere on Earth, followed by the prize-determining final deadline
on 2026-10-15 AoE. Round 1 is now closed; the next published public deadline is
2026-09-12 AoE. Recheck upstream before any schedule-critical launch. The
owner-reported Round-1 result confirms that the baseline was evaluated, but it
does not identify hidden-topology outcomes or validate a local treatment.

## What can contribute now

The exact private-artifact reproduction command and generated-bundle layout are
in [`DEVELOPMENT_V2_RESULTS_HANDOFF.md`](DEVELOPMENT_V2_RESULTS_HANDOFF.md). The
durable aggregate result and exploratory evidence boundaries are in
[`2026-08-21-development-v2-a100-results.md`](../research/2026-08-21-development-v2-a100-results.md).

Useful unpaid work now follows the frozen Round-2 funnel: synthetic causal
mechanics, experiment-owned candidate audit, then a separately approved fresh
development panel. The first question tests whether the complete public
progress tuple--feasibility, penalty, and sensitivity loss--can improve
progress/restart decisions relative to the protected raw-total-loss clock
without selecting against the observed Stage-A panel. The
validated supervised toy-signal harness result and its strict learning-only
claim boundary are in
[`2026-08-27-supervised-toy-signal-results.md`](../research/2026-08-27-supervised-toy-signal-results.md).
The validated immediate-reward contextual-bandit harness and its strict
learning-only claim boundary are in
[`2026-08-27-contextual-bandit-toy-signal-results.md`](../research/2026-08-27-contextual-bandit-toy-signal-results.md).
The validated two-step delayed-credit harness, its model-specific caveat, and
its strict learning-only claim boundary are in
[`2026-08-28-two-step-delayed-credit-results.md`](../research/2026-08-28-two-step-delayed-credit-results.md).
The validated JAX boundary map and exact one-batch pure-transition result are in
[`2026-08-27-normal-path-jax-boundary-results.md`](../research/2026-08-27-normal-path-jax-boundary-results.md).
The validated full-surface information-boundary result that motivated the
systems rung is in
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
