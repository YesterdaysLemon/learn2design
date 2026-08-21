# Patience-200 restart screen plan — 2026-08-21

## Decision

The next optimizer question is whether the existing no-prior candidate's restart
threshold is too high to operate inside a 600-second evaluation. The current
packaged candidate remains unchanged at patience 600. No performance advantage
for patience 200 has been established.

Run two separately frozen stages:

1. one instrumented, outcome-selected mechanics run that may answer only
   whether a patience-200 restart fires and resets correctly; then
2. only after the mechanics gate passes, an uninstrumented paired development
   screen comparing patience 200 with patience 600 on eight topologies and two
   fresh optimizer seeds.

The untouched `submission-like-v1` panel remains reserved for a later gate. A
development-screen pass cannot authorize a submission by itself.

This report contains aggregates and configuration only. Source histories,
candidate arrays, telemetry, logs, device identifiers, and generated outputs
remain private and outside Git.

## Why this mechanism

Authenticated `development-v2` no-prior histories were audited without treating
members, batches, or optimizer seeds as independent topologies. Aggregate
mechanics were:

- 341–426 Objective calls per run, median 390.5;
- first finite-feasible result at median 110.483 seconds, with 30/32 runs
  feasible in the first population;
- final best result at median 541.266 seconds;
- 32/32 runs improved after 300 seconds and 25/32 improved after 480 seconds;
- median gain after 480 seconds was 0.106268;
- optimizer-seed absolute final-loss difference had median 0.167837 and maximum
  0.647653.

The configured patience of 600 was unreachable in every archived 600-second
run; the largest observed per-member stall count was 305. Retrospective replay
showed why a smaller threshold must still be conservative:

| Patience | Runs with a predicted trigger | Member traces | Eventual winning-member paths interrupted |
| ---: | ---: | ---: | ---: |
| 150 | 31/32 | 90/256 | 4/32 |
| 175 | 27/32 | 57/256 | 1/32 |
| 200 | 18/32 | 27/256 | 0/32 |

Patience 200 is therefore the smallest bounded intervention supported by this
diagnostic. It is post-hoc selected and must be tested on fresh stochastic
paths. The replay does not predict a performance benefit.

## Independent model review

A read-only OMP council used Qwen as proposer and Claude as hostile reviewer.
Only checked-in code and aggregate evidence were made available; no private raw
artifact was uploaded or sent to either model. The useful consensus was to
instrument the existing optimizer and change one mechanism at a time. A more
aggressive patience-100/150 proposal and an unsupported claim about Adam reset
bias were rejected after comparison with the authenticated trajectories and
implementation.

Council reports are retained outside Git as temporary host-local review
artifacts. They are advisory; the frozen local plan and tests are authoritative.

## Stage 1 — outcome-selected mechanics only

Configuration:

- panel: `restart-mechanics-v1`;
- topology: `FEECAACFB-LSLSLSLSHSLS`;
- optimizer seed: 11;
- arm: `no_prior_p200`;
- budget: 600 seconds, population 8, full-vmap;
- telemetry: `member-v1`;
- predicted first trigger from the archived path: 400.3757300376892 seconds;
- worker timeout: 1,200 seconds;
- maximum accepted worker wall time: 825 seconds.

This topology/seed was chosen because it had the earliest retrospective p200
trigger. Its loss is excluded from every effect estimate, plot, promotion
criterion, and later panel. The mechanics stage passes only if all artifacts
validate, a restart occurs at the exact 200-stall boundary, the member state and
Adam age/stall reset and generation increment are coherent with the checked-in
moment-reset branch, and at least one full post-restart evaluation is recorded.
Any error, timeout, missing post-restart observation, invalid artifact, or wall
time above 825 seconds retains patience 600.

Telemetry is disabled for all performance-scored runs because its host transfers
occur inside the Objective clock.

## Stage 2 — paired performance screen

### Outcome-blind topology selection

Exclude the mechanics topology, then rank the remaining `development-v1`
topology strings by

```text
SHA256(UTF8("L2D-PATIENCE-200-SCREEN-V1-2026-08-21" + NUL + topology))
```

and take the first eight. The committed panel contains the domain, ordered
topologies, and every selection digest. Tests recompute the rank and verify
zero overlap with `submission-like-v1`.

### Configuration

- arms: `no_prior_p600` and `no_prior_p200`;
- optimizer seeds: 19 and 23, freshly paired within every topology;
- 8 topologies × 2 seeds × 2 arms = 32 serial runs;
- 600 seconds per run, population 8, full-vmap;
- exact arm order alternates by topology index plus seed index, giving 8/8
  overall arm-first balance and 4/4 balance within each seed;
- all optimizer settings except patience are byte-for-byte equal;
- no optimizer telemetry.

Seeds 19/23 do not occur in the authenticated `development-v2` run records,
which used only 7/11, and repository evidence contains no prior scored use of
19/23 for this candidate. Archived 7/11 controls are context only and are not
pooled into the new estimand.

Topology remains the inference unit (`n=8`). The two seeds are repeated paired
measurements. Do not inspect performance after four topologies; even four of
four wins has a minimum two-sided exact sign probability of 0.125 and cannot
support promotion.

### Frozen decision

Define each topology value as the mean over seeds of

```text
best feasible loss(patience 200) - best feasible loss(patience 600)
```

Negative values favor patience 200. Advance only if every condition holds:

- 32/32 runs, 16/16 seed pairs, and 8/8 topology blocks validate;
- all 16 seed pairs are finite-comparable in both arms; treatment-only,
  control-only, or neither-finite censoring is not evaluable for this loss gate
  and retains patience 600;
- zero control-only finite-feasible seed pairs;
- no topology has a lower finite-feasibility rate under patience 200;
- at least 6/8 topology-level wins;
- topology median difference is at most -0.05;
- topology mean difference is below zero;
- topology p90 regret is at most 0.5;
- the mean paired difference is below zero separately for seed 19 and seed 23.

Otherwise retain patience 600. The deterministic 10,000-resample topology
bootstrap, all `2^8` mean sign-flip assignments, separate exact sign test,
leave-one-topology-out results, within-topology arm-first contrast, serial
plan/session-order correlations, and evaluation-throughput log ratios are
reported as exploratory sensitivity analyses only. They cannot replace or
override the frozen rule. A screen pass means only “plan an untouched
submission-like gate”; it does not directly change the submission package.

## Cost and stop rules

The screen contains 5 hours 20 minutes of scored Objective time. Historical
worker overhead projects about 6.5 rental hours including mechanics, setup,
validation, and packaging. At the live secure A100 prices checked during
planning, the estimate was about $9.04 at $1.39/hour or $10.34 at $1.59/hour.
The provider-side eight-hour ceiling bounds those rates at $11.12 and $12.72.

Immediately before provisioning:

- require a visible Runpod balance of at least $15;
- require price at most $1.60/hour;
- configure a provider-native stop at eight hours;
- bind that UTC stop time and a 30-minute evacuation reserve into both plans;
- reject a deadline more than eight hours ahead, allowing only 60 seconds of
  clock/setup tolerance;
- refuse the screen unless at least seven hours remain before that stop;
- stop after the mechanics stage on any failed criterion;
- stop the scored stage at its first worker failure or integrity failure;
- terminate the GPU immediately after the locally received package and sidecar
  pass checksum and ZIP validation.

Do not run unrelated compute, add an optimizer/model, use spot capacity, or
adapt the panel/rule after observing loss.

The screen plan additionally authenticates and binds the complete passed
mechanics package, mechanics plan ID, exact Git revision, package/manifest
digests, record digest, history digest, and telemetry digest. It cannot be
constructed through the official CLI before the mechanics gate passes. Both
restart profiles are non-resumable so a first failure or elapsed-time guard
cannot be erased by a new invocation.

## Evidence boundary

This plan targets one plausible implementation bottleneck: restarts never fire
under the current competition-screen budget. It does not establish that
restarts help, that patience 200 is optimal, or that the retained candidate is
submission-ready. Seed instability and topology heterogeneity remain material.
The next evidence gate after this screen must follow its frozen action and must
not reuse the development panel as confirmation evidence.
