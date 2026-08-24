# H100 coverage-robustness screen — frozen pre-result design

> Superseded as the first paid gate by the lower-cost
> [`2026-08-24-h100-coverage-triage-plan.md`](2026-08-24-h100-coverage-triage-plan.md).
> This original design remains frozen for provenance; its panel is reserved by
> the Stage-A policy for a separately approved independent Stage B.

## Status and evidence boundary

This document defines a pre-result development screen. It contains no UIFO
outcomes for the treatment below and does not authorize paid compute.

The owner-reported Round-1 submission remains the immutable baseline:

- project revision: `5ce3cdb2ddf4c505622a0aeef805936a4ea607d7`;
- ZIP SHA-256:
  `4cc0dbc65a3e61ca5358c18655c432caf478fbdfc07f10512553781f8822924b`;
- default initializer: feasibility anchor plus seven seeded random members;
- patience: 600;
- compilation charged to the Objective clock.

The local `artifacts/generated/submission.zip` and its manifest retain those
bytes. The experimental source path is separate and still defaults to the
Round-1 behavior. A treatment is selected only by the paired harness.

## Official environment delta

The official starter repository advanced from the previously pinned
`d9b1bd7d6f2c4df335bc7725755b02aa5f6f942c` to
`1bb7f54737dec6a08b59879a8831d125f08f8a0b`. The submission interface is
unchanged, but the official evaluator now specifies one NVIDIA H100 SXM5 80 GB,
16 Intel Xeon 8468 vCPUs, 200 GiB RAM, Ubuntu 22.04, and CUDA 13. The current
Objective contract explicitly permits its public `warmup_*` methods before
`start_logging()`.

Historical A100 evidence remains valid for the comparisons it measured. It is
not current hardware-aligned evidence and is not silently relabeled as H100
evidence.

## Causal question

Does deterministic marginal coverage of the seven non-anchor starts reduce
initial-population sensitivity relative to the current seeded random starts?

Only initialization mode differs between arms:

| Property | Control | Treatment |
| --- | --- | --- |
| Arm | `no_prior` | `coverage_balanced` |
| Member 0 | feasibility anchor | identical feasibility anchor |
| Members 1–7 | existing random draw | midpoint Latin-hypercube transform of that draw |
| Optimizer | BatchedRestartAdam | identical |
| Population | 8 | 8 |
| Patience | 600 | 600 |
| Restarts | unchanged | unchanged |
| Scored budget | 1,200 seconds | 1,200 seconds |
| Pre-clock warmup | enabled | enabled |

For a random suffix with `m` members, each parameter column is ranked with a
stable sort and mapped to the unit-space levels

```text
(rank + 0.5) / m,  rank = 0, ..., m - 1.
```

The levels are then mapped back to the existing unbounded logit coordinates.
This is a midpoint Latin hypercube in every marginal parameter dimension. It
uses all seven available random slots, unlike exact antithetic pairing, which
would leave an unpaired singleton after reserving the anchor.

Both arms call `Objective.random_params_unbounded(population_size)` exactly
once before the first evaluation. The transform performs no additional random
draw. Therefore the Objective RNG state immediately after initialization is
identical. Divergent later restart timing is part of the treatment's downstream
effect and is not described as a shared RNG path.

Every scored record retains SHA-256 hashes of the seven pre-transform suffix
members actually used by both arms. The discarded random slot replaced by the
common anchor is intentionally outside this claim. The sealed validator
requires exact suffix-hash equality across each pair, requires the control
suffix to match those hashes, reconstructs the treatment's exact rank mapping
from the authenticated control suffix, and requires the anchor hashes to agree.

The opt-in screen also blocks the completed population ready before logging in
both arms. This removes treatment-only rank/logit dispatch from the first
scored evaluation while leaving the packaged/default Round-1 timing path
unchanged.

The common warmup calls the public Objective helper for every evaluation batch
shape used by the configured full-vmap or diagnostic chunk path, then calls
`start_logging()`. The frozen screen uses only full-vmap batch size 8.
`dfbench==0.3.3` discards the returned JAX arrays, so asynchronous device work
is not guaranteed complete when the helper returns. This is therefore an
identical best-effort compilation warmup in both arms, not a proven completion
barrier or an assertion that all warmup execution is outside the Objective
clock. Because the policy is common, it cannot explain the paired loss
difference.

## Frozen panel and pairing

Panel: `coverage-robustness-v1`.

- 12 explicit size-3 UIFO topologies;
- source SHA-256:
  `e3385a6f4939445e869d71dbf7f1bd5119aa25eebbb89e083b1ec9be336f7309`;
- deterministic round-robin selection across readout, squeezer-count, and
  directional-interior strata;
- exactly 6 detector and 6 homodyne readouts;
- exactly 4 low, 4 middle, and 4 high members on each balance axis;
- zero exact topology overlap with the official archive, `development-v1`,
  `confirmation-v1`, and `submission-like-v1`;
- the restart panels are subsets of `development-v1`, so this also excludes
  every previously scored restart topology.

Fresh optimizer seeds are `[37, 41]`. An outcome-blind filename/configuration
scan of the repository and the external private evidence root found no prior
scored use of either seed on 2026-08-24.

Pairing is fixed as:

```text
pair_order_policy = alternate_topology_and_seed
seed_order_policy = mirrored_sweeps
```

Seed 37 traverses the panel forward and seed 41 backward. Every topology gets
one random-first and one coverage-first pair. Across 24 topology-seed pairs,
each arm runs first exactly 12 times.

## Frozen execution profile

| Field | Value |
| --- | --- |
| Profile | `coverage-robustness-screen-v1` |
| Policy | `coverage-robustness-development-screen-v1` |
| Runs | 12 topologies × 2 seeds × 2 arms = 48 |
| Execution | serial, one visible GPU |
| GPU | exact `NVIDIA H100 80GB HBM3`, MIG disabled |
| Minimum GPU memory | 75,000 MiB |
| Objective budget | 1,200 seconds per run |
| Total scored time | 57,600 seconds = 16.0 H100-hours |
| Worker timeout | 2,100 seconds |
| Session ceiling | 22 hours |
| Provider stop horizon | at most 26 hours, with 30-minute evacuation reserve |
| Population | 8, full-vmap |
| Frequencies | 50 |
| Persistent JAX cache | disabled |
| Telemetry | none |
| Target losses | 4.0, 1.0, 0.5, 0.0 |
| Failure allowance | one terminal worker failure; no resume or rerun |

The 1,200-second screen is bounded development evidence, not an official
four-hour performance claim.

## Purchase envelope

The read-only Runpod pod catalog on 2026-08-24 reported the exact H100 SXM type
as available at low stock, with CUDA 13 availability and secure-cloud price
`$3.29/GPU-hour`. The frozen provider envelope is:

- secure cloud;
- maximum hourly GPU price: `$3.29`;
- maximum provider time: 22 hours;
- maximum total provider charge: `$75.00`;
- one exact H100 SXM GPU;
- no spot/preemptible substitution, H100 PCIe, H100 NVL, A100, or multi-GPU
  split.

Catalog price and stock must be refreshed immediately before authorization. If
either exceeds the frozen envelope, do not provision and do not silently alter
the experiment.

## Estimand and frozen decision

For topology `t`, with seeds treated as repeated measurements:

```text
D_t = mean_seed(
    best_feasible_loss(coverage_balanced)
    - best_feasible_loss(no_prior)
)
```

Negative values favor coverage balancing. Topology is the inference unit.

Promotion is only to a separately frozen official-budget confirmation. Every
criterion below must pass:

1. All 48 records complete and revalidate; all 24 seed pairs and all 12
   topology blocks are complete.
2. Every run is physically feasible and finite-feasible, so all 24 paired
   differences exist.
3. At least 9 of 12 topology values satisfy `D_t < -1e-12`; absolute values at
   most `1e-12` are ties.
4. Median `D_t <= -0.05`.
5. Mean `D_t < 0`.
6. The seed-37 and seed-41 mean differences are each below zero.
7. The random-first and coverage-first stratum mean differences are each below
   zero.
8. The topology p90 regret is at most `0.5`.
9. The median seed-pair evaluation-count ratio, coverage over random, is at
   least `0.95`.
10. For every topology, total treatment evaluations divided by total control
    evaluations across its two seeds is at least `0.90`.

If the complete valid screen misses any criterion, retain the submitted random
start. A worker error, interruption, integrity failure, or terminal partial
attempt is `not_evaluable`; preserve it and do not rerun under a new plan or
select a replacement initializer after inspecting outcomes.

Bootstrap output is descriptive only and cannot override this decision.

## Pre-launch gates

Paid execution remains blocked until all of the following are true:

- an encrypted second copy of the private historical evidence bundle exists
  and verifies;
- the experimental revision is clean and reviewed;
- the complete local test suite and deterministic alternate-path package build
  pass;
- the exact candidate package, source files, revision, panel, live exclusion
  audit, provider stop, and plan SHA-256 are bound outside Git;
- the exact 249-member outcome-blind validator, exact chronology and used-suffix
  raw-draw-to-treatment checks, zero-project-import history-first replay,
  replay comparator, and comparator-issued summary-unlock CLI pass their
  synthetic archive and hostile-review tests;
- the owner explicitly approves at most `$75.00` total provider spend, including
  the mechanics smoke;
- after approval and provisioning, a cold H100 mechanics smoke confirms exact
  device identity, CUDA 13, memory fit, public warmup dispatch before logging,
  and admitted history without interpreting loss; it must not claim a
  device-completion barrier, and any failure stops the paid attempt.

The local sealed-replay implementation, hostile forged-chronology and
unrelated-LHS regressions, synthetic 249-member archive test, and
whole-repository verification were complete on 2026-08-24. A clean committed
candidate package, encrypted second evidence copy, live provider identity, the
cold H100 smoke, and explicit spend approval remain pre-launch gates. No paid
resource was created while writing or implementing this plan.
