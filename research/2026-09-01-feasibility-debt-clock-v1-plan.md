# Feasibility-debt restart clock v1 - frozen plan

Date: 2026-09-01

Study ID: `feasibility-debt-clock-v1`

Status: plan frozen before candidate implementation or result observation

## Decision question

Can an experiment-owned `BatchedRestartAdam` variant replace only the
pre-feasibility member progress clock with a clock driven by the public UIFO
constraint penalty, while preserving the protected optimizer's objective
calls, gradients, Adam updates, incumbent selection, budgets, timing guards,
and random draws until the first causally expected restart?

The falsifiable positive is deliberately mechanical:

1. an explicit `total_loss` compatibility mode is byte-identical to the
   protected optimizer on every frozen trace;
2. in `feasibility_debt` mode, a lane whose total loss improves while its
   constraint debt is flat reaches the frozen restart boundary, whereas the
   protected clock does not;
3. falling constraint debt prevents that restart;
4. after a lane first becomes feasible, its progress rule is exactly the
   protected total-loss rule; and
5. no other state or public-API boundary differs before the declared restart.

A pass supports only the implementation and causal isolation of this one
restart-clock treatment. It is not evidence that the treatment improves UIFO,
the Round-1 score, any hidden topology, or a submission.

## Independence and provenance

This is a new ordinary-runtime checkpoint. It does not execute through the
parked private laboratory controller and it does not reuse or select against
the family, schedules, seeds, thresholds, cases, source, raw output, or
development diagnostics of either retired `constraint-aware-progress` study.
The terminal isolated-runtime diagnostic is not rerun or repaired.

The treatment follows only three already validated public facts:

- UIFO's total loss is `sensitivity_loss + penalty`;
- its public auxiliary tree contains `is_feasible`, `penalty`,
  `sensitivity_loss`, `violations`, and grouped raw powers; and
- the protected optimizer currently resets a member's stall count whenever
  finite total loss improves by `minimum_improvement`.

The protected `submission/` tree, its defaults, the owner-uploaded archive,
all generated topology panels, private histories, and official data are
read-only and out of scope.

## Candidate boundary

Implementation is limited to a new module beneath `experiments/candidates/`
and focused tests. The class is named
`FeasibilityDebtBatchedRestartAdam`. It must expose the protected optimizer's
public arguments plus one required keyword, `progress_mode`, whose only legal
values are `total_loss` and `feasibility_debt`. No mode becomes a packaged
default in this study.

The candidate may inherit the protected helper methods but must own its
`optimize` loop so that its source can later be packaged without monkeypatching
or private runtime hooks. A checked source-delta verifier must reject changes
outside these exact semantic regions:

1. declaration of the three treatment progress arrays;
2. the full-batch member-progress transition;
3. reset of those arrays for restarted members;
4. treatment progress fields added to experiment-only telemetry; and
5. validation of `progress_mode` and the required auxiliary leaves.

Initialization, anchor construction, population sampling, learning rates,
objective evaluation, global feasible incumbent selection, partial-tail
handling, gradient sanitation and clipping, Adam arithmetic, time/budget
guards, restart sampling, restart alternation, callback order, and return
behavior must otherwise remain source- and trace-equivalent.

## Frozen progress transition

For population size `P`, treatment state contains:

- `ever_feasible: bool[P]`, initialized `False`;
- `best_infeasible_debt: float[P]`, initialized `+inf`; and
- `best_feasible_loss: float[P]`, initialized `+inf`.

The debt is exactly the scalar public `aux["penalty"]`. The candidate does not
read private problem attributes, topology metadata, histories, or hidden
constraint state. It makes no extra objective call.

For each member in a complete population batch:

```text
feasible_now = bool(aux.is_feasible)
valid_loss = isfinite(total_loss)
valid_debt = isfinite(penalty) and penalty >= 0

infeasible_improved = (
    not ever_feasible
    and not feasible_now
    and valid_loss
    and valid_debt
    and penalty < best_infeasible_debt - minimum_improvement
)

feasible_improved = (
    feasible_now
    and valid_loss
    and total_loss < best_feasible_loss - minimum_improvement
)

improved = infeasible_improved or feasible_improved
ever_feasible = ever_feasible or (feasible_now and valid_loss)
best_infeasible_debt = penalty if infeasible_improved else prior value
best_feasible_loss = total_loss if feasible_improved else prior value
stall = 0 if improved else stall + 1
```

Once `ever_feasible` is true, later infeasible observations cannot reset that
member's clock. When a member restarts, all three treatment fields reset to
their initial values, just as the protected member-best loss and Adam age do.
An evaluation-limited partial tail is logged but does not mutate progress,
apply an update, draw restart randomness, or trigger a restart.

In `total_loss` mode, progress state and transitions are exactly the protected
`member_best_loss` and `stalled_steps` rules, including non-finite behavior.

## Typed auxiliary contract

Before a treatment progress transition, the evaluated full batch must contain:

| path | type and shape |
| --- | --- |
| `loss` | finite-or-nonfinite floating array `[P]` |
| `gradient` | floating array `[P,D]` |
| `aux.is_feasible` | Boolean array `[P]` |
| `aux.penalty` | floating array `[P]` |
| `aux.sensitivity_loss` | floating array `[P]` |
| `aux.violations` | floating array `[P,C]`, `C >= 1` |
| `aux.power_values.hard` | floating array with leading dimension `P` |
| `aux.power_values.soft` | floating array with leading dimension `P` |
| `aux.power_values.detector` | floating array with leading dimension `P` |

Missing leaves, incorrect leading dimensions, Boolean impostors, or negative
finite penalties fail closed before any optimizer update or restart decision.
Non-finite penalties are valid observations but cannot count as progress.
`total_loss` compatibility mode must accept exactly what the protected path
accepts and may not add treatment-only validation.

## Fresh deterministic fixtures

All fixtures are synthetic and newly specified here. They use dimension `D=2`,
population `P=3`, learning rates `0.04` through `0.12`, gradient clip `1.0`,
restart scale `0.25`, safety seconds `0`, and `minimum_improvement=1e-7` unless
a case states otherwise. The fixed optimizer seeds are:

| case family | seed |
| --- | ---: |
| compatibility/no restart | 91021 |
| compatibility/restart | 91031 |
| flat debt | 91033 |
| falling debt | 91079 |
| feasibility switch | 91121 |
| partial tail and chunks | 91139 |

The seeds were chosen before execution and are not selected against any
observed metric.

Each scripted batch returns the same row value for all three members, plus a
fixed gradient row `[0.6,-0.8]`. `sensitivity_loss` is exactly
`total_loss-penalty`; `violations` is `[penalty]`; each power group is the
typed one-column value used only to authenticate pass-through. The evaluator
records candidate parameters, all returned leaves, evaluation counts, callback
order, and random-sample commitments.

### Complete cases

1. `protected_compatibility_no_restart`: five full batches with total losses
   `[9,8,7,6,5]`, penalties `[3,2.5,2,1.5,1]`, all infeasible, patience 8.
   Protected and candidate `total_loss` projections must be byte-identical.
2. `protected_compatibility_restart`: five full batches with total loss and
   penalty both constant at `4`, all infeasible, patience 2. Protected and
   candidate `total_loss` projections, including restart draws and post-restart
   states, must be byte-identical.
3. `flat_debt_divergence`: five full batches with total losses
   `[8,7,6,5,4]`, penalty `2` throughout, all infeasible, patience 3. The
   protected clock records an improvement on every batch and never restarts.
   Treatment stalls are `[0,1,2,3,0]` per member and its only first-generation
   restart occurs after zero-based batch 3. Before that restart, parameters,
   losses, gradients, Adam states and ages, global incumbent state, budget,
   callback order, and random transcript are equal; only declared progress
   state and stall fields may differ.
4. `falling_debt_control`: five full batches with total losses
   `[8,7,6,5,4]` and penalties `[3,2.5,2,1.5,1]`, all infeasible, patience 3.
   Treatment stalls remain zero and no restart occurs.
5. `feasibility_switch_control`: six full batches with
   `(loss,penalty,feasible)` equal to `(8,3,F)`, `(7,2,F)`, `(6,0,T)`,
   `(5,0,T)`, `(5,0,T)`, and `(5,1,F)`, patience 3. Treatment improvements are
   `[T,T,T,T,F,F]`, `ever_feasible` becomes true only on batch 2, and the final
   infeasible observation cannot reset the clock.
6. `partial_tail_control`: three full batches followed by a two-member partial
   tail under an eleven-evaluation budget and patience 2. The tail consumes the
   exact remaining budget, emits no update or restart, and leaves all progress
   state equal to the state after the third full batch.
7. `chunk_projection_equivalence`: replay cases 4 and 6 with chunk sizes
   `None`, `1`, and `2`. Complete logical-batch projections must be byte-identical
   after reassembly and evaluation counts must not overshoot.
8. `auxiliary_fail_closed_matrix`: independently delete each required leaf,
   change every required leading dimension, substitute integer feasibility,
   supply negative finite penalty, and supply a scalar penalty for `P>1`.
   Every attack must be rejected before update/restart state changes. A NaN
   penalty must be accepted as an observation but produce `improved=False`.
9. `source_delta_and_process_replay`: the source-delta verifier accepts only
   the declared regions. Two fresh CPU processes reproduce the complete
   sanitized case projection byte-for-byte with zero stderr and no network,
   file, subprocess, or credential access from the fixture.

## Result contract and stopping rule

The sanitized projection contains only the study ID, exact source revisions
and hashes, Boolean case outcomes, exact counts, first treatment restart batch,
mode-parity root, pre-divergence root, process-replay root, and final action.
It contains no topology, candidate vector, raw trajectory, environment value,
credential, or timing sample.

The study passes only if all nine complete cases pass exactly. The success
action is:

```text
approve_feasibility_debt_candidate_for_fresh_panel_planning
```

Any mismatch, malformed output, nondeterminism, unapproved source delta,
unexpected file/network/process effect, or test failure has action:

```text
park_feasibility_debt_candidate
```

There is no threshold relaxation, case removal, seed swap, repair against an
observed result, or second terminal execution. Implementation conformance fixes
are allowed only before the committed pre-result boundary.

## Verification sequence

1. Commit this plan and the minimal handoff change; that commit is the freeze
   boundary.
2. Implement the experiment-owned candidate, source verifier, fixture, and
   focused tests without running result-bearing cases.
3. Commit a clean pre-result implementation boundary.
4. Run focused static and contract tests, then at most one full repository
   pass if the focused boundary is clean.
5. Invoke the complete deterministic projection once in the ordinary locked
   local CPU environment.
6. Record only the sanitized terminal result and never rerun this ID.

No private controller resume is needed or authorized. No GPU, provider,
official dataset, private panel, protected artifact, portal, merge, or paid
endpoint is used.

## Route to a new number

Even a complete pass does not produce a score comparable to the owner-reported
Round-1 `0.444293`. It permits the next plan-only gate: freeze a newly generated,
archive-disjoint paired development screen comparing the protected baseline
and this experiment-owned candidate under identical topologies, seeds, runtime,
and hardware. That screen will produce a real generated-panel mean loss and
paired difference, but only a later public-leaderboard evaluation can produce
a hidden score directly comparable to `0.444293`.

Before any accelerator screen, the exact panel commitment, run count, runtime,
device, promotion rule, untouched follow-up set, provider-cleanup rule, and
maximum dollar charge must be frozen and separately approved by the owner.

## Claim boundary

This study may establish only that a public-penalty progress clock is correctly
implemented, causally isolated, deterministic, and ready for a fresh generated
panel. It cannot establish that constraint debt predicts future feasibility,
that earlier restarts help UIFO, that the candidate beats the protected
optimizer, or that any score moves toward `0.14`. It includes no official data,
private outcome evidence, generated topology, model training, RL, accelerator,
paid compute, packaging change, or portal action.
