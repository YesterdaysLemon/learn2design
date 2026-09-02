# Feasibility-debt candidate screen v1 - frozen pre-result plan

Date: 2026-09-01

Study ID: `feasibility-debt-candidate-screen-v1`

Status: frozen by the commit containing this file. No official dataset bytes or
private outcomes have been opened for this study; no panel, implementation,
GPU resource, scored run, result, or candidate package exists.

## Predecessor, rationale, and narrow question

The one terminal local-CPU invocation of `feasibility-debt-clock-v3` passed all
nine frozen cases twice at revision
`6d2a36f5a41687dd8e38a2aa15120c84dc9b535b`. Its process replay root is
`c440ede7eb07afd889f907bc3326df55b01c3af473a729de3933fa3d65590b8d` and
its candidate source SHA-256 is
`ca7abd365c5d1172dab2f47fccdf0afa3df9652e75cc2003385312cec48844d6`.
That result validates only the synthetic transition and transport. V3 is
terminal and will never be rerun.

The paid-screen question is:

> On a new deterministic eight-topology panel, can Stage 1 select one complete
> predeclared challenger that then beats the exact Round-1 lifecycle on all
> four untouched Stage-2 topologies and by at least `0.05` mean paired loss?

Stage 1 selects; it does not confirm. The only promotion evidence is the four
untouched Stage-2 topology pairs. Stage-1 and pooled eight-topology quantities
are selection-conditional descriptions and can never rescue a failed Stage-2
gate.

This is not a retry or top-up of a terminal patience or coverage study. V3 is a
materially new mechanism: before first feasibility, public non-negative
penalty is the restart-progress clock; after first feasibility, control passes
permanently to the protected total-loss clock. Patience 200 therefore has a
new pre-feasibility meaning. C and D below are complete new bundles, the old
patience and coverage decisions remain final, every old panel is excluded, and
no old result selected this panel, seed, threshold, or rule. D is explicitly an
exploratory composition and receives no evidentiary credit from the older
coverage result.

A pass can justify owner review of one experiment-owned bundle for Round 2. It
cannot establish a hidden score, four-hour performance, general superiority,
causal attribution to one bundle component, or progress from the owner-reported
`0.444293` toward `0.14`.

## Exact arms and lifecycle identities

Every arm uses population 8, one feasibility anchor, seven suffix members,
full-vmap evaluation, no semantic prior, and these optimizer settings except
for the stated patience:

```text
learning_rate_low = 0.03
learning_rate_high = 0.15
minimum_improvement = 1e-7
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
gradient_clip_norm = 1.0
restart_noise_scale = 0.35
safety_seconds = 2.0
batch_time_safety_factor = 1.5
batch_time_window = 8
```

The frozen optimizer-settings source is
`experiments/uifo_paired/optimizer_settings.py` at SHA-256
`bba97635abcc1599257826bb6eb775fe3cdeb6f9ddb98fb615397d353642f858`.

| ID | Module and class | `algorithm_str` | Patience | Initial suffix | Warmup | Progress |
| --- | --- | --- | ---: | --- | --- | --- |
| `A_round1_control` | exact ZIP member `submission.py:BatchedRestartAdam` | `batched_restart_adam` | 600 | seeded random | false | protected total loss |
| `B_round1_warmup` | `submission.submission:BatchedRestartAdam` | `batched_restart_adam` | 600 | seeded random | true | protected total loss |
| `C_v3_random` | `experiments.candidates.feasibility_debt_clock_v3:FeasibilityDebtBatchedRestartAdamV3` | `feasibility_debt_batched_restart_adam_v3` | 200 | seeded random | true | `feasibility_debt` |
| `D_v3_coverage` | same V3 module and class | `feasibility_debt_batched_restart_adam_v3` | 200 | coverage-balanced transform of the same seeded random suffix | true | `feasibility_debt` |

Arm A is loaded in an isolated module namespace from the authenticated archive,
not substituted with the later current class. The complete owner-reported
Round-1 identity is:

- evaluated repository revision:
  `5ce3cdb2ddf4c505622a0aeef805936a4ea607d7`;
- `artifacts/generated/submission.zip` SHA-256:
  `4cc0dbc65a3e61ca5358c18655c432caf478fbdfc07f10512553781f8822924b`;
- protected manifest SHA-256:
  `99e18299ba6dd232f203cc2e59131cea61cd83ff8b0a9bceab6c56c764ddd86a`;
- exact three archive members only: `requirements.txt`
  `776cf62a1ac0727d4975a6434936449ff008540a3cfbbe5ddad579ecfd9e23d1`,
  `semantic_prior.json`
  `c08cefb94f0285d9681ab8125c23545cc93c7231d0b5aefe849a80c74a4f4312`,
  and `submission.py`
  `45f18e17f3e9e0855629079c311a4c01318a3d2e6700db4719e2a54d61ffea76`.

Arm B binds the current protected source SHA-256
`0fefbaaf18d9831895d788df45c92cbaf4522da7c54d8f78646e449ffa9374c9`.
Arms C and D bind the V3 source hash above and the required exact keyword
`progress_mode="feasibility_debt"`. The coverage transform may change only the
seven non-anchor suffix rows.

Before launch, a separate immutable source lock must bind, for every arm, the
module bytes, class, `algorithm_str`, complete kwargs, population mode,
progress mode, package/runtime closure, worker, runner, analyzer, and panel.
Each generated config must repeat and authenticate those fields. A missing,
extra, or mismatched field makes the attempt not evaluable.

The public rules permit Objective-provided warmup before `start_logging()`.
Warmup is part of the B/C/D treatment bundles, not a separately identified
cause. It may compile only the exact value/gradient/auxiliary shapes. It may not
sample randomness, inspect or emit a loss, mutate parameters, advance the
Objective clock or evaluation count, create history, or alter budget state.
Capture-only instrumentation around the returned raw population must prove the
same raw-population byte hash for A/B/C within each topology/seed pair and the
same pre-transform raw draw for D. Parameter, RNG/draw-count, clock, history,
evaluation-count, and budget-state receipts must be identical immediately
before and after warmup. Source review must independently prove that the
warmup path contains no random sampling.

Evaluation counts and evaluation rates are reported descriptively. They are
not selection or promotion gates because warmup and implementation form part
of the complete treatment bundle.

## Canonical locks and receipts

Every lock and receipt is path-free canonical JSON. Its exact top-level schema
is
`{"payload":object,"receipt_type":string,"schema_version":1,"study_id":string}`
with no extra key. `study_id` is the exact study ID above. Bytes are Python
`json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
allow_nan=False).encode("utf-8")` with no BOM or trailing newline. A sibling
sidecar contains lowercase `SHA256(receipt_bytes)` followed by exactly two
spaces, the fixed basename, and LF. Parsing, canonical reserialization, sidecar
verification, exact key sets, and expected receipt type must all pass.
Fixed basenames are `source-lock.json`, `runtime-lock.json`,
`panel-commitment.json`, `split-receipt.json`, and `selection-receipt.json`;
each sidecar appends `.sha256` to that basename.

The source-lock payload has exactly `revision`, `arm_profiles`, `sources`,
`runtime_lock_sha256`, `worker_sha256`, `orchestrator_sha256`,
`production_analyzer_sha256`, `reference_analyzer_sha256`, and
`panel_commitment_sha256`. `arm_profiles` is ordered A, B, C, D and repeats the
module ID, class, `algorithm_str`, exact kwargs, population mode, progress mode,
source digest, and package-closure digest from this plan. `sources` is sorted by
logical ID and every row has exactly `logical_id`, `sha256`, and `size_bytes`.
Arm A's normalized logical module ID is `round1_zip::submission.py`; its fixed
isolated Python module name is `l2d_round1_control_submission`. Neither host
path enters an identity or digest preimage.

The split-receipt payload has exactly `panel_sha256`, `candidate_rows`,
`legal_split_rows`, `chosen_stage1_indices`, `chosen_stage2_indices`,
`stratum_counts`, and `independent_verification`. The selection-receipt payload
has exactly `panel_sha256`, `split_receipt_sha256`, `stage1_archive_sha256`,
`ordered_run_ids`, `challenger_rows`, `eligible_ids`, `finalist`, `action`, and
`stage2_outcome_opened`. Run IDs are in frozen execution order; candidate and
split rows are in generator/index order; `eligible_ids` follows B, C, D
priority. `finalist` is one exact arm ID or null. Every generated config binds
the applicable lock and receipt SHA-256 values.

## Outcome-blind panel and split commitment

The panel ID is `feasibility-debt-candidate-screen-v1`. Its eight size-3 UIFO
topologies are a deterministic function of only committed public topology
metadata and exact exclusion sets:

- official archive SHA-256:
  `149f6aac17aff2e33750b4e1b6cebd3cef1c39d47ae49a3a7ed77315cb7838a7`;
- topology generator upstream reference:
  `1bb7f54737dec6a08b59879a8831d125f08f8a0b`;
- frozen reference feature/generation source `tools/build_topology_panels.py`
  SHA-256:
  `b3a85a9f66a2f35f13f143da3c580c91fe5a642b489e4940841306cae9a91229`;
- candidate topology-seed range: exactly the 4,096 consecutive integers
  beginning at `2026090100`;
- exact topology-string exclusion against the official archive and all seven
  checked-in prior panels;
- de-duplication by exact topology string before feature selection; and
- the existing public `readout`, `squeezer_bin`, and `directional_bin`
  functions and round-robin stratum order, with no outcome or parameter field
  loaded.

The first eight eligible unique candidates in the exact round-robin output
order form the panel; "generator order" nowhere means raw seed order.
Stage membership is then chosen without outcomes by enumerating all 70 four-of-
eight Stage-1 subsets. A legal split has exactly two D and two H readouts per
stage and, for every squeezer and directional bin, an absolute stage-count
difference at most one. Among legal splits, minimize lexicographically:

1. the absolute difference between Stage-1 and Stage-2 sums of zero-based
   selection indices;
2. the absolute difference between Stage-1 and Stage-2 sums of generator seed
   integers; and
3. SHA-256 of canonical JSON containing the sorted Stage-1 indices.

Canonical JSON is exactly the UTF-8 bytes, with no BOM or trailing newline, of
Python `json.dumps({"stage1_indices": indices}, sort_keys=True,
separators=(",", ":"), ensure_ascii=True, allow_nan=False)`. The last item is
an ascending hexadecimal tie-break. Generator-seed-sum imbalance in item 2 is
the only lineage balance claim; Stage 2 is a mechanically balanced convenience
holdout, not a representative-sample or population-inference claim. If no
legal split exists,
the study is not evaluable; there is no replacement, expanded range, or manual
choice. The committed receipt must list generator order, topology lineage,
readout and bins, every legal split and objective tuple, the chosen split, and
per-stage stratum counts.

The frozen prior-panel hashes are:

| Panel | SHA-256 |
| --- | --- |
| `development-v1.json` | `d5f660261e413f59b179d4fadf1f157b30f117aa265fd230d1d130bd6d69246b` |
| `confirmation-v1.json` | `52fe189709b27e2abb7de659fae0c080faf25b89f3ce66a3b1a13025be221dba` |
| `submission-like-v1.json` | `d85227f216528d635e56a93094e661721f62f379808707f310bf4da60d8fa57b` |
| `coverage-robustness-v1.json` | `e3385a6f4939445e869d71dbf7f1bd5119aa25eebbb89e083b1ec9be336f7309` |
| `coverage-triage-v1.json` | `f400cdc3a947cd076ce9bd9f48a2dafcb98dfd3f9f938a74ceb11ca88c360972` |
| `restart-mechanics-v1.json` | `2bc42026f52c09d85625ecce8d3ce0729c1efa06d0716511ed18d9d59c9f91c6` |
| `restart-screen-v1.json` | `dd1404e7b260c93a141b303c1a7f88f9ef02ceba03f109523708b2a8ed54b5d3` |

Implementation must add an isolated, single-panel, append-only generator. The
old source hash above is a reference dependency, not a claim that the old
multi-panel CLI implements this contract. The new path and hash must enter the
separate clean pre-result source lock. It must independently reproduce all
seven hashes, pass the exact upstream reference explicitly rather than inherit
the old default, verify the 4,096-seed range, exclude every old topology,
refuse an existing output path, and never rewrite an old panel or `audit.json`.
A second no-project-import verifier must reconstruct the exclusion and split
receipt.

Reading official data is not authorized by this plan. After exact owner
approval, the generator may open only the archive topology-string column for
exclusion. It may not load losses, parameters, sensitivity, power, complexity,
history, or any private outcome panel. Raw panel and receipt bytes are generated
artifacts and remain outside Git in content-addressed, backed-up private task
storage. Before paid provisioning, a dated sanitized research record must
commit only their SHA-256 values, zero-overlap counts, seed lineage, feature-bin
counts, chosen split indices, and independent-verification result; it must
contain no topology string or official row.

## Stage-aware execution and selection

All runs are serial on one exact secure H100. Within a topology block each arm
uses the same topology and optimizer seed. Stage 1 uses optimizer seed
`20260901`; Stage 2 uses `20260902`. Seed, topology set, and session phase are
therefore confounded with stage. The paired arm contrast within each stage is
the only admissible comparison; there is no seed, drift, order, or cross-stage
causal claim.

Stage 1 executes 16 runs on its four panel members. In ascending chosen member
index, the arm orders are the fixed pairwise-balanced Williams rows:

```text
row 0: A B C D
row 1: B A D C
row 2: C D A B
row 3: D C B A
```

Each arm occupies every absolute within-stage position once and every pair
precedes one another twice. Stage 1 must finish, seal, authenticate, and be
replayed before any Stage-2 configuration is materialized or outcome opened.

For challenger `c` on Stage-1 topology `t`, define

```text
d1[c,t] = best_feasible_loss(c,t) - best_feasible_loss(A,t)
```

Negative favors the challenger. For a finite vector of paired differences,
`maximum_harm = max(0.0, max(d))`. A challenger is eligible only if all 16 valid
Stage-1 records exist, every run is physically and finite feasible, it wins at
least 3 of 4 pairs, its arithmetic mean `d1` is below zero, and its maximum
harmful `d1` is at most `0.5`. Select the eligible challenger with the smallest
mean `d1`; exact ties within `1e-12` resolve by B, then C, then D.

If and only if all 16 records are valid and no challenger is eligible, the
terminal action is `retain_round1_control_stage1_failed`; Stage 2 does not run.
Any earlier error is instead not evaluable. Production selection, a separate
history-first replay, and a sealed selection-receipt verifier must agree before
Stage 2 can start.

The `stage1_failed` result envelope has `finalist=null`, omits Stage-2 and
pooled metrics as `not_applicable`, sets `stage2_outcome_opened=false`, and
contains the complete authenticated Stage-1 selection table. Production,
history-first, and detached projections must agree on that exact branch.

Stage 2 executes eight runs on the four untouched members: A and the selected
finalist only, ordered A/finalist, finalist/A, A/finalist, finalist/A in
ascending member index. Define `d2[t]` analogously. No other challenger,
replacement arm, seed, topology, top-up, or retry may enter Stage 2.

## Frozen projection, metrics, and promotion rule

The canonical history-row fields are strict-Boolean `is_feasible`, numeric-or-
null `loss`, finite non-decreasing numeric `time_seconds`, and positive non-
decreasing integer `eval_count_after_call`. Every expected candidate-call row
must exist in exact call order. A missing field/row, wrong type, non-finite
time, decreasing time/count, truncated call ledger, or early worker termination
invalidates the whole run.

Physical feasibility is an `any` reduction over candidate-call rows: a run is
physically feasible iff at least one row has `is_feasible is True`.
`best_feasible_loss` is the minimum `float(loss)` over rows where
`is_feasible is True`, `loss` is numeric but not Boolean, and
`math.isfinite(float(loss))` is true. Null or non-finite loss in an otherwise
well-typed intermediate row is excluded rather than imputed and does not by
itself invalidate the run; promotion still requires at least one finite
feasible row. A non-numeric non-null loss invalidates the run. The record-level
`metrics.has_feasible`, `metrics.has_finite_feasible`,
`metrics.best_feasible_loss`, and `objective_accounting.eval_count` must exactly
replay this row rule, with final count equal to the last
`eval_count_after_call`. The independent replay must implement this rule from
history bytes without importing its production implementation.

A run without at least one finite feasible row is not evaluable; there is no
imputation, fallback value, or censor substitution. The score projection is an
ordinary arithmetic mean across topology values, matching the public
functional but not the hidden topologies or four-hour-per-topology budget.

For any paired difference `d`, `abs(d) <= 1e-12` is a tie, `d < -1e-12` is a
win, and `d > 1e-12` is a loss. The median is the middle value for odd `n` and
the arithmetic mean of the two middle sorted values for even `n`. Linear p90
harm applies that rule to sorted `max(d, 0.0)` values and index
`h=(n-1)*0.9`, interpolating between floor and ceiling indices. The descriptive
paired-topology bootstrap targets the arithmetic mean paired difference of the
fixed selected finalist; it never reselects. It uses NumPy PCG64 with seed
`20260903`, 10,000 complete topology-block resamples with replacement within
the reported Stage-1, Stage-2, or pooled set, and linear 2.5th/97.5th
percentiles. It is selection-conditional and cannot change an action.

Per-run `evaluation_count` is the authenticated final cumulative Objective
candidate-evaluation count in the complete logged history. Per-run
`evaluation_rate` is that count divided by the authenticated elapsed logged
seconds from `start_logging()` through the final history row. That duration
must be finite and strictly positive; zero or negative duration invalidates the
run. Arm summaries
report the arithmetic mean and median of raw counts and rates over that arm's
topologies; no ratio is a gate.

The result reports each Stage-1 arm mean; A and finalist Stage-2 means; all
paired differences, win/tie/loss counts, mean, median, p90 harm, maximum harm,
evaluation counts and rates; the descriptive bootstrap; and pooled eight-
topology summaries labeled `selection_conditional_descriptive_only`.

The finalist passes only if:

1. all 16 Stage-1 and all 8 Stage-2 records, histories, configs, logs, source
   identities, initial-population receipts, runtime receipts, and selection
   receipts authenticate;
2. every run is physically feasible and has a finite best feasible loss;
3. the finalist wins all 4 untouched Stage-2 topology pairs;
4. its Stage-2 arithmetic mean difference is at most `-0.05`; and
5. the production analyzer, independent no-project-import history-first replay,
   and detached-summary projection agree on every run, topology value, metric,
   criterion, and action. Floats use
   `math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)`; integers, Booleans,
   strings, nulls, keys, sequence lengths, and ordering agree exactly.

No Stage-1 or pooled quantity is a promotion criterion.

Pass action: `review_selected_bundle_for_round2_candidate_integration`.

Failure action: `retain_round1_control`.

Not-evaluable action: `retain_round1_control_attempt_not_evaluable`.

Even on a pass, the finalist mean is a small operational holdout number. It
must appear beside the same-panel A mean and paired delta and must never be
described as a new score comparable to `0.444293`. Only a later organizer
evaluation on hidden topologies can produce a genuinely comparable score.

## Runtime, smoke, cost, and fail-closed cleanup

| Item | Frozen value |
| --- | --- |
| Planned scored runs | 24 = 16 Stage 1 + 8 Stage 2 |
| Objective budget | 600 logged seconds per run |
| Total scored Objective time | 14,400 seconds = 4.0 H100-hours |
| Execution | serial; one worker and one GPU at a time |
| Hardware | exact secure `NVIDIA H100 80GB HBM3`; MIG disabled |
| Runtime | Python 3.12 patch release frozen pre-result; `dfbench==0.3.3`; exact CUDA-13/JAX 0.9.0.1 lock; caches disabled |
| Repository locks | `pyproject.toml` SHA-256 `8f412a28ddbeb284464f6c49ce2cf7c86f7e227881d8aa07f84780637a7ff5f4`; `uv.lock` SHA-256 `3e98259ff36b73445a93cbaeeb51d4454677677a8a2661b431add0a37996c013` |
| Cold smoke | one D-arm, 120-second loss-blind smoke; topology seed `2026095000`, optimizer seed `20260900` |
| Smoke timeout | 300 seconds |
| Scored-worker timeout | 900 seconds |
| Valid-run wall limit | 720 seconds; a later successful exit is not evaluable |
| Provider horizon | at most 25,200 seconds from provider RUNNING or first billable instant, whichever is earlier |
| Dispatch deadline | hard horizon minus 1,800 seconds |
| Cleanup reserve | final 1,800 seconds; no study dispatch |
| Price ceiling | `$3.29` per H100-hour |
| Maximum GPU charge | `$23.03` |
| Maximum all-in provider charge | `$25.00` |
| Provider objects | one secure pod, at most 40 GiB ephemeral container disk; no network volume, endpoint, or template |

The clean pre-result environment lock must additionally freeze the exact
Python patch version and interpreter SHA-256, immutable base-image digest (a
tag is insufficient), Linux kernel and NVIDIA driver versions, CUDA runtime
and library package names/versions/hashes, JAX/JAXLIB/PJRT/plugin wheel names
and hashes, `dfbench` wheel hash, every installed distribution and its
`.dist-info/RECORD` file-tree root, environment-variable allowlist, and device
model/MIG receipt. Raw host paths and device UUID stay private; their canonical
logical identities and digests enter the lock. Any mismatch rejects before
smoke.

The smoke topology is generated from its frozen seed and must be disjoint from
the official archive, all prior panels, and this new panel. If it overlaps,
the attempt is not evaluable; there is no replacement smoke seed. The smoke
must load and authenticate every arm/profile boundary, exercise D's most
complex runtime path, emit no optimizer loss or candidate, and finish before
any scored dispatch. Its success authorizes scored dispatch but is not result
evidence.

A read-only Runpod catalog check on 2026-09-01 reported the exact secure GPU at
`$3.29/hour`, CUDA 13 available, and LOW pod availability. The earlier 32-run
H100 study used 7.117350 provider-hours for 5.333333 scored hours; scaling that
utilization to 24 runs projects 5.338012 provider-hours and about `$17.58`.
The envelope is a guard, not a spending target.

`T0` is the first provider timestamp at which the sole task-owned pod becomes
RUNNING. `B0` is the earliest provider-billable timestamp, falling back to the
provider create-receipt timestamp if no distinct billable timestamp exists.
The hard horizon `H` is the earliest of `T0+25,200s`, `B0+25,200s`, and the
time at which the provider's quoted time-integrated all-in charge would reach
`$25.00`. An independent controller watchdog must enforce deletion by `H`;
the study profile must set this seven-hour maximum explicitly and reject the
older eight-hour default.

The complete successful paid path has these non-transfer wall-time buckets:

| Bucket | Maximum seconds |
| --- | ---: |
| post-RUNNING source/runtime preflight | 300 |
| cold smoke | 300 |
| 16 valid Stage-1 workers | 11,520 = 16 x 720 |
| Stage-1 seal/transfer, production selection, independent history-first replay, and sealed selection verifier | 900 |
| Stage-2 materialization and authentication | 120 |
| 8 valid Stage-2 workers | 5,760 = 8 x 720 |
| Stage-2 seal/transfer, production analyzer, independent history-first replay, and detached-summary projection | 900 |
| terminal seal and evacuation index | 300 |
| **total before cleanup reserve** | **20,100** |

On a full 25,200-second horizon, the frozen success envelope is 20,100 seconds
before the dispatch deadline `H-1,800s`, leaving 3,300 seconds unallocated
rather than promised. If `H-T0 < 21,900s` because billable setup consumed that
slack, the controller performs cleanup without smoke or scored dispatch.
The final reserve is separately capped at 900 seconds for evidence evacuation
and hash verification and 900 seconds for deletion plus inventory/billing
receipts. A valid scored worker must finish within 720 seconds; the 900-second
process timeout is only a fail-closed kill boundary, and the first timeout
halts the attempt. No bucket may borrow from the cleanup reserve.

Before smoke, every scored run, replay, transfer, hash, or cleanup step, the
controller checks that its frozen operation bound fits before the applicable
deadline. No smoke or scored run may begin unless its complete bound ends by
`H-1,800s`. The last 1,800 seconds permit no study dispatch.

Before provisioning, recheck exact secure SKU, CUDA 13, price, account
capacity, clean resource inventory, source locks, panel receipts, backup,
green CI, zero fixed/setup charge, and the enforceable hard stop. The quoted
combined hourly rate for the pod and at most 40 GiB ephemeral disk must not
exceed `$3.5714285714`; no separately billed object may exist. Provider billing
semantics and the B0 receipt must be machine-readable before smoke. A changed
SKU, price, cap, panel,
source, revision, runtime, or preflight condition voids approval. Community
cloud, another accelerator, multi-GPU execution, Docker fallback, or a higher
cap requires a new plan and approval.

The first worker error, timeout, interruption, provider guard, parse failure,
integrity mismatch, incomplete record, or malformed evidence immediately halts
all new dispatch, prevents Stage 2, and sets the canonical not-evaluable
action. A JSON/packet parse failure must still create a closed receipt with the
raw-output SHA-256 and no raw output content. `stage1_failed` is legal only
after all 16 valid Stage-1 records prove that no challenger is eligible. There
is no resume, rerun, replacement, top-up, substitution, extension, or
outcome-guided repair.

Cleanup order is fixed: authenticate the task process tree by PID, parent PID,
start timestamp, executable SHA-256, and command-line SHA-256; terminate it
bottom-up; record that every listed PID is gone and no descendant remains;
evacuate and hash all sealed evidence; delete the pod and every task-owned
volume, endpoint, template, and other object listed in the pre-provision
resource manifest; then obtain a zero-resource inventory and final billing
receipt before the hard horizon. If descendant enumeration or termination
fails, the controller still deletes the pod before the horizon but records the
attempt not evaluable. Cleanup failure does not authorize more time or money.

## Implementation and pre-result gate

This plan commit is the contract freeze. A later unpaid checkpoint may create
only a new candidate-screen namespace and profile. Retired IDs, profiles,
configs, hashes, panels, results, and analyzers remain byte-identical. The new
implementation must include:

- the isolated append-only panel generator and independent verifier, with raw
  generated artifacts kept outside Git;
- per-arm module/class/algorithm/source/settings/population/progress bindings;
- a stage-aware orchestrator that materializes Stage 2 only from an
  authenticated Stage-1 selection receipt;
- a dedicated seven-hour provider profile and fail-closed raw-output receipt;
- sealed archive packaging, a production analyzer, and a genuinely separate
  no-project-import history-first replay;
- exact source, package, environment, panel, run-order, initial-population,
  selection, cost, and cleanup manifests; and
- regression tests proving every retired profile is unchanged.

Implementation may make code conform to this plan but may not relax or alter a
panel rule, arm, seed, order, metric, threshold, timeout, cap, stop action, or
claim. It must reach a separate clean commit, pass hostile pre-result review,
and obtain green CI before provisioning. It must not edit `submission/`, any
prior panel/result, terminal study, or protected artifact. No result-bearing
fixture, UIFO run, official topology access, or GPU action is permitted by the
plan-only checkpoint.

## Approval and claim boundary

Freezing this plan does not authorize official-data access or spend. One later
owner approval may cover exactly two conditional scopes:

1. read only the official archive topology-string column at the frozen archive
   hash to generate, hash-lock, back up, and independently verify the private
   disjoint panel and smoke identity, while committing only the sanitized
   commitment record defined above; and
2. after that panel/receipt, exact implementation and source locks, verified
   backup, green CI, clean provider inventory, and unchanged catalog/cost
   preflight exist, provision one exact secure H100 under the `$25.00` all-in
   and seven-hour caps. Provisioning is initially only for the loss-blind smoke;
   scored dispatch is conditional on that smoke passing.

Failure of any condition voids the paid scope rather than widening it.
Task-owned provider cleanup is included. Portal upload, candidate integration
into `submission/`, merge, destructive action outside task-owned cleanup,
private outcome access, and any later paid run remain separate owner decisions.

The official timeline currently lists optional public-leaderboard deadlines on
2026-09-12 and 2026-09-29 AoE and the prize-determining final deadline on
2026-10-15 AoE. Recheck upstream before any launch or upload.
