# V3 synchronous TD terminal result

Status: **passed**

Study ID: `multistep-td-action-prefix-v3`

Date: 2026-08-29

## Decision

The tenth guarded local study passed all nineteen frozen CPU cases in one
terminal invocation. The blank tabular learner propagated a terminal toy signal
backward through three bootstrap boundaries with the exact four-sweep
synchronous TD update. Its frozen post-fit policy returned `1.0` on train,
validation, and test regimes without held-out updates, while the constant,
feedback-only myopic, no-bootstrap, and seeded-random comparators remained below
the precommitted positive gate. Transition-target, reward-origin, and complete
signal-ablation controls all returned to `0.5` on held-out regimes and rejected
that gate.

The controller's terminal action is
`synthetic_four_step_synchronous_td_propagation_confirmed_for_harness`.

This supports only the fixed synthetic synchronous-TD harness and its
deliberately predictive toy propagation signal. The public signed signal
intentionally encodes evaluator truth. A hand-programmed oracle could therefore
solve the family without delayed learning, so this result does not establish
that delayed credit is necessary or that a public shortcut is absent.

## Provenance and controller receipt

- frozen plan:
  [`2026-08-28-multistep-td-action-prefix-v3-plan.md`](2026-08-28-multistep-td-action-prefix-v3-plan.md)
- clean pre-result implementation record:
  [`2026-08-28-multistep-td-action-prefix-v3-pre-result-implementation.md`](2026-08-28-multistep-td-action-prefix-v3-pre-result-implementation.md)
- terminal revision:
  `9d8c64887c730043d2da7c313ac9240fd3f3e85c`
- controller cycle:
  `20260829T123207Z-d4c8709f5fba`
- immutable private result and sidecar SHA-256:
  `c6e7cecd8d6e9fa7e12aee116f141522321f400a9286940b38a2023e54f5d86f`
- complete-family SHA-256:
  `c3e093639b05690016f8f39ed7dba75c0b493e30d1508664ec7225748c744c11`
- fixed random-comparator stream SHA-256:
  `8fe33e0832d9fe4b705f97bf20d8559223e8af217ca8f6a97c587b8bea9e6803`

The controller authenticated the committed revision, frozen source approvals,
registry contract, protected submission tree, and protected local artifacts
before running the dedicated worker. It ran once on local CPU with credentials
scrubbed, worker networking disabled, bounded output, a global lease, and a
one-hour process-tree timeout. The result file matched its immutable SHA
sidecar, the terminal event and state ledger recorded the same digest, and the
controller returned to `awaiting_study` with failure streak zero, no active
cycle, no stop marker, and no remaining lease.

## Sanitized aggregate result

### Family, partition, and information boundary

| frozen case | sanitized terminal evidence |
|---|---|
| `generator_partition` | 4 train, 2 validation, and 2 test regimes; 2,048 / 1,024 / 1,024 episodes; zero generator RNG calls; `structure_kind = none` |
| `complete_family_replay` | all 122,880 legal rows replayed exactly: 57,344 nonterminal, 65,536 terminal, 61,440 predecessor nodes, and 122,880 unique transition keys; all 25 corruptions rejected |
| `target_swap_twin` | 122,880 evaluator targets flipped while frozen public bytes and their digest remained identical; all 8,192 required terminal rewards changed |
| `realized_path_disjointness` | all 16,384 realized public rows and 4,096 public paths were split-disjoint with identity fields excluded |
| `typed_episodic_contract` | immutable float64 six-field observations, int8 binary actions, float64 binary rewards, exact four-step event/done order, and all 8 invalid actions rejected |
| `lazy_information_boundary` | policy input was observation-only; 16,384 legal lazy resolutions completed in exact order; all 20 physical timing/reentrancy attacks were rejected without changing the table |
| `pending_transition_authentication` | wrong identity, duplicate append, and cross-episode append were rejected and cleared fail-closed |
| `keyed_trace_authentication` | all 20 malformed, reordered, identity, type, donor, origin, and component attacks were rejected; independent trace and TD digests authenticated |
| `train_only_source_boundary` | zero held-out updates, zero exploding-source operations, zero inverse-train operations, exact absent/exploding/lazy behavior, and an unchanged held-out learner state |

### TD propagation, comparators, and attribution

| frozen case | sanitized terminal evidence |
|---|---|
| `synchronous_td_order` | exact positive-cell counts `[2, 4, 6, 8]` and writes `[60, 60, 60, 60]` over four sweeps; 32 aggregate lookups per sweep; 2,048 legal terminal reads; all invalid sweeps rejected |
| `multistep_td_recovery` | behavior return `0.0625`, reward sum `128`, and regret `1920`; post-fit train, validation, test, and minimum held-out-regime return all `1.0`; 8,192 train transitions and zero held-out updates |
| `baseline_replay` | constant-zero, constant-one, feedback-only myopic, and no-bootstrap test return `0.5`; seeded-random test return `0.0703125`; independent replay exact |
| `all_boundary_terminal_dependency` | changed cells advanced exactly `[1, 2, 3, 4]` across the four sweeps for both probe signs; 256 selected terminal-scalar reads, zero unselected reads, and the opposite chain unchanged |
| `transition_target_control` | validation and test return `0.5`, true test gap `0.5`, canonical behavior exact, successor multiset preserved, and the complete positive gate rejected |
| `reward_origin_control` | validation and test return `0.5`, true test gap `0.5`, exact per-cell counts and means, unchanged reward multiset, no early origin materialization, and the complete positive gate rejected |
| `signal_attribution_control` | fresh refit and true-policy ablation both returned `0.5` on validation and test; only the declared signal changed and the complete positive gate was rejected |

### Complete projection and isolation

| frozen case | sanitized terminal evidence |
|---|---|
| `control_difference_whitelists` | canonical, target-swap, transition-target, reward-origin, and signal-ablation projections covered every legal row; 100 protected-field mutations were rejected |
| `sanitized_result_contract` | the bounded projection authenticated all 18 non-process case schemas and rejected 50 forbidden raw/path/container samples |
| `process_isolation` | two fresh worker projections matched exactly and passed |

All nineteen cases passed. No case, seed, split, threshold, permutation, mapping,
or stopping rule was changed after the plan was frozen, and the terminal fixture
must never be rerun.

## Claim boundary

The result establishes only that this particular synthetic harness can:

- keep evaluator-only identities and terminal scalars behind the frozen physical
  boundary while authenticating public transitions;
- apply the declared synchronous tabular TD targets and propagate one terminal
  scalar across three bootstrap boundaries; and
- distinguish the canonical toy association from the frozen comparator and
  negative-control constructions.

It is not evidence for delayed-credit necessity, absence of a target-correlated
public shortcut, general or production RL, meta-RL, hidden-topology learning,
UIFO value, optimizer quality, native-rewrite value, accelerator speed,
competition performance, candidate value, or score. It used no official data,
private topology panel, portal, leaderboard, cloud resource, GPU, Docker, SSH,
paid endpoint, or money, and it did not change the protected submission.

## Protected-submission verification

Before and after the guarded invocation, the protected submission source tree
remained at Git tree
`e2b495b6f1bf9f5c8f0b36ae5bc095f6df8e7588`. The existing protected local
artifacts remained:

- `artifacts/generated/submission.zip`:
  `4cc0dbc65a3e61ca5358c18655c432caf478fbdfc07f10512553781f8822924b`
- `artifacts/generated/submission.manifest.json`:
  `99e18299ba6dd232f203cc2e59131cea61cd83ff8b0a9bceab6c56c764ddd86a`

Neither artifact was rebuilt, overwritten, uploaded, or opened for selection.

## Next gate

The next unresolved learning rung is online bootstrapped control with an
explicit behavior-policy exploration contract and train-only updates on a fresh
topology-independent toy family. The next checkpoint is plan-only: choose one
new versioned study ID and freeze the complete generator, typed trajectory,
action-selection, exploration, bootstrap-target, update-order, held-out,
comparator, attribution, leakage, stopping, and claim contracts before any
learner implementation or execution.

The narrow future question should be whether a small deterministic tabular
on-policy control learner can acquire a deliberately learnable multi-step choice
from its own frozen exploratory behavior, retain it on untouched held-out
generator regimes, beat precommitted constant, myopic, no-bootstrap, and
seeded-random baselines, and lose the association under identically evaluated
transition-target, behavior-assignment, and signal-attribution controls. A toy
pass would validate only that online-control harness. Meta-RL, native rewrites,
official-data training, candidate integration, accelerator benchmarking, and
paid training remain later owner gates.
