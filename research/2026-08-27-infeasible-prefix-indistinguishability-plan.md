# Infeasible-prefix indistinguishability plan - 2026-08-27

## Frozen decision

Study ID: `infeasible-prefix-indistinguishability-v1`

Status: frozen before retained execution.

The only question is whether a deterministic online restart rule that observes
only finite loss and the feasibility flag can simultaneously guarantee both:

1. a restart by a finite bound `B` on every forever-infeasible, strictly
   improving path; and
2. no restart before feasibility on every path whose first feasible observation
   could occur after the same observable prefix.

The submitted patience-600/no-prior candidate remains protected. This analytic
study does not implement a restart treatment, choose a tenure threshold, compare
competition candidates, use a topology, or estimate optimizer performance.

Run exactly the six deterministic cases below on the local JAX CPU backend. The
observable paths, action-vector family, bounds, exhaustive counts, invariants,
and decision actions are fixed in the clean pre-result revision. The worker
runs with network access disabled and writes its immutable result only beneath
the sibling private local-lab root through `tools/run_local_lab.py`.

## Fixture identity

- observable fields: exact finite loss and `is_feasible` only;
- shared-prefix bounds: `1, 2, 3, 5, 8, 13`;
- maximal bound: `B = 13`;
- late crossing: the first feasible observation is `B + 1`;
- loss path: an exact integer loss that decreases by one at every observation;
- policy projection: the restart/no-restart action vector realized by a
  deterministic rule on the one shared prefix;
- maximal exhaustive family: all `2^13 = 8192` action vectors;
- positive-control signal: one additional Boolean certificate visible at
  observation `B`, false on the forever path and true on the late-crossing path.

At every step the rule observes the current `(loss, is_feasible)` pair and then
decides whether to restart. It receives no path identity, hidden future label,
objective identity, wall clock, randomness, budget lookahead, or other auxiliary
signal. `B` is fixed before either path is constructed. Both copies of the rule
begin in the identical deterministic internal state, so identical observations
inductively produce identical later state and actions. The theorem concerns one
target lane: "restart by B" means that lane undergoes a restart after an
observation no later than `B`, while "preserve" means no restart or equivalent
state-changing event touches that lane before its first feasible observation.
Restarting some unrelated population member does not satisfy the bound. The
forever path is the exact integer mathematical extension of each tested finite
prefix; it is not claimed to be a realizable infinite float32 descent.

Every deterministic online rule has exactly one realized action vector on the
shared observable prefix. The fixture enumerates those realized vectors; it
does not claim to enumerate the rule's behavior on unrelated paths.

## Complete case set and invariants

### 1. `shared_prefix_identity`

Construct a forever-infeasible strictly improving path and a path whose first
feasible observation is `B + 1`. The case passes only if the two observations
are byte-identical through `B`, both prefix hashes match, every prefix loss is
finite and strictly improves, and the two observations at `B + 1` differ only
in information that has then become observable.

### 2. `action_vector_exhaustion`

Enumerate all 8,192 restart/no-restart action vectors through `B = 13`. A
vector satisfies the bounded-restart obligation exactly when it contains a
restart. It preserves the late crossing exactly when it contains no restart.
The case passes only if all vectors are visited and zero vectors satisfy both
obligations.

### 3. `witness_partition`

The exhaustive family must partition exactly into 8,191 vectors that restart
by `B` but destroy the late crossing, one all-no-restart vector that preserves
the crossing but violates the bound, zero joint satisfiers, and zero vectors in
neither class. The counts must sum to 8,192.

### 4. `boundary_sweep`

Repeat the exact enumeration at `B` equal to `1, 2, 3, 5, 8, 13`. At every
bound, require `2^B - 1` bounded-only vectors, one preservation-only vector,
zero joint satisfiers, and zero unclassified vectors. This pins the argument to
an arbitrary finite bound rather than a quirk of 13.

### 5. `extra_signal_positive_control`

Break prefix identity by exposing the declared certificate at observation `B`.
Use the deterministic rule "restart at `B` exactly when the certificate is
false." The case passes only if it restarts the forever path by `B`, preserves
the late-crossing path until `B + 1`, and confirms that the two observable
prefixes are no longer identical. This is a fixture control, not a proposed
optimizer signal or policy.

### 6. `process_isolation`

Run the complete timing-free proof projection in two fresh,
credential-scrubbed, network-disabled CPU worker processes. Their JSON values
and SHA-256 digests must be byte-for-byte identical.

## Stopping and decision rule

The study is terminal after one controller invocation. Do not repeat, top up,
drop a case, alter a bound, add a signal, change the policy family, or relax an
invariant after any result is observed.

Pass only if all six cases complete and every declared invariant passes. The
frozen success action is:

```text
synthetic_identical_prefix_obstruction_confirmed
```

Any failed case has the frozen action:

```text
park_infeasible_prefix_boundary_research
```

A timeout, malformed result, nondeterminism, source drift, dirty worktree,
lease collision, or controller error also parks the laboratory. It is not a
study failure that can be repaired by rerunning the terminal fixture.

## Success and failure actions

A pass confirms only a deterministic, pathwise online-information limit under
identical observable prefixes. If a rule must restart the forever path by `B`,
it must take the same pre-`B + 1` action on the indistinguishable late-crossing
path; preserving every possible late crossing therefore precludes a finite
pathwise restart bound. Universality comes from that non-anticipating prefix
lemma; the finite enumeration is a hostile reference check of its finite action
projection, not an exhaustive test of arbitrary program implementations.

The claim does not cover randomized probability guarantees, expected restart
time, rules given richer pre-feasibility signals, a finite chosen crossing
horizon, distributional assumptions, or optimizer performance. It neither
recommends the current clock nor justifies a feasibility-aware replacement.

A failure or park ends mutation and requires owner review. No candidate,
submission artifact, portal, leaderboard, official dataset, private panel,
GPU, cloud resource, or paid endpoint is authorized by either outcome.
