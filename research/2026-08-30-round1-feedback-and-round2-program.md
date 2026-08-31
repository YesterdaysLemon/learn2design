# Round-1 feedback intake and Round-2 local program

Date: 2026-08-30

Status: frozen plan-only checkpoint

First study ID: `constraint-aware-progress-toy-v1`

## External observation

The owner supplied the organizing team's Round-1 result email. The sanitized
aggregate observation is:

- Round-1 score: `0.444293` (lower is better);
- best reported Round-1 score: `0.019674`;
- placement: 14th of 43 evaluated participants; and
- the organizer invited a Round-2 submission.

The repository cannot independently authenticate the email or portal record,
so these values remain explicitly owner-reported. They are public-feedback
context, not a replacement for the checked-in experiment ledger. No sender,
email address, account identifier, message header, or portal detail is stored.

The official competition README says the leaderboard score is the arithmetic
mean best feasible loss over ten hidden topologies. Its currently published
timeline lists the second public submission on 2026-09-12 AoE, the third on
2026-09-29 AoE, and the final on 2026-10-15 AoE. Recheck the upstream rules
before any schedule-critical launch or upload.

## Owner decision and preserved baseline

The owner authorized a local Round-2 research pivot and a two-hour laboratory
cadence. This opens outcome-blind candidate research in experiment code. It
does not authorize a portal action, official-data use, private outcome-panel
selection, paid compute, GPU provisioning, a merge, or overwriting the retained
submission.

The Round-1 package remains immutable:

- submission revision:
  `5ce3cdb2ddf4c505622a0aeef805936a4ea607d7`;
- ZIP SHA-256:
  `4cc0dbc65a3e61ca5358c18655c432caf478fbdfc07f10512553781f8822924b`;
- packaged policy: patience 600, no semantic prior, random starts; and
- terminal H100 action: `retain_random_start_candidate`.

Do not rerun or top up any terminal study. In particular, do not reopen the
failed coverage treatment, tune a subgroup against its eight observed
topologies, or open the untouched `coverage-robustness-v1` panel. The frozen
online-SARSA V4 plan remains intact but is paused, not failed, amended, or
quarantined. It is not the live priority while this Round-2 program is active.

## Evidence-led diagnosis

The hidden result is a strong reason to seek a material method change, but it
does not identify which hidden topology failed or which mechanism will help.
Three checked-in facts narrow the first move:

1. The packaged optimizer tracks member progress using total finite loss.
   Finite but infeasible loss improvements reset the stall clock.
2. The authenticated UIFO public auxiliary contract exposes current
   `sensitivity_loss`, `penalty`, `violations`, and raw grouped power values,
   while the current optimizer directly consumes only `is_feasible`.
3. The semantic prior, patience-200 rule, and midpoint coverage-balanced starts
   each failed their frozen promotion gates. Their observed panels are closed.

Therefore the first new hypothesis is not another initializer or a retuned
patience value. It is that current public constraint progress can make the
member-progress and restart decision less easily distracted by improving
sensitivity inside an increasingly infeasible basin. This is plausible, not
established: the auxiliary signals are diagnostics, not certificates of future
feasibility.

## Round-2 funnel

Work proceeds in strict stages:

1. **Synthetic causal mechanics.** Freeze and run the toy study below. It may
   validate only the proposed decision rule and its controls.
2. **Candidate implementation audit.** If stage 1 passes, implement the rule
   only in an experiment-owned candidate adapter, prove public-API and budget
   parity, and keep `submission/` unchanged.
3. **Fresh generated development screen.** Before any accelerator launch,
   freeze a new archive-disjoint development panel, paired seeds, topology-level
   rule, runtime stack, stop conditions, and cost ceiling. This requires a new
   explicit owner approval.
4. **Untouched promotion screen.** Only a treatment that passes stage 3 may be
   evaluated once on a separately sealed panel under another frozen gate.
5. **Artifact and portal review.** Promotion into a new ZIP, building the final
   artifact, or uploading it are separate owner actions.

The public score may prioritize urgency, but it may not be used as a local
threshold, imputed topology outcome, or claim that a local treatment will reach
the Round-1 best score.

## Frozen analytic question

> On a deterministic smooth constrained multi-basin toy family, can a
> public-aux-guided progress policy reduce final feasible regret relative to the
> protected raw-total-loss progress policy under identical Adam dynamics,
> objective-call budgets, initial populations, and random transcripts, while
> losing that advantage when the constraint signal is shuffled or ablated and
> avoiding material harm when sensitivity and feasibility are aligned?

This plan is the freeze boundary. No learner, fixture, worker, candidate, or
result is executed or observed in this checkpoint.

## Complete toy family and exact reference

All arithmetic is IEEE-754 binary64. The fixture operates in unbounded
coordinates `u in R^3`. Its branch-stable sigmoid is
`1/(1+exp(-u))` for nonnegative `u` and `exp(u)/(1+exp(u))` otherwise; active
coordinates are `x = 4*sigmoid(u)-2`.

Each world is identified by the four-bit tuple `(A,B,K,T)` in lexicographic
order and uses:

- `a = 0.80 + 0.20*A`, `b = -0.50 + 1.00*B`,
  `k = 0.50 + 0.50*K`, and `t = 0.10 + 0.06*T`;
- `c = -0.25` when `A xor B` is zero, otherwise `+0.25`;
- canonical tilt `q=t`;
- `sensitivity_loss = (x0^2-a^2)^2 + q*x0 +
  k*(x1-b)^2 + 0.5*(x2-c)^2`;
- canonical `violation=max(0,-x0)`,
  `penalty=0.02*violation^2`, and `is_feasible=(x0>=0)`; and
- `loss=sensitivity_loss+penalty`.

The constrained reference is not `(a,b,c)`. For canonical `q>0`, its `x0` is
the unique derivative root in `[a/2,a]`; for aligned-control `q=-t`, it is the
unique derivative root in `[a,min(1.5*a,1.99)]`. Starting from the stated
bracket, perform exactly 80 binary64 bisections on
`d(z)=4*z*(z^2-a^2)+q`: if `d(mid)<=0`, replace the lower endpoint, otherwise
replace the upper endpoint. The reference is the final midpoint with
`x1=b,x2=c`. An independently written scalar oracle must reproduce every
reference value to `1e-12` before optimizer construction.

The fixed anchor is active point `(0,0,0)`. For canonical and aligned worlds,
the denominator `anchor_sensitivity-reference_sensitivity` must be finite and
strictly positive. Normalized feasible gap is the un-clipped quotient
`(best_feasible_sensitivity-reference_sensitivity)/denominator`. A value below
`-1e-10` fails the reference contract instead of being clipped. If an arm has
no feasible observation, its gap is exactly one. Values above one remain above
one so harm cannot be hidden.

Worlds with even bit parity are the eight development worlds; worlds with odd
bit parity are the eight untouched held-out worlds. Formula construction and
the scalar oracle are separate code paths. They must prove all identities,
balanced splits, finite values, reference brackets, denominator positivity,
and evaluator joins before constructing an optimizer. No state or policy is
fit across worlds.

There are exactly three family variants over the same sixteen identities:

- `canonical`, as defined above;
- `aligned`, changing only `q` from `t` to `-t`; and
- `impossible`, changing only the feasibility threshold to `x0>=2.25` and
  violation to `max(0,2.25-x0)`.

The impossible family has no feasible active point, does not claim a feasible
reference, and always assigns gap one after verifying zero feasible joins.

## Frozen transcript and source isolation

The four paired seeds are `2026083001`, `2026083003`, `2026083007`, and
`2026083011`. For each seed, construct one world-independent NumPy `PCG64`
from `SeedSequence([20260830,seed])`. The guarded worker runtime is frozen to
CPython `3.13.14` x64 and `numpy==2.5.1`; the implementation checkpoint must
commit the interpreter, NumPy package, `PCG64`, and `SeedSequence` identities
and hashes before result execution. Consume standard-normal binary64 values in
exactly this row-major order:

1. one `(7,3)` initial suffix after the all-zero anchor lane; then
2. for each batch `0..63`, one `(8,3)` fresh-restart array followed by one
   `(8,3)` perturbation array, whether or not either array is used.

That is exactly 3,093 binary64 values per transcript and 12,372 values over four
seed transcripts. The hash preimage is the ASCII domain
`L2D-constraint-progress-transcript-v1` plus a zero byte, little-endian
`uint64` seed, and the little-endian float64 values above. The aggregate root
preimage is the same domain plus `/root`, a zero byte, then the four individual
lowercase transcript hashes in seed order, each followed by LF. The root, draw
count, array shapes, individual hashes, and runtime identities must be committed
in source and registry at the clean pre-result revision before any objective
result.

Every world, family, order, and arm for a given seed receives an immutable copy
of the same transcript. Thus transcript bytes cannot encode world, family, or
split. Mutable generator state and any seed-dependent policy branch are
illegal. A live transcript-fingerprint probe must be denied at the optimizer
adapter; the sealed provider applies released restart rows without exposing a
seed or transcript identity.

Forward arm order is
`[protected_raw_progress,constraint_lexicographic_progress,
shuffled_progress_control,ablated_progress_control,no_restart_comparator]`;
reverse order is its exact reverse. Each arm is executed once in each order.
Order twins must produce byte-identical keyed trajectory projections before
their seed value can enter a world aggregate. This equality uses a normalized
twin projection that omits the `order` tag; the two order-tagged identity roots
must remain distinct and correctly joined.

Development and held-out execution are separate fresh child processes for each
family variant, for exactly six phase receipts per outer launch. A development
child receives only the eight development world records; an exploding held-out
sentinel is connected to its source API. A held-out child receives only the
eight held-out records, the immutable contract/transcript roots, and no
development metrics, state, action, or output. Each trajectory resets all
optimizer state. The phase receipts commit source keys, attempted reads, and
zero forbidden reads.

### Capability and information boundary

`WorldRecord` is evaluator/environment-only. The environment closure may read
its coefficients and threshold to calculate one evaluation, but it exposes no
record, coefficient, threshold, reference, family, split, world index, bit
tuple, seed, or identity field to an optimizer arm.

At action time the optimizer core receives exactly:

- current unbounded parameter rows, fixed learning rates, moments, ages,
  progress states, stalls, and budget fraction;
- total loss and the sanitized total gradient for each evaluated row;
- only its arm-specific three-field decision tuple; and
- from a sealed physical-incumbent tracker shared by all arms, one Boolean and
  one unbounded center row indicating the best canonical feasible observation.

It receives no canonical auxiliary tuple when the shuffled or ablated adapter
is active. It receives no sensitivity/penalty/feasibility leaf except through
the declared decision adapter, and it cannot call the environment, evaluator,
reference solver, scorer, trace joiner, source, or oracle directly. The sealed
incumbent tracker may consume canonical feasibility and sensitivity but returns
only the common presence/center capability used by the frozen restart rule.

The transcript stays inside a sealed restart-draw provider. Only after the
restart mask and current round are committed may it release the current batch's
declared fresh and perturbation rows. It never exposes future or unused values
to the progress or update consumer, while its final receipt still authenticates
every unused draw.

Trace identities and canonical aux remain collector-only. Reference values,
denominators, gaps, family/split/world labels, coefficients, thresholds, and
source receipts remain evaluator-only. The worker oracle receives completed
immutable records only after action selection ends. An implementation must use
distinct frozen dataclasses for `OptimizerPacket`, `CollectorEnvelope`, and
`EvaluatorEnvelope`; passing a `WorldRecord` or `ObservationRecord` to the
optimizer is illegal. Live denial probes for every forbidden field and method
must traverse the actual optimizer adapter and produce zero state mutations.

## Exact optimizer chronology

Every arm uses population eight, anchor lane zero, the seven transcript suffix
lanes, `geomspace(0.03,0.15,8)` learning rates, 64 batches, toy patience eight,
Adam `beta1=0.9`, `beta2=0.999`, `epsilon=1e-8`, improvement tolerance `1e-7`,
gradient clip norm one, and restart-noise base scale `0.35`.

The analytic total-loss gradient is computed in active coordinates as
`[4*x0*(x0^2-a^2)+q+penalty_dx0,
2*k*(x1-b),x2-c]`, where `penalty_dx0=-0.04*violation` below the family
threshold and zero otherwise, then multiplied componentwise by
`4*sigmoid(u)*(1-sigmoid(u))`. Gradient records must have binary64 dtype and
shape three; their values may be nonfinite only until the declared sanitizer,
which maps every NaN and infinity to exact zero.
Each member's clip scale is `min(1,1/(sqrt(sum(g^2))+1e-12))`.

Every trajectory starts with `u[0]=(0,0,0)`, `u[1:8]` equal to the transcript
suffix, all moments exact zero, all Adam ages and stalls integer zero, every
progress state encoded as unobserved, global-feasible presence false, global
center `(0,0,0)`, and restart round integer zero. Each batch performs exactly:

1. evaluate all eight current `u` rows and authenticate their typed aux;
2. record the best canonical physically feasible observation and its `u`;
3. compute the arm's progress decision, setting stall to zero on improvement
   and otherwise incrementing it by one;
4. sanitize/clip the total gradient; set `age1=age0+1`,
   `m1=.9*m0+.1*g`, `v1=.999*v0+.001*g^2`,
   `mhat=m1/(1-.9^age1)`, `vhat=v1/(1-.999^age1)`, then
   `u1=u0-lr*mhat/(sqrt(vhat)+1e-8)` in unbounded coordinates;
5. set restart mask to `stall>=8` after that update; and
6. replace every selected member and reset its moments, age, progress sentinel,
   and stall to zero.

The physical-incumbent tracker scans feasible finite batch rows by ascending
member index and selects the lexicographic minimum
`(sensitivity_loss,member_index)`. It replaces the cross-batch incumbent only
when sensitivity loss is strictly smaller in binary64; an exact tie retains the
earlier `(batch,member)` source. Presence, source batch/member, sensitivity,
unbounded center, and center hash are committed in the `BatchReceipt` before a
restart decision can consume the center.

There are no evaluations outside the 64 complete batches; the last post-update
parameters are not scored. Each trajectory therefore has exactly 512 objective
evaluations and 512 member transitions.

When any restart occurs, progress is `(batch_index+1)/64` and scale is
`0.35*max(0.10,1-progress)`. The batch perturbation array is centered by column
and divided by its population standard deviation with `ddof=0` plus `1e-6`.
For a batch with a nonempty mask, selection uses the current restart-round value
before incrementing it; after all replacements the round increments by exactly
one. If a canonical feasible incumbent exists, selected members with
`(member_index+restart_round)%2==0` use incumbent plus scaled normalized noise;
the rest receive their same-index row of the precommitted fresh array directly
as unbounded `u`. Without a feasible incumbent every selected member receives
its fresh row. Lane zero is restart-eligible under the same rule after its
batch-zero anchor observation. No-restart still authenticates all unused
arrays.

A restart resets the member progress sentinel, so its next finite valid
observation improves. Partial batches, conditional random draws, optional
centering, and implementation-defined tie behavior are forbidden.

Because the zero anchor is canonically feasible at batch zero, canonical and
aligned results test only post-anchor feasible-quality progress and restart
behavior. They do not test time to first feasibility or pre-feasible incumbent
search. The no-feasible-incumbent branch is exercised only by the impossible
control and supports no performance claim.

## Arms and interventions

Canonical feasibility and best-feasible scoring are never altered. The
progress function receives one explicit decision tuple
`(decision_is_feasible,decision_penalty,decision_sensitivity_loss)`:

1. `protected_raw_progress` ignores that tuple; improvement is finite total
   loss smaller than member best by more than `1e-7`.
2. `constraint_lexicographic_progress` receives the canonical tuple. A feasible
   tuple always improves over an infeasible best. Two feasible tuples compare
   sensitivity loss. Two infeasible tuples compare penalty first, then
   sensitivity only when penalty differs by at most `1e-7`; the selected
   component must improve by more than `1e-7`. A feasible best is never
   replaced by an infeasible tuple.
3. `shuffled_progress_control` receives all three tuple fields from donor
   member `(member_index+1)%8` in the same authenticated batch. Parameters,
   total loss, gradients, canonical aux, physical incumbent, and scoring remain
   canonical.
4. `ablated_progress_control` receives exact tuple `(false,0.0,0.0)` for every
   member and batch; all canonical fields and scoring remain unchanged.
5. `no_restart_comparator` uses protected raw progress and Adam but forces the
   restart mask false after authenticating the same chronology and transcript.

A fixed two-step sentinel must traverse the actual progress consumer before
the main trajectories. Its stored `ProgressState` is exactly
`(mode=lex,observed=true,feasible=false,first=1.0,second=5.0)`; its canonical
observations are `(false,0.9,6.0)` then `(false,0.8,7.0)`; its cyclic donor
observations are `(false,1.1,0.0)` then `(false,1.2,0.0)`; and its ablated
observations are `(false,0.0,0.0)` twice. The resulting decisions are exactly
`[true,true]`, `[false,false]`, and `[true,false]`. Literal pass flags or
disconnected control helpers are forbidden. Allowed static differences are
only arm ID, progress-comparator selection, tuple adapter, and the declared
restart-enabled Boolean. Dynamic differences must descend from those fields or
their resulting optimizer state.

## Typed evidence and exact counts

Type codes are `B` JSON Boolean, `I` exact JSON integer, `F` finite binary64
JSON number, `S` bounded ASCII string, `H` lowercase 64-character SHA-256,
`N` exact JSON null, and `A[T;n]` fixed-order array of `n` values of type `T`.
Raw analytic gradients may be nonfinite only before record construction; the
record stores their three-bit nonfinite mask and the sanitized finite vector.

Every record is an ordered object with exactly these fields:

- `WorldRecord(family:S,world:I,bits:A[I;4],split:S,a:F,b:F,k:F,t:F,c:F,
  threshold:F,reference_x0:F|N,reference_sensitivity:F|N,
  denominator:F|N)`;
- `ObservationRecord(family:S,world:I,seed:I,order:S,arm:S,batch:I,member:I,
  u:A[F;3],x:A[F;3],loss:F,gradient:A[F;3],
  gradient_nonfinite:A[B;3],canonical_is_feasible:B,sensitivity:F,
  penalty:F,violation:F,decision_source:S,decision_donor_member:I,
  decision_is_feasible:B,decision_penalty:F,decision_sensitivity:F)`;
- `ProgressState(mode:S,observed:B,feasible:B,first:F,second:F)`, where
  unobserved is exactly `(mode,false,false,0.0,0.0)`, raw observed is
  `(raw,true,false,best_total_loss,0.0)`, feasible lexicographic is
  `(lex,true,true,best_sensitivity,0.0)`, and infeasible lexicographic is
  `(lex,true,false,best_penalty,best_sensitivity)`;
- `DecisionTuple(is_feasible:B,penalty:F,sensitivity:F)`;
- `TransitionRecord(family:S,world:I,seed:I,order:S,arm:S,batch:I,member:I,
  progress_before:ProgressState,progress_after:ProgressState,stall_before:I,
  stall_after:I,adam_age_before:I,adam_age_after:I,update_applied:B,
  restart_triggered:B,restart_kind:S,restart_round:I,center_source:S,
  state_before_sha256:H,state_after_sha256:H)`;
- `BatchReceipt(family:S,world:I,seed:I,order:S,arm:S,batch:I,
  incumbent_present:B,incumbent_sensitivity:F|N,incumbent_source_batch:I,
  incumbent_source_member:I,incumbent_center_sha256:H,
  restart_round_before:I,restart_round_after:I,restart_mask:A[B;8],
  fresh_draw_sha256:H,perturb_draw_sha256:H)`;
- `TrajectoryRecord(family:S,world:I,seed:I,order:S,arm:S,evaluations:I,
  transitions:I,best_feasible_sensitivity:F|N,gap:F,
  normalized_twin_sha256:H,event_root_sha256:H)`;
- `PhaseReceipt(family:S,split:S,world_keys:A[I;8],attempted_reads:I,
  forbidden_reads:I,forbidden_payload_rows:I,sentinel_connected:B,
  input_root_sha256:H,output_root_sha256:H)`;
- `WorldAggregateRecord(family:S,world:I,arm:S,seed_gaps:A[F;4],
  mean_gap:F)`; and
- `AttackReceipt(attack_id:S,injection_path:S,rejection_code:S,
  consumer_reached:B,state_mutations:I)`.

Exact enums are the three family names above; split `development|heldout`;
order `forward|reverse`; the five arm names above; decision source
`unused|canonical|cyclic-donor|ablated`; progress mode `raw|lex`; restart kind
`none|fresh|incumbent`; and center source `none|fresh|global-feasible`.
Unused donor and restart-round integers are `-1`. All strings are at most 64
ASCII bytes.

Identity order is family, world, seed, order, arm, batch, member. Duplicate,
missing, reordered, or cross-identity joins fail closed. The fixture path emits
typed records; a separately written worker oracle reconstructs formulas,
progress decisions, Adam transitions, restarts, trajectory metrics, and roots
without importing fixture construction, progress, update, or aggregation
helpers. The controller recomputes numeric performance gates from the bounded
240-row aggregate array and recomputes structural relations from case metrics,
approved source hashes, exact count formulas, the two independent outer
projections, phase receipts, attack receipts, and committed roots. It does not
pretend the aggregate rows alone prove chronology or isolation.

### Canonical encodings and roots

Before JSON encoding, every finite negative zero becomes positive zero. Ordered
objects use the field order above and are encoded with `ensure_ascii=true`,
`allow_nan=false`, separators `(',',':')`, UTF-8, and one LF. A record-set root
preimage is ASCII `L2D-constraint-progress-v1/`, the exact record type, one zero
byte, then canonical JSON lines in the identity order above. State hashes use
record type `OptimizerState` and exact fields
`u:A[F;3],m:A[F;3],v:A[F;3],age:I,stall:I,progress:ProgressState,
incumbent_present:B,incumbent_sensitivity:F|N,
incumbent_source_batch:I,incumbent_source_member:I,
incumbent_center:A[F;3],restart_round:I`.

A present incumbent-center hash preimage is ASCII
`L2D-constraint-progress-v1/incumbent-center`, one zero byte, then exactly three
little-endian binary64 unbounded-center values after converting negative zero
to positive zero. An absent incumbent uses `incumbent_sensitivity=null`, source
batch and member `-1`, state center `[0.0,0.0,0.0]`, transition center source
`none`, and SHA-256 of ASCII
`L2D-constraint-progress-v1/absent-incumbent` plus one zero byte.

Fresh- and perturb-draw hash preimages are respectively ASCII
`L2D-constraint-progress-v1/fresh-draw` or
`L2D-constraint-progress-v1/perturb-draw`, one zero byte, little-endian
`uint64` seed, little-endian `int32` batch, then exactly one raw row-major
`(8,3)` little-endian binary64 transcript slice. These draw hashes preserve the
committed transcript bits, including any signed-zero bit. Their shapes and
slice offsets are replayed against the per-seed transcript before a
BatchReceipt is accepted.

Each trajectory event root contains its eight ordered ObservationRecords,
eight ordered TransitionRecords, then one BatchReceipt for each batch
`0..63`. The normalized twin hash
uses the same records after omitting only the `order` field; donor identity,
decision source, transcript use, restarts, and state hashes remain included.
Family, schema, intervention, source, attack, trajectory, aggregate, and final
projection roots use their same-named record sets and domains. Root-of-roots
preimages list lowercase child hashes in their declared record order, each
followed by LF. No root may omit a declared field or receipt.

The two-step sentinel is a separate `SentinelRecord` domain with exact fields
`sentinel_id:S="progress-consumer-v1",ordinal:I,member:I=0,
stored:ProgressState,canonical_tuple:DecisionTuple,donor_member:I=1,
donor_tuple:DecisionTuple,ablated_tuple:DecisionTuple,
canonical_decision:B,donor_decision:B,ablated_decision:B`. Ordinals are zero
and one and use the exact stored/tuple values already frozen above. Both rows
must be consumed by the production progress adapter and included in the
intervention root.

Forward/reverse equality is checked through the normalized twin hashes before
the four seed gaps are admitted to a world aggregate.

Per outer projection the exact totals are:

- 48 world records, four transcript records, and six phase receipts;
- 1,920 trajectories (`3*16*4*2*5`);
- 122,880 complete batches and BatchReceipts (`1,920*64`);
- 983,040 observations and 983,040 transitions; and
- 240 world aggregates (`3*16*5`).

Development and held-out metrics are equal-weight means of four paired seed
gaps within each world, then equal-weight means across the exact eight worlds.
Orders are reproduction twins, not extra inference units. Missing, duplicate,
nonfinite, or extra rows fail rather than being imputed.

The sanitized result has exact top-level keys in order:
`study_id`, `plan_revision`, `study_revision`, `contract_sha256`,
`transcript_root_sha256`, `status`, `action`, `world_aggregates`, and `cases`.
`plan_revision` is the 40-hex commit containing this frozen file;
`study_revision` is the later clean implementation commit. `contract_sha256`
is SHA-256 over ASCII `L2D-constraint-progress-v1/contract`, a zero byte, the
exact frozen plan blob, a zero byte, and canonical JSON of the normalized
study-registry object (including cases, sources, hashes, actions, and worker)
at `study_revision`. Every case contains exact
keys `case_id`, `passed`, and `metrics`; the controller rejects extras.

The twelve case metric objects have these exact ordered field/type schemas:

1. `world_records:I, constrained_references:I, reference_exclusions:I,
   development_worlds_per_family:I, heldout_worlds_per_family:I,
   formula_mismatches:I, reference_mismatches:I,
   nonpositive_denominators:I, duplicate_world_keys:I,
   implementation_root_sha256:H, oracle_root_sha256:H, roots_equal:B`.
2. `transcripts:I, values_per_transcript:I, transcript_values:I,
   trajectories:I, evaluations:I, unequal_arm_counts:I,
   order_twin_mismatches:I, committed_root_sha256:H,
   observed_root_sha256:H, roots_equal:B`.
3. `observations:I, schema_valid_observations:I, join_failures:I,
   capability_attacks:I, capability_rejected:I,
   capability_state_mutations:I, canonical_decisions:A[B;2],
   donor_decisions:A[B;2], ablated_decisions:A[B;2],
   implementation_schema_root_sha256:H, oracle_schema_root_sha256:H,
   implementation_intervention_root_sha256:H,
   oracle_intervention_root_sha256:H, roots_equal:B`.
4. `batches:I, batch_receipts:I, transitions:I, replay_mismatches:I,
   order_mismatches:I, reset_mismatches:I, incumbent_tie_mismatches:I,
   incumbent_state_mismatches:I, restart_events:I,
   implementation_state_root_sha256:H, oracle_state_root_sha256:H,
   roots_equal:B`.
5. `development_aggregates:I, development_receipts:I,
   heldout_receipts:I, forbidden_reads:I, heldout_source_in_development:I,
   development_outputs_in_heldout:I, implementation_source_root_sha256:H,
   oracle_source_root_sha256:H, roots_equal:B`.
6. `treatment_mean_gap:F, baseline_mean_gap:F, mean_improvement:F,
   heldout_wins:I, heldout_ties:I, heldout_losses:I,
   maximum_signed_world_harm:F, mean_gate:B, win_gate:B, harm_gate:B`.
7. `treatment_mean_gap:F, no_restart_mean_gap:F, mean_improvement:F,
   minimum_arm_evaluations:I, maximum_arm_evaluations:I,
   evaluation_parity:B, transcript_parity:B, comparator_gate:B`.
8. `control_mean_gap:F, baseline_mean_gap:F, mean_improvement:F,
   heldout_wins:I, maximum_signed_world_harm:F,
   substituted_mean_gate:B, substituted_win_gate:B,
   substituted_harm_gate:B, positive_gate_recovered:B`.
9. exactly the case-8 schema for the ablated arm.
10. `treatment_mean_gap:F, baseline_mean_gap:F,
    absolute_mean_difference:F, maximum_signed_world_harm:F,
    trajectories:I, mean_gate:B, harm_gate:B`.
11. `trajectories:I, observations:I, feasible_observations:I,
    nonunit_gaps:I, references_used:I, false_feasible_joins:I`.
12. `launches:I, projections_equal:B, maximum_stdout_bytes:I,
    stderr_bytes:I, surviving_children:I, attacks:I, attacks_rejected:I,
    attack_state_mutations:I, implementation_attack_root_sha256:H,
    oracle_attack_root_sha256:H, roots_equal:B`.

The capability matrix contains exactly twenty-two forbidden optimizer accesses:
family, split, world, bits, seed, `a`, `b`, `k`, `t`, `c`, threshold,
reference-x0, reference-sensitivity, denominator, gap, environment call,
evaluator call, oracle call, source call, and canonical aux bypass. Each is a
live adapter call; the final two accesses are a future-transcript read and a
transcript-fingerprint query. Every probe has exact rejection code
`capability-denied`, a consumer receipt, and zero optimizer-state mutations.

## Cases and gates

The future registry contract contains exactly these twelve ordered cases:

1. `family_replay`: 48 worlds, 32 constrained references, 16 impossible
   reference exclusions, eight development and eight held-out identities per
   family, zero formula/reference mismatches, positive denominators, unique
   keys, and equal fixture/oracle family roots.
2. `transcript_commitment`: four transcripts, 3,093 values each, 12,372 values,
   the committed root, exact arrays, 1,920 trajectories, 983,040 evaluations,
   equal arm counts, and byte-identical forward/reverse projections.
3. `typed_aux_and_intervention`: 983,040 schema-valid observation rows, zero
   join failures, all 22 live capability attacks rejected with zero state
   mutation, exact consumer sentinel decisions
   `[true,true]/[false,false]/[true,false]`, and equal fixture/oracle schema and
   intervention roots.
4. `chronology_replay`: 122,880 batches and BatchReceipts, 983,040 transitions,
   zero replay, order, reset, incumbent-tie, or incumbent-state mismatches,
   exact restart/reset rules, and equal independent state roots. Result-derived
   restart counts are reported but are not literal gates.
5. `development_and_source_isolation`: exactly 120 development world
   aggregates, three development and three held-out phase receipts, zero
   forbidden reads, no held-out source in development, no development output in
   held-out, and exact source roots. Development gaps are descriptive only.
6. `heldout_primary`: canonical arm 2 mean gap is at least `0.05` below arm 1,
   arm 2 wins at least six of eight canonical held-out world aggregates, and no
   canonical held-out world gap is more than `0.15` worse than arm 1. The harm
   value is exactly `max_world(arm2_mean_gap-arm1_mean_gap)` and must be at most
   `0.15`.
7. `restart_comparators`: canonical arm 2 mean held-out gap is at least `0.05`
   below arm 5, with exact evaluation and transcript parity for all five arms.
8. `shuffled_signal_control`: substitute arm 3 for arm 2 in every case-6
   clause. Its substituted three-clause conjunction must be false, and its mean
   improvement over arm 1 must be strictly less than `0.05`.
9. `ablated_signal_control`: substitute arm 4 for arm 2 under the exact same
   rule as case 8; its conjunction must be false and mean improvement strictly
   less than `0.05`.
10. `aligned_control`: over the aligned family's eight held-out worlds, absolute
    arm-2 minus arm-1 mean-gap difference is at most `0.03`, and no world harm
    exceeds `0.10`; signed harm is exactly
    `max_world(arm2_mean_gap-arm1_mean_gap)`. All sixteen worlds, four seeds,
    two orders, and five arms must nevertheless be present.
11. `impossible_control`: all 640 impossible-family trajectories and 327,680
    observations contain zero canonical feasible observations, every gap is
    exactly one, no feasible reference is used, and false-feasible joins are
    zero.
12. `process_and_sanitizer`: two fresh credential-scrubbed, network-disabled
    CPU outer projections are byte-identical, each has the totals above, stderr
    is empty, stdout is at most 1,048,576 bytes, no child survives, and all
    twelve attacks below are rejected through their real consumer.

The ordered malformed matrix is exact:

| attack ID | injection | rejection code |
| --- | --- | --- |
| `nan-loss` | observation `loss=float('nan')` | `nonfinite-loss` |
| `gradient-dtype` | gradient replaced by three ASCII strings | `gradient-dtype` |
| `gradient-shape` | gradient length changed from three to two | `gradient-shape` |
| `feasible-type` | Boolean feasibility replaced by integer one | `feasible-type` |
| `negative-penalty` | decision penalty changed to `-1e-9` | `negative-penalty` |
| `duplicate-observation` | duplicate the exact batch-0/member-0 row | `duplicate-key` |
| `missing-member` | remove batch-0/member-7 | `missing-member` |
| `wrong-arm` | replace arm with `unknown-arm` | `arm-identity` |
| `cross-seed` | change one transition seed to `2026083003` | `cross-seed-join` |
| `cross-order` | change one transition order tag | `cross-order-join` |
| `extra-result-field` | append top-level key `unexpected` | `result-schema` |
| `transcript-hash` | flip the first hex digit of one transcript hash | `transcript-hash` |

Every attack starts from a separately reconstructed valid minimal envelope,
keyed `canonical/world-0/seed-2026083001/forward/
constraint_lexicographic_progress/batch-0/member-0` (with the declared
batch member set where required),
enters the production parser and named consumer, and emits one ordered
`AttackReceipt` with `consumer_reached=true` and `state_mutations=0`. Expected
receipt paths are respectively `observation.loss`, `observation.gradient`,
`observation.gradient`, `observation.canonical_is_feasible`,
`observation.decision_penalty`, `observation.identity`,
`observation.identity`, `observation.arm`, `transition.seed`,
`transition.order`, `result`, and `transcript.sha256`.

World is the inference unit. A win is a world mean gap smaller by more than
`1e-12`; absolute difference at most `1e-12` is a tie. Every positive and
negative-control gate is required. Case pass values are recomputed relations,
not worker assertions. No post-result threshold, seed, world, control, or
subgroup change is permitted.

## Stopping rule and actions

Implementation is a later checkpoint. It may add only a dedicated synthetic
fixture and worker beneath `experiments/local_lab`, focused tests, the exact
registry contract, pre-result transcript commitment, approved source hashes,
and controller validation required for this ID. It must prove the allowed diff,
obtain a clean commit and green CI, and pass hostile read-only audits without
executing the fixture, worker, or optimizer for result-bearing metrics.

One still-later guarded invocation completes the study with exactly two fresh
outer projections. A pass records action
`constraint_progress_mechanics_ready_for_candidate_audit`. Any failed gate,
recovered negative control, malformed or nondeterministic output, timeout,
source drift, or process survivor records
`park_round2_constraint_progress_research`. There is no retry, top-up, seed
swap, threshold relaxation, same-ID repair, or selection against a terminal
result.

## Claim boundary

A pass may say only that the exact public-aux-guided rule improved the frozen
synthetic family under matched toy budgets, retained its aligned control, and
lost its gate under the declared signal interventions. It would justify only a
separate experiment-owned candidate implementation audit.

It would not establish UIFO improvement, hidden-topology generalization,
leaderboard gain, a Round-2 score, proximity to the best entrant, superiority
of RL, or permission to inspect a private panel, alter the protected package,
spend money, provision an accelerator, merge a PR, or upload a submission.
