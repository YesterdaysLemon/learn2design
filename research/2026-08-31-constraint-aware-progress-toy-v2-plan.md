# Constraint-aware progress toy v2

Date: 2026-08-31

Status: frozen plan-only checkpoint

Study ID: `constraint-aware-progress-toy-v2`

## Lineage and freeze boundary

This is a fresh study ID after `constraint-aware-progress-toy-v1` encountered a
pre-result infrastructure exception. V1 produced no authenticated scientific
result, no optimizer aggregate, and no completed-study entry. It is permanently
quarantined and mechanically refused by the controller.

The one-shot
[`constraint-progress-startup-forensics-v1` result](2026-08-31-constraint-progress-startup-forensics-v1-results.md)
established that the reproduced standard-library Windows Job, pipe, gate,
timeout, output-cap, and survivor boundary is deterministic and host-feasible.
It did not import the scientific runtime or identify the later V1-specific
failure layer. The checked-in V1 worker also already called its child gate
before `_load_runtime()`, so neither import order nor any other later layer is
an established cause. V2 therefore tests one outcome-blind infrastructure
hardening hypothesis: isolate framing, supervision, and exception projection in
a small standard-library bootstrap so a later worker failure becomes a closed
sanitized stage code rather than raw stderr. This is not claimed as a causal
fix. It does not select a family, seed, threshold, arm, control, or claim
against an observed scientific result.

The complete scientific predecessor contract is the LF-normalized blob
[`2026-08-30-round1-feedback-and-round2-program.md`](2026-08-30-round1-feedback-and-round2-program.md)
with SHA-256
`9a9c4536a28ee6fdea8f74387be5975943eaab551a3c959f41e4d3d49ba86c96`.
Every scientific definition, typed record, chronology rule, formula, count,
join, comparator, intervention, threshold, stopping action, and claim boundary
in that blob is incorporated into V2 unchanged, subject only to this closed
identity translation:

1. study ID `constraint-aware-progress-toy-v1` becomes
   `constraint-aware-progress-toy-v2`;
2. only the outer contract-digest domain changes from
   `L2D-constraint-progress-v1/contract` to
   `L2D-constraint-progress-v2/contract`;
3. source/module/mode names use the V2 ID; and
4. the V1 monolithic recursive worker entry/supervision boundary is replaced by
   the frozen V2 bootstrap boundary below.

All transcript, record-set, state, draw, incumbent, sentinel, and root-of-roots
domains—including literal `L2D-constraint-progress-v1` and
`progress-consumer-v1` labels—remain unchanged protocol labels. This preserves
the already frozen transcript root and makes evidence-semantic drift
detectable. V2 is distinguished by its study ID, plan/study revisions, outer
contract digest, approved source hashes, and worker mode.

No other difference is permitted. If implementation cannot satisfy that
closed translation, V2 is rejected before registration or execution rather
than amended. This document is the freeze boundary; no result-bearing code is
executed in this checkpoint.

## Falsifiable question

> On the same deterministic smooth constrained multi-basin toy family, can the
> same public-aux-guided progress policy reduce final feasible regret relative
> to the same protected raw-total-loss progress policy under identical Adam
> dynamics, objective-call budgets, initial populations, and random
> transcripts, while losing that advantage when the constraint signal is
> shuffled or ablated and avoiding material harm when sensitivity and
> feasibility are aligned?

A pass can validate only this frozen synthetic causal mechanism. V2 is not an
estimate of Round-2 score and is not a candidate evaluation.

## Exact preserved scientific identity

The following table restates the result-bearing identity. Every value is exact.

| Component | Frozen V2 value |
| --- | --- |
| arithmetic | IEEE-754 binary64 |
| active coordinates | branch-stable `x = 4*sigmoid(u)-2`, `u in R^3` |
| world identities | all `(A,B,K,T) in {0,1}^4`, lexicographic |
| coefficients | `a=.80+.20*A`, `b=-.50+1.00*B`, `k=.50+.50*K`, `t=.10+.06*T`, `c=-.25` iff `A xor B == 0`, otherwise `.25` |
| canonical sensitivity | `(x0^2-a^2)^2 + t*x0 + k*(x1-b)^2 + .5*(x2-c)^2` |
| canonical constraint | `violation=max(0,-x0)`, `penalty=.02*violation^2`, feasible iff `x0>=0` |
| aligned family | only the tilt changes from `t` to `-t` |
| impossible family | only threshold changes to `x0>=2.25` and violation to `max(0,2.25-x0)` |
| split | even bit parity development; odd bit parity untouched held-out |
| seeds | `2026083001`, `2026083003`, `2026083007`, `2026083011` |
| transcript RNG | NumPy `SeedSequence([20260830,seed])` and `PCG64` |
| transcript draws | one `(7,3)` suffix, then 64 ordered `(8,3)` fresh and `(8,3)` perturb arrays; 3,093 values per seed |
| transcript root | `9c250412d296b7e60a5ab0e02f4cf69925d165bb6c3f61e3a29e00b475d99edd` |
| population and batches | population 8; 64 complete batches |
| evaluations | 512 per trajectory; no post-final-update evaluation |
| initialization | lane zero `u=(0,0,0)` plus the seven transcript suffix rows |
| learning rates | `geomspace(.03,.15,8)` |
| Adam | `beta1=.9`, `beta2=.999`, `epsilon=1e-8` |
| progress tolerance | `1e-7` |
| toy patience | 8 |
| gradient clip | member L2 norm one with denominator offset `1e-12` |
| restart noise | `.35*max(.10,1-(batch+1)/64)` |
| arms | protected raw, constraint lexicographic, cyclic-donor shuffled, ablated, no-restart |
| orders | exact forward arm order and exact reverse twin |
| inference unit | world; order is a reproduction twin, not an inference unit |

The constrained scalar reference remains 80 bisections of
`d(z)=4*z*(z^2-a^2)+q` on the predecessor's exact brackets, followed by its
exact midpoint. Formula and scalar-oracle paths remain independent and must
agree to `1e-12`. Normalized feasible gap, absent-feasible gap one, no clipping,
negative-gap rejection below `-1e-10`, and impossible-family exclusion are
unchanged.

The runtime remains CPython `3.13.14` x64 on `AMD64` with `numpy==2.5.1`,
executable SHA-256
`ad169f4cb4bfb78c7a5c030a4529c19d6643276778e33994c93e145b6191c3ec`,
NumPy initializer SHA-256
`a6958cb364663b7acce81ccfd58eeb65a2b34d5376157f924777b97211a73be4`,
metadata SHA-256
`6ae45122ee97050e48849438320430d05f01814f72e66e69cbeed027d2c6a1e8`,
PCG64 module SHA-256
`210bd962e911039f1639d0137f6e41444e37db23aba1622635d9dba8abc6a1c9`,
and SeedSequence module SHA-256
`08355a330efec79a840b5767bb5356ad21e3b0f14acce9a3c969208626daad7f`.

## Exact optimizer and intervention contract

The V1 chronology is unchanged: evaluate all eight rows; update the sealed
physical feasible incumbent; consume the arm-specific progress tuple; perform
the exact Adam update; derive the restart mask after that update; then replace
and reset selected members. Restart-round selection, tie retention, centering,
standardization, fresh-row indexing, reset state, and transcript consumption
are byte-for-byte semantic invariants.

Canonical feasibility and best-feasible scoring are never intervened on.

1. `protected_raw_progress` ignores the decision tuple and compares finite
   total loss with tolerance `1e-7`.
2. `constraint_lexicographic_progress` uses canonical feasibility, penalty,
   and sensitivity with the exact feasible/infeasible lexicographic ordering
   frozen in V1.
3. `shuffled_progress_control` receives the complete tuple from donor member
   `(member+1)%8` in the same authenticated batch.
4. `ablated_progress_control` receives exactly `(false,0.0,0.0)`.
5. `no_restart_comparator` uses protected raw progress and identical Adam but
   forces its authenticated restart mask false.

The production progress adapter must consume the same two-row sentinel and
produce exact decisions `[true,true]`, `[false,false]`, and `[true,false]` for
canonical, donor, and ablated tuples. Static arm differences remain limited to
arm ID, progress comparator, tuple adapter, and restart-enabled Boolean.

## Information, source, and evidence boundary

`WorldRecord` remains environment/evaluator-only. The optimizer receives only
current state, total loss, sanitized total gradient, its declared three-field
decision tuple, budget fraction, and the sealed feasible-incumbent
presence/center capability. It never receives coefficients, threshold,
reference, family, split, world, bits, seed, transcript identity, gap, source,
scorer, oracle, or canonical auxiliary data outside its active adapter.

Development and held-out worlds remain physically separate fresh phase
processes for each family: exactly six phase receipts per projection. A
development process receives only its eight development records and a connected
exploding held-out sentinel. A held-out process receives only its eight
held-out records, immutable contract/transcript roots, and no development
metric, state, action, or output. Every trajectory resets all optimizer state.

The exact V1 record schemas remain normative after the limited identity translation:
`WorldRecord`, `ObservationRecord`, `ProgressState`, `DecisionTuple`,
`TransitionRecord`, `BatchReceipt`, `TrajectoryRecord`, `PhaseReceipt`,
`WorldAggregateRecord`, `AttackReceipt`, `SentinelRecord`, and
`OptimizerState`. Field order, scalar types, enum sets, null sites, signed-zero
normalization, canonical JSON, root construction, identity order, state hashes,
draw hashes, normalized twin hashes, and root-of-roots construction are
unchanged. Implementations must compare their complete schema projection to the
predecessor blob under the four-item identity translation and reject any extra
semantic delta before optimizer construction.

Each complete projection still contains exactly 48 worlds, four transcripts,
six phase receipts, 1,920 trajectories, 122,880 batches and batch receipts,
983,040 observations, 983,040 transitions, and 240 world aggregates. Missing,
duplicate, nonfinite, reordered, or cross-identity rows fail closed.

The same 22 live forbidden-capability accesses and the same 12 malformed-record
attacks must traverse their real consumers, produce their exact predecessor
rejection codes and paths, and leave zero optimizer-state mutations. Literal
pass flags, asserted rejection counts, disconnected sentinels, or mock-only
proofs are forbidden.

## Sole implementation delta: framed, exception-sealed bootstrap

V1 correctly read its recursive child Job gate before `_load_runtime()`, but
its framing, process supervision, scientific dispatch, and top-level exception
behavior lived in one worker module. An unexpected later exception escaped
`main()` and the controller observed forbidden stderr. The raw text remains out
of scope, and no specific later layer is inferred.

V2 instead adds exactly two study sources:

- `experiments/local_lab/constraint_aware_progress_toy_v2.py`: all scientific
  fixture, replay, optimizer, phase, projection, and aggregation logic relocated
  without semantic change; and
- `experiments/local_lab/constraint_aware_progress_toy_v2_worker.py`: a
  standard-library-only bootstrap and process supervisor.

Before entry dispatch the bootstrap's direct import statements are exactly
`from __future__ import annotations`, `import os`, `import struct`, and
`import sys`. Their normal interpreter/standard-library transitive closure is
allowed and is not represented as an empty process. `argparse` is forbidden.
All other standard-library imports (`ctypes`, `hashlib`, `importlib`, `json`,
`pathlib`, `subprocess`, `tempfile`, `threading`, `time`, and typing helpers)
are deferred until after nested framing succeeds, or until a non-nested mode is
authenticated. NumPy, the V2 scientific module, V1, `submission`, JAX, and all
optimizer/evaluator modules are forbidden before that boundary. Module import
is inert.

The entry function first performs a side-effect-free exact comparison of
`sys.argv[1:]` against `("--mode", mode)` where `mode` is one of exactly:

- `constraint-aware-progress-toy-v2`;
- `constraint-aware-progress-toy-v2-projection`;
- `constraint-aware-progress-toy-v2-phase`; or
- `constraint-aware-progress-toy-v2-runtime-probe`.

Malformed or extra arguments emit the closed `environment` bootstrap-failure
receipt below and exit `70` with zero stderr. This manual mode discovery is the
only operation allowed before a nested frame read.

For recursive modes `constraint-aware-progress-toy-v2-projection` and
`constraint-aware-progress-toy-v2-phase`, the bootstrap's first main-path
operation after side-effect-free mode discovery is an exact blocking frame
read. The parent must, in order:

1. create a fresh unnamed kill-on-close Windows Job;
2. spawn `sys.executable -S -P` pointing at the V2 bootstrap with file-backed
   stdout and stderr, pipe stdin, the frozen credential-scrubbed environment,
   and no breakaway flag;
3. assign the child and verify exact membership in that fresh Job;
4. write ASCII gate `L2D-CONSTRAINT-PROGRESS-V2\n`, little-endian `uint32`
   payload length, canonical payload bytes, and EOF; and
5. enforce the predecessor timeout, 1,048,576-byte cap, joined non-daemon input
   writer, zero stderr, clean exit, zero active Job processes, and zero
   survivors.

The projection-mode payload is exactly zero bytes. The phase-mode payload is
the predecessor's canonical compact JSON object with exact ordered keys
`family`, `split`, `contract_sha256`, `transcript_root_sha256`, and `worlds`;
its records and ordering are unchanged. The bootstrap reads exactly the gate,
four length bytes, the declared payload, and one final byte which must be EOF.
The payload cap is exactly 1,048,576 bytes. Short gate, wrong gate, short
length, oversize, short payload, trailing input, assignment failure,
membership failure, timeout, output overflow, writer failure, stderr, nonzero
exit, malformed JSON, or survivor fails before scientific output is accepted.
Only after the full frame, EOF, mode, environment, contract identities, and
packet schema authenticate may the bootstrap import
`constraint_aware_progress_toy_v2` and dispatch the requested mode in that same
already-assigned process. For both recursive modes it must invoke the actual
socket/network-disable gate after frame and contract authentication but before
that import; the environment marker alone is not enforcement. A hostile test
must exercise the real recursive child path and prove network constructors are
denied before scientific dispatch.

The top-level `constraint-aware-progress-toy-v2` mode is still a direct
controller child, matching the V1 path whose dedicated import/runtime preflight
passed. A new V2-specific `constraint-aware-progress-toy-v2-runtime-probe`
path must authenticate the interpreter, package files, NumPy/PCG64 identities,
network gate, and contract environment without invoking V1 or any result mode.
The V1 runtime-probe function and worker remain quarantined and unreachable.
Before the full mode imports the scientific module it disables network and
authenticates the Python/contract environment; the scientific entry validates
the frozen NumPy identities immediately after import and before world or
optimizer construction. The controller continues to capture stdout/stderr to
exclusive private temporary files, enforce the one-hour tree timeout and output
cap, and remove those files after validation.

Scientific entry functions return an object and never write stdout directly;
the bootstrap alone validates, canonicalizes, and writes one result line.
Every exception after entry begins is caught before Python can write a
traceback. The bootstrap emits at most one closed failure JSON object to stdout,
zero stderr, and exits `70`. That object has exact ordered keys and types
`schema_version:I=1`, `study_id:S="constraint-aware-progress-toy-v2"`,
`mode:S` from the four V2 modes or exact `invalid`, and `stage:S` from
`gate|length|payload|environment|import|dispatch|output|cleanup`. It contains no
exception text, path, command, environment value, packet, metric, or raw
payload. This guarantee begins at entry-function execution; an interpreter or
script-load failure before entry is outside the catch boundary, remains a
zero-result infrastructure failure, and is still rejected by the controller's
zero-stderr rule. Parent and controller accept the failure schema only on
nonzero exit, treat it only as sanitized infrastructure failure, and park; it
can never be interpreted as a scientific result or case.

The implementation must prove with hostile tests that monkeypatched forbidden
imports cannot run before membership and frame authentication, that every
framing/exception branch emits zero stderr, and that all bootstrap-created
Jobs, processes, threads, and files are gone. It must not reproduce V1's
terminal invocation or inspect its raw stderr. It must also prove that moving
the process supervisor out of the scientific module changes no scientific
constant, record schema, root preimage, chronology, action stream, aggregate,
or controller gate under the four-item translation.

## Exact twelve result cases

The V2 registry retains the same ordered 12 case IDs and exact predecessor
metric schemas.

1. `family_replay`: 48 worlds; 32 references; 16 exclusions; balanced splits;
   zero mismatches; positive denominators; unique keys; equal roots.
2. `transcript_commitment`: four 3,093-value transcripts; 12,372 values;
   exact root; 1,920 trajectories; 983,040 evaluations; equal arm counts; exact
   forward/reverse twins.
3. `typed_aux_and_intervention`: all 983,040 observations valid; zero join
   failures; all 22 capability attacks rejected without mutation; exact
   sentinel decisions; equal schema/intervention roots.
4. `chronology_replay`: 122,880 batches/receipts and 983,040 transitions; zero
   replay, order, reset, incumbent-tie, or incumbent-state mismatches; equal
   independent roots.
5. `development_and_source_isolation`: 120 development aggregates; three
   development and three held-out receipts; zero forbidden reads or cross-split
   payloads; equal source roots.
6. `heldout_primary`: treatment mean gap improves by at least `0.05`, wins at
   least 6 of 8 canonical held-out worlds, and maximum signed world harm is at
   most `0.15`.
7. `restart_comparators`: treatment mean held-out gap improves by at least
   `0.05` over no-restart with exact evaluation/transcript parity.
8. `shuffled_signal_control`: the complete case-6 conjunction is false and
   mean improvement is strictly below `0.05`.
9. `ablated_signal_control`: the complete case-6 conjunction is false and mean
   improvement is strictly below `0.05`.
10. `aligned_control`: absolute treatment/baseline held-out mean difference is
    at most `0.03`, maximum signed world harm at most `0.10`, with every frozen
    trajectory present.
11. `impossible_control`: all 640 trajectories and 327,680 observations have
    zero feasible observations, unit gaps, no references, and zero false joins.
12. `process_and_sanitizer`: two fresh full projections are byte-identical,
    stderr is zero, stdout is at most 1,048,576 bytes, no child survives, and
    all 12 malformed attacks are rejected through their real consumers with
    zero mutation and equal roots.

World wins retain the `1e-12` tolerance. Every positive and negative gate is
required and independently recomputed by the worker oracle and controller. The
success action remains
`constraint_progress_mechanics_ready_for_candidate_audit`; the failure action
remains `park_round2_constraint_progress_research`.

## Registration, audit, and execution separation

The next checkpoint may implement only the two V2 sources, focused tests, exact
12-case registry entry, V2-specific controller validator and worker allowlist,
five approved source hashes (`dependency_lock`, `fixture_source`,
`lab_protocol`, `study_plan`, `worker_source`), and normalized pinned registry
digest. It must preserve the V1 historical registry entry and explicit
controller refusal. It must not invoke V2 or move private controller state.

The controller's pending-study calculation must exclude every ID in
`QUARANTINED_STUDIES` as well as every completed ID. V1 remains present in the
registry as historical evidence but can never keep the queue pending, become a
completed study, or be selected for execution. A regression test must complete
a later synthetic study with V1 absent from `completed_studies` and prove the
controller reaches the correct post-cycle state without selecting V1.
When V2 is the only nonterminal, nonquarantined registry entry, a passed V2
must leave exact state `awaiting_study` with stop reason
`no_approved_study_pending`.

Before a clean pre-result commit, independent hostile audits must verify:

- the four-item scientific-contract translation has no extra semantic delta;
- the bootstrap import boundary, gate framing, membership chronology, cleanup,
  stderr, and sanitized failure paths are exercised through real consumers;
- the complete family, transcript, chronology, controls, schemas, sources,
  attacks, aggregates, and controller recomputation match this plan; and
- the protected submission tree and artifact hashes are unchanged.

Focused tests run first and at most one full repository verification pass may
run in that implementation checkpoint. A later heartbeat may perform at most
one guarded local-CPU V2 invocation only after clean revision, green CI, source
approval, registry digest, controller `awaiting_study` state, absent stop/lease,
and an owner-authorized atomic resume that preserves the completed ledger and
failure streak and appends a sanitized resume event.

That resume is a dedicated, lease-protected controller operation, not a manual
file edit. It requires current state `parked`, `active_cycle=null`, no stop
marker, no existing lease, unchanged completed ledger, V1 in
`QUARANTINED_STUDIES`, and V2 approved but not completed. Under the global lease
it atomically rewrites only `status` to `awaiting_study`, `stop_reason` to null,
and `updated_utc`; `failure_streak` and `completed_studies` remain byte-for-byte
equal. It then appends exactly one sanitized event with fields
`event_schema_version:1`, `event:"controller_resumed"`,
`from_status:"parked"`, `to_status:"awaiting_study"`,
`reason:"owner_authorized_v1_quarantine_recovery"`,
`retired_study:"constraint-aware-progress-toy-v1"`, and `utc`. The lease is
released only after state and event verification. Any precondition, write,
event, or verification failure leaves the lab parked or stops further action;
it never starts V2 in the resume operation.
Fault-injection tests must prove a state-write or event-append failure cannot
leave an unlogged `awaiting_study` state.

There is no retry, top-up, seed swap, threshold relaxation, same-ID repair,
subgroup selection, or tuning against a terminal result. A pre-result confound
quarantines V2 without invocation. A failed, malformed, timed-out,
nondeterministic, drifted, or surviving-process invocation parks the controller
and ends mutation.

## Claim boundary

A pass may say only that this exact public-aux-guided rule improved the frozen
synthetic family under matched toy budgets, retained its aligned control, and
lost its gate under the declared signal interventions after the framed,
exception-sealed process boundary. It could authorize only a separate
experiment-owned candidate
implementation audit.

It would not establish UIFO or hidden-topology improvement, learning necessity,
leaderboard gain, a Round-2 score, proximity to `0.14`, superiority of RL,
accelerator value, or permission to use official data, inspect private outcome
panels, alter the protected package, spend money, provision hardware, merge a
PR, build an upload artifact, or interact with the portal.
