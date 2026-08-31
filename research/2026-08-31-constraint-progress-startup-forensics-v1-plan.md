# Constraint-progress startup forensics v1

Date: 2026-08-31

Status: frozen non-result diagnostic plan

Checkpoint ID: `constraint-progress-startup-forensics-v1`

## Owner decision and failed-study disposition

The owner explicitly authorized recovery work and a return to local Round-2
research after the guarded `constraint-aware-progress-toy-v1` attempt parked.
Permitted controller metadata records one start at revision
`4098ce1ba25a163e96ac3cf735b7cd7e419bc64c`, followed about two seconds later
by `RuntimeError: local-lab worker emitted forbidden stderr`. The controller
completed its dedicated runtime preflight and entered `cycle_started`, but it
did not authenticate a terminal result and did not add the study ID to
`completed_studies`.

This is an infrastructure failure, not a positive or negative scientific
result. No optimizer aggregate, case result, or competition claim may be
inferred. The V1 plan and implementation remain immutable evidence. V1 is
policy-quarantined and must never be invoked again. Mechanical retirement from
the runnable registry, or an equivalent explicit controller refusal, is
mandatory before any owner-authorized unpark; a fabricated completed-study
record is forbidden.

Raw private stderr, cycle files, histories, and topology evidence remain out of
scope. The only facts used here are the sanitized controller state and event
projection permitted by the laboratory protocol.

## Frozen question

> Can a standard-library-only probe reproduce the two-level Windows Job,
> process, pipe, gate, timeout, output-cap, and survivor boundaries used before
> the failed worker could reach fixture or optimizer construction, and report
> whether a deterministic failure occurs in that reproduced boundary without
> observing any result-bearing metric?

This document is the freeze boundary. The diagnostic may conform to this
contract but may not expand into the failed worker, fixture, learner, or study
result path.

## Scope and exclusions

The probe may use only Python's standard library and Windows process/Job APIs.
It may create and destroy only its own ephemeral processes, pipes, temporary
files, threads, and unnamed Jobs. Pre-existing operating-system objects are
read-only: the probe may not take ownership, change a DACL, mutate a system
file, or alter any parent Job.

It must not:

- import or execute either V1 source module;
- import NumPy, JAX, `submission`, or any optimizer/evaluator package;
- call `tools/run_local_lab.py` or touch the sibling private lab root;
- read raw stderr from the failed cycle or any other private evidence;
- construct a toy world, transcript, observation, gradient, objective, policy,
  candidate, or score;
- use the official dataset, a topology panel, a GPU, network access, Docker,
  SSH, paid compute, or a provider API; or
- alter `submission/`, the retained ZIP, a terminal record, or the failed V1
  plan and implementation.

The only retained output is the closed JSON receipt below and its SHA-256
sidecar in a fresh OS-temp directory outside both project roots. No exception text, absolute path,
private environment value, command line, child payload, or OS handle is
retained.

## Source, runtime, and environment contract

Implementation adds exactly one source,
`experiments/local_lab/constraint_progress_startup_forensics_v1.py`, with
mutually exclusive `runner`, `parent`, and `child` modes. It contains no
module-level execution outside `if __name__ == "__main__": main()`.

The runner is invoked with the same system Python identity accepted by the V1
runtime preflight, plus `-S -P`. Before launching anything it must verify the
public registry identities: CPython `3.13.14`, `64bit`, machine `AMD64`, and
executable SHA-256
`ad169f4cb4bfb78c7a5c030a4529c19d6643276778e33994c93e145b6191c3ec`.

For parent and child launches the runner projects only these ambient names,
matched case-insensitively and preserving their actual spelling and values:

`COMSPEC`, `LD_LIBRARY_PATH`, `PATH`, `PATHEXT`, `PROGRAMDATA`, `SYSTEMDRIVE`,
`SYSTEMROOT`, `TEMP`, `TMP`, `TMPDIR`, `VIRTUAL_ENV`, and `WINDIR`.

It then overwrites exactly:

- `CUDA_VISIBLE_DEVICES=""`;
- `JAX_PLATFORMS="cpu"`;
- `LEARN2DESIGN_LOCAL_LAB_NETWORK="disabled"`;
- `PYTHONHASHSEED="0"`; and
- `XLA_PYTHON_CLIENT_PREALLOCATE="false"`.

No `L2D_*` value is inherited or added. The environment commitment is
`sha256(b"L2D-startup-forensics-v1/environment\0" + concat)` where `concat`
contains each final `(name.upper(), value)` pair in ascending upper-name order
as UTF-8 name, NUL, UTF-8 value, NUL. Only the digest and pair count are
reported; the runner independently verifies the same digest in both parents
and all children.

The implementation embeds the eventual plan commit as `PLAN_REVISION`. At
execution the runner requires a clean worktree, obtains `PROBE_REVISION` from
`git rev-parse HEAD`, and hashes LF-normalized bytes of this plan and its single
probe source. The contract digest is:

`sha256(b"L2D-startup-forensics-v1/contract\0" + plan_bytes + b"\0" + source_bytes)`.

The probe executes only from a clean committed revision. Tests independently
recompute every identity and digest.

## Exact framing

The literal gate is ASCII `L2D-STARTUP-FORENSICS-V1\n`. A valid child frame is
exactly:

`gate || uint32_le(payload_length) || payload || EOF`.

Payload length is at most `32768`. The child's first main-path operation is an
`read_exact` loop for the gate; it then reads the four-byte length, the exact
payload, and one final byte which must be EOF. EOF before the requested count,
the wrong gate, an oversized length, or a trailing byte is rejected. The child
must catch every declared framing failure, emit a closed receipt on stdout,
emit zero stderr, and exit zero; unexpected exceptions remain nonzero/stderr
and fail the outer probe.

Payload bytes for length `n` are the first `n` bytes of concatenated
`sha256(b"L2D-startup-forensics-v1/payload\0" + uint32_le(index)).digest()` for
indices starting at zero. There are no random seeds or selectable parameters.

Canonical JSON is UTF-8, ASCII-escaped, finite, compact
(`separators=(",", ":")`), insertion-ordered exactly as frozen below, followed
by one LF. Every parser rejects duplicate object keys. JSON `null` is permitted
only where a field below is explicitly typed `|null`.

## Exact two-level process topology

The runner performs two fresh, sequential outer runs. For each it:

1. creates an unnamed outer Windows Job with
   `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`;
2. launches `sys.executable -S -P <probe> --mode parent` with repository root
   as working directory, the frozen environment, pipe stdin, and file-backed
   stdout/stderr;
3. assigns the parent to the outer Job and verifies parent membership;
4. only then writes the literal outer gate
   `L2D-STARTUP-FORENSICS-OUTER-V1\n` and closes stdin; and
5. enforces a 60-second process-tree timeout and 64-KiB combined output cap,
   then verifies zero stderr, clean exit, zero active outer-Job processes, and
   zero survivors after Job closure.

The parent's first main-path operation reads the exact outer gate. It then runs
the eleven child cases below in order. Every child launch includes exact
`--mode child --case <case_id>` arguments from the frozen case enum. For every
inner-Job case the parent creates a fresh unnamed kill-on-close Job, launches
the child with `CREATE_NEW_PROCESS_GROUP`, assigns it to the inner Job, and
verifies `IsProcessInJob(child_handle, inner_job_handle)` **before** writing
any child frame. Child-side membership is intentionally not inferred: every
child also belongs to the outer Job, and a null-Job query cannot distinguish
the two. All input writers are non-daemon threads and must join.

## Complete child cases

There are exactly eleven fresh child launches per parent and twenty-two total:

1. `gate_only_empty`: no inner Job; valid contiguous frame; payload length `0`;
   expected child status `accepted`.
2. `nested_empty`: inner Job required; valid contiguous frame; length `0`;
   expected `accepted`.
3. `nested_fragmented`: inner Job required; valid frame and length `4097`;
   the entire frame is written in fragment sizes
   `(1,2,3,5,8,13,21,34,55,89)` repeated to completion; expected `accepted`.
4. `nested_large`: inner Job required; valid contiguous frame; length `32768`;
   expected `accepted`.
5. `nested_wrong_gate`: inner Job required; replace only the gate's final LF
   with `!`, then send a valid zero-length suffix and EOF; expected `rejected`
   with child error `gate_read`.
6. `nested_truncated_gate`: inner Job required; send all but the final gate byte
   and EOF; expected `rejected` with child error `gate_read`.
7. `nested_no_gate`: inner Job required; send EOF immediately; expected
   `rejected` with child error `gate_read`.
8. `nested_short_length`: inner Job required; send a valid gate followed by
   only three zero length bytes and EOF; expected `rejected` with child error
   `length_read`.
9. `nested_short_payload`: inner Job required; send a valid gate, declared
   length `5`, four deterministic payload bytes, and EOF; expected `rejected`
   with child error `payload_read`.
10. `nested_oversized_length`: inner Job required; send a valid gate, declared
    length `32769`, and EOF; expected `rejected` with child error `payload_cap`.
11. `nested_trailing_input`: inner Job required; send a valid zero-length frame
    followed by byte `0x01` and EOF; expected `rejected` with child error
    `trailing_input`.

Thus each successful parent records eleven launches, ten inner-Job assignments,
ten parent-side pre-gate inner memberships, four accepted frames, seven rejected controls, zero stderr bytes,
and zero surviving descendants. The gate-only case proves framing independently
of nested assignment; the seven malformed controls prove every declared framing
failure is observable independently of Job failures.

## Closed child and parent schemas

Every child emits exactly these ordered keys and types:

1. `schema_version: int` exact `1`;
2. `case_id: str` from the eleven-case enum and equal to the authenticated
   launch argument;
3. `environment_sha256: str` lowercase 64-hex;
4. `status: str` in `accepted|rejected`;
5. `error_code: null|str`, null iff accepted, otherwise in
   `gate_read|length_read|payload_read|payload_cap|trailing_input`;
6. `payload_bytes: int` in `0..32768`;
7. `payload_sha256: str` lowercase 64-hex, equal to the accepted payload hash;
   every rejected receipt is sanitized to zero payload bytes and SHA-256 of
   empty bytes regardless of how much malformed input preceded rejection.

Every parent emits exactly these ordered keys and types:

1. `schema_version: int` exact `1`;
2. `checkpoint_id: str` exact checkpoint ID;
3. `environment_pairs: int` positive;
4. `environment_sha256: str` lowercase 64-hex;
5. `outer_membership_before_gate: bool`;
6. `children: list[ChildObservation]` in frozen case order, length `0..11` on
   infrastructure failure and exactly `11` on pass;
7. `child_launches: int`;
8. `inner_assignments: int`;
9. `inner_memberships_before_gate: int`;
10. `accepted_frames: int`;
11. `rejected_frames: int`;
12. `child_stderr_bytes: int`;
13. `surviving_descendants: int`;
14. `passed: bool`;
15. `error_code: null|str`.

`ChildObservation` contains exactly these ordered keys:

1. `case_id: str`;
2. `inner_job_required: bool`;
3. `inner_job_assigned: bool`;
4. `membership_before_gate: bool`;
5. `write_mode: str` in
   `contiguous|fragmented|wrong_gate|truncated_gate|no_gate|short_length|short_payload|oversized_length|trailing_input`;
6. `expected_payload_bytes: int`;
7. `expected_payload_sha256: str` lowercase 64-hex;
8. `child_receipt: object|null`, with the exact child schema when present;
9. `stdout_bytes: int`;
10. `stderr_bytes: int`;
11. `return_code: int|null`;
12. `surviving_descendants: int`;
13. `passed: bool`;
14. `error_code: null|str`.

Parent/observation infrastructure errors use only this precedence-ordered enum:
`outer_gate_read`, `job_create`, `job_limit`, `child_spawn`, `job_assign`,
`job_membership`, `gate_write`, `timeout`, `output_cap`, `writer_join`,
`child_stderr`, `child_exit`, `child_schema`, `child_relation`, `job_query`, or
`survivor`. The first error stops further child launches. Counts and the child
list describe only completed attempts; `passed` must be false and `error_code`
must be the earliest code. No literal success field substitutes for a real
consumer receipt.

On a parent infrastructure failure, the observation for the in-flight case is
appended and later cases are omitted. `child_launches` counts only successful
`Popen` returns; `inner_assignments` and `inner_memberships_before_gate` count
only completed calls. The partial observation is frozen by stage:

- `job_create|job_limit|child_spawn`: assignment and membership false,
  `child_receipt=null`, stdout/stderr zero, `return_code=null`, survivors zero;
- `job_assign`: launch counted, assignment/membership false, receipt null,
  captured byte counts retained, and the post-cleanup integer return code;
- `job_membership`: launch and assignment counted, membership false, receipt
  null, captured bytes, and post-cleanup return code;
- `gate_write|timeout|output_cap|writer_join|child_stderr|child_exit|child_schema`:
  all reached launch/assignment/membership facts retained, receipt null,
  captured bytes, and post-cleanup return code;
- `child_relation`: the schema-valid child receipt is retained, but the
  observation fails its independently recomputed relation;
- `job_query`: all earlier facts are retained and
  `surviving_descendants == -1` denotes that no count was available; and
- `survivor`: all earlier facts are retained with the observed strictly
  positive survivor count.

For the no-inner-Job case, assignment and membership are always false and no
inner-Job error is legal. For every other failed observation whose survivor
query succeeds, `surviving_descendants` is the observed nonnegative count after
forced cleanup. `accepted_frames` and `rejected_frames` count only schema-valid
child receipts with the corresponding status. `child_stderr_bytes` always
equals the sum of retained observation byte counts.

For a passing parent, `child_launches == len(children) == 11`,
`inner_assignments == inner_memberships_before_gate == 10`,
`accepted_frames == 4`, `rejected_frames == 7`, and every count equals the
corresponding sum over observations. Each observation's `case_id` equals its
child receipt's ID. Accepted receipts equal the frozen payload length/hash;
every rejected receipt has `payload_bytes == 0` and the SHA-256 of empty bytes.
`child_stderr_bytes` is the sum over observations, and every observation and
parent survivor count is zero.

## Runner envelope and independent validation

The runner emits exactly these ordered keys:

1. `schema_version: int` exact `1`;
2. `checkpoint_id: str` exact checkpoint ID;
3. `plan_revision: str` lowercase 40-hex;
4. `probe_revision: str` lowercase 40-hex;
5. `plan_sha256: str` lowercase 64-hex;
6. `probe_source_sha256: str` lowercase 64-hex;
7. `contract_sha256: str` lowercase 64-hex;
8. `python_executable_sha256: str` exact frozen hash;
9. `python_version: str` exact `3.13.14`;
10. `python_architecture: str` exact `64bit`;
11. `machine: str` exact `AMD64`;
12. `runs: list[OuterObservation]` length `0..2` on runner failure and exactly
    `2` on pass;
13. `runs_equal: bool`;
14. `passed: bool`;
15. `error_code: null|str`;
16. `receipt_root_sha256: str` lowercase 64-hex.

`OuterObservation` has ordered keys `body_sha256: str`, `body_bytes: int`,
`stderr_bytes: int`, `return_code: int|null`, `surviving_processes: int`, and
`body: ParentReceipt|null`. Runner-only errors use the precedence-ordered enum
`runtime_identity`, `dirty_worktree`, `outer_job_create`, `outer_job_limit`,
`outer_spawn`, `outer_job_assign`, `outer_job_membership`, `outer_gate_write`,
`outer_timeout`, `outer_output_cap`, `outer_stderr`, `outer_exit`,
`outer_schema`, `outer_relation`, `outer_job_query`, or `outer_survivor`.

Runner partials are equally closed. `runtime_identity|dirty_worktree` produce
`runs=[]`. Every attempted outer run appends one `OuterObservation`; an
`outer_job_create|outer_job_limit|outer_spawn` failure has `body=null`, zero
bytes, `body_sha256` equal to SHA-256 of empty bytes, `return_code=null`, and
zero survivors. After a successful spawn, byte
counts and the post-cleanup return code are retained; `body` is non-null only
after exact parent-schema parsing, and `body_sha256` is the SHA-256 of its
canonical bytes. Until that parse succeeds, `body=null`, `body_sha256` is
SHA-256 of empty bytes, and `body_bytes` is still the exact captured stdout
byte count (including zero or an over-cap count). `stderr_bytes` is always the
exact captured count. After forced cleanup, `surviving_processes` is the
observed nonnegative outer-Job count for every post-spawn failure except
`outer_job_query`, which uses
`surviving_processes=-1`; `outer_survivor` retains the strictly positive count;
all other successful queries retain the observed nonnegative count.
`runs_equal` is true iff there are exactly two non-null parent bodies and their
canonical bytes are identical; otherwise it is false. Runner `passed` is true
iff `runs_equal` and both complete outer observations pass, and `error_code` is
null iff passed.

The runner independently parses and recomputes every case expectation, count,
membership relation, payload hash, environment equality, child/parent pass,
outer cleanup, and two-run byte equality. The receipt root is
`sha256(b"L2D-startup-forensics-v1/receipt\0" + canonical_parent_1 + b"\0" + canonical_parent_2)`;
on an early failure the missing canonical parent is the empty byte string. The
root never includes itself.

The final runner JSON is written once with exclusive creation to a fresh
`tempfile.mkdtemp` OS directory whose resolved path is outside both the
checkout and `learn2design-local-lab`; its lowercase digest plus LF is written
once to `result.json.sha256`. A verifier re-reads both files, rejects duplicate
keys, recomputes provenance and every relation, and checks the sidecar before
any sanitized conclusion is committed. After the conclusion is recorded, the
runner removes only that verified temporary directory and its two files.

## Stopping rule and actions

The checkpoint passes only if both fresh outer launches are byte-identical,
all eleven child cases have their exact expected accepted/rejected disposition,
all parent-side pre-gate membership chronology receipts are exact, provenance and
environment commitments match, output/stderr limits hold, all Jobs report zero
active processes, and no process survives.

- **Pass:** record the sanitized diagnostic result, mechanically remove V1
  from the runnable registry or add an explicit controller refusal, then freeze
  a fresh `constraint-aware-progress-toy-v2` plan in a separate commit. V2 must
  restate the scientific contract before execution, preserve the V1 family,
  seeds, thresholds, and claim boundary without selecting against a scientific
  result (none exists), and change only the process boundary justified by this
  diagnostic. V2 implementation and invocation remain later checkpoints.
- **Fail:** record only the earliest sanitized infrastructure code, keep the
  controller parked, mechanically remove V1 from the runnable registry or add
  an explicit controller refusal, do not create V2, and require a fresh
  owner-reviewed process-boundary plan.

Both actions keep V1 quarantined. Registry retirement or explicit controller
refusal must be committed and green before any owner-authorized atomic state
transition from `parked`; the completed ledger and failure streak are preserved
and an append-only sanitized resume event is required. The diagnostic runs at
most once after implementation and hostile audit. The failed V1 controller
path is never rerun.

## Claim boundary

A pass can establish only that the exact reproduced two-level process boundary
is host-feasible and whether a failure occurs inside that probe. It cannot
identify an unexercised V1-specific layer as the cause, validate the
constraint-aware progress mechanism, support an optimizer or candidate change,
show hidden-topology generalization, estimate leaderboard improvement or
proximity to score `0.14`, or authorize spending, private evidence, merge,
packaging, or upload.
