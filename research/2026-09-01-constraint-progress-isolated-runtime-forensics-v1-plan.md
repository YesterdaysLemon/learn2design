# Constraint-progress isolated runtime forensics v1

Date: 2026-09-01

Status: frozen plan-only checkpoint

Checkpoint ID: `constraint-progress-isolated-runtime-forensics-v1`

## Lineage and freeze boundary

The one guarded `constraint-aware-progress-toy-v2` invocation failed before
cycle start with the sanitized controller condition
`constraint-progress isolated runtime probe failed`. It produced no result,
sidecar, completed-study entry, optimizer metric, or scientific evidence. V2
is retired under its no-retry rule; V1 remains permanently quarantined.

This is a fresh, standalone, non-scientific checkpoint. It reproduces only the
isolated CPython launch, scrubbed environment, staged imports, network denial,
runtime identity, canonical-output, timeout, cap, and cleanup envelope through
new code. It must not import, execute, copy into a temporary module, or call
the controller, V1, V2, either scientific fixture, `submission`, JAX, official
data, private result files, or topology evidence.

The exact failed-attempt identity is frozen as:

- repository revision
  `9413cd4982cab74887fa8c7dc3dd4bf9c4d8508a`;
- V2 plan revision `c5314afaa50490e39c53669d971114d280e43c07`;
- V2 contract SHA-256
  `621ade24962abd16ea4c3902691ae1781067572618c0639785fcadbfcb5b585f`;
- runtime CPython `3.13.14`, `64bit`, `AMD64`;
- executable SHA-256
  `ad169f4cb4bfb78c7a5c030a4529c19d6643276778e33994c93e145b6191c3ec`;
- NumPy `2.5.1` and the four already frozen NumPy file/identity hashes from
  the V2 registry; and
- controller envelope: `sys.executable -S -P`, checkout cwd, file-backed
  stdout/stderr, 60-second timeout, 16,384-byte combined cap, fresh process
  group, and credential-scrubbed environment.

No seed, stage, threshold, expected disposition, or successor change may be
selected against development output from the diagnostic. This plan commit is
the freeze boundary. This checkpoint adds no probe implementation and executes
no child process.

## Falsifiable question

> Can a fresh independent probe reproduce the failed call's non-scientific
> isolated runtime envelope twice with byte-identical, closed per-stage
> receipts, thereby either locating one deterministic earliest failing stage
> or establishing that every frozen stage is currently operational, while
> emitting zero stderr and leaving no process or temporary file behind?

The question is about infrastructure reproducibility only. It cannot validate
the constraint-aware progress mechanism or a candidate.

## Exact source and execution boundary

Implementation is limited to one new source:

`experiments/local_lab/constraint_progress_isolated_runtime_forensics_v1.py`.

The module is import-inert and uses the standard library in its parent path.
The parent launches the same source only in a dedicated child mode. Each child
command is exactly:

```text
sys.executable -S -P <committed-probe-path> --child <case-id>
```

The parent independently constructs the scrubbed environment. It may inherit
only `COMSPEC`, `LD_LIBRARY_PATH`, `PATH`, `PATHEXT`, `PROGRAMDATA`,
`SYSTEMDRIVE`, `SYSTEMROOT`, `TEMP`, `TMP`, `TMPDIR`, `VIRTUAL_ENV`, and
`WINDIR`; it then sets empty `CUDA_VISIBLE_DEVICES`, CPU-only JAX markers,
`LEARN2DESIGN_LOCAL_LAB_NETWORK=disabled`, `PYTHONHASHSEED=0`, and the three
exact failed-attempt `L2D_*` identities above. Receipts commit environment key
sets and scalar equality only, never environment values or paths.

Each child gets exclusive file-backed stdout and stderr, a 60-second timeout,
a 16,384-byte combined cap, and a fresh process group. The parent must kill the
complete process tree on any error, join all threads, verify termination, and
remove only its own fresh temporary root. Network constructors are replaced
and exercised before any non-standard-library import. No network call may
succeed.

The runner writes one result JSON and SHA-256 sidecar by exclusive creation to
a fresh `tempfile.mkdtemp` directory outside the checkout and sibling private
lab root, re-reads and independently validates both, emits only a sanitized
projection, and removes the verified temporary directory. It runs once after
implementation, hostile audit, clean commit, and focused tests.

## Complete cumulative case matrix

Each run launches the following nine cases in this exact order. A case executes
every earlier stage before stopping after its named target, so the matrix has a
monotone prefix and identifies the earliest deterministic failure.

1. `argv_bootstrap`: exact child arguments, interpreter flags, and import-inert
   entry discovery.
2. `contract_environment`: exact allowed key set, disabled-network marker, and
   exact three `L2D_*` identities.
3. `late_stdlib_imports`: deferred standard-library imports used by the
   runtime envelope.
4. `network_denial`: replace and exercise socket construction, connection,
   resolution, and reverse-resolution entry points.
5. `site_discovery`: import `site`, derive checkout and user-site candidates,
   append only those two resolved candidates, and report only existence and
   membership Booleans plus a domain-separated commitment.
6. `numpy_import`: import NumPy and the PCG64 and SeedSequence defining
   modules after the exact path setup; expose only module/version identities.
7. `runtime_identity`: independently hash the interpreter, NumPy initializer,
   NumPy metadata, PCG64 module, and SeedSequence module and compare every
   scalar with the frozen V2 registry identity.
8. `canonical_output`: canonicalize and size-check the closed identity object
   without paths, exception text, commands, environment values, or raw bytes.
9. `composite`: repeat the complete ordered sequence and require the exact
   runtime identity object expected by the failed controller preflight.

The parent executes two complete sequential runs, for exactly 18 child
launches. It does not change the order or retry an individual case.

## Typed receipts and sanitizer

Every child emits exactly one canonical JSON line and zero stderr. Its ordered
fields are:

1. `schema_version: int` exact `1`;
2. `checkpoint_id: str` exact checkpoint ID;
3. `case_id: str` from the nine-case enum;
4. `target_stage: str` from the nine-stage enum;
5. `reached_stage: str|null`;
6. `status: str` in `passed|failed`;
7. `error_code: str|null`, null iff passed, otherwise the earliest stage;
8. `environment_keys_sha256: str` lowercase 64-hex;
9. `site_commitment_sha256: str|null`;
10. `identity_sha256: str|null`;
11. `identity_matches: bool|null`; and
12. `network_attempts_rejected: int` in `0..6`.

All exceptions are caught inside the new child and collapse to the current
stage code. No exception text, path, traceback, command, PID, environment
value, package file content, or raw payload is emitted.

Each parent `CaseObservation` contains the case ID, exact captured byte counts,
integer return code, parsed child receipt or null, survivor count, and one
closed parent error code from `spawn|timeout|output_cap|stderr|exit|schema|
relation|cleanup`. Each `RunReceipt` contains the nine observations in order,
the first child-stage failure or null, aggregate counts, zero-stderr and
zero-survivor totals, and a run-body SHA-256. The top-level result contains
exact provenance, two run receipts, `runs_equal`, `identified_stage`,
`diagnostic_status`, `action`, and a domain-separated receipt-root SHA-256.

An independent verifier in the same source parses JSON with duplicate-key and
nonfinite rejection, recomputes every schema, relation, hash, prefix,
provenance, byte-count, disposition, action, and sidecar before any conclusion
is accepted.

## Frozen stopping rule and actions

The diagnostic passes only if:

- both complete nine-case runs are byte-identical;
- all 18 launches return schema-valid canonical receipts with zero stderr,
  output within cap, and zero surviving processes;
- reached stages form the exact cumulative prefix;
- if a child stage fails, every later cumulative case reports the same earliest
  stage and every earlier case passes;
- `composite` agrees with the independently derived earliest stage;
- if no stage fails, all nine cases pass and the composite identity exactly
  matches the frozen registry identity; and
- the verified result/sidecar temporary root is completely removed.

Frozen actions are:

- deterministic earliest stage:
  `freeze_constraint_progress_v3_stage_bounded_runtime_plan`;
- all stages operational in both runs:
  `freeze_constraint_progress_v3_reproducible_runtime_plan`;
- nondeterministic, malformed, timed-out, over-cap, stderr, survivor, cleanup,
  provenance, or verification failure:
  `park_constraint_progress_runtime_research`.

The first two actions authorize only a fresh V3 plan. V3 must independently
restate the scientific contract, preserve the complete family, seeds,
thresholds, cases, and claim boundary, and limit its process delta to what this
diagnostic supports. V2 remains retired even if every stage is operational.
The failure action ends mutation and requires owner review. This diagnostic is
single-shot and is never rerun.

## Claim boundary

A passing diagnostic may identify a deterministic infrastructure stage or show
that the frozen runtime envelope was operational on these two fresh runs. It
cannot identify the historical root cause beyond that observation, validate a
scientific fixture, establish optimizer or hidden-topology improvement,
estimate a competition score or movement toward `0.14`, authorize candidate
integration, use official/private evidence, spend money, invoke a GPU, merge,
package, upload, or interact with the portal.
