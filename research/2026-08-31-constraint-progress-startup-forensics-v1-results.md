# Constraint-progress startup forensics v1 result

Date: 2026-08-31

Status: passed terminal non-result diagnostic; never rerun

Checkpoint ID: `constraint-progress-startup-forensics-v1`

## Outcome

The one permitted invocation of the frozen standard-library-only Windows
startup probe passed. Both fresh outer runs were byte-identical, all eleven
child cases per run had their frozen disposition, every required Job
assignment and pre-gate membership check succeeded, stderr remained empty,
and no process survived cleanup.

This establishes that the reproduced two-level Windows Job, process, pipe,
gate, timeout, output-cap, and survivor boundary is feasible and deterministic
on this host. The earlier `constraint-aware-progress-toy-v1` exception therefore
occurred in an unexercised V1-specific layer after this boundary, such as worker
bootstrap or runtime import. This diagnostic does not identify which later
layer failed and did not execute or evaluate the optimizer.

## Frozen provenance

- plan revision: `59f3c5a5ab5a985b1a67477b68eb6eb9976c2f3f`
- probe revision: `1c871b922d421e9b2d0cea05586015955ba673e0`
- plan SHA-256: `8f217f5eb9a32227cbc58b1ad94d4261f958a6b93b3e820b00ec675dcf62739d`
- probe source SHA-256: `557f89cc5eee0ef644773b3706966bb644b84c156d2167bf991ca661892352ae`
- contract SHA-256: `9e19eba687d0089a915b27d53956384d889459509abfd34fdd9e4f0a022db60e`
- executable SHA-256: `ad169f4cb4bfb78c7a5c030a4529c19d6643276778e33994c93e145b6191c3ec`
- runtime: CPython `3.13.14`, `64bit`, `AMD64`
- receipt root SHA-256: `bf1b4bb7e0ca8b6c72a481f5dfe88e4e61447e3ae594566b41e040024e633497`

## Sanitized aggregate receipt

There were exactly two sequential outer runs. Each returned zero, emitted zero
stderr bytes, left zero surviving processes, and produced the same 7,856-byte
canonical parent body with SHA-256
`371dd6204aba91242353c16be8eef5272dca0ac733acfffcc4879557825888ab`.

Each parent recorded:

- 14 projected environment pairs with commitment
  `15de24903dc13f559cbce67a22cfd4004eed753fe5f645eda2b574e774c1ba04`;
- successful outer membership before its gate;
- 11 child launches;
- 10 inner-Job assignments and 10 verified pre-gate memberships;
- four accepted valid frames and seven rejected malformed controls;
- zero child stderr bytes and zero surviving descendants; and
- terminal `passed: true` with no error code.

The runner exclusively created, re-read, authenticated, and sidecar-verified
its temporary result before emitting this projection. It then removed only its
own verified temporary directory and files. No result file remains in either
project root.

## Audit and verification

Three independent Luna/max read-only hostile audits passed the final source and
contract for family/chronology, evidence/schema, and Windows runtime
feasibility. Focused verification passed:

```text
python -m py_compile experiments/local_lab/constraint_progress_startup_forensics_v1.py
uv run --frozen --group dev --group integration pytest -q tests/test_constraint_progress_startup_forensics_v1.py
23 passed
```

The one allowed full-suite attempt reached the new probe tests, then exposed a
known historical source-lock assertion for the quarantined V1 entry because
`docs/AUTONOMOUS_LAB.md` had truthfully changed after V1 parked. It later made
no progress in a pre-existing JAX/initializer area and was interrupted. The
historical approval is retained rather than refreshed: V1 must not be made
runnable by blessing post-failure protocol bytes. Controller refusal is the
required mechanical retirement.

## Disposition and claim boundary

`constraint-aware-progress-toy-v1` remains a pre-result infrastructure failure,
not a completed or failed scientific study. It must never be rerun and must
not be added to the completed ledger. The controller now refuses that ID before
output validation, repository inspection, lease acquisition, state access, or
worker launch. Its registry entry remains only as immutable historical
contract evidence.

A fresh V2 plan may preserve V1's synthetic family, seeds, thresholds, and
claim boundary while changing only the process boundary justified here.
Neither this diagnostic nor a future toy result estimates movement from the
owner-reported Round-1 score `0.444293` toward `0.14`, supports a candidate
change, or authorizes official data, private outcome evidence, paid compute,
GPU use, packaging, upload, or merge.
