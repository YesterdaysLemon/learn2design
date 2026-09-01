# Constraint-aware progress toy v2 preflight failure

Date: 2026-09-01

Status: parked pre-result infrastructure failure; never retry V2

## Outcome

The one owner-authorized guarded invocation of
`constraint-aware-progress-toy-v2` failed closed during the controller's
isolated runtime preflight. The exact sanitized condition was:

```text
RuntimeError: constraint-progress isolated runtime probe failed
```

The controller had authenticated the clean repository, approved V2 sources,
normalized registry, protected submission tree and artifacts, and the
`awaiting_study` control state before reaching this boundary. The failure
occurred before `_begin_cycle`, before full-study dispatch, optimizer
construction, or metric execution, and before any result or sidecar write. It
does not reveal which internal runtime-probe stage failed and is not a
scientific result.

## Provenance and control receipt

- invocation revision:
  `9413cd4982cab74887fa8c7dc3dd4bf9c4d8508a`
- V2 frozen plan revision:
  `c5314afaa50490e39c53669d971114d280e43c07`
- V2 contract SHA-256:
  `621ade24962abd16ea4c3902691ae1781067572618c0639785fcadbfcb5b585f`
- controller event: `preflight_parked`
- event UTC: `2026-09-01T19:50:29Z`
- private cycle ID: retained only in the private control ledger
- postcondition: `status=parked`, `active_cycle=null`
- stop reason:
  `RuntimeError: constraint-progress isolated runtime probe failed`
- failure streak: `2`
- completed studies: the same ten earlier passed IDs; V1 and V2 absent
- result file: absent
- result sidecar: absent
- stop marker: absent
- lease and lock directory after readback: absent

The failed call consumed local CPU only, returned in about one second, and did
not use a GPU, paid endpoint, official data, private topology evidence, or the
competition portal. The checkout remained clean and draft PR #40 remained
green.

## Disposition

V2 is a pre-result infrastructure failure, not a completed scientific study.
Its invocation has nevertheless observed a preflight outcome, so the frozen
no-retry rule applies. Never resume and rerun V2, repair it in place, refresh
its approvals, select a same-ID change against this failure, or inspect its raw
private process output. Retain its registry entry only as historical contract
evidence and add a mechanical controller refusal before any future recovery.

The next admissible work is the fresh
[`constraint-progress-isolated-runtime-forensics-v1` plan](2026-09-01-constraint-progress-isolated-runtime-forensics-v1-plan.md).
It may reproduce only the non-scientific isolated runtime envelope through a
new source and ID. It must not import or execute V1, V2, their scientific
fixtures, or any result path.

This failure produced no new score and provides no estimate of movement from
the owner-reported Round-1 score `0.444293` toward `0.14`.
