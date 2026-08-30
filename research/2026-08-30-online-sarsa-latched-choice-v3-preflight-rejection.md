# Online SARSA latched-choice v3 preflight rejection

Date: 2026-08-30

Study ID: `online-sarsa-latched-choice-v3`

Plan commit: `e3d743e902ed297c1a209cd8594b9d0d2d7f2ecc`

Disposition: `rejected_before_terminal_execution`

Next action: `freeze_fresh_online_sarsa_latched_choice_v4`

## What happened

The V3 contract was committed in a plan-only checkpoint. The repository,
controller, stop-marker, lease, protected-source, revision, and PR-CI guards
were clean at the start of the separate implementation checkpoint. Before any
fixture, worker, registry, allowlist, source-approval, or controller change, a
hostile runtime audit found that the frozen Windows isolation profile cannot be
established on this host without destructively rewriting core operating-system
file security.

No V3 fixture, worker, learner, registry entry, controller allowlist, source
approval, private result, sidecar, controller state transition, lease, terminal
event, or evidentiary claim exists. The guarded controller was not invoked and
remains `awaiting_study`. The only executable check was a read-only
infrastructure probe of the exact frozen Python import closure; it did not run
study, fixture, worker, learner, or result code and observed no result-bearing
metric.

## Blocking finding

The frozen plan defines the Python DLL set as every PE module returned by
`K32EnumProcessModulesEx(LIST_MODULES_ALL)` after the exact import sequence,
excluding only the Python executable. It then requires every executable,
parent, DLL, DLL parent, and standard-library object to have an exact protected
self-relative security descriptor owned by the controller SID, with null
group, no SACL, exact control bits, and only the frozen worker, controller, and
LocalSystem ACEs.

The exact import probe necessarily loaded Windows KnownDLLs from
`C:/Windows/System32`, including `ntdll.dll`, `KERNEL32.DLL`, and
`KERNELBASE.dll`. Those actual files are owned by TrustedInstaller on this host
and carry six non-inherited operating-system and application-package access
rules. They do not and must not match the controller-owned three-principal
descriptor frozen by V3.

Copying Python or ordinary extension DLLs into a private scratch tree cannot
remove these KnownDLL mappings from the complete loaded-module snapshot: the
Windows loader maps the core System32 images before the bootstrap can run.
Meeting the frozen descriptor rule would therefore require taking ownership of
and replacing the DACLs on core operating-system files and parents. That is an
unsafe destructive system mutation, is outside the authorized repository and
private-lab scope, and is explicitly not an admissible implementation step.
An asserted descriptor, skipped module, reduced DLL closure, inherited system
ACL, or Python-level monkeypatch would weaken the immutable contract.

The plan itself requires pre-result quarantine when the host cannot establish
and probe the exact AppContainer, ACL, Job, and native-loader profile. V3 is
therefore rejected before implementation rather than registered with a
weakened or fictitious isolation proof.

## Independent audit disposition

Three read-only hostile audits were completed before the disposition. The
family and chronology audit found the generator counts, SARSA order, permit
counts, control arithmetic, and scalar schema internally consistent. The
runtime audit independently reproduced the System32 KnownDLL closure and found
no contract-faithful non-destructive route to the frozen DACL profile. A third
evidence-boundary audit also found that several child/final projection and
attack-witness preimages were not closed enough for the promised independent
outer reconstruction. The runtime finding alone is terminal, so none of those
frozen requirements was relaxed, implemented, or selected against.

## Frozen boundary

V3 has no pass/fail result and contributes no evidence. It does not validate
online SARSA, exploration, bootstrapped control, held-out retention, any
negative control, Windows isolation, RL, hidden topology, an optimizer,
candidate performance, accelerator behavior, or a competition score. The ten
earlier terminal local studies remain the complete approved evidence set.

The V3 plan, family, regimes, schedule, tokens, seeds, thresholds, cases,
controls, runtime profile, and this preflight diagnostic are quarantined. Do
not repair, register, allowlist, execute, import, reuse, or select a successor
against them.

## Successor requirement

A successor requires a unique fresh versioned study ID and a newly committed
plan before implementation. Its infrastructure contract must be proven
host-feasible before the plan freezes and must not require changing ownership
or access rules on Windows system files. It must independently define any new
family, online chronology, controls, scalar schema, runtime boundary, stopping
actions, and synthetic-harness-only claim without selecting against V1, V2, or
V3 development diagnostics. Plan, exact implementation audit, and any single
guarded invocation remain separate checkpoints.
