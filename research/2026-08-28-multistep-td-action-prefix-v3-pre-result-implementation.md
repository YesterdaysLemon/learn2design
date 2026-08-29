# V3 pre-result implementation checkpoint

Status: **implemented and cleanly audited before any guarded study invocation**

Study ID: `multistep-td-action-prefix-v3`

Frozen plan commit: `bc76a1c`

Implementation commits: `42b3c31`, `870c233`

Date: 2026-08-28

## Scope and result boundary

This checkpoint implemented the already frozen V3 contract. It did not invoke
`tools/run_local_lab.py`, write a private result or sidecar, transition the
private controller state, append a terminal event, or produce terminal study
evidence. The controller remains `awaiting_study`.

The implementation is confined to the deterministic topology-independent
fixture, its network-disabled worker, the exact registry contract, controller
validation, and focused tests. It does not touch the submitted package or its
defaults.

## Implemented contract

The fixture now implements the complete four-action family and independently
replays every legal predecessor, action, successor, terminal reward, and done
outcome. Public successor state contains only the target-independent action
prefix; evaluator truth remains separate. Genuine target-swapped twins retain
identical frozen public bytes.

The exercised train and held-out paths use capability-bound counting sources.
Absent, exploding, and lazy held-out sentinels are connected to the source and
environment boundaries. Pending transitions use sink-bound one-use step
authorizations, and the fitter accepts only an authenticated copied projection
of the behavior trace. Comparators use issued one-use public-feedback scopes.

The learner is the frozen blank tabular four-sweep synchronous fitted-TD
kernel. Complete transition-target, reward-origin, signal-ablation, comparator,
terminal-scalar dependency, physical timing, source isolation, trace
authentication, control-difference, sanitization, and process-isolation checks
are implemented without changing a frozen case, threshold, regime, seed,
mapping, split, stopping rule, or claim.

## Pre-result verification

- All 18 non-process cases passed the exact frozen projection. Canonical train,
  validation, and test recovery and every negative-control aggregate matched
  the precommitted analytic values.
- The focused six-test V3 file initially passed five tests and exposed one
  controller-only schema defect: list-valued attack metadata in the registry
  was being mistaken for the integer `attack_classes` result aggregate. No
  fixture or learner case failed.
- The controller validator was corrected to distinguish the explicit frozen
  result containers from registry-only metadata. Twenty-one focused controller
  validator and end-to-end transition tests then passed, and the exact formerly
  failing real-result validation test passed on rerun.
- Two fresh network-disabled worker processes matched the complete local
  projection. The strict timing and trace attacks, bounded result sanitizer,
  and worker network-before-import gate passed in the focused file.
- The source sanitizer rejected all 50 frozen malicious samples. The broader
  controller adversarial projection passed 70 focused attacks before the final
  narrowed container correction.
- Three independent Luna-max static audits found no remaining family/TD,
  boundary/authentication, or registry/schema blocker. The schema auditor
  repeated its review after the controller correction and found the explicit
  container closure complete.
- `git diff --check` was clean apart from Git's informational Windows
  line-ending warnings.

The approved committed-source digests are:

| source | committed SHA-256 |
|---|---|
| `uv.lock` | `5aa38f61873af4713dd88514227eb28aceaaade949215bef65d8125ab45834d0` |
| V3 fixture | `620d9246a1d27c0b2915367db07648a62eb573d2369b03a240e77b015b078460` |
| lab protocol | `680b5b555b41f3e8b8c280c32574caf720ee1f31ff651ca741eb76376097b2af` |
| frozen V3 plan | `bdf1aa2257a48c96a21bef7b1662c73f8a6ab05c44e1eadc8a010af88ba99cb0` |
| V3 worker | `be69c77b63234d8ed11d599961cabac11b2304b5f235301954c06a825abe2ce7` |

The normalized registry digest pinned by the controller is
`d0946cdd96affba3878219d66041d4656de4e4894d62ea85ceef4cbcd4c4925b`.

## Claim boundary and next gate

This checkpoint establishes only that the frozen synthetic implementation and
its pre-result safeguards conform under development tests. It is not a passed
terminal study and makes no claim about delayed-credit necessity, absence of a
public shortcut, general or production RL, hidden structure, UIFO, optimizer
quality, competition performance, candidate value, accelerators, or score.

The next gate is one later, single guarded local-CPU invocation of V3, after a
fresh clean-state, source-approval, lease, stop-marker, and CI check. A passed
invocation may support only the fixed synthetic synchronous-TD harness and toy
propagation mechanics. Any failed, malformed, timed-out, nondeterministic, or
drifted invocation must park the controller and stop mutation.
