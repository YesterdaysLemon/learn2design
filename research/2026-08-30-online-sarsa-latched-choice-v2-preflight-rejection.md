# Online SARSA latched-choice v2 preflight rejection

Date: 2026-08-30

Study ID: `online-sarsa-latched-choice-v2`

Plan commit: `2250fe103b3a4252d9255d7f987981a3916e3e09`

Disposition: `rejected_before_terminal_execution`

Next action: `freeze_fresh_online_sarsa_latched_choice_v3`

## What happened

The v2 contract was committed in a plan-only checkpoint. During the separate
implementation checkpoint, hostile read-only audit found that the proposed
implementation did not exercise several mechanisms required by the frozen
contract. These were substantive evidence confounds, not result-format or
preflight-conformance defects that could be accepted with narrower wording.

The task-owned fixture, worker, focused tests, registry entry, and controller
wiring existed only as an uncommitted pre-result skeleton. They were removed
after the audit. The fixture, worker, learner, and guarded controller were
never invoked. No result-bearing metric was observed, and no private result,
sidecar, controller transition, lease, terminal event, source approval, or
evidentiary claim exists for this ID. The private controller remains
`awaiting_study`.

## Blocking findings

The audit established all of the following independent blockers:

- complete-family corruption probes did not route each named mutation through
  the required independent replay validator;
- policy, update, reentrancy, capability, chronology, bootstrap, and keyed
  trace rejection counts included asserted outcomes rather than proof through
  their exact live consumers;
- the beacon-ablation path rejected the required zeroed beacon before the
  ablation-specific validator could authorize it, so a frozen negative control
  was unreachable;
- train-only source isolation, held-out spies, independent myopic and
  no-bootstrap comparators, and the two process-order/fresh-worker checks were
  not connected to exercised constructions;
- the negative-control positive-gate vectors and intervention difference
  contracts did not authenticate the complete causal and noncausal field
  obligations; and
- sanitizer, file-isolation, child-timeout, and public process-isolation paths
  left bypasses that could admit an unauthenticated or incomplete projection.

Any one of these defects is sufficient to reject the implementation. Together
they mean that a green terminal projection could have reported proof that its
mechanisms never actually exercised. Repairing the already frozen ID against
these implementation diagnostics would also make the successor selected
against quarantined development evidence. The controller was therefore not
allowlisted or invoked.

## Frozen boundary

V2 has no pass/fail result and contributes no evidence. It does not validate
online SARSA, exploration, bootstrapped control, held-out retention, any
negative control, RL, hidden topology, an optimizer, candidate performance,
accelerator behavior, or a competition score. The ten earlier terminal local
studies remain the complete approved evidence set.

The v2 plan, family, regimes, seeds, schedule, tokens, thresholds, cases,
discarded implementation, generator commitments, and all development-only
diagnostics are quarantined. Do not repair, register, allowlist, execute,
import, reuse, or select a successor against them.

## Successor requirement

A successor requires a unique fresh study ID and a newly committed plan before
implementation. It must be designed independently of v1 and v2 diagnostics.
Before freezing another broad terminal schema, the plan must reduce each
claimed mechanism to an executable proof path: every rejection enters its
named real consumer, every source and comparator is physically exercised,
every control is reachable and identically evaluated, and no pass field is a
literal substitute for the mechanism it claims to authenticate. Plan,
implementation audit, and any single guarded invocation remain separate
checkpoints.
