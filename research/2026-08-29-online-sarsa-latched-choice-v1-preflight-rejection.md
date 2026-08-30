# Online SARSA latched-choice v1 preflight rejection

Date: 2026-08-29

Study ID: `online-sarsa-latched-choice-v1`

Disposition: `rejected_before_terminal_execution`

Next action: `freeze_fresh_online_sarsa_latched_choice_v2`

## What happened

The v1 plan was committed before implementation. During the separate
implementation checkpoint, a hostile read-only audit found that the frozen
result contract assigns mutually incompatible scalar types to the same fields.
The contradiction cannot be resolved by implementation without changing the
already frozen contract.

An incomplete task-owned fixture skeleton and focused development checks
existed briefly before this conflict was confirmed. They were never registered
or allowlisted and were removed rather than preserved as a selectable fixture.
No worker was created, the guarded controller was never invoked, and no private
result, sidecar, controller transition, lease, or terminal event exists. The
private controller remains `awaiting_study`. Any development-only diagnostics
from the discarded skeleton are quarantined and cannot be used to select a
successor.

## Blocking finding

The plan requires exact integer rejection counts:

- `selection_attacks_rejected == 10`;
- `update_attacks_rejected == 11`;
- `authorization_attacks_rejected` equal to the thirteen frozen mutation
  classes; and
- `static_mutations_rejected == 52`.

The same plan later declares every field ending in `_rejected` to be an exact
JSON Boolean and declares every remaining case field to be an exact JSON
integer. JSON cannot represent any of the required counts as both an exact
Boolean and an exact integer, and the plan explicitly rejects Boolean/integer
substitution. Therefore no result can satisfy the complete frozen contract.

Treating the fields as counts would violate the Boolean rule. Treating them as
Booleans would violate the exact-count gates. Renaming or splitting the fields
would alter the frozen required-field sets. None is an admissible preflight
conformance fix.

## Frozen boundary

No pass/fail claim exists for v1. It does not validate online SARSA, exploration,
bootstrapped control, held-out retention, any negative control, RL, an optimizer,
candidate performance, hidden structure, accelerator behavior, or a competition
score. The ten earlier terminal local studies remain the complete approved
evidence set.

The v1 plan, family, regimes, seeds, schedule, thresholds, cases, discarded
skeleton, and any development diagnostics are quarantined. Do not repair,
register, allowlist, execute, import, reuse, or select a successor against them.

## Successor requirement

A successor requires a fresh versioned study ID and a newly committed plan
before any implementation or learner execution. Its result schema must include
an explicit field-by-field scalar-type table that is consistent with every
count and equality gate. The fresh plan must otherwise restate, independently
and without selecting against v1 diagnostics, the complete topology-independent
family, online exploration and update order, held-out isolation, comparators,
controls, stopping actions, and synthetic-harness-only claim boundary.
