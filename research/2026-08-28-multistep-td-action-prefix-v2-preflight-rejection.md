# Multi-step TD action-prefix v2 preflight rejection

Date: 2026-08-28

Study ID: `multistep-td-action-prefix-v2`

Disposition: `rejected_before_terminal_execution`

Next action: `freeze_fresh_multistep_td_action_prefix_v3`

## What happened

The v2 plan was committed before implementation. A fresh, self-contained
action-prefix fixture and network-disabled worker were then implemented. Its
development-only projection reported all seventeen non-process cases passing,
and two fresh worker projections matched the local projection. These checks
are diagnostics, not evidence.

Three independent read-only pre-result audits found that several passing cases
could not establish the exact frozen claims. The guarded controller was never
invoked. No private result, hash sidecar, state transition, or terminal event
exists. The v2 worker is absent from the controller allowlist, v2 is absent
from `studies.json`, and the protected submission was not changed.

## Blocking findings

1. The target-swap case did not construct a target-swapped twin. It compared
   each public observation with itself or a byte copy, so it could pass without
   proving that flipping only evaluator truth preserved every public byte.
2. The nominally myopic comparator read `episode.target` and called the
   evaluator reward formula directly instead of fitting only authenticated
   behavior feedback. Its baseline result therefore was not an independent
   feedback-only comparator.
3. The absent/exploding held-out cases and lazy pre-action cases were vacuous.
   Their sentinels were never connected to the training or environment path;
   invalid action validation raised before any lazy materializer could be
   tested.
4. The pinned legal-family projection included every action row, successor
   digest, terminal reward, and `done` value, but omitted the promised explicit
   predecessor key and successor type/layout/legality contract. It also did not
   fail closed before learner construction or prove full realized-path
   disjointness.
5. The updater authenticated only successor phase and prefix, not the exact
   pending successor identity. The keyed trace omitted independently mutable
   observation/action layout metadata and did not strictly type donor and
   origin identities, leaving malformed recombinations that could pass.
6. Several timing checks substituted protocol-stage errors for the promised
   physical lazy-scalar, later-transition, origin-resolution, or nested
   reentrancy attacks. The TD dependency check covered only one of the three
   bootstrap boundaries.
7. Canonical/control difference checks and the all-row ablation case did not
   compare the complete unaffected observation, transition, feedback, target,
   key, link, donor, and origin projections. Control rejection also did not
   evaluate the entire frozen positive gate.
8. The seeded-random comparator omitted the train-tag replay, and deterministic
   replay checks did not independently compare every baseline action trace.

## Frozen boundary

No terminal pass/fail claim exists for v2. Its passing development projection
does not validate multi-step value propagation, delayed credit, RL, an
optimizer, candidate performance, hidden structure, accelerator behavior, or
a competition score. The nine earlier terminal local studies remain the
entire approved evidence set.

The v2 source, worker, plan, generator commitment, thresholds, regimes, seeds,
permutations, cases, and observed development diagnostics are quarantined.
They must not be repaired in place, registered, executed, or reused as a
successor fixture.

## Requirements for a successor

A successor requires a fresh versioned ID and a newly committed plan before
any learner executes. At minimum it must:

- fail closed on a complete independent family replay before constructing a
  learner, with explicit predecessor keys, successor type/layout/legality,
  terminal outcomes, unique keys, and realized-path disjointness;
- construct target-swap twins only from already frozen public bytes, then flip
  evaluator truth and authenticate the exact permitted outcome changes;
- connect operation-counting held-out and lazy successor/reward sentinels to
  the real source and environment boundaries, while keeping held-out data out
  of the train-only learner API;
- bind every pending update to an exact predecessor/action/successor identity
  and strictly authenticate typed component metadata, donor/origin identities,
  malformed records, and physical reentrancy/timing attacks;
- fit all comparators only from authenticated behavior feedback and replay the
  exact train, validation, and test action streams independently;
- compare canonical and each control under an explicit complete-field
  difference whitelist, including every legal-row signal ablation; and
- test terminal-scalar propagation across all three bootstrap boundaries and
  require each negative control to reject the full positive gate.

No v3 threshold, regime, seed, permutation, schema, attack, or fixture may be
selected against v2's observed development diagnostics.
