# Current handoff

Updated: 2026-08-21

This is the authority for the project's next action. Dated files in `research/`
are the evidence record, not a competing task list.

## Current state

- The submission candidate is deterministic and packages successfully.
- The topology-conditioned neural initializer failed its offline control gate.
- The frozen `development-v2` A100 screen completed 64/64 runs and validated at
  all 16 topology inference units.
- Production replay, an independent history-based calculation, and the archived
  summary agree.
- The semantic prior failed the frozen policy: 7/0/9 topology wins/ties/losses,
  mean difference `+0.029737199613027163`, action
  `retain_no_prior_candidate`.
- The packaged submission now defaults to no-prior initialization, matching
  the frozen `retain_no_prior_candidate` action. The historical harness still
  selects each comparison arm explicitly.
- Authenticated no-prior trajectories show that the configured patience of 600
  never triggers inside the 600-second screen. Patience 200 is now frozen as a
  post-hoc mechanism candidate; it has no performance claim yet.

## Next decision

Do not run `confirmation-v1`. The next gate is the two-stage bounded restart
study in [`PATIENCE_200_SCREEN_RUNBOOK.md`](PATIENCE_200_SCREEN_RUNBOOK.md): one
loss-blind, instrumented mechanics run, followed only on a mechanics pass by an
uninstrumented eight-topology patience-200 versus patience-600 screen using
fresh seeds 19/23. The user approved a cost-bounded Runpod run, but price,
visible balance, provider stop, and local artifact-evacuation gates still apply.
Even a screen pass authorizes only planning on the untouched submission-like
panel; it does not directly change the packaged candidate.

## What can contribute now

The exact private-artifact reproduction command and generated-bundle layout are
in [`DEVELOPMENT_V2_RESULTS_HANDOFF.md`](DEVELOPMENT_V2_RESULTS_HANDOFF.md). The
durable aggregate result and exploratory evidence boundaries are in
[`2026-08-21-development-v2-a100-results.md`](../research/2026-08-21-development-v2-a100-results.md).

Useful unpaid work now includes ingestion hardening, candidate packaging
correctness, contract checks, and review of the frozen restart plan. Do not claim an
order-of-magnitude gain: every frozen target has
`order_of_magnitude_claim_ready=false`.
