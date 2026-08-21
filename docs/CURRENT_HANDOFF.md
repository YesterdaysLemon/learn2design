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
- The packaged submission still needs a separate focused alignment change; the
  results-evidence work deliberately does not change the algorithm.

## Next decision

First, land the evidence workflow and aggregate report. Then use a separate PR
to make the packaged submission default match `retain_no_prior_candidate`,
rebuild the deterministic submission ZIP, and rerun contract/integration tests.

Do not run `confirmation-v1`: development did not pass. After candidate
alignment, the recommended next evidence gate is a separately frozen
no-prior-only submission-like evaluation on the existing disjoint panel at the
official budget. Do not start that paid, long-running work without a reviewed
plan and explicit owner approval.

## What can contribute now

The exact private-artifact reproduction command and generated-bundle layout are
in [`DEVELOPMENT_V2_RESULTS_HANDOFF.md`](DEVELOPMENT_V2_RESULTS_HANDOFF.md). The
durable aggregate result and exploratory evidence boundaries are in
[`2026-08-21-development-v2-a100-results.md`](../research/2026-08-21-development-v2-a100-results.md).

Useful unpaid work now includes ingestion hardening, candidate packaging
correctness, contract checks, and review of the next run plan. Do not claim an
order-of-magnitude gain: every frozen target has
`order_of_magnitude_claim_ready=false`.
