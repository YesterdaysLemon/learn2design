# Current handoff

Updated: 2026-08-24

This is the authority for the project's next action. Dated files in `research/`
are the evidence record, not a competing task list.

## Current state

- The owner reports that the Round-1 candidate was uploaded on 2026-08-24. The
  repository cannot independently verify the portal receipt. The submitted
  baseline remains revision `5ce3cdb2ddf4c505622a0aeef805936a4ea607d7`
  and ZIP SHA-256
  `4cc0dbc65a3e61ca5358c18655c432caf478fbdfc07f10512553781f8822924b`.

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
  never triggers inside the 600-second development screen.
- The loss-blind restart mechanics predecessor passed and the complete
  `restart-screen-v1` evidence validated at 32 runs, 16 seed pairs, and 8
  topology inference units.
- Production replay, the independent history-only evaluator, and the archived
  restart summary agree: p200 wins/ties/losses 4/0/4, mean difference
  `-0.016933403182594786`, median `+0.05236019711778772`, frozen action
  `retain_patience_600`.
- The separately frozen `submission-like-screen-v1` completed 20/20 runs and
  all 10 two-seed topology blocks on the untouched panel. All runs were
  physically and finite feasible.
- Production replay, a deliberately independent history-first evaluator, and
  the archived summary agree at `1e-12` absolute and relative tolerance. The
  frozen operational action is
  `candidate_evidence_complete_for_submission_review`.
- The private history-only trajectory diagnostic is complete. At the last
  common 6,312-evaluation checkpoint, seed-31/second-sweep minus
  seed-29/first-sweep mean loss was `+0.35777353288148966`; the corresponding
  matched-wall-time contrast was `+0.2955358367854244`. The direction agreed
  on five of six comparable progress fractions. This weakens a
  throughput-only explanation but cannot separate seed from sweep/session
  order.
- The deterministic builder now pins the ZIP creator platform. Windows and
  Linux builds reproduce evaluated candidate SHA-256
  `4cc0dbc65a3e61ca5358c18655c432caf478fbdfc07f10512553781f8822924b`.
- Private source and generated artifacts remain outside Git. This host holds
  the only verified durable evidence bundle. The old provider volumes were
  inventoried, unique pilot evidence was evacuated and hash-verified, and all
  provider pods, endpoints, templates, and volumes were removed.
- The next local candidate adds an opt-in midpoint Latin-hypercube transform of
  the seven random non-anchor starts. The packaged default remains the submitted
  random initializer unless and until evidence promotes the treatment.
- A new 12-topology `coverage-robustness-v1` panel is disjoint from the official
  archive and every prior named panel. The frozen paired screen uses seeds
  37/41, 48 serial 1,200-second runs, exact H100/CUDA-13 provenance, a 22-hour
  provider ceiling, and a `$75` maximum charge.
- The coverage path now has source-lock authentication, exact 249-member archive
  validation, exact used-suffix raw-draw-to-treatment rank binding,
  objective/worker/session chronology checks, independent history-first replay
  with no project imports,
  and an opaque summary unlock that only the replay comparator can issue.
  Synthetic full-archive, forged-chronology, unrelated-LHS, terminal-partial,
  and end-to-end CLI tests pass. Whole-repository verification also passes;
  launch-time clean packaging and real-H100 mechanics evidence remain open.
- The public `dfbench==0.3.3` warmup helpers dispatch before logging but return
  no arrays. Because JAX is asynchronous, this is documented as common
  best-effort compilation warmup, not a proven pre-clock completion barrier.
  Both arms do explicitly block their completed population initialization
  before logging, removing the treatment-only transform timing asymmetry.
- No paid H100 resource has been created.

## Next decision

Do not run `confirmation-v1`, tune patience further on an observed panel, or
interpret the submission-like operational pass as a competitiveness claim.
Retain the packaged patience-600/no-prior candidate; its final package/evidence
review is complete.

Before paid GPU work, create and verify an encrypted second copy of the external
evidence bundle, commit the exact candidate, rebuild its separate package from
that clean revision, rerun the whole suite, and freeze the external plan. Then
ask the owner for explicit approval of a maximum `$75` total secure-cloud
charge. Do not provision from the general budget offer alone.

If approved, provision one exact H100 and run the cold mechanics smoke first.
The smoke must prove exact provider/JAX device identity, the CUDA-13 wheel stack,
cache policy, memory fit, and one admitted history; it is not optimizer evidence.
Any smoke failure stops the paid attempt. Only a passing smoke permits the
unchanged `coverage-robustness-screen-v1` plan to continue. A terminal partial
or worker failure is preserved as `not_evaluable` and is not rerun. See
[`H100_COVERAGE_RUNBOOK.md`](H100_COVERAGE_RUNBOOK.md). The treatment is not
promoted unless every frozen topology-level criterion passes.

## What can contribute now

The exact private-artifact reproduction command and generated-bundle layout are
in [`DEVELOPMENT_V2_RESULTS_HANDOFF.md`](DEVELOPMENT_V2_RESULTS_HANDOFF.md). The
durable aggregate result and exploratory evidence boundaries are in
[`2026-08-21-development-v2-a100-results.md`](../research/2026-08-21-development-v2-a100-results.md).

Useful unpaid work now includes the pre-result robustness plan, backup
verification, ingestion hardening, and contract checks. The validated restart
result and exact private reproduction command are in
[`2026-08-21-patience-200-a100-results.md`](../research/2026-08-21-patience-200-a100-results.md).
Do not claim an order-of-magnitude gain: no restart target supplied a complete
eight-topology comparison.

The exact pre-result profile is in
[`2026-08-22-submission-like-screen-plan.md`](../research/2026-08-22-submission-like-screen-plan.md).
The validated aggregate result and limitations are in
[`2026-08-23-submission-like-screen-a100-results.md`](../research/2026-08-23-submission-like-screen-a100-results.md).
The package, private-data provenance, and provider-cleanup ledger is in
[`2026-08-23-final-submission-review.md`](../research/2026-08-23-final-submission-review.md).
The sealed operating and replay commands are in
[`SUBMISSION_LIKE_SCREEN_RUNBOOK.md`](SUBMISSION_LIKE_SCREEN_RUNBOOK.md).
