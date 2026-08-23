# No-prior submission-like screen — frozen pre-result plan

Date: 2026-08-22

Status: planning and validation only. No hardware has been provisioned and no
result has been observed.

## Decision boundary

The validated candidate remains no-prior initialization with patience 600.
The `restart-screen-v1` action `retain_patience_600` is not reopened by this
study. `confirmation-v1` remains closed. This plan changes neither
`submission/submission.py` nor the deterministic packaged candidate.

The original accelerator plan reserved all ten `submission-like-v1`
topologies for four-hour official-budget runs. That design needs at least 40
scored GPU-hours with one seed and 80 with two, so it cannot fit a $20 runway.
The frozen profile below is therefore a **cost-bounded submission-panel
characterization**, not an official-budget rehearsal and not evidence of
leaderboard competitiveness.

## Exact profile

| Field | Frozen value |
| --- | --- |
| Study profile | `submission-like-screen-v1` |
| Policy | `no-prior-submission-like-screen-v1` |
| Panel | `submission-like-v1`, all 10 topologies |
| Panel SHA-256 | `d85227f216528d635e56a93094e661721f62f379808707f310bf4da60d8fa57b` |
| Arm | `no_prior` only; packaged patience remains 600 |
| Optimizer seeds | 29 and 31 |
| Runs | 20 = 10 topologies × 2 seeds × 1 arm |
| Objective budget | 1,200 seconds per run; 24,000 seconds total |
| Population / frequencies | 8 / 50 |
| Evaluation path | full-vmap; `evaluation_chunk_size=null` |
| Targets | 4.0, 1.0, 0.5, 0.0 |
| Worker / session limits | 2,100 seconds / 32,400 seconds |
| Failures | first timeout, nonzero exit, invalid record, or integrity failure stops |
| Hardware | one on-demand A100 80 GB, MIG disabled, serial execution |
| Runtime | Python 3.12; persistent JAX compilation cache disabled |
| Provider stop / reserve | at most 10 hours / final 1,800 seconds reserved |
| Purchase ceiling | at most $1.60/GPU-hour and $16.00 total provider charge |

Seed 29 traverses the committed panel forward. Seed 31 traverses it backward.
This mirrored order gives every topology one earlier and one later observation
for serial-drift diagnostics. Seeds remain repeated measurements nested inside
topology; neither 20 runs nor history rows are independent inference units.

The exact deterministic `submission.zip`, builder manifest, source-file
digests, source revision, and upstream reference are validated before plan
construction and embedded in every run configuration. Every ZIP member is
safely read and required to match both its builder-manifest digest/size and the
normalized bytes in the clean checkout that the worker imports. The live exclusion audit
must recompute zero overlap against the pinned official dataset and every
development, confirmation, restart-mechanics, and restart-screen panel.

## Estimand and frozen action

For topology `t`, reconstruct from authenticated histories:

```text
L_t = mean(best finite feasible loss(t, seed 29),
           best finite feasible loss(t, seed 31))
```

The primary descriptive result is the arithmetic mean of the ten `L_t`
values. Also report their median and linear p90, a deterministic 10,000-sample
topology-block bootstrap interval (`seed=20260822`), physical and finite
feasibility, the p90 absolute within-topology seed gap, and censor-aware target
outcomes. Complete topology blocks, preserving both seeds, are the bootstrap
resampling units. Seed gap is descriptive and has no pass threshold.

The binary rule is intentionally operational; no defensible absolute
competitiveness margin exists:

- `passed / candidate_evidence_complete_for_submission_review` only if 20/20
  records are revalidated, all 10 two-seed topology blocks are complete, all 20
  runs have a physically feasible finite score, the exact candidate package is
  bound, and all three reproductions agree;
- `failed / retain_candidate_and_investigate_submission_like_reliability` for a
  complete panel that violates a frozen operational criterion;
- `not_evaluable / retain_candidate_attempt_not_evaluable` for an error,
  interruption, timeout, nonzero exit, deadline/wall guard, or incomplete
  terminal attempt.

A pass authorizes only a final package/evidence review. It does not change the
candidate, claim superiority, or automatically submit it. No p-value, target
hit, bootstrap bound, or favorable post-hoc statistic can replace this rule.

## Terminal-attempt protection

After outcome-blind device/runtime preflight and before any result-bearing
worker, the runner atomically creates
`submission-like-screen-v1.terminal-attempt.json` in the persistent output
root, outside the plan-ID directory. Its presence blocks a second plan ID,
another output directory under that root, and `--resume`. The first worker
failure is terminal. Preserve and package partial evidence; do not add seeds,
drop topologies, resume, or rerun after seeing any outcome.

If a worker exits zero but its completed record fails integrity validation, the
runner preserves the original record digest and fields, marks the record as a
terminal `RecordIntegrityError`, rebuilds only structural indexes, and exits the
attempt. This makes `--allow-incomplete` evacuation possible without treating
the invalid record as a valid result.

The output root and terminal receipt must be on the one canonical durable
volume named in the approved runbook. Changing the root to evade the receipt is
a protocol violation. The receipt is evacuated and authenticated separately
from the exact 109-member complete archive.

The runner has a two-step plan contract. `--dry-run` writes the plan once for
review. Result-bearing execution requires that same file and its recorded
SHA-256 via `--approved-plan-sha256`; it rebuilds live candidate/exclusion
inputs, permits only the newly rebuilt creation timestamp to differ, and then
executes the authenticated reviewed plan object. Plan drift fails before the
terminal receipt is claimed.

## Outcome-blind evidence workflow

1. Package without extraction. Generate the ZIP, SHA-256 sidecar, package
   manifest, exact pre-run plan, and terminal receipt on persistent storage.
2. Create a path-free source lock with basename, SHA-256, and byte size for all
   five sources. Record the source-lock SHA-256 out of band before transfer.
3. Transfer to a private directory outside Git. Authenticate the source-lock
   digest before parsing it, then verify all five files and the sidecar filename
   and digest.
4. For a complete package, check duplicate/case-colliding names, traversal, absolute/backslash paths,
   encryption, symlink/special types, ZIP CRCs, entry/total sizes, compression
   ratios, exact 109-member allowlist, exact plan/profile, 20/10/2 hierarchy,
   environment, logs, worker status, and terminal receipt.
5. Load NPZ only with `numpy.load(..., allow_pickle=False)` after bounded NPY
   header/shape/dtype checks. Recompute feasibility, best loss, target hits,
   timing, and evaluation counts from raw histories. Keep `summary.json` sealed.
6. Run the production record summarizer and the no-import, history-first
   reference evaluator. Compare all 20 run outcomes, 10 topology values, four
   target families, aggregates, bootstrap, and all five criteria.
7. Open archived `summary.json` only after raw replay agreement, then require
   production/reference/archive agreement at `1e-12` absolute and relative
   tolerance. Any mismatch fails closed before interpretation.

A terminal partial follows a separate structural path. After source-lock,
sidecar, ZIP, plan, environment, package-state, session, allowlist-subset, and
terminal-receipt authentication, it is reported as `not_evaluable`. That path
does not open `summary.json`, run records, histories, or logs and never attempts
or claims three-way outcome replay.

## Hostile pre-result audit

The pre-result audit required candidate ZIP-to-checkout binding, complete and
failed-result replay tests, exact archived-summary schemas, physical as well as
finite feasibility comparison, a full synthetic 109-member integration
fixture, structural terminal-partial handling, and an immutable reviewed-plan
execution handshake. These are launch blockers: any regression keeps this
profile closed.

Generated receipts, normalized tables, and later figures remain outside Git.
A later results PR may commit only aggregate evidence; it must exclude raw
histories, topology strings, run IDs, candidates, logs, GPU identifiers,
balances, secrets, and provider-local absolute paths.

## Cost, stop, and approval gate

Scored Objective time is 6 hours 40 minutes. Expected rental duration is about
7.5–8 hours including setup, per-process overhead, validation, packaging, and
evacuation. At the $1.60/hour ceiling, expected compute is at most about $12.80;
the provider-native 10-hour stop caps it at $16.00.

Before provisioning, require all of the following:

- this evidence-plan PR merged and green at one clean immutable revision;
- no active paid resource;
- live price at or below $1.60/hour;
- both visible balance and remaining user-authorized cumulative runway at least
  $16.00, with prior charges reconciled rather than inferred from balance;
- the provider-native UTC stop configured no more than 10 hours ahead;
- at least 9.5 hours remaining at study start, including evacuation reserve;
- exact candidate package, exclusion audit, dry plan, and source locations
  reviewed before the terminal receipt can be claimed;
- explicit owner approval for this exact profile and maximum charge.

If any condition fails, do not provision, silently shrink the design, run
`confirmation-v1`, or substitute a new optimizer/model.
