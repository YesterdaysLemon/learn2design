# H100 coverage triage plan — 2026-08-24

Status: frozen precommitment. The single terminal attempt completed on
2026-08-25 and failed the promotion rule; see
[`2026-08-25-h100-coverage-triage-results.md`](2026-08-25-h100-coverage-triage-results.md).
The design below is preserved as the pre-result record and does not authorize a
rerun or Stage B.

This Stage-A screen replaces the proposed `$75` first step with a bounded
reject-or-review gate. It can retain the submitted random-start candidate or
produce a review request for an independent Stage B. That outcome is review-only:
it does not authorize, unlock, or provision Stage B. It cannot promote the
treatment, change the packaged default, or support an official-budget or
leaderboard claim.

## Baseline and question

The owner reports that the Round-1 baseline was uploaded on 2026-08-24. Its
immutable local identity is revision
`5ce3cdb2ddf4c505622a0aeef805936a4ea607d7` and ZIP SHA-256
`4cc0dbc65a3e61ca5358c18655c432caf478fbdfc07f10512553781f8822924b`.
Stage A does not overwrite that package.

The only treatment is the already implemented `coverage_balanced` population
suffix: the seven seeded random non-anchor draws are rank-transformed per
coordinate to midpoint Latin-hypercube logits. The feasibility anchor,
optimizer, patience, objective, paired raw draw, population size, topology,
seed, and wall-clock budget remain common. The control is `no_prior` with the
untransformed suffix.

Question: is the treatment strong and consistent enough at 600 seconds to
justify paying for a separate confirmation? A failure is not evidence that the
treatment is globally ineffective.

## Outcome-blind panel

`coverage-triage-v1.json` was appended after every older generated panel. Its
topology-generation/API provenance is the current official starter-kit revision
`1bb7f54737dec6a08b59879a8831d125f08f8a0b`, the explicit H100-panel override
in `tools/build_topology_panels.py`. Exact archive exclusion was recomputed
against dataset bytes SHA-pinned below, first captured at starter-kit revision
`d9b1bd7d6f2c4df335bc7725755b02aa5f6f942c`. These are separate provenance
layers; the generator's historical default was not rewritten and no older
panel bytes changed:

- official archive SHA-256:
  `149f6aac17aff2e33750b4e1b6cebd3cef1c39d47ae49a3a7ed77315cb7838a7`;
- panel SHA-256:
  `f400cdc3a947cd076ce9bd9f48a2dafcb98dfd3f9f938a74ceb11ca88c360972`;
- 8 unique size-3 topologies, 4 detector and 4 homodyne readouts;
- observed complexity marginals are squeezer low/middle/high `4/2/2` and
  directional low/middle/high `2/4/2`; this is not a full 3-by-3 complexity
  balance and is treated only as a low-cost triage panel;
- zero exact overlap with the archive, `development-v1`, `confirmation-v1`,
  `submission-like-v1`, or `coverage-robustness-v1`;
- all four older generated panel files remained byte-identical after the new
  panel was appended.

No live optimizer result, history, or loss entered the generation rule. The
12-member `coverage-robustness-v1` panel remains untouched as the candidate
panel for a separately reviewed Stage B; it is not automatically selected by a
Stage-A result.

## Exact Stage-A design

| Item | Frozen value |
|---|---|
| Profile | `coverage-triage-screen-v1` |
| Arms | `no_prior`, `coverage_balanced` |
| Topology units | 8 |
| Optimizer seeds | 37, 41 |
| Runs | 32 serial runs = 8 × 2 × 2 |
| Objective budget | 600 seconds per run |
| Scored Objective time | 19,200 seconds = 5.3333 GPU-hours |
| Pair order | alternate by topology and seed; 8/8 arm-first balance |
| Seed order | forward seed-37 sweep, reverse seed-41 sweep |
| Population | full-vmap 8; one anchor plus seven suffix members |
| Hardware | one secure `NVIDIA H100 80GB HBM3`, MIG disabled |
| Runtime | Python 3.12, exact JAX/CUDA-13 wheel contract, caches disabled |
| Worker timeout | 1,200 seconds |
| Main-session ceiling | 25,200 seconds = 7 hours |
| Provider stop | 8 hours, including smoke, with 1,800-second evacuation reserve |
| Price ceiling | `$3.29` per H100-hour |
| Maximum GPU charge | `$26.32` |
| Maximum total provider charge | `$30.00` |

The scored-time estimate is `$17.5467` at the price ceiling. The remaining
envelope covers cold compilation, the loss-blind smoke, packaging, evacuation,
and modest provider overhead. If the exact secure H100 price exceeds `$3.29`,
the profile does not launch. Internal population vmap is required; concurrent
topology, seed, or arm workers are forbidden because they change the timing
estimand.

The mirrored sweeps balance arm order, but optimizer seed remains confounded
with early/late session position. Seed- and order-stratified guards are
required, but no causal seed or drift interpretation is allowed.

## Terminal-attempt rule

The first result-bearing attempt is terminal. Any worker error, timeout,
interrupt, provider deadline guard, integrity failure, or incomplete package is
`not_evaluable`. There is no resume, rerun, replacement topology, seed top-up,
or budget extension. The plan-associated receipt ledger is outside the result
directory so changing the output root does not create another allowed attempt.

## Frozen decision

For topology \(t\), with the two optimizer seeds treated as repeated paired
measurements,

```text
D_t = mean_seed(best_feasible_loss(coverage_balanced)
                - best_feasible_loss(no_prior))
```

Negative favors coverage balancing. Topology is the only inference unit. A
tie satisfies `abs(D_t) <= 1e-12`. Stage A passes only if all 14 gates hold:

1. all 32 records authenticate and revalidate;
2. all 16 pairs and all 8 topology blocks are complete;
3. all runs are physically and finite feasible;
4. every pair has a finite loss difference;
5. at least 7 of 8 topology values are wins;
6. median `D_t <= -0.05`;
7. mean `D_t < 0`;
8. each optimizer-seed mean is below zero;
9. each arm-order mean is below zero;
10. linear topology p90 regret is at most `0.5`;
11. the maximum harmful topology difference is at most `0.5`;
12. the overall median treatment/control evaluation-count ratio is at least
    `0.95`;
13. every topology's aggregate evaluation-count ratio is at least `0.90`;
14. the complete-record replay gate is active rather than a partial preview.

The topology-block bootstrap interval is descriptive and cannot override these
gates. Under a simple sign null, 7 or more wins among 8 has one-sided
probability `9/256 = 0.03515625`; 6 or more would be `37/256 = 0.14453125`, so
6/8 is not enough to request a Stage-B review.

Pass action: review the separately described Stage-B design and, only if it
remains acceptable, request separate owner approval. This is review-only; it
does not authorize, unlock, provision, or promote Stage B.

Fail action: `retain_random_start_candidate`.

Not-evaluable action: `retain_candidate_attempt_not_evaluable`.

## Stage-B precommitment

Stage A cannot choose its own confirmation after outcomes are visible. Its
hashed decision policy records the following candidate independent design for
review; it is not an executable Stage-B authorization:

- proposed profile ID `coverage-confirmation-screen-v1` (this profile must be
  registered and independently frozen before any Stage-B run);
- untouched panel `coverage-robustness-v1`, SHA-256
  `e3385a6f4939445e869d71dbf7f1bd5119aa25eebbb89e083b1ec9be336f7309`;
- proposed optimizer seeds 43/47, whose outcome-blind freshness must be
  rechecked before Stage-B freeze;
- 12 topology units, 48 serial paired runs, 1,200 seconds each;
- at least 10/12 wins, median at most `-0.05`, mean below zero, both seed and
  order means below zero, p90 and maximum harmful difference at most `0.5`,
  the same evaluation-ratio gates, and topology-bootstrap mean upper bound
  below zero;
- a complete Stage-B plan/profile freeze and separate owner approval before any
  Stage-B provision.

Even a Stage-B pass at 1,200 seconds is a bounded replication, not a four-hour
official-budget result or hidden-leaderboard evidence.

## Outcome access and cost evidence

The Stage-A ZIP contains `summary.commitment.json`, not plaintext
`summary.json`. Packaging emits the exact summary as a separate release file.
The archive validator recomputes every result from pickle-free histories while
that release remains withheld. A deliberately independent, no-project-import
history-first replay must match production. Only a comparator-issued agreement
may open the exact release whose SHA-256 and byte size were committed before
analysis.

This is a procedural outcome-access boundary, not adversarial cryptographic
secrecy: the detached release is plaintext, the comparator token is only an
accidental-misuse barrier, and a trusted operator could bypass either. The
durable guarantees are the pre-analysis hashes and independent three-way
replay agreement.

After the pod and every related resource are deleted, a post-cleanup Runpod
billing receipt records provider hours, GPU charge, total charge, and cleanup.
The triage source lock authenticates six external files, including that receipt,
and rejects usage above 8 hours, `$26.32` GPU charge, or `$30` total. The
source-lock digest must be recorded independently before analysis; recomputing
it after changing evidence is not authentication.

## Claim boundary and launch authority

- A Stage-A pass only justifies reviewing Stage B.
- A Stage-A failure retains the submitted random-start baseline.
- No pooled Stage-A/Stage-B estimate is permitted.
- No equivalence, non-inferiority, global superiority, speedup, causal drift,
  official-budget, or leaderboard claim is permitted.
- The owner has approved this local refreeze, not paid provisioning. Launch
  requires a new explicit approval after the clean revision, exact candidate,
  plan hash, current H100 price/stock, backup, and smoke command are presented.

Operational steps are in
[`docs/H100_COVERAGE_TRIAGE_RUNBOOK.md`](../docs/H100_COVERAGE_TRIAGE_RUNBOOK.md).
