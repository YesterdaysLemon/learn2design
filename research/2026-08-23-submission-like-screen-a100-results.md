# Submission-like A100 screen: validated aggregate result

Date: 2026-08-23

Status: complete and independently validated. The frozen operational gate
passed. This report records aggregate evidence only; source archives, raw
histories, normalized private rows, logs, candidate arrays, and generated plots
remain outside Git.

## 1. Integrity validation

The single terminal `submission-like-screen-v1` attempt ran from clean revision
`adac59accb5b28986f52e0ef0385bc52f9a685bc`, plan ID
`6ec8e68d6b331a3f`, and the previously reviewed no-prior/patience-600 candidate.
The external provenance values are:

| Object | SHA-256 |
| --- | --- |
| Complete ZIP | `c4b8ffc90a8d98db47d366e6a6c3e32e0c7d3b149869f5e695b7ad996d5fd5df` |
| ZIP checksum sidecar | `7d1b807e7d2752e8d6711c733fe0c728654c737bac10a3349e9f29d2e4412400` |
| Package manifest | `1cddff19ff36cd741faf775454cd91e44645d4e2c4c23a2bc1af16e0eb689f3e` |
| Reviewed plan | `a9b5b36f78fc5542a321f062455f0bcce5d6c6b51485600a62ed6a5185ee8075` |
| Terminal-attempt receipt | `5d69f19969cd1fed06a177df2b155e5b94ea2fe316a8b2eeea50f722849a6787` |
| Out-of-band source lock | `4b165759dd025fc757d57ccb1c74d8f63212d6f2281c86a5b1fa98d5272699d2` |

The source-lock digest was recorded before transfer. The local workflow first
authenticated that digest and then parsed the lock, verified the five named
sources and the checksum sidecar's filename and contents, and inspected the ZIP
without extraction. It rejected duplicate or case-colliding names, traversal,
absolute or backslash paths, encryption, symlinks/special files, oversized
entries or totals, and suspicious compression ratios. All 109 members passed
CRC validation and the exact profile-bound allowlist.

The archive contained exactly 20 run records, 20 configs, 20 pickle-free NPZ
histories, 20 stdout logs, 20 stderr logs, and the required session, summary,
index, manifest, package-state, and preflight evidence. All run/history/log
digests matched. The study had zero worker errors, interruptions, timeouts, or
nonzero exits. The exact one-arm hierarchy was preserved: ten topologies, seeds
29 and 31 nested within topology, and 20 serial runs. The topology, not the run
or history row, is the inference unit.

While `summary.json` remained sealed, the validator recomputed physical and
finite feasibility, best feasible loss, target hits, timing, and evaluation
counts from authenticated histories loaded only with
`numpy.load(..., allow_pickle=False)`.

## 2. Frozen-policy reproduction

Two genuinely different calculations were required before the archive summary
could be opened:

1. The production replay used the repository's plan/config validation and
   `summarize_submission_like_records` implementation.
2. The small reference evaluator did not import the production decision or
   aggregation helpers. It reconstructed each run from history first, averaged
   the two seeds within topology, recomputed the topology aggregates and frozen
   decision, and compared every run, topology, target family, bootstrap value,
   and criterion.

The production replay and independent reference agreed. Only then was the
archived summary opened; all three results matched at `1e-12` absolute and
relative tolerance.

| Frozen quantity | Recomputed value |
| --- | ---: |
| Complete runs | 20 / 20 |
| Complete topology blocks | 10 / 10 |
| Physically feasible runs | 20 / 20 |
| Finite-feasible runs | 20 / 20 |
| Mean topology best feasible loss | `3.668145075519928` |
| Median topology best feasible loss | `3.720257129554202` |
| Linear topology p90 loss | `3.9031474950785925` |
| p90 absolute within-topology seed gap | `1.1017258600405546` |
| 95% topology-bootstrap mean interval | `[3.453649953532894, 3.8362434699425836]` |
| Bootstrap seed / resamples | `20260822` / `10000` |
| Frozen status | `passed` |
| Frozen action | `candidate_evidence_complete_for_submission_review` |

All five operational criteria passed. This authorizes final package/evidence
review only. It does not change the candidate, reopen patience 600, establish
superiority or equivalence, estimate hidden-leaderboard rank, or authorize
`confirmation-v1`.

The post-analysis repository minimum build reproduced candidate ZIP SHA-256
`4cc0dbc65a3e61ca5358c18655c432caf478fbdfc07f10512553781f8822924b`,
matching the exact candidate archive bound into the frozen plan.

## 3. Post-hoc sensitivity and exploratory reliability analysis

All resampling and sensitivity calculations retain complete topology blocks;
the two seeds are repeated measurements, not 20 independent observations.
Exploratory p-values are reported as a family of diagnostics and do not create a
new promotion rule.

The topology-block bootstrap median interval was
`[3.5995703116626325, 3.8860660040871053]` with seed `20260823` and 10,000
resamples. Leave-one-topology-out ranges were:

| Quantity | Minimum | Maximum |
| --- | ---: | ---: |
| Mean loss | `3.627605858183076` | `3.7651732238183877` |
| Median loss | `3.7033852916119825` | `3.737128967496422` |
| Linear p90 loss | `3.886596734808225` | `3.917575332464481` |
| p90 absolute seed gap | `0.5629798013613524` | `1.1358904030558616` |

Seed 29's mean loss was `3.520377157127216`; seed 31's was
`3.81591299391264`. Seed 29 was lower on eight topologies and seed 31 on two,
with no ties. Across-topology Pearson correlation was `0.37516440645041826`
(two-sided exploratory `p=0.2854084305154175`), and Spearman correlation was
`0.2727272727272727` (`p=0.44583834154275137`). These are weak, imprecise
consistency diagnostics, not evidence that one optimizer seed is intrinsically
better. Seed 29 was always the first sweep and seed 31 the second, so seed and
sweep phase are not identifiable.

For target loss 4.0, 17/20 runs reached the target. At topology level, both
seeds reached in seven blocks, one seed reached in three, and neither reached
in zero. Two blocks were seed-29-only and one was seed-31-only. Among the seven
both-reached blocks, the descriptive median of the within-topology mean hit
time was `554.9175362586975` seconds and the corresponding median evaluation
count was `2800`. These conditional summaries are selection-biased. Unreached
runs remain right-censored at their observed terminal time and evaluation
count. At targets 1.0, 0.5, and 0.0, no run reached the target; all ten topology
blocks are `neither-reached`. No arbitrary finite hit time was imputed.

The mirrored-order diagnostic found mean later-minus-earlier loss
`0.2955358367854244`; later runs were lower/higher/tied on 2/8/0 topology
blocks. Pair-position gap versus later-minus-earlier loss had Spearman
`rho=0.01818181818181818` (`p=0.9602404181286243`). Pair gap versus log10
throughput contrast had `rho=-0.12727272727272726`
(`p=0.7260570147627894`). Mean log10 later/earlier throughput was
`-0.00010078668915702257`, while the topology-macro mean throughput was
`5.749705671419872` evaluations/second. Strict timezone-aware timestamps were
validated within the completed session and in non-overlapping serial order.
Actual run-start gap versus later-minus-earlier loss had
`rho=0.01818181818181818` (`p=0.9602404181286243`); these gaps include
controller and compilation intervals. Loss contrast versus log10 throughput
contrast had `rho=0.23636363636363633` (`p=0.5108853175152002`). The one-arm
screen has no arm-first contrast. These diagnostics do not identify causal
drift.

The subsequent history-only matched-resource diagnostic found the same broad
direction after controlling evaluation progress. At the last checkpoint shared
by all histories, 6,312 evaluations, the topology-level seed-31-minus-seed-29
mean difference was `+0.35777353288148966` and the median was
`+0.2017395070749819`; seed 29 was lower on eight topologies and seed 31 on two.
At the last common wall-time checkpoint, `1197.446573973` seconds, the mean
difference was `+0.2955358367854244`, the median was
`+0.16334283552654782`, and the lower counts were again 8/2. The contrast
direction agreed on five of six comparable progress fractions, and the mean
absolute contrast difference between axes was `0.0457895698453755`.

Persistence at matched evaluation count is inconsistent with a simple
throughput-only explanation. It does not establish that seed 29 is better:
seed 29 was always the first sweep and seed 31 always the second. Random-start
sensitivity, sweep/session order, and topology heterogeneity remain confounded.
No value was imputed before a run first achieved finite feasibility, and the
diagnostic did not alter the frozen action.

## 4. Limitations

- There are ten independent topology units and only two repeated optimizer
  seeds.
- The 1,200-second worker budget is shorter than the official competition
  budget.
- A one-arm operational screen cannot estimate an optimizer contrast or show
  that the current candidate is competitive.
- Seed and sweep phase are perfectly confounded. The observed later-sweep loss
  shift could reflect stochastic search variation, session drift, or both.
- Target-hit summaries below complete attainment are censored; conditional
  both-reached summaries do not describe the full panel.
- No equivalence or non-inferiority margin was preregistered. Failure to find a
  difference would not establish equivalence.
- The matched-resource diagnostic is post hoc. Its checkpoints and the proposed
  search-robustness direction were not part of the frozen operational decision.

## 5. Recommended next experiment

The history-only diagnostic is complete and weakens a throughput-only
explanation. The next useful gate is a separately frozen, small
search-robustness change on a new disjoint panel. A leading candidate is to
replace unconstrained random starts with antithetic or coverage-balanced starts
while keeping the same optimizer, population size, patience, and resource
budget. That hypothesis targets the observed initial-population sensitivity
without adding an optimizer or model.

Freeze the implementation, paired randomness, topology units, resource cap,
stop conditions, and decision rule before launch. Do not reuse this screen as
confirmation, run `confirmation-v1`, or tune a new rule against the observed
panel.

## Evidence ledger

Independently verified: source-lock-first external authentication, exact ZIP
structure and CRCs, bounded pickle-free histories, record/history/log checksums,
the complete 20-run/10-topology hierarchy, history-derived run outcomes, two
independent frozen replays, and the sealed archived comparison.

Matched frozen summary: all listed aggregates, target families, bootstrap
results, five operational criteria, status, and action.

Confirmatory conclusion: the no-prior/patience-600 candidate has complete
operational evidence for final submission review under this bounded profile.

Exploratory finding: final loss was worse in the later sweep on eight of ten
topologies and the direction persisted at matched evaluation count. This
weakens a throughput-only explanation, but seed and sweep phase remain
confounded. The effect is a search-robustness lead, not an algorithm decision.

Unresolved: official-budget performance, leaderboard competitiveness, causal
session drift, and whether a frozen coverage-balanced initial population
improves a new disjoint panel. Cross-platform outer-ZIP reproducibility is now
closed: Windows and Linux builds both reproduce evaluated candidate SHA-256
`4cc0dbc65a3e61ca5358c18655c432caf478fbdfc07f10512553781f8822924b`.

Exact analysis and focused verification commands:

```powershell
uv run --frozen --group dev --group integration --group analysis `
  python tools/analyze_submission_like.py $archive `
  --checksum "$archive.sha256" `
  --package-manifest "$archive.manifest.json" `
  --plan $plan `
  --source-lock $sourceLock `
  --expected-source-lock-sha256 $recordedSourceLockSha256 `
  --terminal-attempt-receipt $terminalReceipt `
  --output $analysisOutput

uv run --frozen --group dev --group integration --group analysis pytest -q `
  tests/test_submission_like_ingestion.py `
  tests/test_submission_like_reference_analysis.py `
  tests/test_submission_like_plan.py
```

The generated Markdown/HTML report, normalized tables, safe aggregate JSON,
private diagnostics, and five figures stay in the external analysis directory.
All five figures and the rendered report were visually inspected.
