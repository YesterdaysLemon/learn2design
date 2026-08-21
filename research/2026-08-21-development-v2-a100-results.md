# Development-v2 A100 results — 2026-08-21

## Decision

The frozen `development-v2` screen completed, validated, and failed its
predeclared promotion policy. The required action is
`retain_no_prior_candidate`; do not run `confirmation-v1`.

This report contains aggregates only. The source ZIP, histories, candidate
arrays, configs, logs, GPU identifiers, and generated analysis outputs remain
outside Git.

## 1. Integrity validation

The analysis authenticated these external files before parsing study results:

| File | SHA-256 |
| --- | --- |
| `development-v2.zip` | `7b28509299e81e1f5151c4854bb5591d022de44e17b12c81e71fa2a08eabce24` |
| `development-v2.zip.manifest.json` | `533f28f44e69ab9efe817520daca1962ecc6aa2d16a2ef709d4c88a120a7fafc` |
| `development-v2.zip.sha256` | `74c6e781fcf3c74144ee764cf3a7cb8a4ea14c6e05da2da5dcc8382b80c4c4cf` |
| `development-v2-plan.json` | `a6cd004891809f2ecc07370c31cfae876836d924b8928ffbacda59fb5c7c6108` |

The checksum sidecar's contents, filename, and digest matched the actual ZIP.
The external package manifest's source path was treated as provenance only.

Before the archived summary was opened, synthetic tests covered CRC corruption,
duplicate and case-colliding names, traversal and absolute paths, backslashes,
symlink-like entries, per-entry and total size limits, suspicious compression,
malformed JSON, missing artifacts, checksum failures, duplicate run IDs, broken
pairings, incorrect topology hierarchy, and pickle-bearing NPZ files.

The real archive then passed:

- all 329 ZIP CRCs, canonical paths, regular-file types, size limits, and a
  maximum observed compression ratio of 19.31;
- exactly 64 records, 64 configs, 64 histories, 64 stdout logs, 64 stderr logs,
  the five required root JSON/JSONL files, and four preflight artifacts;
- every history and log checksum recorded by its run;
- `numpy.load(..., allow_pickle=False)` and exact NPZ schema/dtype checks;
- per-run metrics recomputed from all 195,912 history rows;
- 64 clean worker exits, no timeouts, errors, or interruptions;
- plan `49ff0e783f4f6a10`, source revision
  `dbd557b713ab657ac971957369d89eb67649d09f`, frozen profile and decision
  policy, 16 topologies, seeds 7 and 11, 32 complete pairs, balanced arm-first
  order, population 8, 600-second budgets, full-vmap execution, Python 3.12,
  one A100 80GB PCIe with MIG disabled, disabled persistent compilation cache,
  and no forbidden JAX/XLA/CUDA overrides.

The archive was streamed in place and never extracted.

## 2. Frozen-policy reproduction

Two distinct calculations ran before reading `summary.json`:

1. **Production replay** reused the plan/config validation and
   `experiments.uifo_paired.analysis.summarize_records` implementation.
2. **Independent reference calculation** rebuilt run outcomes from history rows
   and independently paired arms, averaged the two optimizer seeds within each
   topology, calculated topology differences and feasibility discordance,
   bootstrapped complete topology blocks, and evaluated the frozen criteria. It
   does not import decision or aggregation helpers from `analysis.py`.

Both calculations agreed at every topology. Only then was `summary.json` opened;
all three agreed. Comparisons used absolute and relative tolerance `1e-12` to
cover last-bit Python 3.12/3.13 differences in a few target-time logarithms.

| Frozen result | Reproduced value |
| --- | ---: |
| Complete runs | 64 / 64 |
| Complete optimizer-seed pairs | 32 / 32 |
| Complete topology units | 16 / 16 |
| Finite-feasible pairs in both arms | 32 / 32 |
| Feasibility discordance | 0 |
| Semantic-prior wins / ties / losses | 7 / 0 / 9 |
| Mean topology difference | +0.029737199613027163 |
| Median topology difference | +0.01409399791234045 |
| 95% frozen bootstrap CI for mean | [-0.05009649678888578, 0.11516577927277213] |
| Frozen bootstrap seed / resamples | 20260819 / 10,000 |
| Topology p90 regret | 0.2009830174305911 |
| Frozen status | `failed` |
| Frozen action | `retain_no_prior_candidate` |

Positive differences favor `no_prior`. The confirmatory conclusion is narrow:
`semantic_prior` did not pass the preregistered development screen. This is not
evidence that every semantic prior is ineffective and is not an equivalence or
non-inferiority result.

## 3. Phase 3 — post-hoc sensitivity and exploratory analysis

The primary exploratory estimand remains the topology-level mean over optimizer
seeds of `semantic_prior - no_prior` best feasible loss. The topology is the
inference unit (`n=16`); neither 64 runs, 32 seed pairs, nor history rows were
treated as independent units.

### Tests and effect uncertainty

- The exact mean sign-flip test enumerated all `2^16 = 65,536` sign assignments
  using `abs(mean(s_i d_i))`; two-sided `p=0.5107421875`.
- The separate exact two-sided sign test for direction only used the 7-versus-9
  split; `p=0.803619384765625`.
- Wilcoxon signed-rank sensitivity used
  `scipy.stats.wilcoxon`, exact two-sided mode, `zero_method="wilcox"`, no
  continuity correction, zero zero-differences, and no tied absolute ranks;
  `p=0.433197021484375`. Its symmetry assumption makes it sensitivity evidence,
  not a preferred promotion test.
- A 10,000-resample complete-topology-block bootstrap gave mean CI
  `[-0.05110890746055037, 0.11153523141742168]` and median CI
  `[-0.029180718616148127, 0.11620360620125547]` with deterministic exploratory
  seeds. Both optimizer seeds stayed inside every resampled topology block.
- The standardized mean difference was `0.17199822319966201`, with exploratory
  bootstrap CI `[-0.33326637300739365, 0.7052258470696733]`.

No p-value above is a revised promotion rule, and the interpretation does not
select whichever analysis looks most favorable.

### Robustness and heterogeneity

Leave-one-topology-out results stayed on the no-prior side for the mean and
median: mean range `[0.0027973489656134554, 0.05422257383059761]`, median range
`[0.013746492703460289, 0.014441503121220611]`, 6–7 semantic-prior wins, and p90
regret range `[0.15161107844434163, 0.20812025213001062]`.

Descriptively, four topologies showed at least 0.05 benefit and six showed at
least 0.05 harm. The full topology-difference range was
`[-0.33754341365052953, 0.43383495932423277]`. These are post-hoc extremes, not
confirmed subgroups. Seed signs were opposite in 14 of 16 topologies and both
harmful in the remaining two; the seed-difference correlation was
`-0.4084745355990786` (`p=0.11622978752269619`). The substantial seed
instability prevents a strong topology-specific benefit claim.

### Existing censor-aware target summaries

The existing target summaries and `order_of_magnitude_claim_ready` rule were
reused. No unreached target was assigned an arbitrary finite time.

| Target | Both reached | Semantic only | No-prior only | Neither | Claim ready |
| ---: | ---: | ---: | ---: | ---: | --- |
| 4.0 | 8 | 7 | 8 | 9 | no |
| 1.0 | 0 | 0 | 0 | 32 | no |
| 0.5 | 0 | 0 | 0 | 32 | no |
| 0.0 | 0 | 0 | 0 | 32 | no |

At target 4.0 only three complete topology blocks supported the existing
censor-aware ratio calculation; topology inference was not ready. The time and
evaluation log-ratio bootstrap upper bounds were above the ten-times-faster
threshold. The data do not support a general claim that `semantic_prior`
reaches useful targets sooner.

### Run order, session drift, and throughput

Serial run order and session time gave the same descriptive Spearman
`rho=-0.02058823529411765` (`p=0.9396737397890448`). The within-topology
arm-first contrast—semantic-first pair difference minus no-prior-first pair
difference—had mean `+0.16810713495023408`, median
`+0.1907315873726354`, and exact sign-flip `p=0.2159423828125`. This possible
order sensitivity is exploratory and does not revise the frozen outcome.

Evaluation throughput was essentially balanced: mean topology log10 evaluation
ratio `-0.00008802729324980372`, median `0.000004179079985965322`, and bootstrap
CI `[-0.001891121277787451, 0.0018183764210536759]`. The loss result is not
explained by a material aggregate throughput advantage.

### Narrow conclusion checks

- Average final feasible loss improvement: **not supported**; the observed mean
  is positive and uncertainty spans zero.
- Median-topology improvement: **not supported**; the observed median is positive
  and uncertainty spans zero.
- Finite-feasibility improvement: **not supported**; all 32 pairs were
  finite-feasible in both arms.
- Faster useful-target attainment: **not supported generally**; censoring is
  substantial and no target passes the existing claim rule.
- Meaningful topology-dependent benefit: **descriptively possible but not
  established**; apparent benefits are seed-unstable and post hoc.

## 4. Limitations

- There are only 16 independent topology units and two optimizer seeds per
  topology.
- Targets at 1.0 and below are completely censored; target 4.0 remains mixed and
  topology-incomplete for comparative inference.
- Exploratory tests form a family and were not preregistered as confirmatory.
- No equivalence margin was preregistered. Failure to reject is not equivalence
  or non-inferiority.
- The A100 development screen is competition-aligned for this comparison but is
  not a hidden leaderboard or official-budget submission result.

## 5. Recommended next experiment

After a separate PR makes the packaged candidate follow the frozen
`retain_no_prior_candidate` action, the next evidence gate should be a
**no-prior-only submission-like evaluation on the already frozen disjoint
submission-like panel at the official budget**. This addresses the most relevant
remaining uncertainty: absolute performance and feasibility of the retained
candidate under submission-like conditions. Require the existing archive
exclusion, serial execution, cache policy, provenance, artifact-integrity, and
topology-level reporting gates before any claim.

Do not launch that study from this report. It is paid, long-running work and
requires a separately frozen run plan and explicit owner approval. Do not run
`confirmation-v1`, add an optimizer/model, or revise the failed development
policy.

## Reproduction

From the repository root in PowerShell, with the private results directory kept
as a sibling of the repository:

```powershell
$resultsRoot = (Resolve-Path ..\learn2design-runpod-results).Path
$analysisOutput = Join-Path $resultsRoot 'development-v2-analysis-replay'

uv run --frozen --group integration --group analysis `
  python tools/analyze_uifo_results.py `
  (Join-Path $resultsRoot 'development-v2.zip') `
  --plan (Join-Path $resultsRoot 'development-v2-plan.json') `
  --output $analysisOutput
```

The command refuses an existing output directory and any output under the Git
repository. It writes normalized CSV/JSON tables, three-way replay evidence,
the exploratory report, and five PNG figures outside Git.
