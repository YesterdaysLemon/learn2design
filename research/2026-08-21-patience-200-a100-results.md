# Patience-200 bounded A100 screen — validated result

Date: 2026-08-21

Study revision: `811ade10288562481fcacbf99306cd44ff0d4886`

Screen plan: `4af939af65a5314f`

Frozen policy: `patience-200-development-screen-v1`

## 1. Integrity validation

The loss-blind `restart-mechanics-v1` predecessor was authenticated and passed
its frozen mechanics action, `run_restart_screen_v1`. Its loss was not used or
reported. The subsequent uninstrumented `restart-screen-v1` package was then
validated without extraction and without opening `summary.json`.

The authenticated source basenames and SHA-256 values are:

| Source | SHA-256 |
| --- | --- |
| `restart-screen-v1.zip` | `7dcf9baa6f4ce08f68248207b001984275e08851557e0394b22ebf798661251e` |
| `restart-screen-v1.zip.sha256` | `ae74633e6227d7bd0692cb02cf9aa470457a15c13e46bb414668c2bf4bd18fc0` |
| `restart-screen-v1.zip.manifest.json` | `1ad2bee26d896928f41af8c79ce694e2e5080e474ce475842cb5542ffc0d148a` |
| `restart-screen-v1-plan.json` | `f093055089c580c49ee540f93f80d5eedbaf0e6be2b01fc8a619d12437698450` |

Validation accepted exactly 169 members: 32 records, 32 configs, 32 pickle-free
NPZ histories, 32 stdout logs, 32 stderr logs, the fixed session/package files,
and preflight evidence. ZIP paths, types, duplicates, encryption, compression
methods, CRCs, per-entry/total sizes, and compression ratios passed. Total
uncompressed size was 3,093,251 bytes and the maximum compression ratio was
12.271. The sidecar filename and digest both matched the ZIP.

All 32 workers completed with zero timeout or nonzero return code. The hierarchy
was exactly 16 topology-seed pairs and 8 topology inference units, using seeds
19 and 23, balanced arm-first order, serial execution, population 8, 600-second
budgets, full-vmap evaluation, cache-disabled Python 3.12, and one A100 80 GB
PCIe with MIG disabled. Every history and log digest matched its record; all
metrics were recomputed from histories. `summary.json` stayed sealed throughout
these checks.

## 2. Frozen-policy reproduction

Before any post-hoc test, the production summarizer and a deliberately small,
history-first reference evaluator independently paired arms, averaged the two
optimizer seeds within topology, and recomputed every frozen criterion. Their
8 topology values, 16 seed pairs, and 11 decision criteria matched. Only then
was the archived summary opened; its frozen fields also matched. The final full
production/reference/archive comparison passed at absolute and relative
tolerance `1e-12`.

Patience-200 minus patience-600 is the signed effect, so negative favors 200:

| Frozen quantity | Result |
| --- | ---: |
| Complete runs / seed pairs / topologies | 32 / 16 / 8 |
| Finite-feasible in both arms | 16/16 seed pairs |
| p200 wins / ties / losses | 4 / 0 / 4 |
| Topology mean difference | -0.016933403182594786 |
| Topology median difference | +0.05236019711778772 |
| Topology p90 regret | 0.19527321150173893 |
| Frozen 95% topology-bootstrap mean CI | [-0.1960204729323901, 0.14039272215043327] |
| Seed-19 / seed-23 mean differences | -0.0003617288410160402 / -0.03350507752417353 |
| Bootstrap seed / resamples | 20260821 / 10,000 |

Completion, comparability, feasibility guards, mean-below-zero, p90-regret,
and both-seed-mean criteria passed. The required minimum of six topology wins
failed, and the required median of at most -0.05 failed. The frozen status and
action are therefore:

```text
failed / retain_patience_600
```

No exploratory result below changes that action.

## 3. Phase 3 — post-hoc sensitivity and exploratory analysis

The exact paired sign-flip statistic was explicitly
`abs(mean(s_i * d_i))`, enumerating all `2^8 = 256` topology sign assignments.
It gave two-sided `p=0.8515625`. The separate exact two-sided sign test, which
discards magnitudes and tests direction only, gave `p=1.0` for the 4/4 split.

The exact SciPy 1.16.1 Wilcoxon signed-rank sensitivity gave `p=0.84375`, with
`zero_method="wilcox"`, zero discarded differences, zero absolute-rank ties,
and no continuity correction. This is only a sensitivity analysis because its
signed-rank interpretation requires a symmetric difference distribution.

Topology-block bootstraps preserved both optimizer seeds. The exploratory 95%
intervals were `[-0.2006390070529963, 0.14153457178793447]` for the mean and
`[-0.14809485695048563, 0.14421091142180775]` for the median. The standardized
mean difference was -0.06493. None supplies evidence of a stable average or
median improvement.

Seed behavior was inconsistent: two topologies favored p200 under both seeds,
three favored p600 under both, and three changed sign. The cross-seed Pearson
correlation was 0.1860 (`p=0.6591`). Leave-one-topology-out means ranged from
-0.06427 to +0.05677, medians from -0.01526 to +0.11998, p200 wins from 3 to 4,
and p90 regret from 0.13515 to 0.21229.

Descriptive topology means ranged from -0.53284 to +0.31442. Three topologies
showed at least 0.05 benefit and four at least 0.05 harm, but these are
post-hoc anonymized patterns, not confirmed subgroups.

### Censor-aware target hitting

| Target | Both reached | p200 only | p600 only | Neither | Complete topologies |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4.0 | 10 | 1 | 4 | 1 | 3/8 |
| 1.0 | 0 | 0 | 0 | 16 | 0/8 |
| 0.5 | 0 | 0 | 0 | 16 | 0/8 |
| 0.0 | 0 | 0 | 0 | 16 | 0/8 |

Unreached targets remain censored; no arbitrary finite times were imputed. No
target had all eight complete topology blocks, so topology-level time and
evaluation-count intervals were not estimable. In the three complete
topologies at target 4.0, the descriptive mean log10 p200/p600 ratios were
+0.02082 for time and +0.05119 for evaluations; complete-case selection makes
these unsuitable for a general claim.

Serial run order and session time each had Spearman rho 0.2143 (`p=0.6103`).
The within-topology arm-first contrast was +0.25910 (exact sign-flip
`p=0.09375`). The mean topology log10 evaluation-count ratio was -0.000291 with
95% bootstrap interval `[-0.001124, 0.000652]`; the mean log10 evaluation-rate
ratio was -0.000134. These diagnose possible drift only and do not alter the
frozen decision.

The data do not support a claim that p200 improves average final feasible loss,
the median topology, finite-feasibility rates, or useful-target arrival. The
heterogeneity is too unstable and post-hoc to establish a topology-dependent
benefit. Failure to reject is not equivalence or non-inferiority; no equivalence
margin was preregistered.

## 4. Limitations

This was a development screen with only eight independent topologies and two
optimizer seeds per topology. It selected between two patience settings and
cannot confirm its own choice. Strong target censoring prevents a complete
target-time comparison. The arm-first diagnostic is imprecise and potentially
interesting, but choosing a favorable order interpretation after seeing the
screen would be outcome-dependent. Hidden competition performance remains
unknown.

## 5. Recommended next experiment

Retain the packaged patience-600/no-prior candidate and stop tuning patience on
this development panel. The next evidence gate should be a separately frozen,
no-prior-only submission-like evaluation on the existing disjoint panel. Freeze
the candidate, topology panel, official-style budget, outcome schema, and a hard
cost/stop envelope before launch; make no algorithm change during the gate.

This recommendation is deliberately not `confirmation-v1`: no treatment won a
development promotion. It is also not authorization to spend GPU budget from
this report. A cheaper targeted diagnostic would be reasonable only if it asks
a narrower operational question without reusing this panel for selection.

## Reproduction and private outputs

The authenticated command is:

```powershell
$privateRoot = (Resolve-Path '..\learn2design-runpod-results\patience-200-20260821').Path
$source = Join-Path $privateRoot 'provider-evacuation-811ade1'
$output = Join-Path $privateRoot 'restart-screen-v1-analysis-811ade1-v3'

uv run --frozen --group dev --group integration --group analysis `
  python tools/analyze_restart_screen.py `
  (Join-Path $source 'restart-screen-v1.zip') `
  --checksum (Join-Path $source 'restart-screen-v1.zip.sha256') `
  --package-manifest (Join-Path $source 'restart-screen-v1.zip.manifest.json') `
  --plan (Join-Path $source 'restart-screen-v1-plan.json') `
  --output $output
```

The output directory contains the integrity and frozen-replay receipts,
production/reference/archive JSON, full exploratory JSON, normalized CSV tables
with a data dictionary, a Markdown and HTML report, and four PNG figures. It is
outside Git. Raw histories, logs, candidate arrays, GPU identifiers, secrets,
and absolute provider paths are not committed.

Repository verification for the evidence implementation was:

```text
uv sync --frozen --group dev --group integration --group analysis
uv run --frozen --group dev --group integration --group analysis pytest -q
python tools/build_submission.py
git diff --check
```

The full suite passed 156 tests. The lean integration environment passed 52
tests with the analysis-only rendered-report test skipped. The deterministic
submission ZIP remained
`4cc0dbc65a3e61ca5358c18655c432caf478fbdfc07f10512553781f8822924b`.
All four plots and the rendered HTML report were visually inspected; the browser
reported zero console warnings or errors.
