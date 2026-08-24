# Final candidate and evidence review

Date: 2026-08-23

Status: complete. This review changed analysis, packaging reproducibility, and
documentation only. It did not change the evaluated patience-600/no-prior
algorithm or authorize `confirmation-v1`.

## Evidence and trajectory review

The sealed `submission-like-screen-v1` result remains unchanged: 20/20 runs and
10/10 topology blocks completed, all runs were physically and finite feasible,
and production replay, the history-first reference calculation, and the
archived summary agree. The frozen action remains
`candidate_evidence_complete_for_submission_review`.

The post-hoc history-only diagnostic compared the repeated seeds at matched
evaluation counts and matched wall times, retaining topology as the inference
unit. No missing feasible value was imputed.

| Diagnostic | Recomputed result |
| --- | ---: |
| Final common evaluation checkpoint | 6,312 evaluations |
| Mean seed-31 minus seed-29 loss at that checkpoint | `+0.35777353288148966` |
| Median difference | `+0.2017395070749819` |
| Seed-29 / seed-31 lower topology counts | 8 / 2 |
| Final common wall-time checkpoint | `1197.446573973` seconds |
| Mean seed-31 minus seed-29 loss at that checkpoint | `+0.2955358367854244` |
| Median difference | `+0.16334283552654782` |
| Axis direction agreement | 5 / 6 comparable fractions |
| Mean absolute contrast difference between axes | `0.0457895698453755` |

The later-sweep gap persists when evaluation count is matched, so an
evaluation-throughput-only explanation is unlikely. Seed 29 was always the
first sweep and seed 31 the second, however, so random seed and session order
remain perfectly confounded. This is a search-robustness lead, not a causal
finding or a new promotion rule.

## Package review

The deterministic builder now pins the ZIP creator platform to the value used
by the evaluated archive. Windows and Linux builds both reproduce SHA-256
`4cc0dbc65a3e61ca5358c18655c432caf478fbdfc07f10512553781f8822924b`.
The flat archive still contains only `requirements.txt`, `semantic_prior.json`,
and `submission.py`; keeping the inert prior data preserves the exact evaluated
bytes. Contract checks confirmed fixed timestamps and creator metadata. A
pattern scan of the built archive found no private host paths, provider paths,
credential markers, private keys, or GPU identifiers.

## Private-data provenance and provider cleanup

The two old provider volumes were inventoried through authenticated,
short-lived workers. On the populated volume, the complete development-v2,
restart-mechanics, and restart-screen archives and their sidecars matched the
authenticated local copies byte for byte. Four small smoke/timing pilot
directories were the only research evidence not already sealed locally. They
were evacuated into one external archive, verified at 126,645 bytes and 76
entries with SHA-256
`ff2d3a6e82c7f176d40f90b981c583779b3047632ce8d822e5d18958b8fecb8f`.
The second volume was empty. After verification, both volumes and all temporary
pods, endpoints, and templates were removed.

No raw history, candidate array, log, dataset, device identifier, secret, or
provider-local absolute path was added to Git. This host is now the only
verified durable holder of the private evidence. An encrypted second backup is
therefore the smallest operational obligation before more paid research.

## Verification

The focused submission/ingestion/reference/plan suite passed 73 tests. The
repository minimum suite passed 222 tests with two intentional skips across 224
collected tests. `git diff --check` passed, and the minimum deterministic build
reproduced the evaluated candidate hash above. The complete sealed analyzer was
also rerun against the external archive; raw replay and archived-summary
comparison both matched, and all five generated plots plus the rendered report
were visually inspected.

```powershell
uv sync --frozen --group dev --group integration --group analysis
uv run --frozen --group dev --group integration --group analysis pytest -q `
  tests/test_submission_contract.py `
  tests/test_submission_like_ingestion.py `
  tests/test_submission_like_reference_analysis.py `
  tests/test_submission_like_plan.py

uv sync --frozen --group dev --group integration
uv run --frozen --group dev --group integration pytest -q
python tools/build_submission.py
git diff --check
```

## Next research gate

Keep the current package as the submission baseline. Before spending GPU time,
freeze one narrow robustness change on a new disjoint panel. The leading
hypothesis is that antithetic or otherwise coverage-balanced random starts can
reduce initial-population sensitivity while keeping BatchedRestartAdam,
population size 8, patience 600, and the resource budget unchanged.

The pre-result plan must fix the code revision, panel, topology inference unit,
paired randomness, resource cap, stop conditions, and decision rule. The
current screen must not be reused as confirmation, and `confirmation-v1`
remains closed.

## Evidence ledger

- Independently verified: sealed one-arm replay, matched-resource trajectories,
  package byte identity across operating systems, private archive hashes, and
  final zero-resource provider state.
- Unchanged decision: retain patience-600/no-prior; operational evidence is
  complete for submission review.
- Exploratory finding: the later-sweep gap survives evaluation matching, which
  motivates an initial-population robustness test but does not identify cause.
- Unresolved: official-budget and hidden-topology competitiveness, causal
  seed/session effects, and whether coverage-balanced starts improve a new
  panel.
- Private outputs: remain outside Git; all five plots and the rendered analysis
  report were visually inspected.
