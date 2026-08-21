# Development-v2 results handoff

The private source artifacts and generated outputs stay in the sibling
`learn2design-runpod-results/` directory. Nothing in that directory belongs in
Git.

## Reproduce

```powershell
$resultsRoot = (Resolve-Path ..\learn2design-runpod-results).Path
$analysisOutput = Join-Path $resultsRoot 'development-v2-analysis-replay'

uv sync --frozen --group dev --group integration --group analysis
uv run --frozen --group dev --group integration --group analysis `
  pytest -q tests/test_uifo_results_ingestion.py
uv run --frozen --group integration --group analysis `
  python tools/analyze_uifo_results.py `
  (Join-Path $resultsRoot 'development-v2.zip') `
  --plan (Join-Path $resultsRoot 'development-v2-plan.json') `
  --output $analysisOutput
```

The validated generated bundle is under
`learn2design-runpod-results/development-v2-analysis/`. Start with
`validation.json`, `three_way_comparison.json`, `analysis_report.md`, and
`handoff.json`; normalized tables are under `normalized/`, and visually checked
plots are under `figures/`.

## Decision and next gate

The frozen status/action is `failed` / `retain_no_prior_candidate`. Do not run
`confirmation-v1`. First align the packaged submission with `no_prior` in a
separate PR. The recommended later evidence gate is a separately frozen,
owner-approved no-prior submission-like evaluation on the existing disjoint
panel; this handoff does not launch it.
