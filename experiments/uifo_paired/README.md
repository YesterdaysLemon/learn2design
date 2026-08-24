# Paired UIFO evaluation

This harness runs one topology, optimizer seed, and algorithm arm per isolated process. It is designed for accelerator machines and refuses CPU execution unless `--allow-cpu` is explicit.

The initial causal comparison is:

- `no_prior`: feasibility anchor plus seven seeded random population members;
- `semantic_prior`: the identical initial draw, with member 1 replaced by the checked-in semantic prior.

An optional `adam` arm reproduces the organizer-style single-start Adam loop locally. It remains available for bounded diagnostics but is not part of the frozen paid primary panel.

The harness never uses `Objective.best_loss` as the score. It saves full batched loss and feasibility histories, then calculates the minimum finite loss among physically feasible candidates.

The current restart implementation resets each member's Adam age with its moments. It also submits a smaller final population when an evaluation cap is not divisible by the population size and uses recent device-complete batch durations for its time-budget tail guard. These are correctness and budget-accounting properties, not evidence that restarts improve UIFO loss; the paired artifacts record the relevant timing defaults and bind the exact source revision.

## Dry-run a plan

```bash
uv run --group integration python tools/run_uifo_paired.py \
  --topology-seeds 1001 1002 \
  --optimizer-seeds 7 11 \
  --max-time 600 \
  --dry-run
```

## Run or resume

```bash
uv run --group integration python tools/run_uifo_paired.py \
  --topology-seeds 1001 1002 \
  --optimizer-seeds 7 11 \
  --max-time 600 \
  --output artifacts/generated/uifo-paired

uv run --group integration python tools/run_uifo_paired.py \
  --topology-seeds 1001 1002 \
  --optimizer-seeds 7 11 \
  --max-time 600 \
  --output artifacts/generated/uifo-paired \
  --resume
```

For a bounded optimizer-mechanics study, `--optimizer-telemetry member-v1`
adds a separate pickle-free `optimizer-telemetry/<run_id>.npz` artifact. It
records scalar per-member observations, restart state, gradient norms and
clipping scales without parameters, candidates, gradients, device IDs, or
topology strings. The run record binds its schema, SHA-256, row count, and
callback overhead; resume and packaging revalidate it against the authenticated
candidate history. Telemetry adds host transfers inside the Objective clock, so
use it symmetrically for diagnostics and disable it for performance scoring.
The frozen `development-v2` and `confirmation-v1` profiles explicitly prohibit
telemetry and retain their historical plan IDs.

The bounded restart follow-up uses two additional exact profiles. First,
`restart-mechanics-v1` instruments one outcome-selected patience-200 run and
excludes its loss from inference. Only a passing mechanics summary permits the
uninstrumented `restart-screen-v1` comparison: patience 600 versus 200 on eight
mechanically selected development topologies with fresh paired seeds 19/23.
See [`PATIENCE_200_SCREEN_RUNBOOK.md`](../../docs/PATIENCE_200_SCREEN_RUNBOOK.md).

The completed `submission-like-screen-v1` profile evaluated only the unchanged
no-prior/patience-600 candidate on all ten previously untouched
`submission-like-v1` topologies. Fresh seeds 29/31 ran as forward/reverse
mirrored sweeps at 1,200 seconds each. The terminal attempt completed 20/20
runs and independently validated as
`candidate_evidence_complete_for_submission_review`. Its frozen decision is
operational evidence completeness, not an optimizer comparison or
official-budget claim. The terminal receipt forbids another attempt. See
[`SUBMISSION_LIKE_SCREEN_RUNBOOK.md`](../../docs/SUBMISSION_LIKE_SCREEN_RUNBOOK.md).

The unlaunched `coverage-robustness-screen-v1` profile compares the submitted
random suffix with a midpoint Latin-hypercube transform of the identical
pre-transform draw. It uses 12 fresh disjoint topologies, seeds 37/41, exact
H100/CUDA-13 provenance, and 48 serial 1,200-second runs. Records bind the raw
draw hashes; the archive validator independently checks the control suffix,
treatment levels, paired anchor, and exact 249-member package. The production
summary and a no-import history-first reference replay must agree before
`summary.json` can be opened. See
[`H100_COVERAGE_RUNBOOK.md`](../../docs/H100_COVERAGE_RUNBOOK.md). No paid run is
authorized by the checked-in profile.

For an unaudited smoke panel, pass a JSON list through `--topologies-file`. A named panel can use the object form:

```json
{
  "panel_id": "confirmation-v1",
  "topologies": ["AAAAAAAAA-LLLLLLLLLLLD"]
}
```

Confirmation computes archive and prior-panel exclusion from their actual bytes; a boolean inside the panel is deliberately rejected:

```bash
uv run --group integration python tools/run_uifo_paired.py \
  --topologies-file panels/confirmation-v1.json \
  --official-dataset /path/to/dataset.h5 \
  --exclude-prior-panel panels/development-v1.json \
  --require-archive-exclusion \
  --optimizer-seeds 7 11 19 \
  --max-time 600 \
  --dry-run
```

The harness verifies the pinned 75 MB dataset SHA-256, computes exact topology-string intersections, rejects any overlap, and binds the resulting audit plus every source-file digest into the plan. Each explicit topology must contain exactly one `D` or `H` readout; resolved identities are checked for duplicates.

The frozen `panels/` directory contains development, confirmation, and submission-like panels plus a deterministic audit. Rebuild them with `tools/build_topology_panels.py`; live results never enter panel selection.

Runs are serial and arm order rotates between pairs. A frozen study profile validates that the primary pairwise order is balanced, not merely that each arm occupies each marginal position. Each worker has a host timeout (the Objective budget plus 30 minutes by default), durable stdout/stderr, full-process wall time, a strict JSON record, and an atomic compact NPZ candidate history. Existing completed histories are fully revalidated at startup, resume, finalization, and packaging; between workers, the already-validated new record is indexed without reopening every old history or recomputing bootstrap intervals. Final and packaged summaries always include the frozen bootstrap analysis.

The summary collapses optimizer seeds within topology before inference. The predeclared decision has two explicit routes: strict topology-level finite-feasibility dominance, guarded against every reverse seed/topology outcome and harmful observed p90 regret; otherwise, the complete paired-loss gate. Frozen target-loss hits are paired in both wall time and Objective evaluations; unreached thresholds remain explicit censoring. A semantic-only hit can contribute a conservative ratio upper bound using the no-prior arm's last logged time and evaluation count. An order-of-magnitude flag can become true only when every predeclared pair supplies an observed ratio or that conservative upper bound and both topology-bootstrap upper bounds for the log10 ratios are at most -1. No-prior-only and neither-reached pairs cannot pass.

Scored runs launch every isolated worker with persistent JAX compilation caching disabled. The effective cache policy is bound into the manifest, while the inherited host settings are preserved in per-session preflight artifacts. Historical and packaged-default paths still compile inside the Objective clock. The explicitly opt-in H100 coverage profile applies the same public preclock warmup and population-ready boundary to both arms; it does not change the Round-1 default timing path. The rental flags can additionally require exactly one A100 or H100, disabled MIG mode, minimum physical GPU memory and free disk, maximum idle memory/utilization, a shorter worker timeout, and a total session wall limit. The H100 profile also rejects any non-CUDA-13 JAX wheel stack. Use the historical A100 procedure in [`docs/A100_RENTAL_RUNBOOK.md`](../../docs/A100_RENTAL_RUNBOOK.md) or the current coverage procedure in [`docs/H100_COVERAGE_RUNBOOK.md`](../../docs/H100_COVERAGE_RUNBOOK.md).

For memory-limited deployment diagnostics, `--evaluation-chunk-size 1` evaluates population members through the scalar public Objective API while preserving the same optimizer state and initial-population pairing. This is not the packaged default and cannot support a competition-throughput claim; use the default vmap path on A100-class hardware for confirmation.

If a scalar-chunk arm shows a gross latency or memory tail, isolate candidate values before blaming the initializer. The candidate probe recreates the exact population-2 anchor, random member 1, and semantic prior in six fresh processes: forward order, then reverse order. It requires an explicit topology, predeclared idle-device thresholds, no active compute process, and a host timeout of at most five minutes:

```bash
python tools/run_uifo_candidate_probe.py \
  --topology HBHCBBCBG-LDSLSLLSLLSL \
  --optimizer-seed 7 \
  --max-idle-memory-mib 1800 \
  --max-idle-utilization 5 \
  --worker-timeout 180 \
  --output artifacts/generated/uifo-candidate-probe
```

Set the idle thresholds from a clean baseline before the run; do not loosen them after observing a candidate. The probe refuses a dirty tree or existing output directory, samples total-device telemetry throughout each worker, requires an exact parent/worker/telemetry process-ID handshake, and waits for post-worker quiescence. Global memory and utilization thresholds remain mandatory because host-side WDDM processes may not be visible inside WSL. A clean timeout with a validated pre-evaluation candidate milestone is retained as right-censored evidence and does not prevent the reverse-order workers; every other error or provenance failure stops the study. Its two observations per role can diagnose a large candidate-conditioned tail, but they are not optimizer-quality or throughput evidence.

The orchestrator refuses a dirty Git tree and refuses to resume if the revision, plan, upstream reference, semantic-prior bytes, runtime versions, backend, or device identity changed. It also rejects stale/foreign run files and concurrent writers. Commit the harness and configuration before consuming accelerator time.

After a clean completion, `tools/package_uifo_study.py` revalidates every run, history, log, and plan membership before writing a deterministic ZIP, SHA-256 sidecar, and package manifest. If a paid machine must be evacuated after the writer has stopped, `--allow-incomplete` writes an explicitly partial package with missing/error run IDs; it cannot pass as a completed study. A hard-kill lock can be recovered only with the explicit `--resume --recover-stale-lock` combination after the owner process is proven dead; the stale lock is preserved under `recovery/`.

Equal-evaluation studies are useful for diagnosing initialization and restarts, but competition-performance claims require equal wall-clock budgets on size-3 UIFO problems. Treat topology identity as the statistical unit; optimizer seeds are repeated measurements.

## Validate a packaged development result

`tools/analyze_uifo_results.py` authenticates the external ZIP and sidecars,
streams every member without extraction, recomputes run metrics from pickle-free
NPZ histories, performs both the production and independent topology-level
replays, and opens the archived summary only after those replays agree. It then
writes normalized tables and explicitly post-hoc analysis outside Git:

```powershell
$resultsRoot = (Resolve-Path ..\learn2design-runpod-results).Path
uv run --frozen --group integration --group analysis `
  python tools/analyze_uifo_results.py `
  (Join-Path $resultsRoot 'development-v2.zip') `
  --plan (Join-Path $resultsRoot 'development-v2-plan.json') `
  --output (Join-Path $resultsRoot 'development-v2-analysis-replay')
```

The command refuses output under the repository and refuses to overwrite an
existing output directory. Keep source and generated artifacts private; only
aggregate research conclusions belong in Git.

For a sealed `restart-screen-v1` package, use the restart-specific workflow.
It validates the exact 32-run/16-pair/8-topology contract, recomputes production
and independent history-only results while `summary.json` remains sealed, and
opens that summary only after both replays agree:

```powershell
$bundle = (Resolve-Path ..\learn2design-runpod-results\patience-200-20260821\provider-evacuation-811ade1).Path
$output = Join-Path (Split-Path $bundle) 'restart-screen-v1-analysis'
uv run --frozen --group dev --group integration --group analysis `
  python tools/analyze_restart_screen.py `
  (Join-Path $bundle 'restart-screen-v1.zip') `
  --checksum (Join-Path $bundle 'restart-screen-v1.zip.sha256') `
  --package-manifest (Join-Path $bundle 'restart-screen-v1.zip.manifest.json') `
  --plan (Join-Path $bundle 'restart-screen-v1-plan.json') `
  --output $output
```

The generated normalized tables, report, and figures remain outside Git. The
tool refuses any output inside a Git checkout or an existing directory.

For the sealed `submission-like-screen-v1` package, use
`tools/analyze_submission_like.py`. It first authenticates the out-of-band
source-lock digest, all five external sources, the terminal-attempt receipt,
the exact 109-member archive, and every pickle-free history. It then compares
the production record summary with an independent history-first evaluator
before opening archived `summary.json`. Only after exact three-way agreement
does it produce the topology-blocked reliability analysis, normalized private
tables, allowlisted aggregate JSON, report, and four figures. All generated
outputs stay outside Git and every plot must be visually inspected.

The first WSL2 deployment smoke is recorded in `research/2026-08-19-rtx4060-deployment-smoke.md`: scalar UIFO works, while the batched candidate OOMs at population 2 on the 8 GB RTX 4060. A later scalar-chunk attempt is recorded in `research/2026-08-19-low-memory-diagnostic-smoke.md`; one arm completed, but an unrelated CUDA workload contaminated the attempted pair. The clean follow-up in `research/2026-08-19-idle-candidate-probe.md` completed all six workers and did not reproduce a semantic-only latency tail. Use larger-memory hardware for competition-aligned paired studies, and require an otherwise idle device for any local diagnostic comparison.
