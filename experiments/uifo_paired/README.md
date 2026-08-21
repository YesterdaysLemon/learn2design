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

Scored runs launch every isolated worker with persistent JAX compilation caching disabled. The effective cache policy is bound into the manifest, while the inherited host settings are preserved in per-session preflight artifacts. This is required because compilation currently occurs inside the Objective clock. The rental flags can additionally require exactly one A100, disabled MIG mode, minimum physical GPU memory and free disk, maximum idle memory/utilization, a shorter worker timeout, and a total session wall limit. Use the canonical paid-machine procedure in [`docs/A100_RENTAL_RUNBOOK.md`](../../docs/A100_RENTAL_RUNBOOK.md).

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

The first WSL2 deployment smoke is recorded in `research/2026-08-19-rtx4060-deployment-smoke.md`: scalar UIFO works, while the batched candidate OOMs at population 2 on the 8 GB RTX 4060. A later scalar-chunk attempt is recorded in `research/2026-08-19-low-memory-diagnostic-smoke.md`; one arm completed, but an unrelated CUDA workload contaminated the attempted pair. The clean follow-up in `research/2026-08-19-idle-candidate-probe.md` completed all six workers and did not reproduce a semantic-only latency tail. Use larger-memory hardware for competition-aligned paired studies, and require an otherwise idle device for any local diagnostic comparison.
