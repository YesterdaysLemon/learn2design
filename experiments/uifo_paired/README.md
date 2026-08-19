# Paired UIFO evaluation

This harness runs one topology, optimizer seed, and algorithm arm per isolated process. It is designed for accelerator machines and refuses CPU execution unless `--allow-cpu` is explicit.

The initial causal comparison is:

- `no_prior`: feasibility anchor plus seven seeded random population members;
- `semantic_prior`: the identical initial draw, with member 1 replaced by the checked-in semantic prior.

An optional `adam` arm reproduces the organizer-style single-start Adam loop locally. It lives in this experiment package so importing it does not pull in dfbench's unrelated optional algorithm dependencies.

The harness never uses `Objective.best_loss` as the score. It saves full batched loss and feasibility histories, then calculates the minimum finite loss among physically feasible candidates.

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

Runs are serial and arm order rotates between pairs. Each worker has a host timeout (the Objective budget plus 30 minutes by default), durable stdout/stderr, full-process wall time, a strict JSON record, and an atomic compact NPZ candidate history. The aggregate indexes are rebuilt only after validating every completed artifact against its recorded digest and metrics.

For memory-limited deployment diagnostics, `--evaluation-chunk-size 1` evaluates population members through the scalar public Objective API while preserving the same optimizer state and initial-population pairing. This is not the packaged default and cannot support a competition-throughput claim; use the default vmap path on A100-class hardware for confirmation.

The orchestrator refuses a dirty Git tree and refuses to resume if the revision, plan, upstream reference, semantic-prior bytes, runtime versions, backend, or device identity changed. It also rejects stale/foreign run files and concurrent writers. Commit the harness and configuration before consuming accelerator time.

Equal-evaluation studies are useful for diagnosing initialization and restarts, but competition-performance claims require equal wall-clock budgets on size-3 UIFO problems. Treat topology identity as the statistical unit; optimizer seeds are repeated measurements.

The first WSL2 deployment smoke is recorded in `research/2026-08-19-rtx4060-deployment-smoke.md`: scalar UIFO works, while the batched candidate OOMs at population 2 on the 8 GB RTX 4060. A later scalar-chunk attempt is recorded in `research/2026-08-19-low-memory-diagnostic-smoke.md`; one arm completed, but an unrelated CUDA workload contaminated the attempted pair. Use larger-memory hardware for competition-aligned paired studies, and require an otherwise idle device for any local diagnostic comparison.
