# H100 coverage screen runbook

This is a pre-launch operating guide for `coverage-robustness-screen-v1`. It
does not authorize paid compute. The owner must explicitly approve the exact
maximum charge after every local and provider gate below passes.

## Frozen envelope

- one secure-cloud `NVIDIA H100 80GB HBM3`;
- exact one-GPU `nvidia-smi` and JAX-visible identity, MIG disabled;
- Python 3.12 and `jax==jaxlib==jax-cuda13-pjrt==jax-cuda13-plugin==0.9.0.1`;
- CUDA-13 runtime wheel installed and CUDA-12 runtime wheel absent;
- persistent JAX and CUDA driver caches disabled;
- 12 `coverage-robustness-v1` topologies, optimizer seeds 37/41;
- arms `no_prior` and `coverage_balanced`, paired hashes for the seven raw
  non-anchor draws actually used by both arms;
- 48 serial runs at 1,200 Objective seconds each;
- 2,100-second worker timeout, 22-hour session ceiling;
- 30-minute evacuation reserve and provider stop no more than 26 hours away;
- secure price no more than `$3.29/GPU-hour`, 22 provider hours, `$75` total;
- the first result-bearing attempt is terminal: no resume or rerun.

Refresh stock and price immediately before requesting approval. A different
H100 name, cloud type, price, wheel stack, or budget is a new design and must
not be substituted silently.

## Local gates

From a clean focused revision:

```powershell
uv sync --frozen --group dev --group integration
uv run --frozen --group dev --group integration pytest -q
python tools/build_submission.py `
  --output artifacts/generated/coverage-candidate.zip `
  --manifest artifacts/generated/coverage-candidate.manifest.json
```

Do not overwrite `artifacts/generated/submission.zip`; it is the submitted
Round-1 baseline. Verify the new manifest reports a clean revision and bind the
candidate ZIP and manifest through `--candidate-package` and
`--candidate-package-manifest`.

The full suite must include the synthetic 249-member archive, independent
history-first replay with a zero-project-import boundary, production/reference
comparison, exact call/evaluation/time ceilings, exact shared-raw-draw rank
transformation, preflight cross-checks, terminal-partial classification, and a
comparator-issued summary-unlock test.

## Freeze the scored plan and request approval

Set a UTC provider stop no more than 26 hours in the future. From the same clean
revision that built the candidate, create the plan outside Git:

```bash
python tools/run_uifo_paired.py \
  --topologies-file experiments/uifo_paired/panels/coverage-robustness-v1.json \
  --official-dataset "$L2D_DATASET" \
  --exclude-prior-panel experiments/uifo_paired/panels/development-v1.json \
  --exclude-prior-panel experiments/uifo_paired/panels/confirmation-v1.json \
  --exclude-prior-panel experiments/uifo_paired/panels/submission-like-v1.json \
  --exclude-prior-panel experiments/uifo_paired/panels/restart-mechanics-v1.json \
  --exclude-prior-panel experiments/uifo_paired/panels/restart-screen-v1.json \
  --require-archive-exclusion \
  --optimizer-seeds 37 41 \
  --arms no_prior coverage_balanced \
  --max-time 1200 \
  --population-size 8 \
  --target-loss 4.0 --target-loss 1.0 --target-loss 0.5 --target-loss 0.0 \
  --worker-timeout 2100 \
  --max-session-wall 79200 \
  --max-worker-failures 1 \
  --provider-stop-utc "$L2D_PROVIDER_STOP_UTC" \
  --provider-evacuation-reserve 1800 \
  --provider-deadline-maximum-horizon 93600 \
  --candidate-package artifacts/generated/coverage-candidate.zip \
  --candidate-package-manifest artifacts/generated/coverage-candidate.manifest.json \
  --study-profile coverage-robustness-screen-v1 \
  --pair-order-policy alternate_topology_and_seed \
  --seed-order-policy mirrored_sweeps \
  --require-h100 \
  --required-gpu-name "NVIDIA H100 80GB HBM3" \
  --preclock-warmup \
  --minimum-gpu-memory-mib 75000 \
  --max-idle-gpu-memory-mib 1000 \
  --max-idle-gpu-utilization 5 \
  --minimum-free-disk-gib 20 \
  --plan-output "$L2D_RESULTS/coverage-plan.json" \
  --output "$L2D_RESULTS/coverage-screen" \
  --dry-run
```

Record and independently review the printed plan SHA-256, exact 48 runs,
candidate evidence, provider stop, and `$75` envelope. Refresh H100 stock and
price. Then request explicit owner approval for at most `$75` total provider
spend, including the smoke below. Do not provision from a general offer of GPU
budget.

## Cold H100 mechanics smoke

After explicit approval, provision one exact H100 and run a single cold-process
mechanics smoke with caches disabled. Use a fresh topology seed that is not part
of any scored panel and do not interpret its loss:

```bash
export JAX_ENABLE_COMPILATION_CACHE=false
export CUDA_CACHE_DISABLE=1
export CUDA_VISIBLE_DEVICES=0

uv sync --frozen --python 3.12 \
  --group dev --group integration --group accelerator-h100

python tools/run_uifo_paired.py \
  --topology-seeds 2026082999 \
  --optimizer-seeds 7 \
  --arms no_prior \
  --population-size 8 \
  --max-time 120 \
  --worker-timeout 600 \
  --max-session-wall 900 \
  --require-h100 \
  --required-gpu-name "NVIDIA H100 80GB HBM3" \
  --preclock-warmup \
  --minimum-gpu-memory-mib 75000 \
  --max-idle-gpu-memory-mib 1000 \
  --max-idle-gpu-utilization 5 \
  --minimum-free-disk-gib 20 \
  --output "$L2D_RESULTS/h100-warmup-smoke"
```

Accept this only if exactly one idle H100 is visible, Python 3.12 and CUDA 13
are active with no CUDA-12 JAX packages, the
single run completes with a valid admitted history, and every preflight/worker
check passes. The public warmup methods return no arrays, so this proves lawful
dispatch-before-logging mechanics, not that asynchronous device execution
finished before the clock began. Population initialization itself is explicitly
blocked ready before logging in both arms, so the treatment-only rank/logit
transform cannot leak into the first scored evaluation.

Destroy a failed smoke pod after preserving its logs. Do not inspect smoke loss
to modify the scored panel or policy. A failed smoke ends the approved paid
attempt.

A passing smoke permits execution of the exact dry-run command above with
`--dry-run` removed and `--approved-plan-sha256 <reviewed-sha256>` added.
Execution requires the unchanged external plan. The output directory must be
new and outside Git. Never add `--resume`: the terminal receipt forbids it for
this profile.

## Preserve and analyze

Whether complete or terminal-partial, stop the writer before packaging and
evacuate the ZIP, checksum, package manifest, plan, terminal-attempt receipt,
and the exact candidate ZIP/manifest to private durable storage. The plan
already contains the candidate hashes produced by launch-time package
validation. Create the result source lock outside every Git checkout:

```powershell
python tools/create_coverage_source_lock.py `
  --archive $archive `
  --checksum $checksum `
  --package-manifest $packageManifest `
  --plan $plan `
  --terminal-attempt-receipt $terminalReceipt `
  --output $sourceLock
```

Immediately record the printed source-lock SHA-256 in a separate durable or
append-only channel before analysis or transfer. That independently recorded
digest is the authenticity anchor: a digest freshly regenerated after changing
the archive proves only self-consistency and must not be accepted. Replay with:

```powershell
python tools/analyze_coverage_robustness.py `
  --archive $archive `
  --checksum $checksum `
  --package-manifest $packageManifest `
  --plan $plan `
  --terminal-attempt-receipt $terminalReceipt `
  --source-lock $sourceLock `
  --expected-source-lock-sha256 $sourceLockSha256 `
  --output $privateAnalysis
```

The analyzer authenticates the lock before parsing it, retains the expected
lock digest, authenticated result hashes, and plan-bound candidate evidence in
its receipt, validates the exact
archive and histories while `summary.json` is sealed, requires production and
independent replay agreement, and only then opens and compares the archived
summary. A terminal partial returns `not_evaluable` without opening outcomes.
Keep all source and generated analysis artifacts outside Git; only a reviewed,
privacy-safe aggregate report belongs here.
