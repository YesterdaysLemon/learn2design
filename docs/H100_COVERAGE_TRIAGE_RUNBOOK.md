# H100 coverage triage runbook

**Historical closed procedure — do not execute.** The single terminal Stage-A
attempt completed on 2026-08-25 and failed the frozen promotion rule. Its action
is `retain_random_start_candidate`; do not rerun this procedure or provision
the proposed Stage B. The commands below are retained only to reconstruct and
audit the completed attempt. See
[`research/2026-08-25-h100-coverage-triage-results.md`](../research/2026-08-25-h100-coverage-triage-results.md).

This was the operating guide for `coverage-triage-screen-v1`. It never
authorized paid compute by itself. The terminal result closed both this attempt
and its conditional Stage-B path.

## Frozen envelope

- one secure `NVIDIA H100 80GB HBM3`, one full GPU, MIG disabled;
- Python 3.12 and exact CUDA-13/JAX `0.9.0.1` package contract;
- persistent JAX and CUDA caches disabled;
- 8 `coverage-triage-v1` topologies, paired seeds 37/41;
- 32 serial runs at 600 Objective seconds, full-vmap population 8;
- 1,200-second worker timeout and 7-hour main-session ceiling;
- provider-native auto-stop/delete at 8 hours with a 30-minute evacuation
  reserve;
- observed panel marginals are readout D/H `4/4`, squeezer low/middle/high
  `4/2/2`, and directional low/middle/high `2/4/2`; this is a triage panel, not
  a full complexity-balanced panel;
- price at most `$3.29/GPU-hour`, GPU charge at most `$26.32`, and all-in
  provider charge at most `$30`;
- one loss-blind cold smoke inside that cap;
- first result-bearing attempt terminal: no resume, rerun, top-up, or
  replacement.

Refresh exact secure stock and price immediately before asking for approval. A
different GPU name, cloud type, price, panel, seed, wheel stack, or budget is a
new design.

## 1. Local gates and backup

From the focused branch, verify the submitted baseline remains unchanged:

```powershell
(Get-FileHash artifacts/generated/submission.zip -Algorithm SHA256).Hash.ToLowerInvariant()
```

Expected:
`4cc0dbc65a3e61ca5358c18655c432caf478fbdfc07f10512553781f8822924b`.

Create and hash-verify an encrypted second copy of the existing private evidence
bundle before any new paid work. Then, from a clean committed revision:

```powershell
uv sync --frozen --group dev --group integration
uv run --frozen --group dev --group integration pytest -q
uv run --no-sync python tools/build_submission.py `
  --output artifacts/generated/coverage-triage-candidate.zip `
  --manifest artifacts/generated/coverage-triage-candidate.manifest.json
```

Do not overwrite `submission.zip`, `coverage-candidate.zip`, or their manifests.
The new manifest must report the exact clean launch revision. Record its ZIP and
manifest SHA-256 values.

Every later `uv run --no-sync` is deliberate: it uses the environment already
created by the preceding frozen sync and cannot silently change the locked
dependency set.

## 2. Freeze the external plan

From the clean prepared checkout after the locked integration environment has
been synced, keep the dataset, candidate, plan, output, receipt ledger, and
eventual result bundle on durable storage outside Git. Set the provider stop no
more than eight hours away:

```bash
export L2D_DATASET=/workspace/private/dataset.h5
export L2D_RESULTS=/workspace/private/coverage-triage
export L2D_PROVIDER_STOP_UTC=YYYY-MM-DDTHH:MM:SSZ
mkdir -p "$L2D_RESULTS"

uv run --no-sync python tools/run_uifo_paired.py \
  --topologies-file experiments/uifo_paired/panels/coverage-triage-v1.json \
  --official-dataset "$L2D_DATASET" \
  --exclude-prior-panel experiments/uifo_paired/panels/development-v1.json \
  --exclude-prior-panel experiments/uifo_paired/panels/confirmation-v1.json \
  --exclude-prior-panel experiments/uifo_paired/panels/submission-like-v1.json \
  --exclude-prior-panel experiments/uifo_paired/panels/coverage-robustness-v1.json \
  --exclude-prior-panel experiments/uifo_paired/panels/restart-mechanics-v1.json \
  --exclude-prior-panel experiments/uifo_paired/panels/restart-screen-v1.json \
  --require-archive-exclusion \
  --optimizer-seeds 37 41 \
  --arms no_prior coverage_balanced \
  --max-time 600 \
  --population-size 8 \
  --target-loss 4.0 --target-loss 1.0 --target-loss 0.5 --target-loss 0.0 \
  --worker-timeout 1200 \
  --max-session-wall 25200 \
  --max-worker-failures 1 \
  --provider-stop-utc "$L2D_PROVIDER_STOP_UTC" \
  --provider-evacuation-reserve 1800 \
  --provider-deadline-maximum-horizon 28800 \
  --candidate-package artifacts/generated/coverage-triage-candidate.zip \
  --candidate-package-manifest artifacts/generated/coverage-triage-candidate.manifest.json \
  --study-profile coverage-triage-screen-v1 \
  --pair-order-policy alternate_topology_and_seed \
  --seed-order-policy mirrored_sweeps \
  --require-h100 \
  --required-gpu-name "NVIDIA H100 80GB HBM3" \
  --preclock-warmup \
  --minimum-gpu-memory-mib 75000 \
  --max-idle-gpu-memory-mib 1000 \
  --max-idle-gpu-utilization 5 \
  --minimum-free-disk-gib 20 \
  --plan-output "$L2D_RESULTS/coverage-triage-plan.json" \
  --output "$L2D_RESULTS/study" \
  --dry-run
```

Verify the panel SHA-256 is
`f400cdc3a947cd076ce9bd9f48a2dafcb98dfd3f9f938a74ceb11ca88c360972`,
the plan has 32 runs and 16 pairs, arm-first counts are 8/8, scored time is
19,200 seconds, the main ceiling is 25,200 seconds, the provider horizon is
28,800 seconds, and the resource budget is exactly `$30`/8 hours. Record the
printed plan SHA-256 independently.

The panel's topology-generation/API metadata is bound to official starter-kit
revision `1bb7f54737dec6a08b59879a8831d125f08f8a0b`, the explicit override recorded
by the generator. Archive exclusion is independently bound to dataset SHA-256
`149f6aac17aff2e33750b4e1b6cebd3cef1c39d47ae49a3a7ed77315cb7838a7`;
those bytes were first captured at historical revision
`d9b1bd7d6f2c4df335bc7725755b02aa5f6f942c`. Do not conflate those provenance
layers or rewrite older panel bytes.

Refresh the provider catalog. If exact secure H100 HBM3 stock is unavailable or
its price exceeds `$3.29/hour`, stop. Present the clean revision, candidate
hashes, plan hash, price, provider stop, backup receipt, and `$30` cap to the
owner. Provision only after a new explicit yes.

Do not interpret a passing Stage-A summary as Stage-B approval. Before any
Stage-B spend, register and independently freeze its complete profile, plan,
candidate identity, runtime/provider envelope, and outcome-blind seed-freshness
check, then obtain a separate owner approval.

## 3. Cold mechanics smoke after approval

Configure provider-native auto-stop/delete before installing anything. The
single smoke topology is outside all scored panels; never inspect its loss to
change the design.

```bash
export JAX_ENABLE_COMPILATION_CACHE=false
export CUDA_CACHE_DISABLE=1
export CUDA_VISIBLE_DEVICES=0

uv sync --frozen --python 3.12 \
  --group dev --group integration --group accelerator-h100

uv run --no-sync python tools/run_uifo_paired.py \
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
  --output "$L2D_RESULTS/h100-cold-smoke"
```

Accept only one idle exact H100, disabled MIG, Python 3.12, CUDA 13 with no
CUDA-12 JAX packages, valid full-vmap history, and every preflight/worker gate.
A failed smoke ends this paid attempt after logs are preserved and the resource
is deleted.

For a passing smoke, rerun the exact plan command with `--dry-run` removed and
`--approved-plan-sha256 <recorded-sha256>` added. The CLI derives the terminal
receipt ledger as `coverage-triage-screen-v1.terminal-attempt.json` in the
external plan directory, independent of the plan filename and result output
root. Keep that directory on durable storage. Never add `--resume`.

## 4. Stop, package, evacuate, and clean up

When the writer ends, package once outside the study directory:

```bash
uv run --no-sync python tools/package_uifo_study.py \
  "$L2D_RESULTS/study/<plan-id>" \
  --output "$L2D_RESULTS/coverage-triage-screen-v1.zip"
```

For a terminal partial, use the runbook-approved `--allow-incomplete` path and
preserve it as `not_evaluable`; do not restart. The complete Stage-A package
has exactly 169 members. It contains `summary.commitment.json`, not
`summary.json`; preserve the generated
`coverage-triage-screen-v1.zip.summary.json` separately and do not open it.

Evacuate and hash-verify the ZIP, SHA sidecar, package manifest, detached summary
release, external plan, terminal receipt, candidate ZIP/manifest, smoke logs,
and provider identifiers. Then delete the pod and every related endpoint,
template, and volume. Confirm no resource remains and capture the final billing
snapshot.

After transferring the evidence directory from Linux to private Windows
storage without renaming files, define every PowerShell path explicitly:

```powershell
$evidenceRoot = "C:\private\learn2design\coverage-triage" # choose the real path
$archive = Join-Path $evidenceRoot "coverage-triage-screen-v1.zip"
$checksum = "$archive.sha256"
$packageManifest = "$archive.manifest.json"
$summaryRelease = "$archive.summary.json"
$plan = Join-Path $evidenceRoot "coverage-triage-plan.json"
$terminalReceipt = Join-Path $evidenceRoot "coverage-triage-screen-v1.terminal-attempt.json"
$billingReceipt = Join-Path $evidenceRoot "runpod-billing-receipt.json"
$sourceLock = Join-Path $evidenceRoot "coverage-triage-source-lock.json"
$privateAnalysis = Join-Path $evidenceRoot "private-analysis"
$providerHours = [double](Read-Host "Observed Runpod provider hours")
$gpuCharge = [double](Read-Host "Observed Runpod GPU charge in USD")
$totalProviderCharge = [double](Read-Host "Observed total Runpod charge in USD")
```

Create the bounded post-cleanup receipt from the observed provider values:

```powershell
uv run --no-sync python tools/create_coverage_triage_billing_receipt.py `
  --plan $plan `
  --provider-hours $providerHours `
  --gpu-charge $gpuCharge `
  --total-provider-charge $totalProviderCharge `
  --resources-deleted `
  --output $billingReceipt
```

The command refuses values above 8 hours, `$26.32` GPU charge, or `$30` total.
It is an operator receipt bound to the captured provider audit, not a provider
signature.

## 5. Source lock and outcome-blind replay

Create the six-file source lock outside Git:

```powershell
uv run --no-sync python tools/create_coverage_triage_source_lock.py `
  --archive $archive `
  --checksum $checksum `
  --package-manifest $packageManifest `
  --plan $plan `
  --terminal-attempt-receipt $terminalReceipt `
  --provider-billing-receipt $billingReceipt `
  --output $sourceLock
```

Record the printed source-lock SHA-256 in a separate durable or append-only
channel before analysis. A hash regenerated after evidence changes is not an
authenticity anchor.

Run the procedurally outcome-blind replay. Supplying the detached summary path
does not make the analyzer open it early: it authenticates and independently
replays the raw histories before calling the release gate. The detached file is
plaintext, not encrypted; a trusted operator must refrain from opening it.

```powershell
uv run --no-sync python tools/analyze_coverage_triage.py `
  --archive $archive `
  --checksum $checksum `
  --package-manifest $packageManifest `
  --plan $plan `
  --terminal-attempt-receipt $terminalReceipt `
  --provider-billing-receipt $billingReceipt `
  --source-lock $sourceLock `
  --expected-source-lock-sha256 "<previously-recorded-sha256>" `
  --summary-release $summaryRelease `
  --output $privateAnalysis
```

The analyzer authenticates the source lock before parsing outcomes, validates
the exact 169-member archive and billing cap, recomputes production and
independent history-first summaries, then releases only the precommitted summary
and requires three-way agreement. A terminal partial returns `not_evaluable`
without opening the detached summary. Keep all raw and generated artifacts
private and outside Git.
