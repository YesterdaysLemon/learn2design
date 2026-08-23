# Submission-like screen runbook

This is the operating and sealed-replay guide for the frozen, cost-bounded
`submission-like-screen-v1` profile. Its single terminal attempt completed on
2026-08-23; the launch section is retained for provenance and is not
authorization to create another Pod. Read the frozen plan and validated results
report first.

## Pre-launch approval

Stop unless the evidence-plan PR is merged, the checkout is clean, all checks
pass, no paid resource is active, live A100 price is at most $1.60/hour, and
both verified remaining authorized runway and visible balance are at least
$16.00. Configure a provider-native stop no more than ten hours ahead. Obtain
explicit owner approval for a maximum $16.00 charge before provisioning.

Use exactly one on-demand, non-preemptible A100 80 GB with MIG disabled. Do not
use spot capacity, another GPU type, multiple GPUs, or confirmation-v1.

## Prepare immutable inputs

Use private paths outside Git for the official dataset, deterministic candidate
bundle, plan, study directory, package, source lock, and generated analysis.

```powershell
$repo = (Resolve-Path '.').Path
$private = (Resolve-Path '..\learn2design-runpod-results').Path
$candidate = Join-Path $private 'submission-like-screen-v1-candidate'

python tools/build_submission.py `
  --output (Join-Path $candidate 'submission.zip') `
  --manifest (Join-Path $candidate 'submission.manifest.json')

uv run --frozen --group dev --group integration pytest -q
python tools/build_submission.py
git status --short
git rev-parse HEAD
```

The second build is the repository minimum check; the first creates the exact
private bundle that will be embedded in the plan. They must have the same ZIP
SHA-256. Never commit either generated bundle.

## Freeze and review the exact plan on the approved host

Only after the owner approves provisioning, transfer the private candidate
bundle to the approved Linux A100 host. Set the official dataset, canonical
durable result root, and the provider-native stop. Replace the example stop
timestamp below with the exact configured UTC value. The following Bash array
is the single source for both plan review and execution; do not edit it between
the two commands.

```bash
export L2D_RESULTS=/workspace/learn2design-results/submission-like-screen-v1
export L2D_DATASET=/workspace/private/Learn2Design2026_dataset.hdf5
export L2D_CANDIDATE=/workspace/private/submission-like-screen-v1-candidate
export L2D_PLAN="$L2D_RESULTS/submission-like-screen-v1-plan.json"
read -r -p 'Exact configured provider stop (UTC, YYYY-MM-DDTHH:MM:SSZ): ' L2D_PROVIDER_STOP_UTC
export L2D_PROVIDER_STOP_UTC

L2D_FROZEN_ARGS=(
  --topologies-file experiments/uifo_paired/panels/submission-like-v1.json \
  --official-dataset "$L2D_DATASET" \
  --exclude-prior-panel experiments/uifo_paired/panels/development-v1.json \
  --exclude-prior-panel experiments/uifo_paired/panels/confirmation-v1.json \
  --exclude-prior-panel experiments/uifo_paired/panels/restart-mechanics-v1.json \
  --exclude-prior-panel experiments/uifo_paired/panels/restart-screen-v1.json \
  --require-archive-exclusion \
  --optimizer-seeds 29 31 \
  --arms no_prior \
  --seed-order-policy mirrored_sweeps \
  --population-size 8 \
  --max-time 1200 \
  --target-loss 4.0 --target-loss 1.0 --target-loss 0.5 --target-loss 0.0 \
  --worker-timeout 2100 \
  --max-session-wall 32400 \
  --max-worker-failures 1 \
  --provider-stop-utc "$L2D_PROVIDER_STOP_UTC" \
  --provider-evacuation-reserve 1800 \
  --provider-deadline-maximum-horizon 36000 \
  --candidate-package "$L2D_CANDIDATE/submission.zip" \
  --candidate-package-manifest "$L2D_CANDIDATE/submission.manifest.json" \
  --study-profile submission-like-screen-v1 \
  --require-a100 \
  --minimum-gpu-memory-mib 75000 \
  --max-idle-gpu-memory-mib 1000 \
  --max-idle-gpu-utilization 5 \
  --minimum-free-disk-gib 20 \
  --plan-output "$L2D_PLAN" \
  --output "$L2D_RESULTS/study"
)

uv run --frozen --no-sync --python 3.12 \
  python tools/run_uifo_paired.py "${L2D_FROZEN_ARGS[@]}" --dry-run

export L2D_REVIEW_PLAN_SHA256="$(sha256sum "$L2D_PLAN" | awk '{print $1}')"
export L2D_PLAN_ID="$(
  python -c 'import json,sys; print(json.load(open(sys.argv[1]))["plan_id"])' \
    "$L2D_PLAN"
)"
export L2D_STUDY_DIRECTORY="$L2D_RESULTS/study/$L2D_PLAN_ID"

test "$(git status --porcelain)" = ""
test "$(sha256sum "$L2D_PLAN" | awk '{print $1}')" = "$L2D_REVIEW_PLAN_SHA256"
printf 'review revision=%s profile=%s plan_id=%s plan_sha256=%s\n' \
  "$(git rev-parse HEAD)" submission-like-screen-v1 \
  "$L2D_PLAN_ID" "$L2D_REVIEW_PLAN_SHA256"
```

Review the printed revision, profile, plan ID, plan SHA-256, candidate hashes,
panel/exclusion evidence, provider stop, and resource budget. Record the plan
SHA-256 outside the Pod. The execution command below accepts only that existing
byte-authenticated plan; it rebuilds all live inputs, permits only the creation
timestamp to differ, and then executes the reviewed plan object.

```bash
read -r -p 'Paste the out-of-band reviewed plan SHA-256: ' L2D_APPROVED_PLAN_SHA256
export L2D_APPROVED_PLAN_SHA256
test "$L2D_APPROVED_PLAN_SHA256" = "$L2D_REVIEW_PLAN_SHA256"

uv run --frozen --no-sync --python 3.12 \
  python tools/run_uifo_paired.py "${L2D_FROZEN_ARGS[@]}" \
  --approved-plan-sha256 "$L2D_APPROVED_PLAN_SHA256"
```

The runner refuses a dirty tree, plan drift, another terminal attempt, and
resume. The first error, interruption, wall/deadline guard, timeout, or nonzero
worker exit ends the attempt. A completed worker record that fails integrity
validation is retained with its original digest and converted to a terminal
error record so partial packaging remains possible. Package partial evidence
and stop; never rerun.

## Package, lock, and evacuate

After the writer stops, package the plan-ID study directory to a new path
outside Git. Use `--allow-incomplete` only for a terminal partial attempt.

```bash
export L2D_ARCHIVE="$L2D_RESULTS/submission-like-screen-v1.zip"
export L2D_SOURCE_LOCK="$L2D_RESULTS/submission-like-screen-v1-source-lock.json"

uv run --frozen --no-sync --python 3.12 \
  python tools/package_uifo_study.py \
  "$L2D_STUDY_DIRECTORY" --output "$L2D_ARCHIVE"

python tools/create_submission_like_source_lock.py \
  --archive "$L2D_ARCHIVE" \
  --checksum "$L2D_ARCHIVE.sha256" \
  --package-manifest "$L2D_ARCHIVE.manifest.json" \
  --plan "$L2D_PLAN" \
  --terminal-attempt-receipt \
  "$L2D_RESULTS/study/submission-like-screen-v1.terminal-attempt.json" \
  --output "$L2D_SOURCE_LOCK"
```

For a stopped terminal partial, use the same packaging command with
`--allow-incomplete`; use `--recover-stale-lock` only when the writer is proven
dead and its lock must be preserved. The local analyzer authenticates only
structure, source hashes, the terminal receipt, and the incomplete state. It
does not open `summary.json`, run records, or histories and reports
`not_evaluable`; three-way outcome replay applies only to a complete archive.

Record the printed source-lock SHA-256 before transfer. Download the archive,
sidecar, manifest, plan, terminal receipt, and source lock to a private local
directory outside Git. Terminate the Pod only after local source hashes, ZIP
structure, CRCs, and the complete allowlist or terminal-partial allowlist subset
pass. For a complete study, keep the provider volume until the full three-way
replay succeeds. A partial remains terminal and not evaluable even after its
structural authentication passes.

## Sealed local replay

```powershell
$bundle = (Resolve-Path '..\learn2design-runpod-results\submission-like-screen-v1').Path
$archive = Join-Path $bundle 'submission-like-screen-v1.zip'
$plan = Join-Path $bundle 'submission-like-screen-v1-plan.json'
$sourceLock = Join-Path $bundle 'submission-like-screen-v1-source-lock.json'
$terminalReceipt = Join-Path $bundle 'submission-like-screen-v1.terminal-attempt.json'
$analysisOutput = Join-Path $bundle 'analysis-generated'
$recordedSourceLockSha256 = Read-Host 'Paste the source-lock SHA-256 recorded before transfer'

uv run --frozen --group dev --group integration --group analysis `
  python tools/analyze_submission_like.py $archive `
  --checksum "$archive.sha256" `
  --package-manifest "$archive.manifest.json" `
  --plan $plan `
  --source-lock $sourceLock `
  --expected-source-lock-sha256 $recordedSourceLockSha256 `
  --terminal-attempt-receipt $terminalReceipt `
  --output $analysisOutput
```

This command does not extract the archive. For a complete archive it
authenticates and recomputes raw histories, matches production and independent
reference results, and only then opens the archived summary. After three-way
agreement it writes private normalized diagnostics, an allowlisted aggregate
JSON, a Markdown/HTML report, and four figures under the external output
directory. Every generated plot must be visually inspected. For a terminal
partial it writes only structural integrity evidence and leaves outcomes
sealed. Generated output remains outside Git. Stop and open a discrepancy issue
on any mismatch; do not interpret partial outcomes or launch another attempt.
