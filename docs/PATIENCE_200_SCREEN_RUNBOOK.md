# Patience-200 bounded A100 runbook

> Completed and validated on 2026-08-21. The mechanics gate passed; the screen
> failed its frozen policy with action `retain_patience_600`. This file is now
> the execution record, not the next-action authority. See
> [`CURRENT_HANDOFF.md`](CURRENT_HANDOFF.md) and the
> [aggregate result](../research/2026-08-21-patience-200-a100-results.md).

This runbook executes the two frozen `restart-mechanics-v1` and
`restart-screen-v1` profiles. Read
[`2026-08-21-patience-200-screen-plan.md`](../research/2026-08-21-patience-200-screen-plan.md)
first. Do not improvise settings, inspect mid-screen loss, or touch the reserved
`submission-like-v1` panel.

## Purchase and provenance gate

Before provisioning, record the merged screen revision in the tracking issue
and verify it is the checked-out `main` revision. In the visible Runpod console:

- available balance must be at least $15;
- one secure, non-preemptible A100 80GB must cost at most $1.60/hour;
- configure an eight-hour provider-native stop;
- attach durable storage with at least 20 GiB free.

Use one GPU only. The repository, official dataset, and result root must be on
the machine before starting. Set `L2D_REPO`, `L2D_DATASET`, `L2D_RESULTS`,
`L2D_SCREEN_REVISION`, and the exact UTC provider stop time in
`L2D_PROVIDER_STOP_UTC`. Verify the pinned dataset independently:

```bash
echo "149f6aac17aff2e33750b4e1b6cebd3cef1c39d47ae49a3a7ed77315cb7838a7  $L2D_DATASET" \
  | sha256sum --check

git fetch origin
git checkout main
git pull --ff-only origin main
test "$(git rev-parse HEAD)" = "$L2D_SCREEN_REVISION"
test -z "$(git status --porcelain)"

python -m pip install uv==0.11.8
uv python install 3.12
uv sync --frozen --python 3.12 \
  --group dev --group integration --group accelerator

unset LD_LIBRARY_PATH XLA_FLAGS
unset JAX_COMPILATION_CACHE_DIR JAX_ENABLE_COMPILATION_CACHE
export CUDA_VISIBLE_DEVICES=0
uv run --frozen --python 3.12 \
  --group integration --group accelerator \
  python tools/check_a100_readiness.py --output-root "$L2D_RESULTS"
```

Proceed only if readiness reports exactly one A100 80GB, disabled MIG, idle
memory/utilization within the profile, disabled persistent JAX cache, and enough
disk. Do not record device identifiers in Git.

## Stage 1 — mechanics plan and run

Generate and inspect the exact one-run plan:

```bash
uv run --frozen --python 3.12 \
  --group integration --group accelerator \
  python tools/run_uifo_paired.py \
  --topologies-file experiments/uifo_paired/panels/restart-mechanics-v1.json \
  --official-dataset "$L2D_DATASET" \
  --exclude-prior-panel experiments/uifo_paired/panels/submission-like-v1.json \
  --require-archive-exclusion \
  --optimizer-seeds 11 \
  --arms no_prior_p200 \
  --arm-patience no_prior_p200=200 \
  --optimizer-telemetry member-v1 \
  --population-size 8 \
  --max-time 600 \
  --target-loss 4.0 --target-loss 1.0 \
  --target-loss 0.5 --target-loss 0.0 \
  --worker-timeout 1200 \
  --max-session-wall 1800 \
  --max-worker-failures 1 \
  --provider-stop-utc "$L2D_PROVIDER_STOP_UTC" \
  --provider-evacuation-reserve 1800 \
  --study-profile restart-mechanics-v1 \
  --require-a100 \
  --minimum-gpu-memory-mib 75000 \
  --max-idle-gpu-memory-mib 1000 \
  --max-idle-gpu-utilization 5 \
  --minimum-free-disk-gib 20 \
  --output "$L2D_RESULTS/restart-mechanics-v1" \
  --dry-run > "$L2D_RESULTS/restart-mechanics-v1-plan.json"
```

Assert one run, policy `restart-mechanics-v1`, patience 200, telemetry
`member-v1`, and archive/submission-panel exclusion. Then rerun the identical
command without `--dry-run` and redirection. Package the sole plan directory:

```bash
mapfile -t MECHANICS_DIRS < <(
  find "$L2D_RESULTS/restart-mechanics-v1" -mindepth 1 -maxdepth 1 -type d
)
test "${#MECHANICS_DIRS[@]}" -eq 1
MECHANICS_DIR="${MECHANICS_DIRS[0]}"

uv run --frozen --no-sync --python 3.12 \
  python tools/package_uifo_study.py "$MECHANICS_DIR" \
  --output "$L2D_RESULTS/restart-mechanics-v1.zip"

cd "$L2D_RESULTS"
sha256sum --check restart-mechanics-v1.zip.sha256
```

Continue only when `summary.json -> predeclared_decision` is exactly
`status=passed`, `passed=true`, `action=run_restart_screen_v1`. Do not use or
report the pilot loss. On any other result, evacuate the package and stop. The
screen command authenticates this study and package again and binds their plan
ID, revision, package SHA-256, manifest SHA-256, record SHA-256, history SHA-256,
and telemetry SHA-256 into the new plan.

## Stage 2 — uninstrumented screen

Return to the clean repository and generate the exact 32-run plan:

```bash
cd "$L2D_REPO"
uv run --frozen --python 3.12 \
  --group integration --group accelerator \
  python tools/run_uifo_paired.py \
  --topologies-file experiments/uifo_paired/panels/restart-screen-v1.json \
  --official-dataset "$L2D_DATASET" \
  --exclude-prior-panel experiments/uifo_paired/panels/submission-like-v1.json \
  --require-archive-exclusion \
  --optimizer-seeds 19 23 \
  --arms no_prior_p600 no_prior_p200 \
  --arm-patience no_prior_p600=600 \
  --arm-patience no_prior_p200=200 \
  --pair-order-policy alternate_topology_and_seed \
  --mechanics-study-dir "$MECHANICS_DIR" \
  --mechanics-package "$L2D_RESULTS/restart-mechanics-v1.zip" \
  --population-size 8 \
  --max-time 600 \
  --target-loss 4.0 --target-loss 1.0 \
  --target-loss 0.5 --target-loss 0.0 \
  --worker-timeout 1200 \
  --max-session-wall 23400 \
  --max-worker-failures 1 \
  --provider-stop-utc "$L2D_PROVIDER_STOP_UTC" \
  --provider-evacuation-reserve 1800 \
  --study-profile restart-screen-v1 \
  --require-a100 \
  --minimum-gpu-memory-mib 75000 \
  --max-idle-gpu-memory-mib 1000 \
  --max-idle-gpu-utilization 5 \
  --minimum-free-disk-gib 20 \
  --output "$L2D_RESULTS/restart-screen-v1" \
  --dry-run > "$L2D_RESULTS/restart-screen-v1-plan.json"
```

The runner refuses to build this plan without a complete, passed, byte-matched
mechanics predecessor from the same Git revision. It also refuses to start
unless at least seven hours remain before the provider stop: the 6.5-hour study
cap plus a 30-minute evacuation reserve. Require 32 runs, seeds `[19, 23]`, no
telemetry key, 16 complete planned pairs, 8/8 arm-first balance, and the frozen policy ID
`patience-200-development-screen-v1`. Then rerun without `--dry-run` and
redirection. Do not read interim loss or summary fields. Operational monitoring
may inspect only process status, errors, time remaining, disk, and device health.

After completion, package and verify exactly as above using
`restart-screen-v1.zip`. If the writer stops after a failure, package with
`--allow-incomplete`; an incomplete package cannot support a decision.
Both restart profiles are intentionally non-resumable: a worker failure,
interruption, wall guard, or provider-deadline guard makes that attempt terminal
and preserves its partial evidence.

If a hard interruption leaves `.study.lock`, first prove that its same-host
writer PID is dead, then package the terminal evidence with both
`--allow-incomplete --recover-stale-lock`. This packaging-only path preserves
the old lock under `recovery/` and never restarts a worker. A live, foreign-host,
or malformed lock is refused.

## Evacuation and stop

Download each ZIP, SHA-256 sidecar, manifest, and dry-run plan to the private
results directory outside Git. Verify the sidecar contents against the received
ZIP and run safe ZIP validation locally before terminating the GPU. Stop the Pod
immediately after local verification. Preserve the volume until both packages
are authenticated locally; volume deletion is a later, separate provenance
decision.
