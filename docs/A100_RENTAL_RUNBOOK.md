# A100 development-screen runbook

This is the paid-machine procedure for the frozen `development-v1` study. It
uses one visible NVIDIA A100 80 GB GPU, runs serially, and writes every result to
durable storage. Do not improvise a two-GPU split: the current harness has no
validated shard/merge format.

## Purchase envelope

- one A100 PCIe or SXM with at least 75,000 MiB reported memory;
- an on-demand, non-preemptible instance; do not use spot capacity;
- Python 3.12 on Linux and an NVIDIA driver compatible with CUDA 12;
- at least 20 GiB free on a persistent or network-backed output volume;
- 22 hours maximum session time;
- 96 planned runs: 16 topologies × 2 optimizer seeds × 3 arms;
- 16 scored Objective-hours plus setup, compilation, validation, and recovery
  allowance.

The study process exits before starting a worker that cannot fit inside its
79,200-second session cap, but the rental does not terminate itself. Configure
a provider-native 24-hour safety stop or equivalent spending cap when
available, leaving two hours to evacuate artifacts, and stop the instance
manually as soon as the verified package is off-machine.

## 1. Prepare the machine

Use paths on durable storage for the dataset and results. The repository and
virtual environment may be disposable.

```bash
set -euo pipefail

export L2D_REPO=/workspace/learn2design
export L2D_DATASET=/workspace/data/dataset.h5
export L2D_RESULTS=/workspace/results

mkdir -p "$L2D_RESULTS"
git clone https://github.com/YesterdaysLemon/learn2design.git "$L2D_REPO"
cd "$L2D_REPO"
git fetch origin
git checkout main
git pull --ff-only origin main
test -z "$(git status --porcelain)"

python -m pip install uv==0.11.8
uv python install 3.12
uv sync --frozen --python 3.12 \
  --group dev --group integration --group accelerator
```

The accelerator group pins the CUDA 12 JAX plugin matching the locked JAX and
JAXlib versions. The runtime uses the packaged CUDA libraries; unset an
inherited library override before checking JAX:

```bash
unset LD_LIBRARY_PATH XLA_FLAGS
unset JAX_COMPILATION_CACHE_DIR JAX_ENABLE_COMPILATION_CACHE
export CUDA_VISIBLE_DEVICES=0
uv run --frozen --python 3.12 \
  --group integration --group accelerator \
  python tools/check_a100_readiness.py \
  --output-root "$L2D_RESULTS"
```

Proceed only if the JSON status is `ready`, exactly one A100 is JAX-visible,
MIG mode is disabled, physical GPU memory is at least 75,000 MiB, the device is
idle, and free disk is at least 20 GiB. JAX's allocator limit is not used as a
capacity gate because JAX normally reserves only a fraction of the physical
card.

## 2. Verify the official archive

The archive is not committed to this repository. Stage the official
`dataset.h5` from starter-kit revision
`d9b1bd7d6f2c4df335bc7725755b02aa5f6f942c`, then verify its bytes:

```bash
echo "149f6aac17aff2e33750b4e1b6cebd3cef1c39d47ae49a3a7ed77315cb7838a7  $L2D_DATASET" \
  | sha256sum --check
```

Stop if the digest differs. The full run independently repeats the archive and
panel-exclusion audit.

## 3. Run the deployment ladder

The deployment topology is a mechanics-only seeded topology outside the frozen
development panel. Run population 2, then 4, then 8. Do not advance after an
OOM, worker error, non-idle preflight, or missing admitted history.

```bash
for population in 2 4 8; do
  uv run --frozen --python 3.12 \
    --group integration --group accelerator \
    python tools/run_uifo_paired.py \
    --topology-seeds 2026082999 \
    --optimizer-seeds 7 \
    --arms no_prior \
    --population-size "$population" \
    --max-time 120 \
    --worker-timeout 600 \
    --max-session-wall 900 \
    --require-a100 \
    --minimum-gpu-memory-mib 75000 \
    --max-idle-gpu-memory-mib 1000 \
    --max-idle-gpu-utilization 5 \
    --minimum-free-disk-gib 20 \
    --output "$L2D_RESULTS/smoke-p$population"
done
```

Inspect each `summary.json`, worker log, and `session.json`. Population 8 must
complete through the default full-vmap path. These smoke results are deployment
evidence only, not optimizer-performance evidence.

## 4. Freeze and inspect the paid plan

The target thresholds—4.0, 1.0, 0.5, and 0.0—were selected before live runs
from the official archive's stored-loss range. They are diagnostic hitting-time
thresholds, not claimed leaderboard values.

```bash
uv run --frozen --python 3.12 \
  --group integration --group accelerator \
  python tools/run_uifo_paired.py \
  --topologies-file experiments/uifo_paired/panels/development-v1.json \
  --official-dataset "$L2D_DATASET" \
  --require-archive-exclusion \
  --optimizer-seeds 7 11 \
  --arms adam no_prior semantic_prior \
  --population-size 8 \
  --max-time 600 \
  --target-loss 4.0 --target-loss 1.0 \
  --target-loss 0.5 --target-loss 0.0 \
  --worker-timeout 1200 \
  --max-session-wall 79200 \
  --require-a100 \
  --minimum-gpu-memory-mib 75000 \
  --max-idle-gpu-memory-mib 1000 \
  --max-idle-gpu-utilization 5 \
  --minimum-free-disk-gib 20 \
  --output "$L2D_RESULTS/development-v1" \
  --dry-run > "$L2D_RESULTS/development-v1-plan.json"

python - "$L2D_RESULTS/development-v1-plan.json" <<'PY'
import json, sys
plan = json.load(open(sys.argv[1], encoding="utf-8"))
assert len(plan["runs"]) == 96
assert plan["configuration"]["jax_compilation_cache_policy"] == "disabled"
assert plan["configuration"]["target_losses"] == [4.0, 1.0, 0.5, 0.0]
print(plan["plan_id"], len(plan["runs"]))
PY
```

Persistent JAX compilation caching is deliberately disabled. Compilation occurs
inside the Objective clock, so sharing compiled executables across isolated
workers would give later arms more scored evaluations and bias the comparison.

## 5. Run or resume

Remove only the final `--dry-run` redirection from the command above to start
the study. Keep the terminal attached through `tmux` or another durable session.
The output directory gains a deterministic plan-ID subdirectory.

If the process is interrupted, rerun the exact command with `--resume`. Resume
fails closed if the code revision, device, runtime, plan, dataset, prior, or
cache policy changed. Resume is supported only on the same host and GPU UUID;
do not assume a replacement instance can continue a preempted study. If a hard
kill left `.study.lock`, first confirm no study or GPU worker process is alive,
then add both `--resume --recover-stale-lock`; the old lock is preserved under
`recovery/`.

Do not run unrelated GPU or CPU-heavy jobs on the rented machine during the
study. Do not run two study workers concurrently on the same GPU.

## 6. Validate and evacuate artifacts

After `session.json` reports `complete`, identify the sole plan directory and
create a validated deterministic package:

```bash
mapfile -t STUDY_DIRS < <(
  find "$L2D_RESULTS/development-v1" -mindepth 1 -maxdepth 1 -type d
)
test "${#STUDY_DIRS[@]}" -eq 1
STUDY_DIR="${STUDY_DIRS[0]}"
uv run --frozen --no-sync --python 3.12 \
  python tools/package_uifo_study.py "$STUDY_DIR" \
  --output "$L2D_RESULTS/development-v1.zip"
cd "$L2D_RESULTS"
sha256sum --check development-v1.zip.sha256
```

Download these three files before stopping the rental:

- `development-v1.zip`
- `development-v1.zip.sha256`
- `development-v1.zip.manifest.json`

Verify the checksum again on the local machine. Keep the persistent volume
until the local ZIP opens successfully and its manifest, summaries, histories,
and logs are present. Then terminate the GPU instance; deleting a Pod and
deleting its persistent volume are separate provider actions.
