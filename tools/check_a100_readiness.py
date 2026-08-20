"""Fail-fast machine check before starting a paid A100 study."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from experiments.uifo_paired.runner import (
    _rental_preflight,
    _validate_cache_disabled_runtime,
    _validate_required_a100,
    cache_disabled_jax_environment,
    environment_fingerprint,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--minimum-gpu-memory-mib", type=int, default=75_000)
    parser.add_argument("--max-idle-gpu-memory-mib", type=int, default=1_000)
    parser.add_argument("--max-idle-gpu-utilization", type=int, default=5)
    parser.add_argument("--minimum-free-disk-gib", type=float, default=20)
    args = parser.parse_args()
    if not args.output_root.is_dir():
        parser.error("--output-root must already exist on durable storage")

    configuration = {
        "require_a100": True,
        "minimum_gpu_memory_mib": args.minimum_gpu_memory_mib,
        "max_idle_gpu_memory_mib": args.max_idle_gpu_memory_mib,
        "max_idle_gpu_utilization_percent": args.max_idle_gpu_utilization,
        "minimum_free_disk_gib": args.minimum_free_disk_gib,
    }
    rental = _rental_preflight(args.output_root, configuration)
    child_environment = cache_disabled_jax_environment()
    os.environ.clear()
    os.environ.update(child_environment)
    runtime = environment_fingerprint()
    _validate_cache_disabled_runtime(runtime)
    _validate_required_a100(runtime)
    print(
        json.dumps(
            {
                "status": "ready",
                "rental_preflight": rental,
                "runtime_environment": runtime,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
