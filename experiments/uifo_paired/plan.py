"""Deterministic paired-run planning."""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import UTC, datetime

VALID_ARMS = ("adam", "no_prior", "semantic_prior")
TOPOLOGY_PATTERN = re.compile(r"^[A-H]{9}-[LSDH]{12}$")


def build_plan(
    *,
    topology_seeds: list[int] | None,
    topologies: list[str] | None,
    optimizer_seeds: list[int],
    arms: list[str],
    max_time_seconds: float | None,
    max_evals: int | None,
    population_size: int,
    n_frequencies: int,
    target_losses: list[float] | None = None,
    allow_cpu: bool = False,
    worker_timeout_seconds: float | None = None,
    topology_panel: dict[str, object] | None = None,
    evaluation_chunk_size: int | None = None,
    require_a100: bool = False,
    minimum_gpu_memory_mib: int | None = None,
    max_idle_gpu_memory_mib: int | None = None,
    max_idle_gpu_utilization_percent: int | None = None,
    minimum_free_disk_gib: float | None = None,
    max_session_wall_seconds: float | None = None,
) -> dict[str, object]:
    """Build a stable plan with AB/BA-style arm-order rotation."""
    if bool(topology_seeds) == bool(topologies):
        raise ValueError("provide exactly one of topology_seeds or topologies")
    if not optimizer_seeds:
        raise ValueError("at least one optimizer seed is required")
    if any(seed < 0 for seed in optimizer_seeds):
        raise ValueError("optimizer seeds must be non-negative")
    if len(set(optimizer_seeds)) != len(optimizer_seeds):
        raise ValueError("optimizer seeds must be unique")
    if not arms or len(set(arms)) != len(arms):
        raise ValueError("arms must be a non-empty unique list")
    unknown = sorted(set(arms) - set(VALID_ARMS))
    if unknown:
        raise ValueError(f"unknown arms: {unknown}")
    if max_time_seconds is None and max_evals is None:
        raise ValueError("at least one budget is required")
    if max_time_seconds is not None and (
        not math.isfinite(max_time_seconds) or max_time_seconds <= 0
    ):
        raise ValueError("max_time_seconds must be finite and positive")
    if max_evals is not None and max_evals <= 0:
        raise ValueError("max_evals must be positive")
    if population_size < 2:
        raise ValueError("population_size must be at least two")
    if evaluation_chunk_size is not None and not (
        1 <= evaluation_chunk_size <= population_size
    ):
        raise ValueError(
            "evaluation_chunk_size must be between one and population_size"
        )
    if n_frequencies < 1:
        raise ValueError("n_frequencies must be positive")
    if require_a100 and allow_cpu:
        raise ValueError("require_a100 and allow_cpu are mutually exclusive")
    if minimum_gpu_memory_mib is not None and minimum_gpu_memory_mib <= 0:
        raise ValueError("minimum_gpu_memory_mib must be positive")
    if max_idle_gpu_memory_mib is not None and max_idle_gpu_memory_mib < 0:
        raise ValueError("max_idle_gpu_memory_mib must be non-negative")
    if max_idle_gpu_utilization_percent is not None and not (
        0 <= max_idle_gpu_utilization_percent <= 100
    ):
        raise ValueError(
            "max_idle_gpu_utilization_percent must be between zero and 100"
        )
    if minimum_free_disk_gib is not None and (
        not math.isfinite(minimum_free_disk_gib) or minimum_free_disk_gib <= 0
    ):
        raise ValueError("minimum_free_disk_gib must be finite and positive")
    if max_session_wall_seconds is not None and (
        not math.isfinite(max_session_wall_seconds)
        or max_session_wall_seconds <= 0
    ):
        raise ValueError("max_session_wall_seconds must be finite and positive")
    if not require_a100 and any(
        value is not None
        for value in (
            minimum_gpu_memory_mib,
            max_idle_gpu_memory_mib,
            max_idle_gpu_utilization_percent,
        )
    ):
        raise ValueError("GPU rental constraints require require_a100")
    targets = [float(target) for target in (target_losses or [])]
    if any(not math.isfinite(target) for target in targets):
        raise ValueError("target losses must be finite")
    if worker_timeout_seconds is None:
        if max_time_seconds is None:
            raise ValueError("worker_timeout_seconds is required without max_time")
        worker_timeout_seconds = max_time_seconds + 30 * 60
    if not math.isfinite(worker_timeout_seconds) or worker_timeout_seconds <= 0:
        raise ValueError("worker_timeout_seconds must be finite and positive")
    if max_time_seconds is not None and worker_timeout_seconds <= max_time_seconds:
        raise ValueError("worker timeout must exceed the Objective time budget")
    if max_evals is not None and set(arms) != {"adam"}:
        if max_evals % population_size:
            raise ValueError("max_evals must be divisible by population_size")

    topology_specs = []
    if topology_seeds:
        if any(seed < 0 for seed in topology_seeds):
            raise ValueError("topology seeds must be non-negative")
        if len(set(topology_seeds)) != len(topology_seeds):
            raise ValueError("topology seeds must be unique")
        topology_specs = [
            {"kind": "seed", "value": int(seed)} for seed in topology_seeds
        ]
    else:
        assert topologies is not None
        if len(set(topologies)) != len(topologies):
            raise ValueError("topology strings must be unique")
        for topology in topologies:
            if not TOPOLOGY_PATTERN.fullmatch(topology):
                raise ValueError(f"invalid size-3 topology string: {topology!r}")
            if sum(token in "DH" for token in topology.split("-", 1)[1]) != 1:
                raise ValueError(
                    "explicit size-3 topology must contain exactly one D/H readout"
                )
            topology_specs.append({"kind": "string", "value": topology})

    runs = []
    pair_index = 0
    for topology_spec in topology_specs:
        for optimizer_seed in optimizer_seeds:
            offset = pair_index % len(arms)
            ordered_arms = arms[offset:] + arms[:offset]
            pair_id = _pair_id(topology_spec, optimizer_seed)
            for order, arm in enumerate(ordered_arms):
                runs.append(
                    {
                        "planned_run_index": len(runs),
                        "run_id": f"{pair_id}__{arm}",
                        "pair_id": pair_id,
                        "run_order_within_pair": order,
                        "topology": topology_spec,
                        "optimizer_seed": int(optimizer_seed),
                        "arm": arm,
                    }
                )
            pair_index += 1

    core = {
        "configuration": {
            "allow_cpu": bool(allow_cpu),
            "arms": arms,
            "evaluation_chunk_size": evaluation_chunk_size,
            "max_evals": max_evals,
            "max_idle_gpu_memory_mib": max_idle_gpu_memory_mib,
            "max_idle_gpu_utilization_percent": max_idle_gpu_utilization_percent,
            "max_time_seconds": max_time_seconds,
            "max_session_wall_seconds": max_session_wall_seconds,
            "minimum_free_disk_gib": minimum_free_disk_gib,
            "minimum_gpu_memory_mib": minimum_gpu_memory_mib,
            "n_frequencies": int(n_frequencies),
            "optimizer_seeds": [int(seed) for seed in optimizer_seeds],
            "population_size": int(population_size),
            "require_a100": bool(require_a100),
            "jax_compilation_cache_policy": "disabled",
            "target_losses": sorted(set(targets), reverse=True),
            "topologies": topology_specs,
            "topology_panel": topology_panel,
            "worker_timeout_seconds": float(worker_timeout_seconds),
        },
        "run_order_policy": "rotate arms once per topology-seed pair",
        "runs": runs,
    }
    plan_id = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    return {
        "format_version": 1,
        "plan_id": plan_id,
        "created_utc": datetime.now(UTC).isoformat(),
        **core,
    }


def _pair_id(topology_spec: dict[str, object], optimizer_seed: int) -> str:
    if topology_spec["kind"] == "seed":
        topology_id = f"tseed{int(topology_spec['value']):010d}"
    else:
        digest = hashlib.sha256(str(topology_spec["value"]).encode()).hexdigest()[:12]
        topology_id = f"topo{digest}"
    return f"{topology_id}__oseed{int(optimizer_seed):010d}"
