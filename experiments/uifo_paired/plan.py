"""Deterministic paired-run planning."""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import UTC, datetime

from experiments.uifo_paired.optimizer_settings import settings_with_patience
from experiments.uifo_paired.optimizer_telemetry import OPTIMIZER_TELEMETRY_MODE
from experiments.uifo_paired.study_profiles import bind_study_profile

RESTART_SCREEN_ARMS = ("no_prior_p600", "no_prior_p200")
COVERAGE_SCREEN_ARMS = ("no_prior", "coverage_balanced")
VALID_ARMS = (
    "adam",
    "no_prior",
    "semantic_prior",
    "coverage_balanced",
    *RESTART_SCREEN_ARMS,
)
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
    require_h100: bool = False,
    required_gpu_name: str | None = None,
    preclock_warmup: bool = False,
    minimum_gpu_memory_mib: int | None = None,
    max_idle_gpu_memory_mib: int | None = None,
    max_idle_gpu_utilization_percent: int | None = None,
    minimum_free_disk_gib: float | None = None,
    max_session_wall_seconds: float | None = None,
    max_worker_failures: int = 1,
    study_profile: str | None = None,
    optimizer_telemetry: str | None = None,
    arm_patience: dict[str, int] | None = None,
    pair_order_policy: str = "rotate_pairs",
    seed_order_policy: str = "listed",
    mechanics_evidence: dict[str, object] | None = None,
    candidate_package_evidence: dict[str, object] | None = None,
    provider_stop_utc: str | None = None,
    provider_evacuation_reserve_seconds: float | None = None,
    provider_deadline_maximum_horizon_seconds: float = 8 * 60 * 60,
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
    if require_a100 and require_h100:
        raise ValueError("require_a100 and require_h100 are mutually exclusive")
    if (require_a100 or require_h100) and allow_cpu:
        raise ValueError("required GPU hardware and allow_cpu are mutually exclusive")
    if required_gpu_name is not None and (
        not isinstance(required_gpu_name, str) or not required_gpu_name.strip()
    ):
        raise ValueError("required_gpu_name must be a non-empty string")
    if required_gpu_name is not None and not (require_a100 or require_h100):
        raise ValueError("required_gpu_name requires an exact GPU model")
    if not isinstance(preclock_warmup, bool):
        raise ValueError("preclock_warmup must be a boolean")
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
    if max_worker_failures < 1:
        raise ValueError("max_worker_failures must be positive")
    if optimizer_telemetry not in (None, OPTIMIZER_TELEMETRY_MODE):
        raise ValueError(
            f"optimizer_telemetry must be {OPTIMIZER_TELEMETRY_MODE!r} or None"
        )
    if optimizer_telemetry is not None and "adam" in arms:
        raise ValueError("optimizer telemetry is only supported for batched arms")
    if (
        "coverage_balanced" in arms
        and study_profile != "coverage-robustness-screen-v1"
    ):
        raise ValueError(
            "coverage_balanced is only valid for "
            "coverage-robustness-screen-v1"
        )
    if pair_order_policy not in ("rotate_pairs", "alternate_topology_and_seed"):
        raise ValueError("unknown pair_order_policy")
    if pair_order_policy == "alternate_topology_and_seed" and len(arms) != 2:
        raise ValueError(
            "alternate_topology_and_seed requires exactly two comparison arms"
        )
    if seed_order_policy not in ("listed", "mirrored_sweeps"):
        raise ValueError("unknown seed_order_policy")
    if seed_order_policy == "mirrored_sweeps" and len(optimizer_seeds) != 2:
        raise ValueError("mirrored_sweeps requires exactly two optimizer seeds")
    if mechanics_evidence is not None and study_profile != "restart-screen-v1":
        raise ValueError(
            "mechanics_evidence is only valid for restart-screen-v1"
        )
    if (
        candidate_package_evidence is not None
        and study_profile
        not in {"submission-like-screen-v1", "coverage-robustness-screen-v1"}
    ):
        raise ValueError(
            "candidate_package_evidence is only valid for package-bound profiles"
        )
    if (provider_stop_utc is None) != (
        provider_evacuation_reserve_seconds is None
    ):
        raise ValueError(
            "provider stop time and evacuation reserve must be supplied together"
        )
    normalized_provider_stop = None
    if provider_stop_utc is not None:
        if not isinstance(provider_stop_utc, str):
            raise ValueError("provider_stop_utc must be an ISO-8601 string")
        try:
            parsed_stop = datetime.fromisoformat(provider_stop_utc.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError("provider_stop_utc is not valid ISO-8601") from error
        if parsed_stop.utcoffset() != UTC.utcoffset(None):
            raise ValueError("provider_stop_utc must be expressed in UTC")
        normalized_provider_stop = parsed_stop.astimezone(UTC).isoformat().replace(
            "+00:00", "Z"
        )
        reserve = provider_evacuation_reserve_seconds
        if (
            isinstance(reserve, bool)
            or not isinstance(reserve, (int, float))
            or not math.isfinite(float(reserve))
            or float(reserve) <= 0
        ):
            raise ValueError(
                "provider evacuation reserve must be finite and positive"
            )
        if (
            isinstance(provider_deadline_maximum_horizon_seconds, bool)
            or not isinstance(
                provider_deadline_maximum_horizon_seconds, (int, float)
            )
            or not math.isfinite(
                float(provider_deadline_maximum_horizon_seconds)
            )
            or float(provider_deadline_maximum_horizon_seconds) <= 0
        ):
            raise ValueError(
                "provider deadline maximum horizon must be finite and positive"
            )
    restart_arms = set(arms) & set(RESTART_SCREEN_ARMS)
    if restart_arms:
        if restart_arms != set(arms):
            raise ValueError("restart-screen arms cannot be mixed with other arms")
        if not isinstance(arm_patience, dict) or set(arm_patience) != set(arms):
            raise ValueError("restart-screen arms require exact per-arm patience")
        arm_optimizer_settings = {
            arm: settings_with_patience(arm_patience[arm]) for arm in arms
        }
    else:
        if arm_patience is not None:
            raise ValueError("arm_patience is only valid for restart-screen arms")
        arm_optimizer_settings = None
    if "coverage_balanced" in arms and set(arms) != set(COVERAGE_SCREEN_ARMS):
        raise ValueError(
            "coverage_balanced must be paired only with the no_prior control"
        )
    if not (require_a100 or require_h100) and any(
        value is not None
        for value in (
            minimum_gpu_memory_mib,
            max_idle_gpu_memory_mib,
            max_idle_gpu_utilization_percent,
        )
    ):
        raise ValueError("GPU rental constraints require an exact GPU model")
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
    if seed_order_policy == "mirrored_sweeps":
        plan_blocks = []
        for seed_index, optimizer_seed in enumerate(optimizer_seeds):
            indexed_topologies = list(enumerate(topology_specs))
            if seed_index:
                indexed_topologies.reverse()
            plan_blocks.extend(
                (topology_index, topology_spec, seed_index, optimizer_seed)
                for topology_index, topology_spec in indexed_topologies
            )
    else:
        plan_blocks = [
            (topology_index, topology_spec, seed_index, optimizer_seed)
            for topology_index, topology_spec in enumerate(topology_specs)
            for seed_index, optimizer_seed in enumerate(optimizer_seeds)
        ]
    for topology_index, topology_spec, seed_index, optimizer_seed in plan_blocks:
        offset = (
            pair_index % len(arms)
            if pair_order_policy == "rotate_pairs"
            else (topology_index + seed_index) % len(arms)
        )
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

    configuration = {
        "allow_cpu": bool(allow_cpu),
        "arms": arms,
        "evaluation_chunk_size": evaluation_chunk_size,
        "max_evals": max_evals,
        "max_idle_gpu_memory_mib": max_idle_gpu_memory_mib,
        "max_idle_gpu_utilization_percent": max_idle_gpu_utilization_percent,
        "max_time_seconds": max_time_seconds,
        "max_session_wall_seconds": max_session_wall_seconds,
        "max_worker_failures": int(max_worker_failures),
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
        "study_profile": study_profile,
    }
    if require_h100:
        configuration["require_h100"] = True
    if required_gpu_name is not None:
        configuration["required_gpu_name"] = required_gpu_name.strip()
    if preclock_warmup:
        configuration["preclock_warmup"] = True
    if optimizer_telemetry is not None:
        configuration["optimizer_telemetry"] = optimizer_telemetry
    if arm_optimizer_settings is not None:
        configuration["arm_optimizer_settings"] = arm_optimizer_settings
    if pair_order_policy != "rotate_pairs":
        configuration["pair_order_policy"] = pair_order_policy
    if seed_order_policy != "listed":
        configuration["seed_order_policy"] = seed_order_policy
    if mechanics_evidence is not None:
        configuration["mechanics_evidence"] = dict(mechanics_evidence)
    if candidate_package_evidence is not None:
        configuration["candidate_package_evidence"] = dict(
            candidate_package_evidence
        )
    if normalized_provider_stop is not None:
        configuration["provider_stop_utc"] = normalized_provider_stop
        configuration["provider_evacuation_reserve_seconds"] = float(
            provider_evacuation_reserve_seconds
        )
        configuration["provider_deadline_maximum_horizon_seconds"] = float(
            provider_deadline_maximum_horizon_seconds
        )
    if study_profile == "submission-like-screen-v1":
        configuration["execution_mode"] = "serial"
        configuration["resource_budget"] = {
            "currency": "USD",
            "gpu_count": 1,
            "maximum_gpu_hourly_price": 1.60,
            "maximum_provider_charge": 16.00,
            "maximum_provider_hours": 10.0,
            "planned_runs": 20,
            "scored_objective_seconds": 24_000,
        }
    elif study_profile == "coverage-robustness-screen-v1":
        configuration["execution_mode"] = "serial"
        configuration["resource_budget"] = {
            "cloud_type": "SECURE",
            "currency": "USD",
            "gpu_count": 1,
            "gpu_type_id": "NVIDIA H100 80GB HBM3",
            "maximum_gpu_hourly_price": 3.29,
            "maximum_provider_charge": 75.00,
            "maximum_provider_hours": 22.0,
            "planned_runs": 48,
            "scored_objective_seconds": 57_600,
        }
    configuration["decision_policy"] = bind_study_profile(study_profile, configuration)
    primary_order = primary_pair_order_counts(runs)
    if study_profile and len(arms) > 1:
        expected_pairs = len(topology_specs) * len(optimizer_seeds)
        if (
            primary_order["complete_primary_pairs"] != expected_pairs
            or primary_order["absolute_imbalance"] != 0
        ):
            raise ValueError(
                f"study profile {study_profile!r} has an incomplete or imbalanced "
                "primary arm order"
            )
    core = {
        "configuration": configuration,
        "run_order_policy": (
            "rotate arms once per topology-seed pair"
            if pair_order_policy == "rotate_pairs"
            else "alternate arm order by topology and optimizer-seed index"
        ),
        "primary_pair_order": primary_order,
        "runs": runs,
    }
    if seed_order_policy != "listed":
        core["optimizer_seed_order_policy"] = seed_order_policy
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


def primary_pair_order_counts(runs: list[dict[str, object]]) -> dict[str, int]:
    """Count the pairwise order of the causal arms, ignoring optional arms."""
    run_arms = {str(run["arm"]) for run in runs}
    if "coverage_balanced" in run_arms:
        comparison_arms = COVERAGE_SCREEN_ARMS
        first_key = "no_prior_first"
        second_key = "coverage_balanced_first"
    elif set(RESTART_SCREEN_ARMS) <= run_arms:
        comparison_arms = RESTART_SCREEN_ARMS
        first_key = "no_prior_p600_first"
        second_key = "no_prior_p200_first"
    else:
        comparison_arms = ("no_prior", "semantic_prior")
        first_key = "no_prior_first"
        second_key = "semantic_prior_first"
    by_pair: dict[str, list[dict[str, object]]] = {}
    for run in runs:
        arm = str(run["arm"])
        if arm not in set(comparison_arms):
            continue
        by_pair.setdefault(str(run["pair_id"]), []).append(run)

    control_first = 0
    treatment_first = 0
    complete_pairs = 0
    for pair_runs in by_pair.values():
        positions = {
            str(run["arm"]): int(run["run_order_within_pair"])
            for run in pair_runs
        }
        if set(positions) != set(comparison_arms):
            continue
        complete_pairs += 1
        if positions[comparison_arms[0]] < positions[comparison_arms[1]]:
            control_first += 1
        else:
            treatment_first += 1
    return {
        "complete_primary_pairs": complete_pairs,
        first_key: control_first,
        second_key: treatment_first,
        "absolute_imbalance": abs(control_first - treatment_first),
    }
