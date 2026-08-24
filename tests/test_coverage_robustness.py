from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.uifo_paired.coverage_analysis import summarize_coverage_records
from experiments.uifo_paired.plan import build_plan
from experiments.uifo_paired.runner import (
    _expected_algorithm_record,
    _initial_population_roles,
    _rental_preflight,
    _run_config,
    _validate_coverage_initial_population,
    _validate_required_cuda13,
    _validate_required_h100,
)


ROOT = Path(__file__).parents[1]
PANEL_PATH = (
    ROOT
    / "experiments"
    / "uifo_paired"
    / "panels"
    / "coverage-robustness-v1.json"
)
PANEL_SHA256 = "e3385a6f4939445e869d71dbf7f1bd5119aa25eebbb89e083b1ec9be336f7309"


def _candidate_evidence() -> dict[str, object]:
    return {
        "format_version": 1,
        "archive_name": "coverage-candidate.zip",
        "archive_sha256": "a" * 64,
        "builder_manifest_name": "coverage-candidate.manifest.json",
        "builder_manifest_sha256": "b" * 64,
        "project_revision": "c" * 40,
        "source_files": [
            {
                "path": "submission.py",
                "sha256": "d" * 64,
                "size_bytes": 1,
            },
            {
                "path": "requirements.txt",
                "sha256": "e" * 64,
                "size_bytes": 1,
            },
        ],
        "upstream_reference": "1bb7f54737dec6a08b59879a8831d125f08f8a0b",
    }


def _plan() -> dict[str, object]:
    panel = json.loads(PANEL_PATH.read_text(encoding="utf-8"))
    return build_plan(
        topology_seeds=None,
        topologies=panel["topologies"],
        optimizer_seeds=[37, 41],
        arms=["no_prior", "coverage_balanced"],
        max_time_seconds=1_200.0,
        max_evals=None,
        population_size=8,
        n_frequencies=50,
        target_losses=[4.0, 1.0, 0.5, 0.0],
        allow_cpu=False,
        worker_timeout_seconds=2_100.0,
        topology_panel={
            "panel_id": panel["panel_id"],
            "topology_count": len(panel["topologies"]),
            "source_sha256": PANEL_SHA256,
            "archive_exclusion_verified": True,
        },
        evaluation_chunk_size=None,
        require_h100=True,
        required_gpu_name="NVIDIA H100 80GB HBM3",
        preclock_warmup=True,
        minimum_gpu_memory_mib=75_000,
        max_idle_gpu_memory_mib=1_000,
        max_idle_gpu_utilization_percent=5,
        minimum_free_disk_gib=20.0,
        max_session_wall_seconds=22 * 60 * 60,
        max_worker_failures=1,
        study_profile="coverage-robustness-screen-v1",
        optimizer_telemetry=None,
        pair_order_policy="alternate_topology_and_seed",
        seed_order_policy="mirrored_sweeps",
        candidate_package_evidence=_candidate_evidence(),
        provider_stop_utc="2026-08-25T12:00:00Z",
        provider_evacuation_reserve_seconds=1_800.0,
        provider_deadline_maximum_horizon_seconds=26 * 60 * 60.0,
    )


def _expected_configs(plan: dict[str, object]) -> dict[str, dict[str, object]]:
    common = plan["configuration"]
    assert isinstance(common, dict)
    return {
        str(run["run_id"]): _run_config(run, common)
        for run in plan["runs"]
    }


def _records(
    plan: dict[str, object], *, winning_topologies: int = 9
) -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    expected = _expected_configs(plan)
    topology_order = {
        json.dumps(topology, sort_keys=True): index
        for index, topology in enumerate(plan["configuration"]["topologies"])
    }
    pair_problems = {}
    records = []
    for run in plan["runs"]:
        config = expected[str(run["run_id"])]
        topology_key = json.dumps(config["topology"], sort_keys=True)
        topology_index = topology_order[topology_key]
        difference = -0.10 if topology_index < winning_topologies else 0.02
        control_loss = 1.0 + topology_index / 100.0
        loss = (
            control_loss
            if config["arm"] == "no_prior"
            else control_loss + difference
        )
        problem = pair_problems.setdefault(
            str(config["pair_id"]),
            {
                "topology_sha256": hashlib.sha256(topology_key.encode()).hexdigest(),
                "topology_string": str(config["topology"]["value"]),
                "n_params": 3,
                "spec": {"test": True},
            },
        )
        anchor_hash = hashlib.sha256(
            f"anchor:{config['pair_id']}".encode()
        ).hexdigest()
        suffix_hashes = [
            hashlib.sha256(
                f"{config['arm']}:{config['pair_id']}:{index}".encode()
            ).hexdigest()
            for index in range(1, 8)
        ]
        records.append(
            {
                "format_version": 1,
                "run_id": config["run_id"],
                "status": "complete",
                "config": config,
                "problem": problem,
                "algorithm": _expected_algorithm_record(config),
                "initial_population_roles": _initial_population_roles(
                    str(config["arm"]), 8
                ),
                "initial_parameter_hashes": [anchor_hash, *suffix_hashes],
                "raw_suffix_parameter_hashes": [
                    hashlib.sha256(
                        f"raw:{config['pair_id']}:{index}".encode()
                    ).hexdigest()
                    for index in range(1, 8)
                ],
                "metrics": {
                    "has_feasible": True,
                    "has_finite_feasible": True,
                    "best_feasible_loss": loss,
                },
                "objective_accounting": {
                    "eval_count": 10_000
                    if config["arm"] == "no_prior"
                    else 9_600
                },
            }
        )
    return records, expected


def test_frozen_coverage_plan_has_exact_pairing_budget_and_h100_contract() -> None:
    plan = _plan()
    configuration = plan["configuration"]

    assert len(plan["runs"]) == 48
    assert plan["primary_pair_order"] == {
        "complete_primary_pairs": 24,
        "no_prior_first": 12,
        "coverage_balanced_first": 12,
        "absolute_imbalance": 0,
    }
    assert configuration["require_a100"] is False
    assert configuration["require_h100"] is True
    assert configuration["required_gpu_name"] == "NVIDIA H100 80GB HBM3"
    assert configuration["preclock_warmup"] is True
    assert configuration["resource_budget"] == {
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
    assert configuration["topology_panel"]["source_sha256"] == PANEL_SHA256
    assert configuration["decision_policy"]["minimum_coverage_topology_wins"] == 9

    configs = _expected_configs(plan)
    assert {
        config["initial_population_mode"] for config in configs.values()
    } == {"random", "coverage_balanced"}
    assert all(config["preclock_warmup"] is True for config in configs.values())


def test_coverage_arm_cannot_run_outside_the_frozen_profile() -> None:
    with pytest.raises(ValueError, match="coverage_balanced is only valid"):
        build_plan(
            topology_seeds=[2026082401],
            topologies=None,
            optimizer_seeds=[7],
            arms=["no_prior", "coverage_balanced"],
            max_time_seconds=10.0,
            max_evals=None,
            population_size=8,
            n_frequencies=50,
        )


def test_coverage_panel_is_disjoint_from_every_prior_named_panel() -> None:
    coverage = set(json.loads(PANEL_PATH.read_text(encoding="utf-8"))["topologies"])
    prior_names = (
        "development-v1.json",
        "confirmation-v1.json",
        "submission-like-v1.json",
        "restart-mechanics-v1.json",
        "restart-screen-v1.json",
    )
    for name in prior_names:
        prior = set(
            json.loads(PANEL_PATH.with_name(name).read_text(encoding="utf-8"))[
                "topologies"
            ]
        )
        assert coverage.isdisjoint(prior), name


def test_coverage_profile_rejects_any_budget_or_hardware_drift() -> None:
    common = {
        "topology_seeds": None,
        "topologies": json.loads(PANEL_PATH.read_text(encoding="utf-8"))[
            "topologies"
        ],
        "optimizer_seeds": [37, 41],
        "arms": ["no_prior", "coverage_balanced"],
        "max_time_seconds": 1_200.0,
        "max_evals": None,
        "population_size": 8,
        "n_frequencies": 50,
        "target_losses": [4.0, 1.0, 0.5, 0.0],
        "worker_timeout_seconds": 2_100.0,
        "topology_panel": {
            "panel_id": "coverage-robustness-v1",
            "topology_count": 12,
            "source_sha256": PANEL_SHA256,
            "archive_exclusion_verified": True,
        },
        "require_h100": True,
        "required_gpu_name": "NVIDIA H100 80GB HBM3",
        "preclock_warmup": True,
        "minimum_gpu_memory_mib": 75_000,
        "max_idle_gpu_memory_mib": 1_000,
        "max_idle_gpu_utilization_percent": 5,
        "minimum_free_disk_gib": 20.0,
        "max_session_wall_seconds": 22 * 60 * 60,
        "study_profile": "coverage-robustness-screen-v1",
        "pair_order_policy": "alternate_topology_and_seed",
        "seed_order_policy": "mirrored_sweeps",
        "candidate_package_evidence": _candidate_evidence(),
        "provider_stop_utc": "2026-08-25T12:00:00Z",
        "provider_evacuation_reserve_seconds": 1_800.0,
        "provider_deadline_maximum_horizon_seconds": 26 * 60 * 60.0,
    }
    with pytest.raises(ValueError, match="max_time_seconds"):
        build_plan(**{**common, "max_time_seconds": 1_199.0})
    with pytest.raises(ValueError, match="preclock_warmup"):
        build_plan(**{**common, "preclock_warmup": False})
    with pytest.raises(ValueError, match="exact GPU model"):
        build_plan(**{**common, "require_h100": False})
    with pytest.raises(ValueError, match="mutually exclusive"):
        build_plan(**{**common, "require_a100": True})


def test_h100_preflight_checks_exactly_one_idle_full_memory_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = {
        "status": "ok",
        "gpus": [
            {
                "index": 0,
                "uuid": "GPU-h100",
                "name": "NVIDIA H100 80GB HBM3",
                "driver_version": "580.00",
                "memory_total_mib": 81_559,
                "mig_mode_current": "Disabled",
                "memory_used_mib": 12,
                "utilization_percent": 0,
            }
        ],
    }
    monkeypatch.setattr(
        "experiments.uifo_paired.runner._nvidia_smi_snapshot",
        lambda *, include_dynamic: snapshot,
    )
    configuration = {
        "require_h100": True,
        "required_gpu_name": "NVIDIA H100 80GB HBM3",
        "minimum_gpu_memory_mib": 75_000,
        "max_idle_gpu_memory_mib": 1_000,
        "max_idle_gpu_utilization_percent": 5,
        "minimum_free_disk_gib": None,
    }

    assert _rental_preflight(tmp_path, configuration)["gpu_idle"] == snapshot
    _validate_required_h100(
        {
            "backend": "gpu",
            "device_count": 1,
            "competition_aligned_h100": True,
            "device_kinds": ["NVIDIA H100 80GB HBM3"],
        },
        "NVIDIA H100 80GB HBM3",
    )
    valid_cuda13 = {
        "python": "3.12.11",
        "versions": {
            "jax": "0.9.0.1",
            "jaxlib": "0.9.0.1",
            "jax-cuda13-pjrt": "0.9.0.1",
            "jax-cuda13-plugin": "0.9.0.1",
            "jax-cuda12-pjrt": "not-installed",
            "jax-cuda12-plugin": "not-installed",
            "nvidia-cuda-runtime": "13.0.96",
            "nvidia-cuda-runtime-cu12": "not-installed",
        },
        "jax_platform_versions": ["cuda 13000"],
    }
    _validate_required_cuda13(valid_cuda13)
    wrong_python = copy.deepcopy(valid_cuda13)
    wrong_python["python"] = "3.13.5"
    with pytest.raises(RuntimeError, match="Python 3.12"):
        _validate_required_cuda13(wrong_python)
    cuda12_installed = copy.deepcopy(valid_cuda13)
    cuda12_installed["versions"]["jax-cuda12-plugin"] = "0.9.0.1"
    with pytest.raises(RuntimeError, match="rejects installed jax-cuda12-plugin"):
        _validate_required_cuda13(cuda12_installed)

    snapshot["gpus"][0]["name"] = "NVIDIA A100 80GB PCIe"
    with pytest.raises(RuntimeError, match="H100"):
        _rental_preflight(tmp_path, configuration)
    with pytest.raises(RuntimeError, match="exactly one"):
        _validate_required_h100(
            {
                "backend": "gpu",
                "device_count": 2,
                "competition_aligned_h100": True,
                "device_kinds": [
                    "NVIDIA H100 80GB HBM3",
                    "NVIDIA H100 80GB HBM3",
                ],
            }
        )
    with pytest.raises(RuntimeError, match="CUDA 13 backend"):
        _validate_required_cuda13(
            {
                "python": "3.12.11",
                "versions": {
                    "jax": "0.9.0.1",
                    "jaxlib": "0.9.0.1",
                    "jax-cuda13-pjrt": "0.9.0.1",
                    "jax-cuda13-plugin": "0.9.0.1",
                    "jax-cuda12-pjrt": "not-installed",
                    "jax-cuda12-plugin": "not-installed",
                    "nvidia-cuda-runtime": "13.0.96",
                    "nvidia-cuda-runtime-cu12": "not-installed",
                },
                "jax_platform_versions": ["cuda 12090"],
            }
        )


def test_runner_validates_midpoint_latin_hypercube_initial_evidence() -> None:
    levels = (np.arange(7, dtype=np.float32) + 0.5) / 7
    unit_suffix = np.stack(
        [np.roll(levels, shift) for shift in range(3)], axis=1
    )
    suffix = np.log(unit_suffix) - np.log1p(-unit_suffix)
    population = np.vstack([np.zeros((1, 3), dtype=np.float32), suffix])

    _validate_coverage_initial_population(population)
    corrupted = population.copy()
    corrupted[1, 0] = corrupted[2, 0]
    with pytest.raises(RuntimeError, match="Latin-hypercube"):
        _validate_coverage_initial_population(corrupted)
    perturbed = population.copy()
    perturbed[1, 0] += 2e-6
    with pytest.raises(RuntimeError, match="Latin-hypercube"):
        _validate_coverage_initial_population(perturbed)


def test_frozen_coverage_decision_passes_only_the_full_strong_synthetic_panel() -> None:
    plan = _plan()
    records, expected = _records(plan, winning_topologies=9)

    summary = summarize_coverage_records(records, expected)

    assert summary["completed_runs"] == 48
    assert summary["complete_optimizer_seed_pairs"] == 24
    assert summary["complete_topologies"] == 12
    assert summary["wins_ties_losses"] == {
        "coverage_balanced_wins": 9,
        "ties": 0,
        "coverage_balanced_losses": 3,
    }
    assert summary["optimizer_seed_mean_differences"] == pytest.approx(
        {"37": -0.07, "41": -0.07}
    )
    assert summary["arm_order_mean_differences"] == pytest.approx(
        {"coverage_first": -0.07, "random_first": -0.07}
    )
    assert summary[
        "overall_median_evaluation_ratio_coverage_over_random"
    ] == pytest.approx(0.96)
    assert summary["predeclared_decision"]["status"] == "passed"
    assert summary["predeclared_decision"]["action"] == (
        "freeze_official_budget_coverage_confirmation"
    )
    assert all(summary["predeclared_decision"]["criteria"].values())

    weak_records, weak_expected = _records(plan, winning_topologies=8)
    weak = summarize_coverage_records(weak_records, weak_expected)
    assert weak["predeclared_decision"]["status"] == "failed"
    assert weak["predeclared_decision"]["action"] == "retain_random_start_candidate"
    assert not weak["predeclared_decision"]["criteria"][
        "minimum_topology_wins_met"
    ]


def test_coverage_summary_rejects_pair_drift_and_marks_execution_errors_terminal() -> None:
    plan = _plan()
    records, expected = _records(plan)
    drifted = copy.deepcopy(records)
    treatment = next(
        record for record in drifted if record["config"]["arm"] == "coverage_balanced"
    )
    treatment["initial_parameter_hashes"][0] = "f" * 64
    with pytest.raises(ValueError, match="anchor hash differs"):
        summarize_coverage_records(drifted, expected)

    raw_drifted = copy.deepcopy(records)
    raw_treatment = next(
        record
        for record in raw_drifted
        if record["config"]["arm"] == "coverage_balanced"
    )
    raw_treatment["raw_suffix_parameter_hashes"][1] = "e" * 64
    with pytest.raises(ValueError, match="pre-transform random draw differs"):
        summarize_coverage_records(raw_drifted, expected)

    errored = copy.deepcopy(records)
    errored[0]["status"] = "error"
    summary = summarize_coverage_records(errored, expected)
    assert summary["predeclared_decision"] == {
        "status": "not_evaluable",
        "passed": False,
        "action": "retain_candidate_attempt_not_evaluable",
        "criteria": summary["predeclared_decision"]["criteria"],
    }
