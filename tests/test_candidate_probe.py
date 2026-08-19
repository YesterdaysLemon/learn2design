from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from experiments.uifo_paired.candidate_probe import (
    OBJECTIVE_SAVE_FIELDS,
    assess_idle_samples,
    build_probe_plan,
    jax_environment_policy,
    parse_compute_processes,
    parse_gpu_sample,
    summarize_probe_records,
    validate_candidate_identity_record,
    validate_completed_probe_record,
    validate_probe_provenance,
)
from experiments.uifo_paired.runner import sha256


TOPOLOGY = "HBHCBBCBG-LDSLSLLSLLSL"


def _plan(**overrides):
    kwargs = {
        "topology": {"kind": "explicit", "value": TOPOLOGY},
        "optimizer_seed": 7,
        "population_size": 2,
        "roles": ["anchor", "random_member_1", "semantic_prior"],
        "repeats": 2,
        "n_frequencies": 50,
        "worker_timeout_seconds": 180,
        "gpu_index": 0,
        "idle_samples": 3,
        "idle_sample_interval_seconds": 0.5,
        "max_idle_memory_mib": 2000,
        "max_idle_utilization_percent": 5,
        "monitor_interval_seconds": 0.5,
        "cooldown_timeout_seconds": 10,
    }
    kwargs.update(overrides)
    return build_probe_plan(**kwargs)


def test_probe_plan_is_exactly_forward_then_reverse() -> None:
    plan = _plan()

    assert [run["role"] for run in plan["runs"]] == [
        "anchor",
        "random_member_1",
        "semantic_prior",
        "semantic_prior",
        "random_member_1",
        "anchor",
    ]
    assert len({run["run_id"] for run in plan["runs"]}) == 6
    assert plan["configuration"]["topology"]["value"] == TOPOLOGY
    assert plan["configuration"]["topology_sha256"] == hashlib.sha256(
        TOPOLOGY.encode()
    ).hexdigest()


def test_probe_plan_rejects_scope_and_timeout_drift() -> None:
    with pytest.raises(ValueError, match="population_size=2"):
        _plan(population_size=3)
    with pytest.raises(ValueError, match="timeout"):
        _plan(worker_timeout_seconds=301)
    with pytest.raises(ValueError, match="exactly two"):
        _plan(repeats=3)
    with pytest.raises(ValueError, match="candidate roles"):
        _plan(roles=["anchor", "semantic_prior"])
    with pytest.raises(ValueError, match="explicit topology"):
        _plan(topology={"kind": "seed", "value": 2026081908})


def test_gpu_telemetry_parsers_are_strict() -> None:
    sample = parse_gpu_sample(
        "0, GPU-abc, NVIDIA GeForce RTX 4060, 1412, 8188, 3\n"
    )
    processes = parse_compute_processes(
        "123, python, 512\n456, python, [N/A]\n"
    )

    assert sample == {
        "gpu_index": 0,
        "gpu_uuid": "GPU-abc",
        "gpu_name": "NVIDIA GeForce RTX 4060",
        "memory_used_mib": 1412,
        "memory_total_mib": 8188,
        "utilization_percent": 3,
    }
    assert processes[0]["used_gpu_memory_mib"] == 512
    assert processes[1]["used_gpu_memory_mib"] is None
    with pytest.raises(RuntimeError, match="must contain"):
        parse_gpu_sample("1412, 8188, 3")
    with pytest.raises(RuntimeError, match="PID"):
        parse_compute_processes("unknown, python, 20")


def test_idle_gate_rejects_load_processes_and_identity_drift() -> None:
    base = {
        "gpu_uuid": "GPU-abc",
        "memory_used_mib": 1400,
        "utilization_percent": 2,
        "compute_processes": [],
    }
    clean = assess_idle_samples(
        [base, {**base, "memory_used_mib": 1500}],
        max_memory_mib=1600,
        max_utilization_percent=5,
    )
    busy = assess_idle_samples(
        [base, {**base, "utilization_percent": 6}],
        max_memory_mib=1600,
        max_utilization_percent=5,
    )
    occupied = assess_idle_samples(
        [{**base, "compute_processes": [{"pid": 12}]}],
        max_memory_mib=1600,
        max_utilization_percent=5,
    )
    changed = assess_idle_samples(
        [base, {**base, "gpu_uuid": "GPU-other"}],
        max_memory_mib=1600,
        max_utilization_percent=5,
    )

    assert clean["passed"] is True
    assert busy["passed"] is False
    assert occupied["passed"] is False
    assert changed["passed"] is False


def test_probe_summary_requires_every_hash_matched_worker() -> None:
    plan = _plan()
    role_hashes = {
        "anchor": "anchor-hash",
        "random_member_1": "random-hash",
        "semantic_prior": "prior-hash",
    }
    records = []
    for run in plan["runs"]:
        records.append(
            {
                "status": "complete",
                "candidate_identity_validated": True,
                "provenance_validated": True,
                "integrity_validated": True,
                "config": run,
                "candidate": {
                    "sha256": role_hashes[run["role"]],
                    "all_candidate_sha256": role_hashes,
                },
                "result": {"call_wall_seconds": 1.0},
                "gpu_monitor": {"peak_memory_used_mib": 2000},
            }
        )

    summary = summarize_probe_records(plan, records)
    assert summary["diagnostic_complete"] is True
    assert summary["candidate_construction_consistent"] is True
    assert summary["performance_inference_ready"] is False
    json.dumps(summary, allow_nan=False)

    censored = copy.deepcopy(records)
    censored[2]["status"] = "error"
    censored[2]["integrity_validated"] = False
    censored[2]["error"] = {"type": "WorkerTimeout", "message": "censored"}
    censored[2]["worker_process"] = {"timed_out": True}
    censored[2]["gpu_monitor"]["timed_out"] = True
    summary = summarize_probe_records(plan, censored)
    assert summary["diagnostic_complete"] is True
    assert summary["all_evaluations_completed"] is False
    assert summary["role_summary"]["semantic_prior"]["timeout_runs"] == 1

    failed = copy.deepcopy(records)
    failed[2]["status"] = "error"
    failed[2]["integrity_validated"] = False
    failed[2]["error"] = {"type": "RuntimeError", "message": "not censored"}
    summary = summarize_probe_records(plan, failed)
    assert summary["diagnostic_complete"] is False

    records[-1]["candidate"]["sha256"] = "changed"
    summary = summarize_probe_records(plan, records)
    assert summary["diagnostic_complete"] is False
    assert summary["candidate_hashes_consistent"] is False

    records[-1]["candidate"]["sha256"] = role_hashes["anchor"]
    records[-1]["candidate"]["all_candidate_sha256"] = {
        **role_hashes,
        "anchor": "different-anchor",
    }
    summary = summarize_probe_records(plan, records)
    assert summary["diagnostic_complete"] is False
    assert summary["candidate_construction_consistent"] is False


def test_completed_probe_validator_binds_process_telemetry_and_milestone(
    tmp_path: Path,
) -> None:
    plan = _plan()
    run = plan["runs"][0]
    gpu_uuid = "GPU-abc"
    gpu_name = "NVIDIA GeForce RTX 4060"
    config = {
        **plan["configuration"],
        **run,
        "expected_gpu_uuid": gpu_uuid,
        "expected_gpu_name": gpu_name,
    }
    environment = {
        "backend": "gpu",
        "device_count": 1,
        "device_kinds": [gpu_name],
        "versions": {"jax": "test"},
        "platform": "test-platform",
        "python": "3.11",
    }
    role_hashes = {
        "anchor": "a" * 64,
        "random_member_1": "b" * 64,
        "semantic_prior": "c" * 64,
    }
    candidate = {
        "role": "anchor",
        "sha256": role_hashes["anchor"],
        "stats": {"n_params": 193},
        "all_candidate_sha256": role_hashes,
        "random_population_sha256": ["d" * 64, role_hashes["random_member_1"]],
    }
    problem = {
        "n_params": 193,
        "topology_string": TOPOLOGY,
        "topology_sha256": plan["configuration"]["topology_sha256"],
    }
    idle = {"passed": True, "gpu_uuid": gpu_uuid}

    for directory in ("logs", "telemetry", "milestones"):
        (tmp_path / directory).mkdir()
    run_id = run["run_id"]
    stdout = tmp_path / "logs" / f"{run_id}.stdout.log"
    stderr = tmp_path / "logs" / f"{run_id}.stderr.log"
    telemetry = tmp_path / "telemetry" / f"{run_id}.jsonl"
    milestone_path = tmp_path / "milestones" / f"{run_id}.json"
    stdout.write_text("", encoding="utf-8")
    stderr.write_text("", encoding="utf-8")
    worker_pid = 12345
    sample = {
        "gpu_uuid": gpu_uuid,
        "memory_used_mib": 2000,
        "utilization_percent": 50,
        "compute_processes": [
            {
                "pid": worker_pid,
                "process_name": "python",
                "used_gpu_memory_mib": 512,
            }
        ],
    }
    telemetry.write_text(json.dumps(sample) + "\n", encoding="utf-8")
    worker_policy = jax_environment_policy()
    worker_policy["CUDA_VISIBLE_DEVICES"] = gpu_uuid
    milestone = {
        "format_version": 1,
        "status": "evaluation_started",
        "worker_pid": worker_pid,
        "environment": environment,
        "jax_environment": worker_policy,
        "config": config,
        "candidate": candidate,
        "problem": problem,
    }
    milestone_path.write_text(json.dumps(milestone), encoding="utf-8")
    record = {
        "format_version": 1,
        "status": "complete",
        "config": config,
        "environment": environment,
        "jax_environment": worker_policy,
        "problem": problem,
        "candidate": candidate,
        "result": {
            "call_wall_seconds": 1.0,
            "gradient_finite_fraction": 1.0,
        },
        "objective_accounting": {"eval_count": 1, "log_call_count": 1},
        "objective_configuration": {
            "max_evals": 1,
            "max_time_seconds": None,
            "save": list(OBJECTIVE_SAVE_FIELDS),
            "save_params_history": False,
            "save_batched_params_history": False,
            "save_time_steps": True,
        },
        "idle_preflight": idle,
        "postflight_idle": idle,
        "gpu_monitor": {
            "worker_pid": worker_pid,
            "timed_out": False,
            "monitoring_error": None,
            "contaminated": False,
            "external_compute_processes": [],
            "sample_count": 1,
            "peak_memory_used_mib": 2000,
            "peak_utilization_percent": 50,
            "worker_compute_context_observed": True,
            "visible_worker_process": {
                "pid": worker_pid,
                "process_name": "python",
            },
            "telemetry": {
                "path": f"telemetry/{run_id}.jsonl",
                "sha256": sha256(telemetry),
            },
        },
        "worker_milestone": {
            "path": f"milestones/{run_id}.json",
            "sha256": sha256(milestone_path),
            "payload": milestone,
        },
        "worker_process": {
            "pid": worker_pid,
            "returncode": 0,
            "timed_out": False,
            "full_wall_seconds": 2.0,
            "stdout": {
                "path": f"logs/{run_id}.stdout.log",
                "sha256": sha256(stdout),
            },
            "stderr": {
                "path": f"logs/{run_id}.stderr.log",
                "sha256": sha256(stderr),
            },
        },
    }

    validate_completed_probe_record(record, config, environment, tmp_path)

    timeout_record = copy.deepcopy(record)
    timeout_record["status"] = "error"
    timeout_record["error"] = {"type": "WorkerTimeout", "message": "censored"}
    timeout_record["worker_process"]["returncode"] = -15
    timeout_record["worker_process"]["timed_out"] = True
    timeout_record["gpu_monitor"]["timed_out"] = True
    timeout_record.pop("result")
    timeout_record.pop("objective_accounting")
    timeout_record.pop("objective_configuration")
    validate_candidate_identity_record(
        timeout_record, config, environment, tmp_path
    )
    validate_probe_provenance(timeout_record, config, environment, tmp_path)

    missing_context = copy.deepcopy(record)
    telemetry.write_text(
        json.dumps({**sample, "compute_processes": []}) + "\n",
        encoding="utf-8",
    )
    missing_context["gpu_monitor"]["telemetry"]["sha256"] = sha256(telemetry)
    with pytest.raises(RuntimeError, match="exclusive worker context"):
        validate_probe_provenance(
            missing_context, config, environment, tmp_path
        )
    telemetry.write_text(json.dumps(sample) + "\n", encoding="utf-8")

    record["worker_process"]["returncode"] = 1
    with pytest.raises(RuntimeError, match="exit cleanly"):
        validate_completed_probe_record(record, config, environment, tmp_path)
