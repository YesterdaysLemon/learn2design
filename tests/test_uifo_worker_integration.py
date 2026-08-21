from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest


pytest.importorskip("jax")
pytest.importorskip("dfbench")
np = pytest.importorskip("numpy")
h5py = pytest.importorskip("h5py")

from experiments.uifo_paired.metrics import summarize_rows
from experiments.uifo_paired.optimizer_telemetry import (
    OPTIMIZER_TELEMETRY_METADATA_SCHEMA,
    OPTIMIZER_TELEMETRY_SCHEMA,
    summarize_optimizer_telemetry,
    validate_optimizer_telemetry,
)
from experiments.uifo_paired.runner import (
    BATCHED_SETTINGS,
    HISTORY_SCHEMA,
    _metric_grids,
    _parameter_hashes,
    audit_topology_exclusion,
    sha256,
    validate_completed_record,
    validate_history_artifact,
)

ROOT = Path(__file__).parents[1]


@pytest.mark.integration
def test_worker_refuses_cpu_before_problem_construction(tmp_path: Path) -> None:
    config = {
        "run_id": "cpu-rejection",
        "pair_id": "cpu-rejection-pair",
        "run_order_within_pair": 0,
        "topology": {"kind": "seed", "value": 1001},
        "optimizer_seed": 7,
        "arm": "no_prior",
        "allow_cpu": False,
        "max_evals": 8,
        "max_time_seconds": None,
        "n_frequencies": 50,
        "population_size": 8,
        "target_losses": [],
    }
    config_path = tmp_path / "config.json"
    output_path = tmp_path / "run.json"
    history_path = tmp_path / "history.npz"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.uifo_paired.runner",
            "--worker-config",
            str(config_path),
            "--worker-output",
            str(output_path),
            "--history-output",
            str(history_path),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert completed.returncode == 1
    assert result["status"] == "error"
    assert result["error"]["type"] == "RuntimeError"
    assert "requires an accelerator" in result["error"]["message"]
    assert not history_path.exists()


@pytest.mark.integration
def test_complete_record_is_recomputed_and_history_tampering_is_rejected(
    tmp_path: Path,
) -> None:
    histories = tmp_path / "histories"
    logs = tmp_path / "logs"
    histories.mkdir()
    logs.mkdir()
    history_path = histories / "run.npz"
    initial_params = np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    np.savez_compressed(
        history_path,
        call_index=np.asarray([0, 0], dtype=np.int32),
        candidate_index=np.asarray([0, 1], dtype=np.int16),
        eval_count_after_call=np.asarray([2, 2], dtype=np.int64),
        time_seconds=np.asarray([1.0, 1.0], dtype=np.float64),
        loss=np.asarray([2.0, 1.0], dtype=np.float64),
        sensitivity_loss=np.asarray([1.9, 0.9], dtype=np.float64),
        penalty=np.asarray([0.1, 0.1], dtype=np.float64),
        is_feasible=np.asarray([False, True], dtype=np.bool_),
        initial_params_unbounded=initial_params,
    )
    arrays = validate_history_artifact(history_path, expected_rows=2)
    assert set(arrays) == set(HISTORY_SCHEMA)

    stdout_path = logs / "run.stdout.log"
    stderr_path = logs / "run.stderr.log"
    stdout_path.write_text("", encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")
    config = {
        "run_id": "run",
        "pair_id": "pair",
        "run_order_within_pair": 0,
        "planned_run_index": 0,
        "topology": {"kind": "seed", "value": 1},
        "optimizer_seed": 7,
        "arm": "no_prior",
        "allow_cpu": True,
        "max_evals": 2,
        "max_time_seconds": None,
        "n_frequencies": 50,
        "population_size": 2,
        "target_losses": [],
    }
    rows = [
        {
            "call_index": 0,
            "candidate_index": index,
            "eval_count_after_call": 2,
            "time_seconds": 1.0,
            "loss": loss,
            "sensitivity_loss": sensitivity,
            "penalty": 0.1,
            "is_feasible": feasible,
        }
        for index, (loss, sensitivity, feasible) in enumerate(
            [(2.0, 1.9, False), (1.0, 0.9, True)]
        )
    ]
    time_grid, eval_grid = _metric_grids(config)
    environment = {"backend": "test"}
    record = {
        "format_version": 1,
        "run_id": "run",
        "status": "complete",
        "config": config,
        "environment": environment,
        "problem": {
            "topology_string": "test-topology",
            "topology_sha256": hashlib.sha256(b"test-topology").hexdigest(),
        },
        "objective_configuration": {
            "max_evals": 2,
            "max_time_seconds": None,
        },
        "algorithm": {
            "module": "submission.submission",
            "class": "BatchedRestartAdam",
            "algorithm_str": "batched_restart_adam",
            "kwargs": {
                **BATCHED_SETTINGS,
                "random_seed": 7,
                "population_size": 2,
                "use_semantic_prior": False,
                "evaluation_chunk_size": None,
            }
        },
        "metrics": summarize_rows(
            rows, time_grid=time_grid, eval_grid=eval_grid
        ),
        "history": {
            "format_version": 1,
            "path": "histories/run.npz",
            "rows": 2,
            "schema": HISTORY_SCHEMA,
            "sha256": sha256(history_path),
        },
        "initial_population_roles": ["anchor", "random"],
        "initial_parameter_hashes": _parameter_hashes(initial_params),
        "objective_accounting": {"log_call_count": 1, "eval_count": 2},
        "worker_process": {
            "full_wall_seconds": 1.0,
            "returncode": 0,
            "timed_out": False,
            "within_official_4h30_container_limit": True,
            "stdout": {
                "path": "logs/run.stdout.log",
                "sha256": sha256(stdout_path),
            },
            "stderr": {
                "path": "logs/run.stderr.log",
                "sha256": sha256(stderr_path),
            },
        },
    }
    validate_completed_record(record, config, history_path, environment)

    telemetry_dir = tmp_path / "optimizer-telemetry"
    telemetry_dir.mkdir()
    telemetry_path = telemetry_dir / "run.npz"
    telemetry_values = {
        "batch_index": [0, 0],
        "member_index": [0, 1],
        "eval_count_after_batch": [2, 2],
        "time_seconds": [1.0, 1.0],
        "evaluation_batch_seconds": [1.0, 1.0],
        "finite_loss": [True, True],
        "feasible": [False, True],
        "observed_member_improved": [True, True],
        "observed_member_best_loss": [2.0, 1.0],
        "stalled_steps_before": [0, 0],
        "stalled_steps_after": [0, 0],
        "adam_age_before": [0, 0],
        "adam_age_after": [1, 1],
        "learning_rate": [0.03, 0.15],
        "gradient_nonfinite_count": [0, 0],
        "gradient_norm": [1.0, 1.0],
        "gradient_clip_scale": [1.0, 1.0],
        "global_feasible_improvement": [False, True],
        "restart_triggered": [False, False],
        "restart_kind": [-1, -1],
        "restart_round": [-1, -1],
        "restart_noise_scale": [np.nan, np.nan],
        "evaluated_generation": [0, 0],
        "next_generation": [0, 0],
        "update_applied": [True, True],
        "budget_progress_fraction": [1.0, 1.0],
    }
    telemetry_arrays = {
        name: np.asarray(telemetry_values[name], dtype=dtype)
        for name, dtype in OPTIMIZER_TELEMETRY_SCHEMA.items()
    }
    telemetry_arrays["callback_seconds"] = np.asarray(
        [0.001], dtype=OPTIMIZER_TELEMETRY_METADATA_SCHEMA["callback_seconds"]
    )
    np.savez_compressed(telemetry_path, **telemetry_arrays)
    validated_telemetry = validate_optimizer_telemetry(
        telemetry_path,
        expected_population_size=2,
        expected_patience=BATCHED_SETTINGS["patience"],
    )
    telemetry_config = {**config, "optimizer_telemetry": "member-v1"}
    telemetry_record = {
        **record,
        "config": telemetry_config,
        "optimizer_telemetry": {
            "format_version": 1,
            "mode": "member-v1",
            "path": "optimizer-telemetry/run.npz",
            "rows": 2,
            "schema": OPTIMIZER_TELEMETRY_SCHEMA,
            "metadata_schema": OPTIMIZER_TELEMETRY_METADATA_SCHEMA,
            "sha256": sha256(telemetry_path),
            "summary": summarize_optimizer_telemetry(validated_telemetry),
        },
    }
    validate_completed_record(
        telemetry_record, telemetry_config, history_path, environment
    )

    telemetry_arrays["feasible"] = np.asarray([False, False], dtype=np.bool_)
    telemetry_arrays["global_feasible_improvement"] = np.asarray(
        [False, False], dtype=np.bool_
    )
    np.savez_compressed(telemetry_path, **telemetry_arrays)
    tampered_telemetry = validate_optimizer_telemetry(
        telemetry_path,
        expected_population_size=2,
        expected_patience=BATCHED_SETTINGS["patience"],
    )
    telemetry_record["optimizer_telemetry"]["sha256"] = sha256(telemetry_path)
    telemetry_record["optimizer_telemetry"]["summary"] = (
        summarize_optimizer_telemetry(tampered_telemetry)
    )
    with pytest.raises(RuntimeError, match="feasibility mismatch history"):
        validate_completed_record(
            telemetry_record, telemetry_config, history_path, environment
        )

    telemetry_path.unlink()

    with history_path.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(RuntimeError, match="digest mismatch"):
        validate_completed_record(record, config, history_path, environment)


@pytest.mark.integration
def test_archive_exclusion_is_computed_from_dataset_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_path = tmp_path / "dataset.h5"
    dtype = np.dtype([("topology_string", "S64")])
    entries = np.asarray(
        [(b"archive-topology-a",), (b"archive-topology-b",)], dtype=dtype
    )
    with h5py.File(dataset_path, "w") as archive:
        archive.create_dataset("entries", data=entries)
    monkeypatch.setattr(
        "experiments.uifo_paired.runner.OFFICIAL_DATASET_SHA256",
        sha256(dataset_path),
    )

    audit = audit_topology_exclusion(["fresh-topology"], dataset_path)
    assert audit["official_dataset"]["overlap_count"] == 0
    assert audit["official_dataset"]["unique_topologies"] == 2

    with pytest.raises(ValueError, match="overlaps the official archive"):
        audit_topology_exclusion(["archive-topology-a"], dataset_path)
