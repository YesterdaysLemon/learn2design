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
from experiments.uifo_paired.runner import (
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
            "kwargs": {
                "random_seed": 7,
                "population_size": 2,
                "use_semantic_prior": False,
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
