from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from experiments.uifo_paired.analysis import summarize_records
from experiments.uifo_paired.metrics import flatten_histories, summarize_rows
from experiments.uifo_paired.package import package_study
from experiments.uifo_paired.plan import build_plan
from experiments.uifo_paired.runner import (
    _nvidia_smi_snapshot,
    _orchestrate_locked,
    _preflight_environment,
    _rebuild_indexes,
    _recover_orphaned_provisional_results,
    _rental_preflight,
    _study_lock,
    _validate_required_a100,
    _validate_resume_manifest,
    atomic_json,
    cache_disabled_jax_environment,
    orchestrate,
    parse_topology_panel,
)

ROOT = Path(__file__).parents[1]


def test_plan_is_deterministic_and_rotates_arm_order() -> None:
    kwargs = {
        "topology_seeds": [1001, 1002],
        "topologies": None,
        "optimizer_seeds": [7],
        "arms": ["no_prior", "semantic_prior"],
        "max_time_seconds": 60.0,
        "max_evals": None,
        "population_size": 8,
        "n_frequencies": 50,
    }
    first = build_plan(**kwargs)
    second = build_plan(**kwargs)

    assert first["plan_id"] == second["plan_id"]
    assert [run["arm"] for run in first["runs"]] == [
        "no_prior",
        "semantic_prior",
        "semantic_prior",
        "no_prior",
    ]
    assert len({run["run_id"] for run in first["runs"]}) == 4


def test_rental_plan_binds_accelerator_and_cache_policy() -> None:
    plan = build_plan(
        topology_seeds=[1001],
        topologies=None,
        optimizer_seeds=[7],
        arms=["adam", "no_prior", "semantic_prior"],
        max_time_seconds=600,
        max_evals=None,
        population_size=8,
        n_frequencies=50,
        require_a100=True,
        minimum_gpu_memory_mib=75_000,
        max_idle_gpu_memory_mib=1_000,
        max_idle_gpu_utilization_percent=5,
        minimum_free_disk_gib=20,
        max_session_wall_seconds=22 * 60 * 60,
    )

    configuration = plan["configuration"]
    assert configuration["require_a100"] is True
    assert configuration["minimum_gpu_memory_mib"] == 75_000
    assert configuration["jax_compilation_cache_policy"] == "disabled"
    assert configuration["max_session_wall_seconds"] == 22 * 60 * 60

    with pytest.raises(ValueError, match="mutually exclusive"):
        build_plan(
            topology_seeds=[1001],
            topologies=None,
            optimizer_seeds=[7],
            arms=["no_prior", "semantic_prior"],
            max_time_seconds=60,
            max_evals=None,
            population_size=8,
            n_frequencies=50,
            allow_cpu=True,
            require_a100=True,
        )


def test_scored_child_environment_strips_hostile_persistent_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JAX_COMPILATION_CACHE_DIR", "/tmp/untrusted-cache")
    monkeypatch.setenv("JAX_ENABLE_COMPILATION_CACHE", "true")
    monkeypatch.setenv("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")
    monkeypatch.setenv("JAX_XLA_GPU_PER_FUSION_AUTOTUNE_CACHE_DIR", "/tmp/other-cache")
    monkeypatch.setenv("XLA_FLAGS", "--example")

    child = cache_disabled_jax_environment()

    assert child["JAX_ENABLE_COMPILATION_CACHE"] == "false"
    assert "JAX_COMPILATION_CACHE_DIR" not in child
    assert "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS" not in child
    assert "JAX_XLA_GPU_PER_FUSION_AUTOTUNE_CACHE_DIR" not in child
    assert child["XLA_FLAGS"] == "--example"
    assert child["CUDA_CACHE_DISABLE"] == "1"

    monkeypatch.setenv("XLA_FLAGS", "--xla_gpu_kernel_cache_file=/tmp/hostile")
    with pytest.raises(RuntimeError, match="cache-related XLA_FLAGS"):
        cache_disabled_jax_environment()


def test_rental_preflight_checks_one_idle_80gb_a100(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = {
        "status": "ok",
        "gpus": [
            {
                "index": 0,
                "uuid": "GPU-test",
                "name": "NVIDIA A100 80GB PCIe",
                "driver_version": "575.00",
                "memory_total_mib": 81_920,
                "mig_mode_current": "Disabled",
                "memory_used_mib": 10,
                "utilization_percent": 0,
            }
        ],
    }
    monkeypatch.setattr(
        "experiments.uifo_paired.runner._nvidia_smi_snapshot",
        lambda *, include_dynamic: snapshot,
    )
    configuration = {
        "require_a100": True,
        "minimum_gpu_memory_mib": 75_000,
        "max_idle_gpu_memory_mib": 1_000,
        "max_idle_gpu_utilization_percent": 5,
        "minimum_free_disk_gib": None,
    }

    result = _rental_preflight(tmp_path, configuration)
    assert result["gpu_idle"] == snapshot

    snapshot["gpus"][0]["memory_used_mib"] = 2_000
    with pytest.raises(RuntimeError, match="idle GPU memory"):
        _rental_preflight(tmp_path, configuration)

    snapshot["gpus"][0]["memory_used_mib"] = 10
    snapshot["gpus"][0]["mig_mode_current"] = "Enabled"
    with pytest.raises(RuntimeError, match="MIG mode disabled"):
        _rental_preflight(tmp_path, configuration)


def test_jax_device_gate_requires_exactly_one_a100() -> None:
    _validate_required_a100(
        {
            "backend": "gpu",
            "device_count": 1,
            "competition_aligned_a100": True,
        }
    )
    with pytest.raises(RuntimeError, match="exactly one"):
        _validate_required_a100(
            {
                "backend": "gpu",
                "device_count": 2,
                "competition_aligned_a100": True,
            }
        )


def test_nvidia_smi_snapshot_parses_static_and_idle_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "experiments.uifo_paired.runner.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout=(
                "0, GPU-abc, NVIDIA A100 80GB PCIe, 575.00, 81920, Disabled, 12, 0\n"
            ),
            stderr="",
        ),
    )

    snapshot = _nvidia_smi_snapshot(include_dynamic=True)

    assert snapshot["status"] == "ok"
    assert snapshot["gpus"][0]["uuid"] == "GPU-abc"
    assert snapshot["gpus"][0]["memory_total_mib"] == 81_920
    assert snapshot["gpus"][0]["mig_mode_current"] == "Disabled"
    assert snapshot["gpus"][0]["memory_used_mib"] == 12


def test_resume_preflight_uses_new_artifacts_without_overwriting_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run(args, **kwargs):
        output_path = Path(args[args.index("--preflight-output") + 1])
        atomic_json(output_path, {"artifact": output_path.name})
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout=f"stdout for {output_path.name}",
            stderr="",
        )

    monkeypatch.setattr("experiments.uifo_paired.runner.subprocess.run", fake_run)

    _preflight_environment(tmp_path, artifact_stem="preflight")
    _preflight_environment(tmp_path, artifact_stem="preflight-resume-session")

    original = json.loads((tmp_path / "preflight.json").read_text(encoding="utf-8"))
    resumed = json.loads(
        (tmp_path / "preflight-resume-session.json").read_text(encoding="utf-8")
    )
    assert original["artifact"] == "preflight.json"
    assert resumed["artifact"] == "preflight-resume-session.json"
    assert (tmp_path / "preflight.stdout.log").read_text(encoding="utf-8") == (
        "stdout for preflight.json"
    )


def test_metrics_use_feasible_nonminimum_member() -> None:
    rows = flatten_histories(
        loss_history=[[0.1, 2.0], [0.0, 1.5]],
        feasible_history=[[False, True], [False, True]],
        time_steps=[1.0, 2.0],
        sensitivity_history=[[0.05, 1.8], [-0.1, 1.3]],
        penalty_history=[[0.05, 0.2], [0.1, 0.2]],
    )
    summary = summarize_rows(rows, targets=[1.75, 1.4])

    assert summary["best_feasible_loss"] == 1.5
    assert summary["first_feasible_loss"] == 2.0
    assert summary["time_to_first_feasible_seconds"] == 1.0
    assert summary["evals_to_best_feasible"] == 4
    assert summary["targets"]["1.75"] == {
        "time_seconds": 2.0,
        "eval_count": 4,
    }
    assert summary["targets"]["1.4"] == {
        "time_seconds": None,
        "eval_count": None,
    }


def test_no_feasible_run_remains_explicit() -> None:
    rows = flatten_histories([[1.0, float("nan")]], [[False, False]], [3.0])
    summary = summarize_rows(rows)

    assert summary["has_feasible"] is False
    assert summary["has_finite_feasible"] is False
    assert summary["best_feasible_loss"] is None
    assert summary["physically_feasible_candidate_fraction"] == 0.0
    assert summary["finite_feasible_candidate_fraction"] == 0.0
    json.dumps(summary, allow_nan=False)


def test_feasible_nan_is_visible_but_not_scored() -> None:
    rows = flatten_histories([[1.0, float("nan")]], [[False, True]], [3.0])
    summary = summarize_rows(rows)

    assert summary["has_feasible"] is True
    assert summary["has_finite_feasible"] is False
    assert summary["best_feasible_loss"] is None
    assert summary["physically_feasible_candidate_fraction"] == 0.5


def test_history_alignment_is_strict() -> None:
    with pytest.raises(ValueError, match="align"):
        flatten_histories([[1.0]], [], [1.0])
    with pytest.raises(ValueError, match="batch shapes"):
        flatten_histories([[1.0, 2.0]], [[True]], [1.0])
    with pytest.raises(ValueError, match="feasibility"):
        flatten_histories([[1.0]], [None], [1.0])


def test_scalar_mixed_and_ragged_histories_flatten_exactly() -> None:
    rows = flatten_histories(
        [1.0, [2.0, 3.0], [4.0, 5.0, 6.0]],
        [True, [False, True], [True, False, True]],
        [1.0, 2.0, 3.0],
    )

    assert len(rows) == 6
    assert [row["candidate_index"] for row in rows] == [0, 0, 1, 0, 1, 2]
    assert [row["eval_count_after_call"] for row in rows] == [1, 3, 3, 6, 6, 6]
    assert summarize_rows(rows)["logged_calls"] == 3


def test_anytime_history_is_reduced_to_fixed_grid() -> None:
    rows = flatten_histories([[2.0], [1.0]], [[True], [True]], [2.0, 5.0])
    summary = summarize_rows(rows, time_grid=[1, 2, 4, 5], eval_grid=[1, 2])

    assert summary["anytime_grid"]["time_seconds"] == {
        "1": None,
        "2": 2.0,
        "4": 2.0,
        "5": 1.0,
    }
    assert summary["anytime_grid"]["eval_count"] == {"1": 2.0, "2": 1.0}


def test_evaluation_cap_must_fit_complete_batches() -> None:
    with pytest.raises(ValueError, match="divisible"):
        build_plan(
            topology_seeds=[1001],
            topologies=None,
            optimizer_seeds=[7],
            arms=["no_prior", "semantic_prior"],
            max_time_seconds=None,
            max_evals=65,
            population_size=8,
            n_frequencies=50,
            worker_timeout_seconds=60,
        )


def test_evaluation_chunk_must_fit_population() -> None:
    with pytest.raises(ValueError, match="evaluation_chunk_size"):
        build_plan(
            topology_seeds=[1001],
            topologies=None,
            optimizer_seeds=[7],
            arms=["no_prior", "semantic_prior"],
            max_time_seconds=60,
            max_evals=None,
            population_size=2,
            n_frequencies=50,
            evaluation_chunk_size=3,
        )


@pytest.mark.parametrize("budget", [float("nan"), float("inf"), -float("inf")])
def test_time_budget_must_be_finite(budget: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        build_plan(
            topology_seeds=[1001],
            topologies=None,
            optimizer_seeds=[7],
            arms=["no_prior", "semantic_prior"],
            max_time_seconds=budget,
            max_evals=None,
            population_size=8,
            n_frequencies=50,
        )


@pytest.mark.parametrize("target", [float("nan"), float("inf"), -float("inf")])
def test_target_losses_must_be_finite(target: float) -> None:
    with pytest.raises(ValueError, match="target losses"):
        build_plan(
            topology_seeds=[1001],
            topologies=None,
            optimizer_seeds=[7],
            arms=["no_prior", "semantic_prior"],
            max_time_seconds=60,
            max_evals=None,
            population_size=8,
            n_frequencies=50,
            target_losses=[target],
        )


@pytest.mark.parametrize(
    "topology",
    ["AAAAAAAAA-LLLLLLLLLLLL", "AAAAAAAAA-DHLLLLLLLLLL"],
)
def test_explicit_topology_requires_exactly_one_readout(topology: str) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        build_plan(
            topology_seeds=None,
            topologies=[topology],
            optimizer_seeds=[7],
            arms=["no_prior", "semantic_prior"],
            max_time_seconds=60,
            max_evals=None,
            population_size=8,
            n_frequencies=50,
        )


def test_confirmation_panel_requires_pinned_archive_provenance(
    tmp_path: Path,
) -> None:
    panel_path = tmp_path / "panel.json"
    panel_path.write_text(
        json.dumps(
            {
                "panel_id": "held-out-v1",
                "topologies": ["AAAAAAAAA-LLLLLLLLLLLD"],
                "archive_exclusion_verified": True,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="self-attested"):
        parse_topology_panel(panel_path)

    panel_path.write_text(
        json.dumps(
            {
                "panel_id": "held-out-v1",
                "topologies": ["AAAAAAAAA-LLLLLLLLLLLD"],
            }
        ),
        encoding="utf-8",
    )
    topologies, metadata = parse_topology_panel(panel_path)
    assert topologies == ["AAAAAAAAA-LLLLLLLLLLLD"]
    assert metadata["archive_exclusion_verified"] is False
    assert metadata["source_sha256"]

    with pytest.raises(ValueError, match="must be computed"):
        parse_topology_panel(panel_path, require_archive_exclusion=True)


def test_study_lock_rejects_a_second_writer(tmp_path: Path) -> None:
    with _study_lock(tmp_path):
        with pytest.raises(RuntimeError, match="already locked"):
            with _study_lock(tmp_path):
                pass
    assert not (tmp_path / ".study.lock").exists()


def test_stale_study_lock_recovery_is_explicit_and_preserved(tmp_path: Path) -> None:
    lock_path = tmp_path / ".study.lock"
    atomic_json(
        lock_path,
        {
            "pid": 2_000_000_000,
            "hostname": platform.node(),
            "created_utc": "2026-08-19T00:00:00+00:00",
        },
    )

    with _study_lock(tmp_path, recover_stale=True):
        assert lock_path.exists()

    recovered = list((tmp_path / "recovery").glob("stale-study-lock-*.json"))
    assert len(recovered) == 1
    assert not lock_path.exists()


def test_orphaned_worker_result_is_preserved_before_rerun(tmp_path: Path) -> None:
    output = tmp_path / "study"
    runs = output / "runs"
    histories = output / "histories"
    runs.mkdir(parents=True)
    histories.mkdir()
    config = {"run_id": "run", "arm": "no_prior"}
    environment = {"backend": "gpu"}
    atomic_json(
        runs / "run.json",
        {
            "format_version": 1,
            "run_id": "run",
            "status": "complete",
            "config": config,
            "environment": environment,
        },
    )
    (histories / "run.npz").write_bytes(b"provisional")

    _recover_orphaned_provisional_results(output, {"run": config}, environment)

    assert not (runs / "run.json").exists()
    assert not (histories / "run.npz").exists()
    recovery = output / "recovery" / "orphaned-workers" / "run"
    assert len(list(recovery.glob("*.json"))) == 1
    assert len(list(recovery.glob("*.npz"))) == 1


def test_completed_study_packages_deterministically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = tmp_path / "study"
    runs_dir = study / "runs"
    runs_dir.mkdir(parents=True)
    histories_dir = study / "histories"
    histories_dir.mkdir()
    run = {
        "planned_run_index": 0,
        "run_id": "pair__adam",
        "pair_id": "pair",
        "run_order_within_pair": 0,
        "topology": {"kind": "seed", "value": 1},
        "optimizer_seed": 7,
        "arm": "adam",
    }
    configuration = {
        "allow_cpu": False,
        "evaluation_chunk_size": None,
        "max_evals": None,
        "max_time_seconds": 60,
        "n_frequencies": 50,
        "population_size": 8,
        "require_a100": True,
        "jax_compilation_cache_policy": "disabled",
        "target_losses": [1.0],
    }
    environment = {
        "backend": "gpu",
        "jax_runtime_configuration": {
            "compilation_cache_dir": None,
            "enable_compilation_cache": False,
        },
    }
    manifest = {
        "format_version": 1,
        "plan_id": "plan",
        "project_revision": "revision",
        "configuration": configuration,
        "runs": [run],
        "environment": environment,
        "runtime_policy": {
            "jax_compilation_cache": {
                "policy": "disabled",
                "effective_environment": {
                    "CUDA_CACHE_DISABLE": "1",
                    "JAX_COMPILATION_CACHE_DIR": None,
                    "JAX_ENABLE_COMPILATION_CACHE": "false",
                    "XLA_FLAGS": None,
                },
            }
        },
    }
    atomic_json(study / "manifest.json", manifest)
    config = {
        **run,
        "allow_cpu": False,
        "evaluation_chunk_size": None,
        "max_evals": None,
        "max_time_seconds": 60,
        "n_frequencies": 50,
        "population_size": 8,
        "require_a100": True,
        "jax_compilation_cache_policy": "disabled",
        "target_losses": [1.0],
    }
    topology = "test-topology"
    atomic_json(
        runs_dir / "pair__adam.json",
        {
            "format_version": 1,
            "run_id": "pair__adam",
            "status": "complete",
            "config": config,
            "environment": environment,
            "metrics": {"has_feasible": True, "best_feasible_loss": 1.0},
            "problem": {
                "topology_sha256": hashlib.sha256(topology.encode()).hexdigest(),
                "topology_string": topology,
                "spec": {},
                "n_params": 1,
            },
        },
    )
    (histories_dir / "pair__adam.npz").write_bytes(b"compressed-history")
    monkeypatch.setattr(
        "experiments.uifo_paired.runner.validate_completed_record",
        lambda *args, **kwargs: None,
    )

    first = package_study(study, tmp_path / "first.zip")
    second = package_study(study, tmp_path / "second.zip")

    assert first["archive"]["sha256"] == second["archive"]["sha256"]
    assert (tmp_path / "first.zip.sha256").is_file()
    assert (tmp_path / "first.zip.manifest.json").is_file()
    with zipfile.ZipFile(tmp_path / "first.zip") as archive:
        assert archive.testzip() is None
        assert "manifest.json" in archive.namelist()
        assert (
            archive.getinfo("histories/pair__adam.npz").compress_type
            == zipfile.ZIP_STORED
        )
        assert "runs.jsonl" in archive.namelist()
        assert "summary.json" in archive.namelist()


def test_session_wall_limit_stops_before_launching_another_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = build_plan(
        topology_seeds=[1001],
        topologies=None,
        optimizer_seeds=[7],
        arms=["adam"],
        max_time_seconds=60,
        max_evals=None,
        population_size=8,
        n_frequencies=50,
        max_session_wall_seconds=1e-9,
    )
    monkeypatch.setattr(
        "experiments.uifo_paired.runner._git",
        lambda *args: "revision" if args == ("rev-parse", "HEAD") else "",
    )
    monkeypatch.setattr(
        "experiments.uifo_paired.runner._rental_preflight",
        lambda output_dir, configuration: {"status": "test"},
    )
    monkeypatch.setattr(
        "experiments.uifo_paired.runner._preflight_environment",
        lambda output_dir, subprocess_environment=None, artifact_stem="preflight": {
            "backend": "gpu",
            "device_count": 1,
            "competition_aligned_a100": True,
            "jax_runtime_configuration": {
                "compilation_cache_dir": None,
                "enable_compilation_cache": False,
            },
        },
    )

    exit_code = _orchestrate_locked(plan, tmp_path, resume=False)

    assert exit_code == 2
    session = json.loads((tmp_path / "session.json").read_text(encoding="utf-8"))
    assert session["status"] == "wall_limit_reached"
    assert not list((tmp_path / "runs").glob("*.json"))


def test_manifestless_nonempty_study_is_never_adopted(tmp_path: Path) -> None:
    (tmp_path / "runs").mkdir()
    with pytest.raises(RuntimeError, match="without its manifest"):
        _orchestrate_locked({"configuration": {}, "runs": []}, tmp_path, False)


def test_artifact_indexes_are_strict_and_collision_safe(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    atomic_json(
        runs_dir / "a.json",
        {
            "run_id": "a",
            "status": "complete",
            "config": {"arm": "no_prior", "pair_id": "pair-a"},
            "metrics": {
                "has_feasible": True,
                "best_feasible_loss": 1.0,
                "diagnostic": float("nan"),
            },
            "problem": {"topology_sha256": "topology-a"},
        },
    )
    atomic_json(
        runs_dir / "b.json",
        {"run_id": "b", "status": "error"},
    )
    _rebuild_indexes(tmp_path)

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in (tmp_path / "runs.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert summary["completed_runs"] == 1
    assert summary["error_runs"] == 1
    assert summary["run_ids"] == ["a", "b"]
    assert records[0]["metrics"]["diagnostic"] is None

    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="--resume"):
        orchestrate({"configuration": {}, "runs": []}, tmp_path, resume=False)


def test_pair_summary_averages_seeds_within_topology() -> None:
    records = []
    for topology_string, seed, control, treatment in [
        ("topology-a", 7, 2.0, 1.0),
        ("topology-a", 11, 4.0, 2.0),
        ("topology-b", 7, 1.0, 2.0),
    ]:
        topology_hash = hashlib.sha256(topology_string.encode()).hexdigest()
        pair_id = f"{topology_string}-{seed}"
        for arm, loss in [("no_prior", control), ("semantic_prior", treatment)]:
            use_prior = arm == "semantic_prior"
            records.append(
                {
                    "run_id": f"{pair_id}-{arm}",
                    "status": "complete",
                    "config": {
                        "arm": arm,
                        "pair_id": pair_id,
                        "optimizer_seed": seed,
                        "topology": {"kind": "string", "value": topology_string},
                        "max_evals": 8,
                        "max_time_seconds": None,
                        "population_size": 3,
                        "n_frequencies": 50,
                        "target_losses": [],
                        "allow_cpu": False,
                        "evaluation_chunk_size": None,
                        "require_a100": False,
                        "jax_compilation_cache_policy": "disabled",
                    },
                    "metrics": {
                        "has_feasible": True,
                        "best_feasible_loss": loss,
                    },
                    "problem": {
                        "topology_sha256": topology_hash,
                        "topology_string": topology_string,
                        "spec": {"kind": "test"},
                        "n_params": 2,
                    },
                    "environment": {"backend": "test"},
                    "objective_configuration": {"save": ["batched_loss"]},
                    "algorithm": {
                        "module": "submission.submission",
                        "class": "BatchedRestartAdam",
                        "algorithm_str": "batched_restart_adam",
                        "kwargs": {
                            "use_semantic_prior": use_prior,
                            "random_seed": seed,
                        },
                    },
                    "initial_population_roles": (
                        ["anchor", "semantic_prior", "random"]
                        if use_prior
                        else ["anchor", "random", "random"]
                    ),
                    "initial_parameter_hashes": (
                        ["anchor", "prior", "random-2"]
                        if use_prior
                        else ["anchor", "random-1", "random-2"]
                    ),
                }
            )

    paired = summarize_records(records)["semantic_prior_vs_no_prior"]
    assert paired["wins_ties_losses"] == {
        "semantic_prior_wins": 1,
        "ties": 0,
        "semantic_prior_losses": 1,
    }
    assert paired["optimizer_seed_pair_diagnostics"][
        "optimizer_seed_pair_wins_ties_losses"
    ] == {
        "semantic_prior_wins": 2,
        "ties": 0,
        "semantic_prior_losses": 1,
    }
    # Topology A contributes its mean (-1.5) once; topology B contributes +1.
    assert paired["topology_macro_mean_difference"] == pytest.approx(-0.25)

    expected_configs = {record["run_id"]: record["config"] for record in records}
    for arm in ("no_prior", "semantic_prior"):
        expected_configs[f"topology-b-11-{arm}"] = {
            **records[-1]["config"],
            "arm": arm,
            "pair_id": "topology-b-11",
            "optimizer_seed": 11,
        }
    planned = summarize_records(records, expected_configs)["semantic_prior_vs_no_prior"]
    assert planned["complete_topologies"] == 1
    assert len(planned["incomplete_topologies"]) == 1
    assert planned["promotion_inference_ready"] is False
    assert planned["topology_macro_mean_difference"] == pytest.approx(-1.5)


def test_pair_summary_rejects_cross_topology_pairing() -> None:
    topology = "topology-a"
    digest = hashlib.sha256(topology.encode()).hexdigest()
    records = []
    for arm, other_topology in [
        ("no_prior", topology),
        ("semantic_prior", "topology-b"),
    ]:
        use_prior = arm == "semantic_prior"
        records.append(
            {
                "run_id": arm,
                "status": "complete",
                "config": {
                    "arm": arm,
                    "pair_id": "pair",
                    "optimizer_seed": 7,
                    "topology": {"kind": "seed", "value": 1},
                    "max_evals": 8,
                    "max_time_seconds": None,
                    "population_size": 2,
                    "n_frequencies": 50,
                    "target_losses": [],
                    "allow_cpu": False,
                    "evaluation_chunk_size": None,
                    "require_a100": False,
                    "jax_compilation_cache_policy": "disabled",
                },
                "metrics": {"has_feasible": True, "best_feasible_loss": 1.0},
                "problem": {
                    "topology_sha256": (
                        digest
                        if not use_prior
                        else hashlib.sha256(other_topology.encode()).hexdigest()
                    ),
                    "topology_string": other_topology,
                    "spec": {},
                    "n_params": 1,
                },
                "environment": {"backend": "test"},
                "objective_configuration": {"save": ["batched_loss"]},
                "algorithm": {
                    "kwargs": {"use_semantic_prior": use_prior},
                },
                "initial_population_roles": (
                    ["anchor", "semantic_prior"] if use_prior else ["anchor", "random"]
                ),
                "initial_parameter_hashes": (
                    ["anchor", "prior"] if use_prior else ["anchor", "random"]
                ),
            }
        )

    with pytest.raises(ValueError, match="problem mismatch"):
        summarize_records(records)


def test_resume_manifest_rejects_revision_drift() -> None:
    existing = {
        "format_version": 1,
        "plan_id": "plan",
        "project_revision": "old",
        "semantic_prior_canonical_sha256": "prior",
        "upstream_reference": "upstream",
    }
    current = {**existing, "project_revision": "new"}

    with pytest.raises(RuntimeError, match="project_revision"):
        _validate_resume_manifest(existing, current)

    existing = {**existing, "environment": {"jax": "old"}}
    current = {**existing, "environment": {"jax": "new"}}
    with pytest.raises(RuntimeError, match="environment"):
        _validate_resume_manifest(existing, current)

    existing = {**existing, "runtime_policy": {"cache": "disabled"}}
    current = {**existing, "runtime_policy": {"cache": "enabled"}}
    with pytest.raises(RuntimeError, match="runtime_policy"):
        _validate_resume_manifest(existing, current)


def test_index_rebuild_rejects_records_outside_the_plan(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    atomic_json(
        runs_dir / "stray.json",
        {
            "format_version": 1,
            "run_id": "stray",
            "status": "error",
            "config": {"run_id": "stray"},
        },
    )
    with pytest.raises(RuntimeError, match="outside plan"):
        _rebuild_indexes(tmp_path, {"expected": {"run_id": "expected"}})


def test_dry_run_cli_does_not_import_simulator() -> None:
    command = [
        sys.executable,
        str(ROOT / "tools" / "run_uifo_paired.py"),
        "--topology-seeds",
        "1001",
        "--optimizer-seeds",
        "7",
        "--max-time",
        "60",
        "--dry-run",
    ]
    completed = subprocess.run(
        command, cwd=ROOT, check=True, capture_output=True, text=True
    )
    plan = json.loads(completed.stdout)

    assert plan["format_version"] == 1
    assert len(plan["runs"]) == 2
