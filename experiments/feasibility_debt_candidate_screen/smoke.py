"""Loss-blind cold-smoke worker for the frozen H100 candidate screen."""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import sys
from typing import Any

from .canonical import canonical_json_bytes, read_receipt, sha256_bytes
from .contract import (
    ARM_ORDER,
    N_FREQUENCIES,
    OFFICIAL_ARCHIVE_SHA256,
    PANEL_ID,
    SMOKE_OPTIMIZER_SEED,
    SMOKE_TOPOLOGY_SEED,
    STUDY_ID,
    arm_spec,
)
from .worker import (
    WorkerError,
    _Instrumentation,
    authenticate_execution_locks,
    load_arm_class,
    verify_warmup_source,
)
from .packet import BoundedTextCapture


class SmokeError(RuntimeError):
    pass


SMOKE_CONFIG_KEYS = {
    "schema_version",
    "study_id",
    "smoke_id",
    "topology_seed",
    "topology_sha256",
    "optimizer_seed",
    "arm_id",
    "arm_profile",
    "source_lock_sha256",
    "runtime_lock_sha256",
    "revision",
    "panel_commitment_sha256",
    "provider_launch_receipt_sha256",
    "resource_manifest_sha256",
    "hard_stop_receipt_sha256",
    "t0_utc",
    "b0_utc",
    "hard_horizon_utc",
    "dispatch_deadline_utc",
    "max_time_seconds",
    "population_size",
    "n_frequencies",
    "allow_cpu",
}

PANEL_COMMITMENT_KEYS = {
    "panel_id",
    "panel_sha256",
    "official_archive",
    "prior_panels",
    "candidate_seed_start",
    "candidate_seed_attempts",
    "eligible_unique_candidates",
    "archive_overlap_count",
    "prior_panel_overlap_count",
    "smoke_topology_seed",
    "smoke_topology_sha256",
    "smoke_overlap_count",
    "upstream_reference",
}

SMOKE_RESULT_KEYS = {
    "status",
    "smoke_id",
    "revision",
    "panel_commitment_sha256",
    "source_lock_sha256",
    "runtime_lock_sha256",
    "provider_launch_receipt_sha256",
    "resource_manifest_sha256",
    "hard_stop_receipt_sha256",
    "hard_horizon_utc",
    "topology_seed",
    "topology_sha256",
    "optimizer_seed",
    "objective_budget_seconds",
    "device_model",
    "arm_boundaries",
    "logged_activity_observed",
    "loss_values_exposed",
    "candidate_values_exposed",
    "history_values_exposed",
}


def validate_smoke_config(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != SMOKE_CONFIG_KEYS:
        raise SmokeError("cold-smoke config schema mismatch")
    spec = arm_spec("D_v3_coverage")
    profile = value["arm_profile"]
    package = profile.get("package_closure_sha256") if isinstance(profile, dict) else None
    exact = {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "smoke_id": f"{STUDY_ID}::cold-smoke",
        "topology_seed": SMOKE_TOPOLOGY_SEED,
        "optimizer_seed": SMOKE_OPTIMIZER_SEED,
        "arm_id": "D_v3_coverage",
        "max_time_seconds": 120.0,
        "population_size": 8,
        "n_frequencies": N_FREQUENCIES,
        "allow_cpu": False,
    }
    if any(value.get(key) != expected for key, expected in exact.items()):
        raise SmokeError("cold-smoke config differs from the frozen profile")
    if not isinstance(package, str) or profile != spec.lock_row(package):
        raise SmokeError("cold-smoke arm profile mismatch")
    for key in (
        "topology_sha256",
        "source_lock_sha256",
        "runtime_lock_sha256",
        "panel_commitment_sha256",
        "provider_launch_receipt_sha256",
        "resource_manifest_sha256",
        "hard_stop_receipt_sha256",
    ):
        digest = value[key]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(token not in "0123456789abcdef" for token in digest)
        ):
            raise SmokeError(f"cold-smoke digest is invalid: {key}")
    revision = value["revision"]
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or any(token not in "0123456789abcdef" for token in revision)
    ):
        raise SmokeError("cold-smoke revision is invalid")
    from .runtime import parse_utc

    t0 = parse_utc(value["t0_utc"])
    b0 = parse_utc(value["b0_utc"])
    hard = parse_utc(value["hard_horizon_utc"])
    dispatch = parse_utc(value["dispatch_deadline_utc"])
    from datetime import timedelta

    if (
        dispatch != hard - timedelta(seconds=1_800)
        or hard > t0 + timedelta(seconds=25_200)
        or hard > b0 + timedelta(seconds=25_200)
    ):
        raise SmokeError("cold-smoke deadline binding mismatch")
    return value


def build_smoke_config(
    *,
    revision: str,
    source_lock_sha256: str,
    runtime_lock_sha256: str,
    package_closure_sha256: str,
    panel_commitment_path: Path,
    provider_launch_receipt_sha256: str,
    resource_manifest_sha256: str,
    hard_stop_receipt_sha256: str,
    deadline_snapshot: dict[str, object],
) -> dict[str, Any]:
    commitment, commitment_sha256 = read_receipt(
        panel_commitment_path,
        expected_study_id=STUDY_ID,
        expected_receipt_type="panel_commitment",
        expected_payload_keys=PANEL_COMMITMENT_KEYS,
    )
    archive = commitment["official_archive"]
    if (
        commitment["panel_id"] != PANEL_ID
        or not isinstance(archive, dict)
        or archive.get("sha256") != OFFICIAL_ARCHIVE_SHA256
        or commitment["smoke_topology_seed"] != SMOKE_TOPOLOGY_SEED
        or commitment["smoke_overlap_count"] != 0
    ):
        raise SmokeError("panel commitment does not authorize the frozen smoke")
    value = {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "smoke_id": f"{STUDY_ID}::cold-smoke",
        "topology_seed": SMOKE_TOPOLOGY_SEED,
        "topology_sha256": commitment["smoke_topology_sha256"],
        "optimizer_seed": SMOKE_OPTIMIZER_SEED,
        "arm_id": "D_v3_coverage",
        "arm_profile": arm_spec("D_v3_coverage").lock_row(
            package_closure_sha256
        ),
        "source_lock_sha256": source_lock_sha256,
        "runtime_lock_sha256": runtime_lock_sha256,
        "revision": revision,
        "panel_commitment_sha256": commitment_sha256,
        "provider_launch_receipt_sha256": provider_launch_receipt_sha256,
        "resource_manifest_sha256": resource_manifest_sha256,
        "hard_stop_receipt_sha256": hard_stop_receipt_sha256,
        "t0_utc": deadline_snapshot["t0_utc"],
        "b0_utc": deadline_snapshot["b0_utc"],
        "hard_horizon_utc": deadline_snapshot["hard_horizon_utc"],
        "dispatch_deadline_utc": deadline_snapshot["dispatch_deadline_utc"],
        "max_time_seconds": 120.0,
        "population_size": 8,
        "n_frequencies": N_FREQUENCIES,
        "allow_cpu": False,
    }
    return validate_smoke_config(value)


def validate_smoke_projection(
    value: object, *, config: dict[str, Any]
) -> dict[str, object]:
    config = validate_smoke_config(config)
    if not isinstance(value, dict) or set(value) != SMOKE_RESULT_KEYS:
        raise SmokeError("cold-smoke result schema mismatch")
    boundary_rows = [
        {
            "arm_id": arm_id,
            "class_name": arm_spec(arm_id).class_name,
            "algorithm_str": arm_spec(arm_id).algorithm_str,
            "source_sha256": arm_spec(arm_id).source_sha256,
        }
        for arm_id in ARM_ORDER
    ]
    exact = {
        "status": "passed",
        "smoke_id": config["smoke_id"],
        "revision": config["revision"],
        "panel_commitment_sha256": config["panel_commitment_sha256"],
        "source_lock_sha256": config["source_lock_sha256"],
        "runtime_lock_sha256": config["runtime_lock_sha256"],
        "provider_launch_receipt_sha256": config[
            "provider_launch_receipt_sha256"
        ],
        "resource_manifest_sha256": config["resource_manifest_sha256"],
        "hard_stop_receipt_sha256": config["hard_stop_receipt_sha256"],
        "hard_horizon_utc": config["hard_horizon_utc"],
        "topology_seed": SMOKE_TOPOLOGY_SEED,
        "topology_sha256": config["topology_sha256"],
        "optimizer_seed": SMOKE_OPTIMIZER_SEED,
        "objective_budget_seconds": 120.0,
        "device_model": "NVIDIA H100 80GB HBM3",
        "arm_boundaries": boundary_rows,
        "logged_activity_observed": True,
        "loss_values_exposed": False,
        "candidate_values_exposed": False,
        "history_values_exposed": False,
    }
    if value != exact:
        raise SmokeError("cold-smoke result differs from the loss-blind contract")
    return value


def _authenticate_panel_commitment(config: dict[str, Any], path: Path) -> None:
    commitment, digest = read_receipt(
        path,
        expected_study_id=STUDY_ID,
        expected_receipt_type="panel_commitment",
        expected_payload_keys=PANEL_COMMITMENT_KEYS,
    )
    if (
        digest != config["panel_commitment_sha256"]
        or commitment["smoke_topology_seed"] != config["topology_seed"]
        or commitment["smoke_topology_sha256"] != config["topology_sha256"]
        or commitment["smoke_overlap_count"] != 0
    ):
        raise SmokeError("cold-smoke panel commitment binding mismatch")


def execute_smoke(
    config: dict[str, Any],
    *,
    repository_root: Path,
    round1_archive: Path,
    round1_manifest: Path,
    runtime_lock_path: Path,
    source_lock_path: Path,
    panel_commitment_path: Path,
    expected_revision: str,
) -> dict[str, object]:
    config = validate_smoke_config(config)
    _authenticate_panel_commitment(config, panel_commitment_path)
    authenticate_execution_locks(
        config,
        repository_root=repository_root,
        round1_archive=round1_archive,
        round1_manifest=round1_manifest,
        runtime_lock_path=runtime_lock_path,
        source_lock_path=source_lock_path,
        expected_revision=expected_revision,
    )
    import jax
    from dfbench import Objective
    from dfbench.problems import UIFOProblem
    from .panel import topology_from_seed

    if jax.default_backend() == "cpu":
        raise SmokeError("cold smoke requires the frozen accelerator")
    device_kinds = [str(getattr(device, "device_kind", "")) for device in jax.devices()]
    if device_kinds != ["NVIDIA H100 80GB HBM3"]:
        raise SmokeError("cold-smoke device identity mismatch")

    boundary_rows: list[dict[str, str]] = []
    classes: dict[str, type[Any]] = {}
    for arm_id in ARM_ORDER:
        spec = arm_spec(arm_id)
        algorithm_class = load_arm_class(
            arm_id,
            repository_root=repository_root,
            round1_archive=round1_archive,
            round1_manifest=round1_manifest,
        )
        if (
            algorithm_class.__name__ != spec.class_name
            or algorithm_class.algorithm_str != spec.algorithm_str
        ):
            raise SmokeError("cold-smoke arm boundary mismatch")
        verify_warmup_source(algorithm_class)
        classes[arm_id] = algorithm_class
        boundary_rows.append(
            {
                "arm_id": arm_id,
                "class_name": spec.class_name,
                "algorithm_str": spec.algorithm_str,
                "source_sha256": spec.source_sha256,
            }
        )

    topology = topology_from_seed(SMOKE_TOPOLOGY_SEED)
    if sha256_bytes(topology.encode("utf-8")) != config["topology_sha256"]:
        raise SmokeError("cold-smoke topology reconstruction mismatch")
    problem = UIFOProblem(size=3, n_frequencies=N_FREQUENCIES, topology=topology)
    if str(problem.topology_string) != topology:
        raise SmokeError("cold-smoke problem topology mismatch")
    objective = Objective(
        problem,
        max_time=120.0,
        max_evals=None,
        save_time_steps=True,
        save_params_history=False,
        save_batched_params_history=False,
        save=[
            "eval_type",
            "batched_loss",
            "batched_sensitivity_loss",
            "batched_penalty",
            "batched_is_feasible",
        ],
        verbose=0,
    )
    spec = arm_spec("D_v3_coverage")
    algorithm_class = classes["D_v3_coverage"]
    instrumentation = _Instrumentation(
        objective,
        expect_warmup=True,
        warmup_source_proof=verify_warmup_source(algorithm_class),
    )
    kwargs = {
        **spec.fixed_kwargs(),
        "random_seed": SMOKE_OPTIMIZER_SEED,
        "initial_population_callback": instrumentation.capture_initial,
        "raw_initial_population_callback": instrumentation.capture_raw,
    }
    algorithm_class().optimize(objective, **kwargs)
    instrumentation.receipt()
    if type(objective.eval_count) is bool or int(objective.eval_count) < 1:
        raise SmokeError("cold smoke completed without objective activity")
    if type(objective.log_call_count) is bool or int(objective.log_call_count) < 1:
        raise SmokeError("cold smoke completed without logged calls")
    return validate_smoke_projection({
        "status": "passed",
        "smoke_id": config["smoke_id"],
        "revision": config["revision"],
        "panel_commitment_sha256": config["panel_commitment_sha256"],
        "source_lock_sha256": config["source_lock_sha256"],
        "runtime_lock_sha256": config["runtime_lock_sha256"],
        "provider_launch_receipt_sha256": config[
            "provider_launch_receipt_sha256"
        ],
        "resource_manifest_sha256": config["resource_manifest_sha256"],
        "hard_stop_receipt_sha256": config["hard_stop_receipt_sha256"],
        "hard_horizon_utc": config["hard_horizon_utc"],
        "topology_seed": SMOKE_TOPOLOGY_SEED,
        "topology_sha256": config["topology_sha256"],
        "optimizer_seed": SMOKE_OPTIMIZER_SEED,
        "objective_budget_seconds": 120.0,
        "device_model": "NVIDIA H100 80GB HBM3",
        "arm_boundaries": boundary_rows,
        "logged_activity_observed": True,
        "loss_values_exposed": False,
        "candidate_values_exposed": False,
        "history_values_exposed": False,
    }, config=config)


def run_smoke_packet(config: dict[str, Any], **kwargs: Any) -> bytes:
    captured_stdout = BoundedTextCapture(max_bytes=1_048_576)
    captured_stderr = BoundedTextCapture(max_bytes=1_048_576)
    with redirect_stdout(captured_stdout), redirect_stderr(captured_stderr):
        smoke = execute_smoke(config, **kwargs)
    if captured_stdout.getvalue() or captured_stderr.getvalue():
        raise SmokeError("cold-smoke runtime emitted unsolicited output")
    return canonical_json_bytes(
        {
            "schema_version": 1,
            "study_id": STUDY_ID,
            "smoke": smoke,
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--round1-archive", type=Path, required=True)
    parser.add_argument("--round1-manifest", type=Path, required=True)
    parser.add_argument("--runtime-lock", type=Path, required=True)
    parser.add_argument("--source-lock", type=Path, required=True)
    parser.add_argument("--panel-commitment", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    raw = args.config.read_bytes()
    try:
        config = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SmokeError("cold-smoke config is not JSON") from error
    if canonical_json_bytes(config) != raw:
        raise SmokeError("cold-smoke config is not canonical JSON")
    packet = run_smoke_packet(
        config,
        repository_root=args.repository_root,
        round1_archive=args.round1_archive,
        round1_manifest=args.round1_manifest,
        runtime_lock_path=args.runtime_lock,
        source_lock_path=args.source_lock,
        panel_commitment_path=args.panel_commitment,
        expected_revision=args.revision,
    )
    sys.stdout.buffer.write(packet)
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
