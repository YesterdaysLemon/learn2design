"""No-project-import history-first replay for the candidate screen.

This module intentionally duplicates the frozen projection and decision rules.
It may import NumPy to parse NPZ bytes, but it must never import ``experiments``,
``submission``, ``tools``, or a production analyzer.
"""

from __future__ import annotations

import math
import hashlib
import io
from pathlib import Path
from typing import Any, Sequence
import zipfile


ARM_ORDER = (
    "A_round1_control",
    "B_round1_warmup",
    "C_v3_random",
    "D_v3_coverage",
)
CHALLENGERS = ARM_ORDER[1:]
ROWS = (
    ARM_ORDER,
    (ARM_ORDER[1], ARM_ORDER[0], ARM_ORDER[3], ARM_ORDER[2]),
    (ARM_ORDER[2], ARM_ORDER[3], ARM_ORDER[0], ARM_ORDER[1]),
    (ARM_ORDER[3], ARM_ORDER[2], ARM_ORDER[1], ARM_ORDER[0]),
)
FIELDS = (
    "call_index",
    "candidate_index",
    "eval_count_after_call",
    "time_seconds",
    "loss",
    "sensitivity_loss",
    "penalty",
    "is_feasible",
)
TOLERANCE = 1e-12
CONFIG_KEYS = {
    "schema_version",
    "study_id",
    "run_id",
    "stage",
    "member_index",
    "execution_position",
    "arm_id",
    "optimizer_seed",
    "topology",
    "topology_sha256",
    "panel_sha256",
    "panel_commitment_sha256",
    "split_receipt_sha256",
    "selection_receipt_sha256",
    "source_lock_sha256",
    "runtime_lock_sha256",
    "revision",
    "arm_profile",
    "max_time_seconds",
    "max_evals",
    "population_size",
    "n_frequencies",
    "allow_cpu",
}

COMMON_KWARGS = {
    "learning_rate_low": 0.03,
    "learning_rate_high": 0.15,
    "minimum_improvement": 1e-7,
    "beta1": 0.9,
    "beta2": 0.999,
    "epsilon": 1e-8,
    "gradient_clip_norm": 1.0,
    "restart_noise_scale": 0.35,
    "safety_seconds": 2.0,
    "batch_time_safety_factor": 1.5,
    "batch_time_window": 8,
    "population_size": 8,
    "use_semantic_prior": False,
    "evaluation_chunk_size": None,
}
PROFILE_ROWS = {
    "A_round1_control": {
        "logical_module_id": "round1_zip::submission.py",
        "python_module_name": "l2d_round1_control_submission",
        "class_name": "BatchedRestartAdam",
        "algorithm_str": "batched_restart_adam",
        "source_sha256": "45f18e17f3e9e0855629079c311a4c01318a3d2e6700db4719e2a54d61ffea76",
        "patience": 600,
        "population_mode": "random",
        "preclock_warmup": False,
        "progress_mode": "protected_total_loss",
    },
    "B_round1_warmup": {
        "logical_module_id": "submission/submission.py",
        "python_module_name": "submission.submission",
        "class_name": "BatchedRestartAdam",
        "algorithm_str": "batched_restart_adam",
        "source_sha256": "0fefbaaf18d9831895d788df45c92cbaf4522da7c54d8f78646e449ffa9374c9",
        "patience": 600,
        "population_mode": "random",
        "preclock_warmup": True,
        "progress_mode": "protected_total_loss",
    },
    "C_v3_random": {
        "logical_module_id": "experiments/candidates/feasibility_debt_clock_v3.py",
        "python_module_name": "experiments.candidates.feasibility_debt_clock_v3",
        "class_name": "FeasibilityDebtBatchedRestartAdamV3",
        "algorithm_str": "feasibility_debt_batched_restart_adam_v3",
        "source_sha256": "ca7abd365c5d1172dab2f47fccdf0afa3df9652e75cc2003385312cec48844d6",
        "patience": 200,
        "population_mode": "random",
        "preclock_warmup": True,
        "progress_mode": "feasibility_debt",
    },
    "D_v3_coverage": {
        "logical_module_id": "experiments/candidates/feasibility_debt_clock_v3.py",
        "python_module_name": "experiments.candidates.feasibility_debt_clock_v3",
        "class_name": "FeasibilityDebtBatchedRestartAdamV3",
        "algorithm_str": "feasibility_debt_batched_restart_adam_v3",
        "source_sha256": "ca7abd365c5d1172dab2f47fccdf0afa3df9652e75cc2003385312cec48844d6",
        "patience": 200,
        "population_mode": "coverage_balanced",
        "preclock_warmup": True,
        "progress_mode": "feasibility_debt",
    },
}


def _expected_profile(arm: str, package: str) -> dict[str, object]:
    if arm not in PROFILE_ROWS or len(package) != 64 or any(
        token not in "0123456789abcdef" for token in package
    ):
        raise ReferenceError("reference package/profile identity mismatch")
    row = dict(PROFILE_ROWS[arm])
    kwargs = {**COMMON_KWARGS, "patience": row["patience"]}
    if arm != "A_round1_control":
        kwargs.update(
            {
                "initial_population_mode": row["population_mode"],
                "preclock_warmup": row["preclock_warmup"],
            }
        )
    if arm in {"C_v3_random", "D_v3_coverage"}:
        kwargs["progress_mode"] = "feasibility_debt"
    return {
        "arm_id": arm,
        **row,
        "kwargs": kwargs,
        "package_closure_sha256": package,
    }


class ReferenceError(RuntimeError):
    pass


def _load(source: Path | bytes) -> list[dict[str, object]]:
    import numpy as np

    names = {f"{name}.npy" for name in FIELDS} | {
        "initial_params_unbounded.npy",
        "raw_params_unbounded.npy",
    }
    target = io.BytesIO(source) if isinstance(source, bytes) else source
    with zipfile.ZipFile(target) as archive:
        members = archive.namelist()
        if len(members) != len(names) or set(members) != names:
            raise ReferenceError("reference NPZ member schema mismatch")
    dtypes = {
        "call_index": np.dtype("int32"),
        "candidate_index": np.dtype("int16"),
        "eval_count_after_call": np.dtype("int64"),
        "time_seconds": np.dtype("float64"),
        "loss": np.dtype("float64"),
        "sensitivity_loss": np.dtype("float64"),
        "penalty": np.dtype("float64"),
        "is_feasible": np.dtype("bool"),
    }
    target = io.BytesIO(source) if isinstance(source, bytes) else source
    with np.load(target, allow_pickle=False) as archive:
        values = {name: np.asarray(archive[name]) for name in dtypes}
        initial = np.asarray(archive["initial_params_unbounded"])
        raw = np.asarray(archive["raw_params_unbounded"])
    if (
        initial.ndim != 2
        or initial.shape[0] != 8
        or raw.shape != initial.shape
        or raw.dtype != initial.dtype
        or not np.all(np.isfinite(initial))
        or not np.all(np.isfinite(raw))
    ):
        raise ReferenceError("reference initial population shape mismatch")
    lengths: set[int] = set()
    for name, dtype in dtypes.items():
        array = values[name]
        if array.ndim != 1 or array.dtype != dtype:
            raise ReferenceError(f"reference history field mismatch: {name}")
        lengths.add(len(array))
    if len(lengths) != 1 or next(iter(lengths)) < 1:
        raise ReferenceError("reference history row lengths mismatch")
    result: list[dict[str, object]] = []
    for offset in range(next(iter(lengths))):
        def maybe(name: str) -> float | None:
            item = float(values[name][offset])
            return item if math.isfinite(item) else None

        result.append(
            {
                "call_index": int(values["call_index"][offset]),
                "candidate_index": int(values["candidate_index"][offset]),
                "eval_count_after_call": int(
                    values["eval_count_after_call"][offset]
                ),
                "time_seconds": float(values["time_seconds"][offset]),
                "loss": maybe("loss"),
                "sensitivity_loss": maybe("sensitivity_loss"),
                "penalty": maybe("penalty"),
                "is_feasible": bool(values["is_feasible"][offset]),
            }
        )
    return result


def project_history_path(path: Path) -> dict[str, Any]:
    rows = _load(path)
    return _project_rows(rows)


def project_history_bytes(content: bytes) -> dict[str, Any]:
    rows = _load(content)
    return _project_rows(rows)


def _project_rows(rows: list[dict[str, object]]) -> dict[str, Any]:
    previous_call = -1
    expected_candidate = 0
    previous_count = 0
    previous_time = -math.inf
    completed_count = 0
    current_call_count = 0
    current_call_size = 0
    finite_feasible: list[float] = []
    physical = False
    for offset, row in enumerate(rows):
        call = row["call_index"]
        candidate = row["candidate_index"]
        count = row["eval_count_after_call"]
        elapsed = row["time_seconds"]
        if (
            isinstance(call, bool)
            or not isinstance(call, int)
            or call < 0
            or isinstance(candidate, bool)
            or not isinstance(candidate, int)
            or candidate < 0
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 1
            or isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(float(elapsed))
            or float(elapsed) < 0
        ):
            raise ReferenceError(f"reference history type failure at {offset}")
        if call == previous_call:
            expected_candidate += 1
            current_call_size += 1
            if count != current_call_count:
                raise ReferenceError("reference within-call count mismatch")
        elif call == previous_call + 1:
            if offset:
                if current_call_count != completed_count + current_call_size:
                    raise ReferenceError("reference completed-call count mismatch")
                completed_count = current_call_count
            expected_candidate = 0
            if offset and count <= previous_count:
                raise ReferenceError("reference call count did not advance")
            current_call_count = count
            current_call_size = 1
        else:
            raise ReferenceError("reference call ledger is incomplete")
        if candidate != expected_candidate:
            raise ReferenceError("reference candidate order mismatch")
        if float(elapsed) < previous_time or count < previous_count:
            raise ReferenceError("reference history chronology mismatch")
        feasible = row["is_feasible"]
        if feasible is not True and feasible is not False:
            raise ReferenceError("reference feasibility is not Boolean")
        loss = row["loss"]
        if loss is not None and (
            isinstance(loss, bool) or not isinstance(loss, (int, float))
        ):
            raise ReferenceError("reference loss is not numeric-or-null")
        if feasible is True:
            physical = True
            if loss is not None and math.isfinite(float(loss)):
                finite_feasible.append(float(loss))
        previous_call = call
        previous_count = count
        previous_time = float(elapsed)
    if current_call_count != completed_count + current_call_size:
        raise ReferenceError("reference final-call count mismatch")
    if previous_time <= 0:
        raise ReferenceError("reference elapsed duration is not positive")
    if previous_time > 600.0:
        raise ReferenceError("reference history exceeds the Objective budget")
    return {
        "rows": len(rows),
        "logged_calls": previous_call + 1,
        "has_feasible": physical,
        "has_finite_feasible": bool(finite_feasible),
        "best_feasible_loss": min(finite_feasible) if finite_feasible else None,
        "evaluation_count": previous_count,
        "elapsed_seconds": previous_time,
        "evaluation_rate": previous_count / previous_time,
    }


def _stage1_order(indices: Sequence[int]) -> list[tuple[int, str]]:
    if (
        len(indices) != 4
        or list(indices) != sorted(set(indices))
        or any(type(index) is not int or index not in range(8) for index in indices)
    ):
        raise ReferenceError("reference Stage-1 indices invalid")
    return [
        (member, arm)
        for member, row in zip(indices, ROWS, strict=True)
        for arm in row
    ]


def _stage2_order(indices: Sequence[int], finalist: str) -> list[tuple[int, str]]:
    if (
        len(indices) != 4
        or list(indices) != sorted(set(indices))
        or any(type(index) is not int or index not in range(8) for index in indices)
    ):
        raise ReferenceError("reference Stage-2 indices invalid")
    if finalist not in CHALLENGERS:
        raise ReferenceError("reference finalist invalid")
    return [
        (member, arm)
        for offset, member in enumerate(indices)
        for arm in (
            (ARM_ORDER[0], finalist) if offset % 2 == 0 else (finalist, ARM_ORDER[0])
        )
    ]


def _authenticate_runs(
    runs: Sequence[dict[str, object]],
    expected: Sequence[tuple[int, str]],
    *,
    stage: int,
    selection_receipt_sha256: str | None = None,
) -> dict[tuple[int, str], dict[str, Any]]:
    if len(runs) != len(expected):
        raise ReferenceError("reference run count mismatch")
    result: dict[tuple[int, str], dict[str, Any]] = {}
    seed = 20260901 if stage == 1 else 20260902
    within_counts: dict[int, int] = {}
    shared: dict[str, object] | None = None
    for position, (packet, identity) in enumerate(zip(runs, expected, strict=True)):
        if not isinstance(packet, dict) or set(packet) != {"config", "history_bytes"}:
            raise ReferenceError("reference run packet schema mismatch")
        config = packet["config"]
        content = packet["history_bytes"]
        if not isinstance(config, dict) or not isinstance(content, bytes):
            raise ReferenceError("reference run packet types mismatch")
        if set(config) != CONFIG_KEYS:
            raise ReferenceError("reference run config schema mismatch")
        member, arm = identity
        within = within_counts.get(member, 0)
        within_counts[member] = within + 1
        exact = {
            "schema_version": 1,
            "study_id": "feasibility-debt-candidate-screen-v1",
            "run_id": f"s{stage}-m{member:02d}-p{within}-{arm}",
            "stage": stage,
            "member_index": member,
            "arm_id": arm,
            "optimizer_seed": seed,
            "execution_position": position,
            "max_time_seconds": 600.0,
            "max_evals": None,
            "population_size": 8,
            "n_frequencies": 50,
            "allow_cpu": False,
        }
        for key, value in exact.items():
            if key in {
                "schema_version",
                "stage",
                "member_index",
                "optimizer_seed",
                "execution_position",
                "population_size",
                "n_frequencies",
            } and type(config.get(key)) is not int:
                raise ReferenceError("reference frozen integer type mismatch")
            if config.get(key) != value:
                raise ReferenceError("reference frozen run order mismatch")
        topology = config["topology"]
        if (
            not isinstance(topology, str)
            or not topology
            or config["topology_sha256"]
            != hashlib.sha256(topology.encode("utf-8")).hexdigest()
        ):
            raise ReferenceError("reference topology binding mismatch")
        profile = config["arm_profile"]
        package = (
            profile.get("package_closure_sha256")
            if isinstance(profile, dict)
            else None
        )
        if not isinstance(package, str) or profile != _expected_profile(arm, package):
            raise ReferenceError("reference arm profile binding mismatch")
        bindings = {
            key: config[key]
            for key in (
                "panel_sha256",
                "panel_commitment_sha256",
                "split_receipt_sha256",
                "source_lock_sha256",
                "runtime_lock_sha256",
                "revision",
            )
        }
        if shared is None:
            shared = bindings
        elif bindings != shared:
            raise ReferenceError("reference stage bindings differ")
        selection = config["selection_receipt_sha256"]
        if (stage == 1 and (selection is not None or selection_receipt_sha256 is not None)) or (
            stage == 2
            and (
                not isinstance(selection_receipt_sha256, str)
                or selection != selection_receipt_sha256
            )
        ):
            raise ReferenceError("reference selection binding mismatch")
        if identity in result:
            raise ReferenceError("reference duplicate run identity")
        result[identity] = project_history_bytes(content)
    return result


def _kind(value: float) -> str:
    if abs(value) <= TOLERANCE:
        return "tie"
    return "win" if value < -TOLERANCE else "loss"


def _summary(values: Sequence[float]) -> dict[str, object]:
    ordered = sorted(values)
    middle = len(ordered) // 2
    median = (
        ordered[middle]
        if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) / 2.0
    )
    harms = sorted(max(value, 0.0) for value in values)
    point = (len(harms) - 1) * 0.9
    low = math.floor(point)
    high = math.ceil(point)
    labels = [_kind(value) for value in values]
    return {
        "differences": list(values),
        "wins": labels.count("win"),
        "ties": labels.count("tie"),
        "losses": labels.count("loss"),
        "mean_difference": sum(values) / len(values),
        "median_difference": median,
        "p90_harm": harms[low] + (harms[high] - harms[low]) * (point - low),
        "maximum_harm": max(0.0, max(values)),
    }


def reference_stage1(
    runs: Sequence[dict[str, object]], indices: Sequence[int]
) -> dict[str, object]:
    projected = _authenticate_runs(runs, _stage1_order(indices), stage=1)
    rows: list[dict[str, object]] = []
    for challenger in CHALLENGERS:
        differences: list[float] = []
        topology_rows: list[dict[str, object]] = []
        for member in indices:
            control = projected[(member, ARM_ORDER[0])]
            treatment = projected[(member, challenger)]
            if not control["has_feasible"] or not control["has_finite_feasible"]:
                raise ReferenceError("reference control is not finite feasible")
            if not treatment["has_feasible"] or not treatment["has_finite_feasible"]:
                raise ReferenceError("reference challenger is not finite feasible")
            difference = float(treatment["best_feasible_loss"]) - float(
                control["best_feasible_loss"]
            )
            differences.append(difference)
            topology_rows.append({"member_index": member, "difference": difference})
        summary = _summary(differences)
        eligible = bool(
            summary["wins"] >= 3
            and summary["mean_difference"] < 0.0
            and summary["maximum_harm"] <= 0.5
        )
        rows.append(
            {
                "arm_id": challenger,
                "topology_rows": topology_rows,
                **summary,
                "eligible": eligible,
            }
        )
    eligible = [row for row in rows if row["eligible"]]
    finalist = None
    if eligible:
        best = min(float(row["mean_difference"]) for row in eligible)
        finalist = next(
            arm
            for arm in CHALLENGERS
            if any(
                row["arm_id"] == arm
                and abs(float(row["mean_difference"]) - best) <= TOLERANCE
                for row in eligible
            )
        )
    return {
        "challenger_rows": rows,
        "eligible_ids": [
            arm
            for arm in CHALLENGERS
            if any(row["arm_id"] == arm and row["eligible"] for row in rows)
        ],
        "finalist": finalist,
        "action": (
            "advance_selected_finalist_to_stage2"
            if finalist is not None
            else "retain_round1_control_stage1_failed"
        ),
        "stage2_outcome_opened": False,
    }


def _bootstrap(values: Sequence[float]) -> list[float]:
    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    generator = np.random.Generator(np.random.PCG64(20260903))
    samples = generator.choice(array, size=(10_000, len(array)), replace=True)
    bounds = np.percentile(np.mean(samples, axis=1), [2.5, 97.5], method="linear")
    return [float(bounds[0]), float(bounds[1])]


def reference_stage2(
    runs: Sequence[dict[str, object]],
    indices: Sequence[int],
    finalist: str,
    selection_receipt_sha256: str,
) -> dict[str, object]:
    projected = _authenticate_runs(
        runs,
        _stage2_order(indices, finalist),
        stage=2,
        selection_receipt_sha256=selection_receipt_sha256,
    )
    values: list[float] = []
    topology_rows: list[dict[str, object]] = []
    for member in indices:
        control = projected[(member, ARM_ORDER[0])]
        treatment = projected[(member, finalist)]
        if not control["has_feasible"] or not control["has_finite_feasible"]:
            raise ReferenceError("reference Stage-2 control invalid")
        if not treatment["has_feasible"] or not treatment["has_finite_feasible"]:
            raise ReferenceError("reference Stage-2 finalist invalid")
        difference = float(treatment["best_feasible_loss"]) - float(
            control["best_feasible_loss"]
        )
        values.append(difference)
        topology_rows.append({"member_index": member, "difference": difference})
    summary = _summary(values)
    passed = summary["wins"] == 4 and summary["mean_difference"] <= -0.05
    return {
        "finalist": finalist,
        "topology_rows": topology_rows,
        **summary,
        "bootstrap_mean_95": _bootstrap(values),
        "passed": passed,
        "action": (
            "review_selected_bundle_for_round2_candidate_integration"
            if passed
            else "retain_round1_control"
        ),
        "stage2_outcome_opened": True,
    }
