"""Third, no-project-import Stage-1/Stage-2 projection from sealed NPZ bytes."""

from __future__ import annotations

import hashlib
import io
import math
from typing import Sequence
import zipfile


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
CONTROL = "A_round1_control"
ARM_ORDER = (
    CONTROL,
    "B_round1_warmup",
    "C_v3_random",
    "D_v3_coverage",
)
CHALLENGERS = (
    "B_round1_warmup",
    "C_v3_random",
    "D_v3_coverage",
)
ROWS = (
    ARM_ORDER,
    (ARM_ORDER[1], ARM_ORDER[0], ARM_ORDER[3], ARM_ORDER[2]),
    (ARM_ORDER[2], ARM_ORDER[3], ARM_ORDER[0], ARM_ORDER[1]),
    (ARM_ORDER[3], ARM_ORDER[2], ARM_ORDER[1], ARM_ORDER[0]),
)
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


class DetachedError(RuntimeError):
    pass


def _project(content: bytes) -> dict[str, object]:
    import numpy as np

    expected = {f"{name}.npy" for name in FIELDS} | {
        "initial_params_unbounded.npy",
        "raw_params_unbounded.npy",
    }
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        names = archive.namelist()
        if len(names) != len(expected) or set(names) != expected:
            raise DetachedError("detached NPZ schema mismatch")
    with np.load(io.BytesIO(content), allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in FIELDS}
        initial = np.asarray(archive["initial_params_unbounded"])
        raw = np.asarray(archive["raw_params_unbounded"])
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
    lengths = set()
    if (
        initial.ndim != 2
        or initial.shape[0] != 8
        or raw.shape != initial.shape
        or raw.dtype != initial.dtype
        or not np.all(np.isfinite(initial))
        or not np.all(np.isfinite(raw))
    ):
        raise DetachedError("detached population shape mismatch")
    for name, dtype in dtypes.items():
        array = arrays[name]
        if array.ndim != 1 or array.dtype != dtype:
            raise DetachedError("detached history dtype mismatch")
        lengths.add(len(array))
    if len(lengths) != 1 or next(iter(lengths)) < 1:
        raise DetachedError("detached history length mismatch")
    previous_call = -1
    previous_count = 0
    completed_count = 0
    call_size = 0
    call_count = 0
    previous_time = -math.inf
    feasible_losses: list[float] = []
    has_feasible = False
    for index in range(next(iter(lengths))):
        call = int(arrays["call_index"][index])
        candidate = int(arrays["candidate_index"][index])
        count = int(arrays["eval_count_after_call"][index])
        elapsed = float(arrays["time_seconds"][index])
        if call == previous_call:
            expected_candidate = call_size
            call_size += 1
            if count != call_count:
                raise DetachedError("detached within-call count mismatch")
        elif call == previous_call + 1:
            if index:
                if call_count != completed_count + call_size:
                    raise DetachedError("detached completed-call count mismatch")
                completed_count = call_count
            expected_candidate = 0
            call_size = 1
            call_count = count
            if index and count <= previous_count:
                raise DetachedError("detached count did not advance")
        else:
            raise DetachedError("detached call ledger is incomplete")
        if (
            candidate != expected_candidate
            or count < 1
            or not math.isfinite(elapsed)
            or elapsed < 0
            or elapsed < previous_time
            or count < previous_count
        ):
            raise DetachedError("detached history chronology mismatch")
        feasible = bool(arrays["is_feasible"][index])
        loss = float(arrays["loss"][index])
        if feasible:
            has_feasible = True
            if math.isfinite(loss):
                feasible_losses.append(loss)
        previous_call = call
        previous_count = count
        previous_time = elapsed
    if (
        call_count != completed_count + call_size
        or previous_time <= 0
        or previous_time > 600.0
    ):
        raise DetachedError("detached terminal history mismatch")
    return {
        "has_feasible": has_feasible,
        "has_finite_feasible": bool(feasible_losses),
        "best_feasible_loss": min(feasible_losses) if feasible_losses else None,
    }


def _summary(values: Sequence[float]) -> dict[str, object]:
    ordered = sorted(values)
    middle = len(ordered) // 2
    median = (ordered[middle - 1] + ordered[middle]) / 2.0
    harms = sorted(max(value, 0.0) for value in values)
    point = (len(harms) - 1) * 0.9
    low = math.floor(point)
    high = math.ceil(point)
    labels = [
        "tie" if abs(value) <= 1e-12 else "win" if value < -1e-12 else "loss"
        for value in values
    ]
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


def _bootstrap(values: Sequence[float]) -> list[float]:
    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64(20260903))
    samples = rng.choice(array, size=(10_000, len(array)), replace=True)
    bounds = np.percentile(np.mean(samples, axis=1), [2.5, 97.5], method="linear")
    return [float(bounds[0]), float(bounds[1])]


def _expected_profile(arm: str, package: str) -> dict[str, object]:
    if arm not in PROFILE_ROWS or len(package) != 64 or any(
        token not in "0123456789abcdef" for token in package
    ):
        raise DetachedError("detached package/profile identity mismatch")
    row = dict(PROFILE_ROWS[arm])
    kwargs = {**COMMON_KWARGS, "patience": row["patience"]}
    if arm != CONTROL:
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


def _project_runs(
    runs: Sequence[dict[str, object]],
    expected: Sequence[tuple[int, str]],
    *,
    stage: int,
    selection_receipt_sha256: str | None = None,
) -> dict[tuple[int, str], dict[str, object]]:
    if len(runs) != len(expected):
        raise DetachedError(f"detached Stage-{stage} run count mismatch")
    seed = 20260901 if stage == 1 else 20260902
    projected: dict[tuple[int, str], dict[str, object]] = {}
    within: dict[int, int] = {}
    shared: dict[str, object] | None = None
    for position, (packet, (member, arm)) in enumerate(
        zip(runs, expected, strict=True)
    ):
        if not isinstance(packet, dict) or set(packet) != {"config", "history_bytes"}:
            raise DetachedError("detached packet schema mismatch")
        config = packet["config"]
        content = packet["history_bytes"]
        if (
            not isinstance(config, dict)
            or set(config) != CONFIG_KEYS
            or not isinstance(content, bytes)
        ):
            raise DetachedError("detached packet/config types mismatch")
        slot = within.get(member, 0)
        within[member] = slot + 1
        exact = {
            "schema_version": 1,
            "study_id": "feasibility-debt-candidate-screen-v1",
            "run_id": f"s{stage}-m{member:02d}-p{slot}-{arm}",
            "stage": stage,
            "member_index": member,
            "execution_position": position,
            "arm_id": arm,
            "optimizer_seed": seed,
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
                "execution_position",
                "optimizer_seed",
                "population_size",
                "n_frequencies",
            } and type(config.get(key)) is not int:
                raise DetachedError("detached frozen integer type mismatch")
            if config.get(key) != value:
                raise DetachedError("detached frozen run order mismatch")
        topology = config["topology"]
        if (
            not isinstance(topology, str)
            or not topology
            or config["topology_sha256"]
            != hashlib.sha256(topology.encode("utf-8")).hexdigest()
        ):
            raise DetachedError("detached topology binding mismatch")
        profile = config["arm_profile"]
        package = (
            profile.get("package_closure_sha256")
            if isinstance(profile, dict)
            else None
        )
        if not isinstance(package, str) or profile != _expected_profile(arm, package):
            raise DetachedError("detached arm profile binding mismatch")
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
            raise DetachedError("detached run bindings differ")
        selection = config["selection_receipt_sha256"]
        if (stage == 1 and (selection is not None or selection_receipt_sha256 is not None)) or (
            stage == 2
            and (
                not isinstance(selection_receipt_sha256, str)
                or selection != selection_receipt_sha256
            )
        ):
            raise DetachedError("detached selection binding mismatch")
        identity = (member, arm)
        if identity in projected:
            raise DetachedError("detached duplicate run identity")
        projected[identity] = _project(content)
    return projected


def detached_stage1(
    runs: Sequence[dict[str, object]], indices: Sequence[int]
) -> dict[str, object]:
    if (
        len(indices) != 4
        or list(indices) != sorted(set(indices))
        or any(type(index) is not int or index not in range(8) for index in indices)
    ):
        raise DetachedError("detached Stage-1 identity mismatch")
    expected = [
        (member, arm)
        for row, member in zip(ROWS, indices, strict=True)
        for arm in row
    ]
    projected = _project_runs(runs, expected, stage=1)
    challenger_rows: list[dict[str, object]] = []
    for challenger in CHALLENGERS:
        differences: list[float] = []
        topology_rows: list[dict[str, object]] = []
        for member in indices:
            control = projected[(member, CONTROL)]
            treatment = projected[(member, challenger)]
            if not all(
                row["has_feasible"] and row["has_finite_feasible"]
                for row in (control, treatment)
            ):
                raise DetachedError("detached Stage-1 finite-feasible gate failed")
            difference = float(treatment["best_feasible_loss"]) - float(
                control["best_feasible_loss"]
            )
            differences.append(difference)
            topology_rows.append({"member_index": member, "difference": difference})
        summary = _summary(differences)
        challenger_rows.append(
            {
                "arm_id": challenger,
                "topology_rows": topology_rows,
                **summary,
                "eligible": bool(
                    summary["wins"] >= 3
                    and summary["mean_difference"] < 0.0
                    and summary["maximum_harm"] <= 0.5
                ),
            }
        )
    eligible = [row for row in challenger_rows if row["eligible"]]
    finalist = None
    if eligible:
        minimum = min(float(row["mean_difference"]) for row in eligible)
        finalist = next(
            challenger
            for challenger in CHALLENGERS
            if any(
                row["arm_id"] == challenger
                and abs(float(row["mean_difference"]) - minimum) <= 1e-12
                for row in eligible
            )
        )
    return {
        "challenger_rows": challenger_rows,
        "eligible_ids": [
            challenger
            for challenger in CHALLENGERS
            if any(
                row["arm_id"] == challenger and row["eligible"]
                for row in challenger_rows
            )
        ],
        "finalist": finalist,
        "action": (
            "advance_selected_finalist_to_stage2"
            if finalist is not None
            else "retain_round1_control_stage1_failed"
        ),
        "stage2_outcome_opened": False,
    }


def detached_stage2(
    runs: Sequence[dict[str, object]],
    indices: Sequence[int],
    finalist: str,
    selection_receipt_sha256: str,
) -> dict[str, object]:
    if (
        finalist not in CHALLENGERS
        or len(indices) != 4
        or list(indices) != sorted(set(indices))
        or any(type(index) is not int or index not in range(8) for index in indices)
    ):
        raise DetachedError("detached Stage-2 identity mismatch")
    expected = [
        (member, arm)
        for offset, member in enumerate(indices)
        for arm in ((CONTROL, finalist) if offset % 2 == 0 else (finalist, CONTROL))
    ]
    projected = _project_runs(
        runs,
        expected,
        stage=2,
        selection_receipt_sha256=selection_receipt_sha256,
    )
    differences: list[float] = []
    topology_rows: list[dict[str, object]] = []
    for member in indices:
        control = projected[(member, CONTROL)]
        treatment = projected[(member, finalist)]
        if not all(
            row["has_feasible"] and row["has_finite_feasible"]
            for row in (control, treatment)
        ):
            raise DetachedError("detached Stage-2 finite-feasible gate failed")
        difference = float(treatment["best_feasible_loss"]) - float(
            control["best_feasible_loss"]
        )
        differences.append(difference)
        topology_rows.append({"member_index": member, "difference": difference})
    summary = _summary(differences)
    passed = summary["wins"] == 4 and summary["mean_difference"] <= -0.05
    return {
        "finalist": finalist,
        "topology_rows": topology_rows,
        **summary,
        "bootstrap_mean_95": _bootstrap(differences),
        "passed": passed,
        "action": (
            "review_selected_bundle_for_round2_candidate_integration"
            if passed
            else "retain_round1_control"
        ),
        "stage2_outcome_opened": True,
    }
