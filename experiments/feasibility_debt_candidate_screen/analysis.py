"""Production history projection and frozen Stage-1/Stage-2 decisions."""

from __future__ import annotations

from dataclasses import dataclass
import io
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

from .contract import (
    ARM_ORDER,
    CHALLENGER_ORDER,
    MAX_TIME_SECONDS,
    N_FREQUENCIES,
    POPULATION_SIZE,
    STAGE1_OPTIMIZER_SEED,
    STAGE2_OPTIMIZER_SEED,
    STUDY_ID,
    arm_spec,
    run_id,
    stage1_order,
    stage2_order,
)


class AnalysisError(RuntimeError):
    """Evidence cannot satisfy the frozen analysis contract."""


HISTORY_FIELDS = (
    "call_index",
    "candidate_index",
    "eval_count_after_call",
    "time_seconds",
    "loss",
    "sensitivity_loss",
    "penalty",
    "is_feasible",
)
TIE_TOLERANCE = 1e-12
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


@dataclass(frozen=True)
class HistoryProjection:
    rows: int
    logged_calls: int
    has_feasible: bool
    has_finite_feasible: bool
    best_feasible_loss: float | None
    evaluation_count: int
    elapsed_seconds: float
    evaluation_rate: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "logged_calls": self.logged_calls,
            "has_feasible": self.has_feasible,
            "has_finite_feasible": self.has_finite_feasible,
            "best_feasible_loss": self.best_feasible_loss,
            "evaluation_count": self.evaluation_count,
            "elapsed_seconds": self.elapsed_seconds,
            "evaluation_rate": self.evaluation_rate,
        }


def _plain_int(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise AnalysisError(f"{label} must be an integer >= {minimum}")
    return value


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnalysisError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise AnalysisError(f"{label} must be finite")
    return result


def project_history(rows: Sequence[dict[str, object]]) -> HistoryProjection:
    """Apply the exact row typing, chronology, and feasibility projection."""
    if not isinstance(rows, (list, tuple)) or not rows:
        raise AnalysisError("history must contain at least one row")
    previous_time = -math.inf
    previous_count = 0
    previous_call = -1
    expected_candidate = 0
    completed_count = 0
    current_call_count = 0
    current_call_size = 0
    feasible_losses: list[float] = []
    any_feasible = False
    for offset, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != set(HISTORY_FIELDS):
            raise AnalysisError(f"history row {offset} schema mismatch")
        call_index = _plain_int(row["call_index"], "call_index")
        candidate_index = _plain_int(row["candidate_index"], "candidate_index")
        evaluation_count = _plain_int(
            row["eval_count_after_call"], "eval_count_after_call", minimum=1
        )
        time_seconds = _finite_number(row["time_seconds"], "time_seconds")
        if time_seconds < 0 or time_seconds < previous_time:
            raise AnalysisError("history time is negative or decreasing")
        if evaluation_count < previous_count:
            raise AnalysisError("history evaluation count decreases")
        if call_index == previous_call:
            expected_candidate += 1
            current_call_size += 1
            if evaluation_count != current_call_count:
                raise AnalysisError(
                    "candidate rows within one call disagree on evaluation count"
                )
        elif call_index == previous_call + 1:
            if offset:
                if current_call_count != completed_count + current_call_size:
                    raise AnalysisError(
                        "completed call evaluation count differs from row count"
                    )
                completed_count = current_call_count
            expected_candidate = 0
            if offset and evaluation_count <= previous_count:
                raise AnalysisError("new call did not advance evaluation count")
            current_call_count = evaluation_count
            current_call_size = 1
        else:
            raise AnalysisError("history call ledger is missing or reordered")
        if candidate_index != expected_candidate:
            raise AnalysisError("history candidate-call order is incomplete")
        if row["is_feasible"] is not True and row["is_feasible"] is not False:
            raise AnalysisError("is_feasible must be a strict Boolean")
        for field in ("sensitivity_loss", "penalty"):
            value = row[field]
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float))
            ):
                raise AnalysisError(f"{field} must be numeric or null")
        loss = row["loss"]
        if loss is not None and (
            isinstance(loss, bool) or not isinstance(loss, (int, float))
        ):
            raise AnalysisError("loss must be numeric or null")
        if row["is_feasible"] is True:
            any_feasible = True
            if loss is not None and math.isfinite(float(loss)):
                feasible_losses.append(float(loss))
        previous_time = time_seconds
        previous_count = evaluation_count
        previous_call = call_index
    if current_call_count != completed_count + current_call_size:
        raise AnalysisError("final call evaluation count differs from row count")
    if previous_time <= 0:
        raise AnalysisError("history elapsed duration must be strictly positive")
    if previous_time > MAX_TIME_SECONDS:
        raise AnalysisError("history exceeds the authenticated Objective budget")
    return HistoryProjection(
        rows=len(rows),
        logged_calls=previous_call + 1,
        has_feasible=any_feasible,
        has_finite_feasible=bool(feasible_losses),
        best_feasible_loss=min(feasible_losses) if feasible_losses else None,
        evaluation_count=previous_count,
        elapsed_seconds=previous_time,
        evaluation_rate=previous_count / previous_time,
    )


def load_history_npz(path: Path) -> list[dict[str, object]]:
    """Load the pickle-free row projection from one authenticated NPZ."""
    return _load_history_npz_source(path)


def load_history_npz_bytes(content: bytes) -> list[dict[str, object]]:
    """Load the same projection directly from sealed archive bytes."""
    return _load_history_npz_source(io.BytesIO(content))


def _load_history_npz_source(source: object) -> list[dict[str, object]]:
    import numpy as np
    import zipfile

    expected_names = {f"{name}.npy" for name in HISTORY_FIELDS} | {
        "initial_params_unbounded.npy",
        "raw_params_unbounded.npy",
    }
    with zipfile.ZipFile(source) as archive:
        names = archive.namelist()
        if len(names) != len(expected_names) or set(names) != expected_names:
            raise AnalysisError("history NPZ member schema mismatch")
    expected_dtypes = {
        "call_index": np.dtype("int32"),
        "candidate_index": np.dtype("int16"),
        "eval_count_after_call": np.dtype("int64"),
        "time_seconds": np.dtype("float64"),
        "loss": np.dtype("float64"),
        "sensitivity_loss": np.dtype("float64"),
        "penalty": np.dtype("float64"),
        "is_feasible": np.dtype("bool"),
    }
    if hasattr(source, "seek"):
        source.seek(0)
    with np.load(source, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in expected_dtypes}
        initial = np.asarray(archive["initial_params_unbounded"])
        raw = np.asarray(archive["raw_params_unbounded"])
    if (
        initial.ndim != 2
        or initial.shape[0] != 8
        or raw.shape != initial.shape
        or raw.dtype != initial.dtype
        or not np.issubdtype(initial.dtype, np.floating)
        or not np.all(np.isfinite(initial))
        or not np.all(np.isfinite(raw))
    ):
        raise AnalysisError("initial population artifact shape mismatch")
    lengths: set[int] = set()
    for name, expected_dtype in expected_dtypes.items():
        array = arrays[name]
        if array.ndim != 1 or array.dtype != expected_dtype:
            raise AnalysisError(f"history NPZ field mismatch: {name}")
        lengths.add(int(array.shape[0]))
    if len(lengths) != 1 or next(iter(lengths)) < 1:
        raise AnalysisError("history NPZ row lengths mismatch")
    result: list[dict[str, object]] = []
    for index in range(next(iter(lengths))):
        def nullable(name: str) -> float | None:
            value = float(arrays[name][index])
            return value if math.isfinite(value) else None

        result.append(
            {
                "call_index": int(arrays["call_index"][index]),
                "candidate_index": int(arrays["candidate_index"][index]),
                "eval_count_after_call": int(
                    arrays["eval_count_after_call"][index]
                ),
                "time_seconds": float(arrays["time_seconds"][index]),
                "loss": nullable("loss"),
                "sensitivity_loss": nullable("sensitivity_loss"),
                "penalty": nullable("penalty"),
                "is_feasible": bool(arrays["is_feasible"][index]),
            }
        )
    return result


def authenticate_run_document(document: dict[str, object]) -> HistoryProjection:
    required = {
        "run_id",
        "config",
        "history_rows",
        "metrics",
        "objective_accounting",
        "runtime",
    }
    if not isinstance(document, dict) or set(document) != required:
        raise AnalysisError("run document schema mismatch")
    config = document["config"]
    if not isinstance(config, dict):
        raise AnalysisError("run config is missing")
    if document["run_id"] != config.get("run_id"):
        raise AnalysisError("run identity mismatch")
    rows = document["history_rows"]
    if not isinstance(rows, list):
        raise AnalysisError("history rows are missing")
    projection = project_history(rows)
    metrics = document["metrics"]
    accounting = document["objective_accounting"]
    runtime = document["runtime"]
    if not isinstance(metrics, dict) or set(metrics) != {
        "has_feasible",
        "has_finite_feasible",
        "best_feasible_loss",
        "evaluation_count",
        "elapsed_seconds",
        "evaluation_rate",
    }:
        raise AnalysisError("record metrics schema mismatch")
    expected_metrics = projection.as_dict()
    expected_metrics.pop("rows")
    expected_metrics.pop("logged_calls")
    for key, expected in expected_metrics.items():
        observed = metrics.get(key)
        if isinstance(expected, float):
            if (
                isinstance(observed, bool)
                or not isinstance(observed, (int, float))
                or not math.isclose(
                    float(observed), expected, rel_tol=1e-12, abs_tol=1e-12
                )
            ):
                raise AnalysisError(f"record metric mismatch: {key}")
        elif type(observed) is not type(expected) or observed != expected:
            raise AnalysisError(f"record metric mismatch: {key}")
    if not isinstance(accounting, dict) or set(accounting) != {
        "eval_count",
        "log_call_count",
    }:
        raise AnalysisError("objective accounting schema mismatch")
    expected_accounting = {
        "eval_count": projection.evaluation_count,
        "log_call_count": projection.logged_calls,
    }
    if any(
        type(accounting.get(key)) is not int or accounting.get(key) != value
        for key, value in expected_accounting.items()
    ):
        raise AnalysisError("objective accounting differs from history")
    if not isinstance(runtime, dict) or set(runtime) != {
        "returncode",
        "timed_out",
        "wall_seconds",
        "stdout_bytes",
        "stderr_bytes",
        "root_pid",
        "parent_pid",
        "process_group_id",
        "start_ticks",
        "executable_sha256",
        "command_line_sha256",
        "timeout_tree_killed",
        "zero_descendants_after_exit",
    }:
        raise AnalysisError("runtime receipt schema mismatch")
    wall = runtime["wall_seconds"]
    if (
        runtime["returncode"] != 0
        or runtime["timed_out"] is not False
        or isinstance(wall, bool)
        or not isinstance(wall, (int, float))
        or not math.isfinite(float(wall))
        or not 0 <= float(wall) <= 720.0
        or runtime["stderr_bytes"] != 0
        or runtime["zero_descendants_after_exit"] is not True
    ):
        raise AnalysisError("run runtime receipt is not valid")
    for key in ("root_pid", "parent_pid", "process_group_id", "start_ticks"):
        _plain_int(runtime[key], f"runtime {key}", minimum=1)
    for key in ("executable_sha256", "command_line_sha256"):
        value = runtime[key]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(token not in "0123456789abcdef" for token in value)
        ):
            raise AnalysisError(f"runtime {key} is not a SHA-256")
    if runtime["timeout_tree_killed"] is not False:
        raise AnalysisError("successful run has a timeout-tree kill")
    return projection


def _validate_stage_documents(
    documents: Sequence[dict[str, object]],
    *,
    stage: int,
    indices: Sequence[int],
    finalist: str | None = None,
    selection_receipt_sha256: str | None = None,
) -> dict[tuple[int, str], HistoryProjection]:
    expected_order = (
        stage1_order(list(indices))
        if stage == 1
        else stage2_order(list(indices), str(finalist))
    )
    if len(documents) != len(expected_order):
        raise AnalysisError(f"Stage {stage} run count mismatch")
    result: dict[tuple[int, str], HistoryProjection] = {}
    expected_seed = STAGE1_OPTIMIZER_SEED if stage == 1 else STAGE2_OPTIMIZER_SEED
    within_counts: dict[int, int] = {}
    shared: dict[str, object] | None = None
    for position, (document, expected) in enumerate(zip(documents, expected_order, strict=True)):
        config = document.get("config") if isinstance(document, dict) else None
        if not isinstance(config, dict):
            raise AnalysisError("run config is missing")
        if set(config) != CONFIG_KEYS:
            raise AnalysisError("run config schema mismatch")
        member_index, arm_id = expected
        within = within_counts.get(member_index, 0)
        within_counts[member_index] = within + 1
        exact = {
            "schema_version": 1,
            "study_id": STUDY_ID,
            "run_id": run_id(stage, member_index, within, arm_id),
            "stage": stage,
            "member_index": member_index,
            "arm_id": arm_id,
            "optimizer_seed": expected_seed,
            "execution_position": position,
            "max_time_seconds": MAX_TIME_SECONDS,
            "max_evals": None,
            "population_size": POPULATION_SIZE,
            "n_frequencies": N_FREQUENCIES,
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
                raise AnalysisError(f"Stage {stage} frozen integer type mismatch: {key}")
            if config.get(key) != value:
                raise AnalysisError(f"Stage {stage} frozen order mismatch: {key}")
        topology = config.get("topology")
        if not isinstance(topology, str) or not topology:
            raise AnalysisError("run topology is invalid")
        from .canonical import sha256_bytes

        if config.get("topology_sha256") != sha256_bytes(topology.encode("utf-8")):
            raise AnalysisError("run topology digest mismatch")
        profile = config.get("arm_profile")
        package_digest = profile.get("package_closure_sha256") if isinstance(profile, dict) else None
        if (
            not isinstance(package_digest, str)
            or profile != arm_spec(arm_id).lock_row(package_digest)
        ):
            raise AnalysisError("run arm profile mismatch")
        binding_keys = (
            "panel_sha256",
            "panel_commitment_sha256",
            "split_receipt_sha256",
            "source_lock_sha256",
            "runtime_lock_sha256",
            "revision",
        )
        current = {key: config.get(key) for key in binding_keys}
        if shared is None:
            shared = current
        elif current != shared:
            raise AnalysisError("stage run bindings are inconsistent")
        selection = config.get("selection_receipt_sha256")
        if (stage == 1 and (selection is not None or selection_receipt_sha256 is not None)) or (
            stage == 2
            and (
                not isinstance(selection_receipt_sha256, str)
                or selection != selection_receipt_sha256
            )
        ):
            raise AnalysisError("stage selection binding mismatch")
        key = (member_index, arm_id)
        if key in result:
            raise AnalysisError("duplicate topology/arm run")
        result[key] = authenticate_run_document(document)
    return result


def classify_difference(value: float) -> str:
    if abs(value) <= TIE_TOLERANCE:
        return "tie"
    return "win" if value < -TIE_TOLERANCE else "loss"


def _difference_summary(values: Sequence[float]) -> dict[str, object]:
    if not values or any(not math.isfinite(value) for value in values):
        raise AnalysisError("paired differences must be finite and non-empty")
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    median = (
        ordered[midpoint]
        if len(ordered) % 2
        else (ordered[midpoint - 1] + ordered[midpoint]) / 2.0
    )
    harms = sorted(max(value, 0.0) for value in values)
    h = (len(harms) - 1) * 0.9
    lower = math.floor(h)
    upper = math.ceil(h)
    p90 = harms[lower] + (harms[upper] - harms[lower]) * (h - lower)
    classes = [classify_difference(value) for value in values]
    return {
        "differences": list(values),
        "wins": classes.count("win"),
        "ties": classes.count("tie"),
        "losses": classes.count("loss"),
        "mean_difference": sum(values) / len(values),
        "median_difference": median,
        "p90_harm": p90,
        "maximum_harm": max(0.0, max(values)),
    }


def bootstrap_mean_interval(values: Sequence[float]) -> list[float]:
    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) < 1 or not np.all(np.isfinite(array)):
        raise AnalysisError("bootstrap values are invalid")
    generator = np.random.Generator(np.random.PCG64(20260903))
    samples = generator.choice(array, size=(10_000, len(array)), replace=True)
    means = np.mean(samples, axis=1)
    bounds = np.percentile(means, [2.5, 97.5], method="linear")
    return [float(bounds[0]), float(bounds[1])]


def select_stage1_finalist(
    documents: Sequence[dict[str, object]], stage1_indices: Sequence[int]
) -> dict[str, object]:
    projections = _validate_stage_documents(
        documents, stage=1, indices=stage1_indices
    )
    control = ARM_ORDER[0]
    challenger_rows: list[dict[str, object]] = []
    for challenger in CHALLENGER_ORDER:
        differences: list[float] = []
        topology_rows: list[dict[str, object]] = []
        for member_index in stage1_indices:
            control_projection = projections[(member_index, control)]
            challenger_projection = projections[(member_index, challenger)]
            if (
                not control_projection.has_feasible
                or not control_projection.has_finite_feasible
                or not challenger_projection.has_feasible
                or not challenger_projection.has_finite_feasible
            ):
                raise AnalysisError("Stage 1 contains a non-finite-feasible run")
            assert control_projection.best_feasible_loss is not None
            assert challenger_projection.best_feasible_loss is not None
            difference = (
                challenger_projection.best_feasible_loss
                - control_projection.best_feasible_loss
            )
            differences.append(difference)
            topology_rows.append(
                {"member_index": member_index, "difference": difference}
            )
        summary = _difference_summary(differences)
        eligible = bool(
            summary["wins"] >= 3
            and summary["mean_difference"] < 0.0
            and summary["maximum_harm"] <= 0.5
        )
        challenger_rows.append(
            {
                "arm_id": challenger,
                "topology_rows": topology_rows,
                **summary,
                "eligible": eligible,
            }
        )
    eligible_rows = [row for row in challenger_rows if row["eligible"]]
    finalist: str | None = None
    if eligible_rows:
        minimum = min(float(row["mean_difference"]) for row in eligible_rows)
        finalist = next(
            challenger
            for challenger in CHALLENGER_ORDER
            if any(
                row["arm_id"] == challenger
                and abs(float(row["mean_difference"]) - minimum) <= TIE_TOLERANCE
                for row in eligible_rows
            )
        )
    return {
        "challenger_rows": challenger_rows,
        "eligible_ids": [
            challenger
            for challenger in CHALLENGER_ORDER
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


def evaluate_stage2(
    documents: Sequence[dict[str, object]],
    stage2_indices: Sequence[int],
    finalist: str,
    selection_receipt_sha256: str,
) -> dict[str, object]:
    projections = _validate_stage_documents(
        documents,
        stage=2,
        indices=stage2_indices,
        finalist=finalist,
        selection_receipt_sha256=selection_receipt_sha256,
    )
    differences: list[float] = []
    topology_rows: list[dict[str, object]] = []
    for member_index in stage2_indices:
        control_projection = projections[(member_index, ARM_ORDER[0])]
        finalist_projection = projections[(member_index, finalist)]
        if (
            not control_projection.has_feasible
            or not control_projection.has_finite_feasible
            or not finalist_projection.has_feasible
            or not finalist_projection.has_finite_feasible
        ):
            raise AnalysisError("Stage 2 contains a non-finite-feasible run")
        assert control_projection.best_feasible_loss is not None
        assert finalist_projection.best_feasible_loss is not None
        difference = (
            finalist_projection.best_feasible_loss
            - control_projection.best_feasible_loss
        )
        differences.append(difference)
        topology_rows.append({"member_index": member_index, "difference": difference})
    summary = _difference_summary(differences)
    passed = summary["wins"] == 4 and summary["mean_difference"] <= -0.05
    return {
        "finalist": finalist,
        "topology_rows": topology_rows,
        **summary,
        "bootstrap_mean_95": bootstrap_mean_interval(differences),
        "passed": passed,
        "action": (
            "review_selected_bundle_for_round2_candidate_integration"
            if passed
            else "retain_round1_control"
        ),
        "stage2_outcome_opened": True,
    }


def arm_descriptive_summary(
    projections: Iterable[HistoryProjection],
) -> dict[str, float]:
    values = list(projections)
    if not values:
        raise AnalysisError("arm summary cannot be empty")
    counts = sorted(value.evaluation_count for value in values)
    rates = sorted(value.evaluation_rate for value in values)

    def median(items: Sequence[float | int]) -> float:
        midpoint = len(items) // 2
        return (
            float(items[midpoint])
            if len(items) % 2
            else (float(items[midpoint - 1]) + float(items[midpoint])) / 2.0
        )

    return {
        "mean_evaluation_count": sum(counts) / len(counts),
        "median_evaluation_count": median(counts),
        "mean_evaluation_rate": sum(rates) / len(rates),
        "median_evaluation_rate": median(rates),
    }
