"""Pickle-free optimizer telemetry for bounded UIFO mechanism studies."""

from __future__ import annotations

import hashlib
import math
import os
import time
import zipfile
from pathlib import Path


OPTIMIZER_TELEMETRY_MODE = "member-v1"
OPTIMIZER_TELEMETRY_SCHEMA = {
    "batch_index": "int32",
    "member_index": "int16",
    "eval_count_after_batch": "int64",
    "time_seconds": "float64",
    "evaluation_batch_seconds": "float64",
    "finite_loss": "bool",
    "loss_float_bits": "int16",
    "feasible": "bool",
    "observed_member_improved": "bool",
    "observed_member_best_loss": "float64",
    "stalled_steps_before": "int32",
    "stalled_steps_after": "int32",
    "adam_age_before": "int32",
    "adam_age_after": "int32",
    "learning_rate": "float64",
    "gradient_nonfinite_count": "int32",
    "gradient_norm": "float64",
    "gradient_clip_scale": "float64",
    "global_feasible_improvement": "bool",
    "restart_triggered": "bool",
    "restart_kind": "int8",
    "restart_round": "int32",
    "restart_noise_scale": "float64",
    "evaluated_generation": "int32",
    "next_generation": "int32",
    "update_applied": "bool",
    "budget_progress_fraction": "float64",
}
OPTIMIZER_TELEMETRY_METADATA_SCHEMA = {"callback_seconds": "float64"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class OptimizerTelemetryCapture:
    """Capture opt-in per-member events without retaining vectors or parameters."""

    def __init__(self) -> None:
        self._columns = {name: [] for name in OPTIMIZER_TELEMETRY_SCHEMA}
        self.callback_seconds = 0.0

    def __call__(self, event: dict[str, object]) -> None:
        import jax
        import numpy as np

        started = time.perf_counter()
        if set(event) != set(OPTIMIZER_TELEMETRY_SCHEMA):
            raise RuntimeError("optimizer telemetry event schema mismatch")
        arrays = {
            name: np.asarray(jax.device_get(event[name]))
            for name in OPTIMIZER_TELEMETRY_SCHEMA
        }
        lengths = set()
        for name, array in arrays.items():
            if array.ndim != 1:
                raise RuntimeError(
                    f"optimizer telemetry event field is not one-dimensional: {name}"
                )
            lengths.add(int(array.shape[0]))
        if len(lengths) != 1 or next(iter(lengths)) < 1:
            raise RuntimeError("optimizer telemetry event field lengths disagree")
        for name, dtype in OPTIMIZER_TELEMETRY_SCHEMA.items():
            self._columns[name].extend(arrays[name].astype(dtype, copy=False).tolist())
        self.callback_seconds += time.perf_counter() - started

    @property
    def rows(self) -> int:
        return len(self._columns["batch_index"])

    def write(self, path: Path) -> dict[str, object]:
        import numpy as np

        if self.rows < 1:
            raise RuntimeError("optimizer telemetry capture is empty")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp.npz")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                **{
                    name: np.asarray(self._columns[name], dtype=dtype)
                    for name, dtype in OPTIMIZER_TELEMETRY_SCHEMA.items()
                },
                callback_seconds=np.asarray(
                    [self.callback_seconds], dtype=np.float64
                ),
            )
        os.replace(temporary, path)
        arrays = validate_optimizer_telemetry(path, expected_rows=self.rows)
        return summarize_optimizer_telemetry(arrays)


def validate_optimizer_telemetry(
    path: Path,
    *,
    expected_sha256: str | None = None,
    expected_rows: int | None = None,
    expected_population_size: int | None = None,
    expected_patience: int | None = None,
) -> dict[str, object]:
    """Validate a strict telemetry NPZ and return its arrays."""
    import numpy as np

    if not path.is_file():
        raise RuntimeError(f"missing optimizer telemetry artifact: {path}")
    if expected_sha256 is not None and _sha256(path) != expected_sha256:
        raise RuntimeError(f"optimizer telemetry digest mismatch: {path}")
    with zipfile.ZipFile(path) as archive:
        archive_names = archive.namelist()
        if len(archive_names) != len(set(archive_names)):
            raise RuntimeError("optimizer telemetry archive has duplicate entries")
        corrupted = archive.testzip()
        if corrupted is not None:
            raise RuntimeError(
                f"optimizer telemetry archive failed CRC validation: {corrupted}"
            )
        names = set(archive_names)
    expected_names = {
        f"{name}.npy"
        for name in (
            *OPTIMIZER_TELEMETRY_SCHEMA,
            *OPTIMIZER_TELEMETRY_METADATA_SCHEMA,
        )
    }
    if names != expected_names:
        raise RuntimeError("optimizer telemetry archive schema mismatch")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {
            name: np.asarray(archive[name]) for name in OPTIMIZER_TELEMETRY_SCHEMA
        }
        metadata = {
            name: np.asarray(archive[name])
            for name in OPTIMIZER_TELEMETRY_METADATA_SCHEMA
        }

    lengths = set()
    for name, dtype in OPTIMIZER_TELEMETRY_SCHEMA.items():
        array = arrays[name]
        if array.ndim != 1 or array.dtype != np.dtype(dtype):
            raise RuntimeError(
                f"optimizer telemetry field has invalid shape/dtype: {name}"
            )
        lengths.add(int(array.shape[0]))
    if len(lengths) != 1:
        raise RuntimeError("optimizer telemetry arrays have inconsistent lengths")
    rows = next(iter(lengths))
    if rows < 1 or (expected_rows is not None and rows != expected_rows):
        raise RuntimeError("optimizer telemetry row count mismatch")
    for name, dtype in OPTIMIZER_TELEMETRY_METADATA_SCHEMA.items():
        value = metadata[name]
        if value.shape != (1,) or value.dtype != np.dtype(dtype):
            raise RuntimeError(
                f"optimizer telemetry metadata has invalid shape/dtype: {name}"
            )
        if np.any(~np.isfinite(value)) or np.any(value < 0):
            raise RuntimeError(f"optimizer telemetry metadata is invalid: {name}")
        arrays[name] = value

    batch = arrays["batch_index"]
    member = arrays["member_index"]
    if batch[0] != 0 or np.any(np.diff(batch) < 0):
        raise RuntimeError("optimizer telemetry batches are not monotonic from zero")
    unique_batches = np.unique(batch)
    if not np.array_equal(unique_batches, np.arange(unique_batches[-1] + 1)):
        raise RuntimeError("optimizer telemetry batches are not contiguous")
    for batch_index in unique_batches:
        batch_rows = batch == batch_index
        members = member[batch_rows]
        if len(np.unique(members)) != len(members) or not np.array_equal(
            np.sort(members), np.arange(len(members))
        ):
            raise RuntimeError("optimizer telemetry member rows are malformed")
        update_values = np.unique(arrays["update_applied"][batch_rows])
        if len(update_values) != 1:
            raise RuntimeError("optimizer telemetry batch mixes update states")
        if expected_population_size is not None:
            if expected_population_size < 2:
                raise ValueError("expected optimizer population must be at least two")
            is_terminal_partial = (
                batch_index == unique_batches[-1] and not bool(update_values[0])
            )
            valid_size = (
                1 <= len(members) < expected_population_size
                if is_terminal_partial
                else len(members) == expected_population_size
            )
            if not valid_size:
                raise RuntimeError("optimizer telemetry batch population is invalid")
        for name in (
            "eval_count_after_batch",
            "time_seconds",
            "evaluation_batch_seconds",
            "budget_progress_fraction",
        ):
            if len(np.unique(arrays[name][batch_rows])) != 1:
                raise RuntimeError(
                    f"optimizer telemetry batch field is inconsistent: {name}"
                )

    nonnegative_fields = (
        "batch_index",
        "member_index",
        "eval_count_after_batch",
        "time_seconds",
        "evaluation_batch_seconds",
        "loss_float_bits",
        "stalled_steps_before",
        "stalled_steps_after",
        "adam_age_before",
        "adam_age_after",
        "learning_rate",
        "gradient_nonfinite_count",
        "evaluated_generation",
        "next_generation",
        "budget_progress_fraction",
    )
    for name in nonnegative_fields:
        values = arrays[name]
        if np.any(~np.isfinite(values)) or np.any(values < 0):
            raise RuntimeError(f"optimizer telemetry field is invalid: {name}")
    if np.any(~np.isfinite(arrays["gradient_clip_scale"])) or np.any(
        arrays["gradient_clip_scale"] < 0
    ) or np.any(
        arrays["gradient_clip_scale"] > 1
    ):
        raise RuntimeError("optimizer telemetry clipping scale is invalid")
    gradient_norm = arrays["gradient_norm"]
    if np.any(np.isnan(gradient_norm)) or np.any(gradient_norm < 0):
        raise RuntimeError("optimizer telemetry gradient norm is invalid")
    if np.any(np.isposinf(gradient_norm) & (arrays["gradient_clip_scale"] != 0)):
        raise RuntimeError("optimizer telemetry overflowing norm was not clipped")
    observed_best = arrays["observed_member_best_loss"]
    if np.any(np.isnan(observed_best)) or np.any(np.isneginf(observed_best)):
        raise RuntimeError("optimizer telemetry observed member best is invalid")
    loss_float_bits = np.unique(arrays["loss_float_bits"])
    if len(loss_float_bits) != 1 or int(loss_float_bits[0]) not in (32, 64):
        raise RuntimeError("optimizer telemetry loss precision is invalid")
    if np.any(arrays["budget_progress_fraction"] > 1):
        raise RuntimeError("optimizer telemetry budget progress is invalid")
    if np.any(arrays["observed_member_improved"] & ~arrays["finite_loss"]):
        raise RuntimeError("optimizer telemetry has a nonfinite member improvement")
    if np.any(
        arrays["global_feasible_improvement"]
        & (~arrays["finite_loss"] | ~arrays["feasible"])
    ):
        raise RuntimeError("optimizer telemetry global improvement is not feasible")

    for name in (
        "eval_count_after_batch",
        "time_seconds",
        "budget_progress_fraction",
    ):
        per_batch = np.asarray(
            [arrays[name][batch == index][0] for index in unique_batches]
        )
        if np.any(np.diff(per_batch) < 0):
            raise RuntimeError(f"optimizer telemetry field is not monotonic: {name}")
    cumulative_rows = 0
    for batch_index in unique_batches:
        batch_rows = batch == batch_index
        cumulative_rows += int(np.count_nonzero(batch_rows))
        if int(arrays["eval_count_after_batch"][batch_rows][0]) != cumulative_rows:
            raise RuntimeError(
                "optimizer telemetry evaluation count disagrees with rows"
            )

    restarted = arrays["restart_triggered"]
    if np.any(restarted & ~arrays["update_applied"]):
        raise RuntimeError("optimizer telemetry restarts without an applied update")
    if np.any(restarted & (arrays["restart_kind"] < 0)) or np.any(
        restarted & (arrays["restart_kind"] > 1)
    ):
        raise RuntimeError("optimizer telemetry restart kind is invalid")
    if np.any(restarted & (arrays["restart_round"] < 0)):
        raise RuntimeError("optimizer telemetry restart round is invalid")
    if expected_patience is not None:
        if expected_patience < 1:
            raise ValueError("expected optimizer patience must be positive")
        expected_restart = arrays["update_applied"] & (
            arrays["stalled_steps_after"] >= expected_patience
        )
        if not np.array_equal(restarted, expected_restart):
            raise RuntimeError(
                "optimizer telemetry restart mask disagrees with patience"
            )
    if np.any(
        restarted
        & (arrays["next_generation"] != arrays["evaluated_generation"] + 1)
    ):
        raise RuntimeError("optimizer telemetry restart generation is invalid")
    if np.any(restarted & (arrays["adam_age_after"] != 0)):
        raise RuntimeError("optimizer telemetry restart did not reset Adam age")
    exploit_restarts = restarted & (arrays["restart_kind"] == 1)
    fresh_restarts = restarted & (arrays["restart_kind"] == 0)
    if np.any(~np.isfinite(arrays["restart_noise_scale"][exploit_restarts])) or np.any(
        arrays["restart_noise_scale"][exploit_restarts] < 0
    ):
        raise RuntimeError("optimizer telemetry exploit scale is invalid")
    if np.any(~np.isnan(arrays["restart_noise_scale"][fresh_restarts])):
        raise RuntimeError("optimizer telemetry fresh restart has a noise scale")

    not_restarted = ~restarted
    if np.any(not_restarted & (arrays["restart_kind"] != -1)) or np.any(
        not_restarted & (arrays["restart_round"] != -1)
    ):
        raise RuntimeError("optimizer telemetry non-restart metadata is invalid")
    if np.any(
        not_restarted
        & (arrays["next_generation"] != arrays["evaluated_generation"])
    ):
        raise RuntimeError("optimizer telemetry generation changed without restart")
    if np.any(~np.isnan(arrays["restart_noise_scale"][not_restarted])):
        raise RuntimeError("optimizer telemetry non-restart has a noise scale")

    restart_rounds = arrays["restart_round"][restarted]
    if len(restart_rounds):
        unique_restart_rounds = np.unique(restart_rounds)
        if not np.array_equal(
            unique_restart_rounds, np.arange(len(unique_restart_rounds))
        ):
            raise RuntimeError("optimizer telemetry restart rounds are not contiguous")
        for restart_round in unique_restart_rounds:
            round_batches = batch[
                restarted & (arrays["restart_round"] == restart_round)
            ]
            if len(np.unique(round_batches)) != 1:
                raise RuntimeError("optimizer telemetry restart round spans batches")

    updated_without_restart = arrays["update_applied"] & not_restarted
    if np.any(
        arrays["adam_age_after"][updated_without_restart]
        != arrays["adam_age_before"][updated_without_restart] + 1
    ):
        raise RuntimeError("optimizer telemetry Adam age did not advance")
    partial = ~arrays["update_applied"]
    if np.any(arrays["adam_age_after"][partial] != arrays["adam_age_before"][partial]):
        raise RuntimeError("optimizer telemetry partial batch changed Adam age")

    for member_index in np.unique(member):
        member_rows = np.flatnonzero(member == member_index)
        if len(np.unique(arrays["learning_rate"][member_rows])) != 1:
            raise RuntimeError("optimizer telemetry member learning rate changed")
        expected_generation = 0
        expected_age = 0
        expected_stall = 0
        for row_index in member_rows:
            if (
                int(arrays["evaluated_generation"][row_index])
                != expected_generation
                or int(arrays["adam_age_before"][row_index]) != expected_age
                or int(arrays["stalled_steps_before"][row_index])
                != expected_stall
            ):
                raise RuntimeError(
                    "optimizer telemetry member state is not continuous"
                )
            if arrays["update_applied"][row_index]:
                expected_stall_after = (
                    0
                    if arrays["observed_member_improved"][row_index]
                    else expected_stall + 1
                )
            else:
                expected_stall_after = expected_stall
            if (
                int(arrays["stalled_steps_after"][row_index])
                != expected_stall_after
            ):
                raise RuntimeError(
                    "optimizer telemetry stalled-step transition is invalid"
                )
            if arrays["restart_triggered"][row_index]:
                expected_generation += 1
                expected_age = 0
                expected_stall = 0
            else:
                expected_age = int(arrays["adam_age_after"][row_index])
                expected_stall = int(arrays["stalled_steps_after"][row_index])

    partial_batches = np.unique(batch[~arrays["update_applied"]])
    if len(partial_batches) > 1 or (
        len(partial_batches) == 1 and partial_batches[0] != unique_batches[-1]
    ):
        raise RuntimeError("optimizer telemetry partial update is not terminal")
    for batch_index in unique_batches:
        if np.count_nonzero(
            arrays["global_feasible_improvement"] & (batch == batch_index)
        ) > 1:
            raise RuntimeError(
                "optimizer telemetry has multiple global improvements in one batch"
            )
    return arrays


def summarize_optimizer_telemetry(arrays: dict[str, object]) -> dict[str, object]:
    """Return aggregate mechanics evidence safe for a strict run record."""
    import numpy as np

    callback_seconds = float(arrays["callback_seconds"][0])
    if not math.isfinite(callback_seconds) or callback_seconds < 0:
        raise RuntimeError("optimizer telemetry callback duration is invalid")
    rows = len(arrays["batch_index"])
    batch_count = int(np.max(arrays["batch_index"])) + 1
    return {
        "rows": rows,
        "batches": batch_count,
        "restart_rows": int(np.count_nonzero(arrays["restart_triggered"])),
        "restart_batches": int(
            len(np.unique(arrays["batch_index"][arrays["restart_triggered"]]))
        ),
        "post_restart_evaluation_rows": int(
            np.count_nonzero(arrays["evaluated_generation"] > 0)
        ),
        "post_restart_evaluation_batches": int(
            len(
                np.unique(
                    arrays["batch_index"][arrays["evaluated_generation"] > 0]
                )
            )
        ),
        "global_improvement_rows": int(
            np.count_nonzero(arrays["global_feasible_improvement"])
        ),
        "clipped_gradient_rows": int(
            np.count_nonzero(arrays["gradient_clip_scale"] < 1.0)
        ),
        "nonfinite_gradient_values": int(
            np.sum(arrays["gradient_nonfinite_count"], dtype=np.int64)
        ),
        "callback_seconds": float(callback_seconds),
    }
