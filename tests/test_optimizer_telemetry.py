from __future__ import annotations

import hashlib
import warnings
import zipfile
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from experiments.uifo_paired.optimizer_telemetry import (
    OPTIMIZER_TELEMETRY_METADATA_SCHEMA,
    OPTIMIZER_TELEMETRY_SCHEMA,
    summarize_optimizer_telemetry,
    validate_optimizer_telemetry,
)
from experiments.uifo_paired.runner import (
    _validate_optimizer_telemetry_against_history,
)


def _valid_arrays() -> dict[str, np.ndarray]:
    values = {
        "batch_index": [0, 0],
        "member_index": [0, 1],
        "eval_count_after_batch": [2, 2],
        "time_seconds": [0.1, 0.1],
        "evaluation_batch_seconds": [0.1, 0.1],
        "finite_loss": [True, True],
        "loss_float_bits": [64, 64],
        "feasible": [False, False],
        "observed_member_improved": [True, True],
        "observed_member_best_loss": [1.0, 2.0],
        "stalled_steps_before": [0, 0],
        "stalled_steps_after": [0, 0],
        "adam_age_before": [0, 0],
        "adam_age_after": [1, 1],
        "learning_rate": [0.03, 0.15],
        "gradient_nonfinite_count": [0, 0],
        "gradient_norm": [1.0, 1.0],
        "gradient_clip_scale": [1.0, 1.0],
        "global_feasible_improvement": [False, False],
        "restart_triggered": [False, False],
        "restart_kind": [-1, -1],
        "restart_round": [-1, -1],
        "restart_noise_scale": [np.nan, np.nan],
        "evaluated_generation": [0, 0],
        "next_generation": [0, 0],
        "update_applied": [True, True],
        "budget_progress_fraction": [0.2, 0.2],
    }
    arrays = {
        name: np.asarray(values[name], dtype=dtype)
        for name, dtype in OPTIMIZER_TELEMETRY_SCHEMA.items()
    }
    arrays["callback_seconds"] = np.asarray(
        [0.01], dtype=OPTIMIZER_TELEMETRY_METADATA_SCHEMA["callback_seconds"]
    )
    return arrays


def _write(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)


def test_optimizer_telemetry_validates_and_summarizes(tmp_path: Path) -> None:
    path = tmp_path / "telemetry.npz"
    _write(path, _valid_arrays())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    arrays = validate_optimizer_telemetry(
        path, expected_sha256=digest, expected_rows=2
    )
    summary = summarize_optimizer_telemetry(arrays)

    assert summary == {
        "rows": 2,
        "batches": 1,
        "restart_rows": 0,
        "restart_batches": 0,
        "post_restart_evaluation_rows": 0,
        "post_restart_evaluation_batches": 0,
        "global_improvement_rows": 0,
        "clipped_gradient_rows": 0,
        "nonfinite_gradient_values": 0,
        "callback_seconds": 0.01,
    }


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("member_index", [0, 0], "member rows"),
        ("gradient_clip_scale", [np.nan, 1.0], "clipping scale"),
        ("adam_age_after", [0, 1], "Adam age did not advance"),
        ("restart_kind", [0, -1], "non-restart metadata"),
    ],
)
def test_optimizer_telemetry_rejects_corrupt_mechanics(
    tmp_path: Path, field: str, replacement: list[object], message: str
) -> None:
    arrays = _valid_arrays()
    arrays[field] = np.asarray(
        replacement, dtype=OPTIMIZER_TELEMETRY_SCHEMA[field]
    )
    path = tmp_path / "corrupt.npz"
    _write(path, arrays)

    with pytest.raises(RuntimeError, match=message):
        validate_optimizer_telemetry(path)


def test_optimizer_telemetry_rejects_checksum_and_schema_corruption(
    tmp_path: Path,
) -> None:
    arrays = _valid_arrays()
    path = tmp_path / "telemetry.npz"
    _write(path, arrays)

    with pytest.raises(RuntimeError, match="digest mismatch"):
        validate_optimizer_telemetry(path, expected_sha256="0" * 64)

    arrays.pop("gradient_norm")
    _write(path, arrays)
    with pytest.raises(RuntimeError, match="archive schema mismatch"):
        validate_optimizer_telemetry(path)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda arrays: arrays.__setitem__(
                "gradient_norm", arrays["gradient_norm"].astype(np.float32)
            ),
            "invalid shape/dtype",
        ),
        (
            lambda arrays: arrays.__setitem__(
                "member_index", arrays["member_index"][:1]
            ),
            "inconsistent lengths",
        ),
        (
            lambda arrays: arrays.__setitem__(
                "unexpected", np.asarray([1, 1], dtype=np.int8)
            ),
            "archive schema mismatch",
        ),
    ],
)
def test_optimizer_telemetry_rejects_shape_dtype_and_extra_fields(
    tmp_path: Path, mutate, message: str
) -> None:
    arrays = _valid_arrays()
    mutate(arrays)
    path = tmp_path / "corrupt.npz"
    _write(path, arrays)

    with pytest.raises(RuntimeError, match=message):
        validate_optimizer_telemetry(path)


def test_optimizer_telemetry_rejects_duplicate_npz_entries(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.npz"
    _write(path, _valid_arrays())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(path, "a") as archive:
            archive.writestr("gradient_norm.npy", archive.read("gradient_norm.npy"))

    with pytest.raises(RuntimeError, match="duplicate entries"):
        validate_optimizer_telemetry(path)


def test_optimizer_telemetry_allows_only_safely_clipped_norm_overflow(
    tmp_path: Path,
) -> None:
    arrays = _valid_arrays()
    arrays["gradient_norm"][0] = np.inf
    arrays["gradient_clip_scale"][0] = 0.0
    path = tmp_path / "overflow.npz"
    _write(path, arrays)

    validate_optimizer_telemetry(path)

    arrays["gradient_clip_scale"][0] = 1.0
    _write(path, arrays)
    with pytest.raises(RuntimeError, match="overflowing norm was not clipped"):
        validate_optimizer_telemetry(path)


def test_history_replay_uses_bound_runtime_loss_precision() -> None:
    boundary_loss = np.float32(0.9999999)
    telemetry = {
        "batch_index": np.asarray([0, 1], dtype=np.int32),
        "member_index": np.asarray([0, 0], dtype=np.int16),
        "evaluated_generation": np.asarray([0, 0], dtype=np.int32),
        "loss_float_bits": np.asarray([32, 32], dtype=np.int16),
        "finite_loss": np.asarray([True, True]),
        "feasible": np.asarray([False, False]),
        "observed_member_improved": np.asarray([True, False]),
        "observed_member_best_loss": np.asarray([1.0, 1.0]),
        "global_feasible_improvement": np.asarray([False, False]),
    }
    history = {
        "loss": np.asarray([1.0, float(boundary_loss)], dtype=np.float64),
        "is_feasible": np.asarray([False, False]),
    }

    _validate_optimizer_telemetry_against_history(
        telemetry, history, minimum_improvement=1e-7
    )

    telemetry["loss_float_bits"] = np.asarray([64, 64], dtype=np.int16)
    with pytest.raises(RuntimeError, match="member improvements mismatch"):
        _validate_optimizer_telemetry_against_history(
            telemetry, history, minimum_improvement=1e-7
        )


def test_optimizer_telemetry_rejects_invalid_callback_duration(
    tmp_path: Path,
) -> None:
    path = tmp_path / "telemetry.npz"
    arrays = _valid_arrays()
    arrays["callback_seconds"] = np.asarray([np.nan], dtype=np.float64)
    _write(path, arrays)

    with pytest.raises(RuntimeError, match="metadata is invalid"):
        validate_optimizer_telemetry(path)
