"""Frozen BatchedRestartAdam settings shared by planning and execution."""

from __future__ import annotations

import math


BATCHED_SETTINGS = {
    "learning_rate_low": 0.03,
    "learning_rate_high": 0.15,
    "patience": 600,
    "minimum_improvement": 1e-7,
    "beta1": 0.9,
    "beta2": 0.999,
    "epsilon": 1e-8,
    "gradient_clip_norm": 1.0,
    "restart_noise_scale": 0.35,
    "safety_seconds": 2.0,
    "batch_time_safety_factor": 1.5,
    "batch_time_window": 8,
}


def settings_with_patience(patience: int) -> dict[str, object]:
    if isinstance(patience, bool) or not isinstance(patience, int) or patience < 1:
        raise ValueError("optimizer patience must be a positive integer")
    return {**BATCHED_SETTINGS, "patience": patience}


def validate_batched_settings(settings: object) -> dict[str, object]:
    """Return a canonical full setting map or reject incomplete variants."""
    if not isinstance(settings, dict) or set(settings) != set(BATCHED_SETTINGS):
        raise ValueError("batched optimizer settings must contain the exact schema")
    result = dict(settings)
    integer_fields = ("patience", "batch_time_window")
    for name in integer_fields:
        value = result[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"batched optimizer setting {name!r} must be positive")
    finite_positive = (
        "learning_rate_low",
        "learning_rate_high",
        "minimum_improvement",
        "epsilon",
        "gradient_clip_norm",
        "batch_time_safety_factor",
    )
    for name in finite_positive:
        value = result[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0
        ):
            raise ValueError(
                f"batched optimizer setting {name!r} must be finite and positive"
            )
    finite_nonnegative = ("restart_noise_scale", "safety_seconds")
    for name in finite_nonnegative:
        value = result[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            raise ValueError(
                f"batched optimizer setting {name!r} must be finite and non-negative"
            )
    for name in ("beta1", "beta2"):
        value = result[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0 < float(value) < 1
        ):
            raise ValueError(f"batched optimizer setting {name!r} must be in (0, 1)")
    if float(result["learning_rate_low"]) > float(result["learning_rate_high"]):
        raise ValueError("learning-rate bounds are reversed")
    if float(result["batch_time_safety_factor"]) < 1:
        raise ValueError("batch_time_safety_factor must be at least one")
    return result
