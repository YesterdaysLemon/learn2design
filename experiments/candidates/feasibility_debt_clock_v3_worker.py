"""Stdlib-only sealed stdout bootstrap for feasibility-debt-clock-v3."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from typing import Any


MAX_ENVELOPE_BYTES = 262_144


def _canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _seal_stdout() -> int:
    result_fd = os.dup(1)
    null_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(null_fd, 1)
    finally:
        os.close(null_fd)
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
    return result_fd


def _write_envelope(result_fd: int, payload: dict[str, Any]) -> None:
    raw = _canonical_json(payload) + b"\n"
    if len(raw) > MAX_ENVELOPE_BYTES:
        raise ValueError("worker envelope exceeds frozen cap")
    view = memoryview(raw)
    while view:
        written = os.write(result_fd, view)
        if written <= 0:
            raise OSError("result descriptor made no progress")
        view = view[written:]


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--transport-probe", action="store_true")
    group.add_argument("--child", action="store_true")
    args = parser.parse_args()

    result_fd = _seal_stdout()
    try:
        if args.transport_probe:
            print("python-noise", flush=True)
            os.write(1, b"fd-noise\n")
            import_noise_module = os.environ.get(
                "FDC_V3_IMPORT_NOISE_MODULE", ""
            )
            if import_noise_module:
                importlib.import_module(import_noise_module)
            payload = {
                "probe": "feasibility-debt-clock-v3",
                "stdout_sealed": True,
            }
        else:
            from experiments.candidates.feasibility_debt_clock_v3_fixture import (
                _worker_projection,
            )

            payload = _worker_projection()
        _write_envelope(result_fd, payload)
        return 0
    except Exception:
        return 2
    finally:
        os.close(result_fd)


if __name__ == "__main__":
    raise SystemExit(main())
