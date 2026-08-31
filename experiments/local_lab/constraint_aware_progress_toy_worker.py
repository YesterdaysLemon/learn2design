"""Isolated worker for the frozen constrained-progress toy study.

The public entry point launches two fresh outer projections.  Each projection
launches six fresh family/split phases.  The fixture and learner are never run
at import time, and this module intentionally emits only bounded aggregate
evidence.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import inspect
import json
import math
import os
import platform
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Mapping, Sequence


SCHEMA_DOMAIN = b"L2D-constraint-progress-v1/"
TRANSCRIPT_DOMAIN = b"L2D-constraint-progress-transcript-v1"
MAX_PACKET_BYTES = 1_048_576
CHILD_TIMEOUT_SECONDS = 60 * 60
CHILD_JOB_GATE = b"L2D-CONSTRAINT-CHILD-JOB-GATE-V1\n"
WORKER_MODULE = "experiments.local_lab.constraint_aware_progress_toy_worker"
SAFE_ENVIRONMENT = {
    "COMSPEC",
    "LD_LIBRARY_PATH",
    "PATH",
    "PATHEXT",
    "PROGRAMDATA",
    "SYSTEMDRIVE",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "TMPDIR",
    "VIRTUAL_ENV",
    "WINDIR",
}


class _NetworkDisabledSocket(socket.socket):
    def connect(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError("network access is disabled in the local laboratory")

    def connect_ex(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError("network access is disabled in the local laboratory")

    def sendto(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError("network access is disabled in the local laboratory")


def _deny_network(*args, **kwargs):
    del args, kwargs
    raise RuntimeError("network access is disabled in the local laboratory")


def _disable_network() -> None:
    socket.socket = _NetworkDisabledSocket
    socket.create_connection = _deny_network
    socket.getaddrinfo = _deny_network
    socket.gethostbyaddr = _deny_network
    socket.gethostbyname = _deny_network
    socket.gethostbyname_ex = _deny_network
    os.environ.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "JAX_PLATFORMS": "cpu",
            "LEARN2DESIGN_LOCAL_LAB_NETWORK": "disabled",
            "MKL_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "PYTHONHASHSEED": "0",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        }
    )


def _load_runtime() -> None:
    global np, fx
    # Child processes start with ``-S -P``.  Add exactly the repository and the
    # host's single user-package directory without processing .pth files,
    # sitecustomize, usercustomize, PYTHONPATH, or the current working directory.
    import site

    root = str(Path(__file__).parents[2].resolve())
    user_packages = str(Path(site.getusersitepackages()).resolve())
    for entry in (root, user_packages):
        if entry not in sys.path:
            sys.path.append(entry)
    import numpy as np_module
    from experiments.local_lab import constraint_aware_progress_toy as fixture

    np = np_module
    fx = fixture


def _reject_duplicate_pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise fx.ContractError("duplicate-json-key")
        value[key] = item
    return value


def _loads_json(value: bytes | str):
    try:
        return json.loads(value, object_pairs_hook=_reject_duplicate_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise fx.ContractError("json-packet") from error


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _process_executable_path() -> Path:
    if os.name != "nt":
        return Path(sys.executable).resolve()
    import ctypes

    buffer = ctypes.create_unicode_buffer(32768)
    length = ctypes.windll.kernel32.GetModuleFileNameW(None, buffer, len(buffer))
    if length <= 0 or length >= len(buffer):
        raise fx.ContractError("runtime-executable-identity")
    return Path(buffer.value)


def _runtime_identity() -> dict[str, str]:
    import importlib.metadata
    import numpy.random._pcg64 as pcg64_module
    import numpy.random.bit_generator as bit_generator_module

    numpy_init = Path(np.__file__).resolve()
    metadata_path = Path(importlib.metadata.distribution("numpy")._path) / "METADATA"
    identity = {
        "machine": platform.machine(),
        "numpy_init_sha256": _sha256_file(numpy_init),
        "numpy_metadata_sha256": _sha256_file(metadata_path),
        "numpy_version": np.__version__,
        "pcg64_identity": f"{np.random.PCG64.__module__}.{np.random.PCG64.__name__}",
        "pcg64_module_sha256": _sha256_file(Path(pcg64_module.__file__).resolve()),
        "python_architecture": platform.architecture()[0],
        "python_executable_sha256": _sha256_file(_process_executable_path()),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "seed_sequence_identity": (
            f"{np.random.SeedSequence.__module__}."
            f"{np.random.SeedSequence.__name__}"
        ),
        "seed_sequence_module_sha256": _sha256_file(
            Path(bit_generator_module.__file__).resolve()
        ),
    }
    if identity != fx.RUNTIME_IDENTITY:
        raise fx.ContractError("runtime-identity")
    return identity


def _positive_zero(value: float) -> float:
    value = float(value)
    return 0.0 if value == 0.0 else value


def _normalize(value: object) -> object:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise fx.ContractError("nonfinite-canonical-value")
        return _positive_zero(value)
    if isinstance(value, tuple):
        return [_normalize(item) for item in value]
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalize(item) for key, item in value.items()}
    return value


def _line(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            _normalize(dict(value)),
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


class _Root:
    def __init__(self, name: str):
        self._hash = hashlib.sha256()
        self._hash.update(SCHEMA_DOMAIN)
        self._hash.update(name.encode("ascii"))
        self._hash.update(b"\0")

    def update(self, value: Mapping[str, object]) -> None:
        self._hash.update(_line(value))

    def update_bytes(self, value: bytes) -> None:
        self._hash.update(value)

    def hexdigest(self) -> str:
        return self._hash.hexdigest()


def _root_of_roots(name: str, roots: Sequence[str]) -> str:
    if any(type(item) is not str or len(item) != 64 for item in roots):
        raise fx.ContractError("child-root")
    return hashlib.sha256(
        SCHEMA_DOMAIN
        + name.encode("ascii")
        + b"\0"
        + b"".join(item.encode("ascii") + b"\n" for item in roots)
    ).hexdigest()


def _state_hash(record: Mapping[str, object]) -> str:
    return hashlib.sha256(
        SCHEMA_DOMAIN + b"OptimizerState\0" + _line(record)
    ).hexdigest()


def _draw_hash(kind: str, seed: int, batch: int, values) -> str:
    if kind not in {"fresh", "perturb"}:
        raise fx.ContractError("draw-kind")
    array = np.asarray(values)
    if array.shape != (8, 3) or array.dtype != np.float64:
        raise fx.ContractError("draw-shape")
    return hashlib.sha256(
        SCHEMA_DOMAIN
        + f"{kind}-draw".encode("ascii")
        + b"\0"
        + struct.pack("<Q", seed)
        + struct.pack("<i", batch)
        + np.ascontiguousarray(array, dtype="<f8").tobytes(order="C")
    ).hexdigest()


def _center_hash(present: bool, center) -> str:
    if not present:
        return hashlib.sha256(
            b"L2D-constraint-progress-v1/absent-incumbent\0"
        ).hexdigest()
    values = np.asarray([_positive_zero(value) for value in center], dtype="<f8")
    return hashlib.sha256(
        b"L2D-constraint-progress-v1/incumbent-center\0"
        + values.tobytes(order="C")
    ).hexdigest()


def _oracle_reference(a: float, q: float) -> float:
    low = a / 2.0
    high = a if q > 0.0 else min(1.5 * a, 1.99)
    low_derivative = 4.0 * low * (low * low - a * a) + q
    high_derivative = 4.0 * high * (high * high - a * a) + q
    if low_derivative > 0.0 or high_derivative < 0.0:
        raise fx.ContractError("oracle-reference-bracket-sign")
    for _ in range(80):
        middle = (low + high) / 2.0
        derivative = 4.0 * middle * (middle * middle - a * a) + q
        if derivative <= 0.0:
            low = middle
        else:
            high = middle
    return float((low + high) / 2.0)


def _oracle_worlds() -> tuple[dict[str, object], ...]:
    records: list[dict[str, object]] = []
    for family in ("canonical", "aligned", "impossible"):
        for world in range(16):
            bits = [
                (world >> 3) & 1,
                (world >> 2) & 1,
                (world >> 1) & 1,
                world & 1,
            ]
            a = float(0.80 + 0.20 * bits[0])
            b = float(-0.50 + 1.00 * bits[1])
            k = float(0.50 + 0.50 * bits[2])
            t = float(0.10 + 0.06 * bits[3])
            c = float(-0.25 if (bits[0] ^ bits[1]) == 0 else 0.25)
            threshold = float(2.25 if family == "impossible" else 0.0)
            split = "development" if sum(bits) % 2 == 0 else "heldout"
            if family == "impossible":
                reference_x0 = reference_sensitivity = denominator = None
            else:
                q = -t if family == "aligned" else t
                reference_x0 = _oracle_reference(a, q)
                reference_lower = a / 2.0
                reference_upper = a if q > 0.0 else min(1.5 * a, 1.99)
                if not reference_lower <= reference_x0 <= reference_upper:
                    raise fx.ContractError("oracle-reference-bracket")
                reference_sensitivity = float(
                    (reference_x0 * reference_x0 - a * a) ** 2
                    + q * reference_x0
                )
                denominator = float(
                    a**4 + k * b * b + 0.5 * c * c - reference_sensitivity
                )
                if not math.isfinite(denominator) or denominator <= 0.0:
                    raise fx.ContractError("oracle-reference-denominator")
            records.append(
                {
                    "family": family,
                    "world": world,
                    "bits": bits,
                    "split": split,
                    "a": a,
                    "b": b,
                    "k": k,
                    "t": t,
                    "c": c,
                    "threshold": threshold,
                    "reference_x0": reference_x0,
                    "reference_sensitivity": reference_sensitivity,
                    "denominator": denominator,
                }
            )
    return tuple(records)


def _world_replay():
    implementation = tuple(item.to_dict() for item in fx.build_world_records())
    oracle = _oracle_worlds()
    if len(implementation) != 48 or implementation != oracle:
        raise fx.ContractError("family-replay")
    implementation_root = _Root("WorldRecord")
    oracle_root = _Root("WorldRecord")
    for left, right in zip(implementation, oracle, strict=True):
        implementation_root.update(left)
        oracle_root.update(right)
    return implementation, implementation_root.hexdigest(), oracle_root.hexdigest()


def _oracle_transcript(seed: int) -> dict[str, object]:
    generator = np.random.Generator(
        np.random.PCG64(np.random.SeedSequence([20260830, seed]))
    )
    suffix = generator.standard_normal((7, 3), dtype=np.float64)
    fresh = np.empty((64, 8, 3), dtype=np.float64)
    perturb = np.empty_like(fresh)
    pieces = [suffix.reshape(-1)]
    for batch in range(64):
        fresh[batch] = generator.standard_normal((8, 3), dtype=np.float64)
        perturb[batch] = generator.standard_normal((8, 3), dtype=np.float64)
        pieces.extend((fresh[batch].reshape(-1), perturb[batch].reshape(-1)))
    values = np.concatenate(pieces).astype("<f8", copy=False)
    digest = hashlib.sha256(
        TRANSCRIPT_DOMAIN
        + b"\0"
        + struct.pack("<Q", seed)
        + values.tobytes(order="C")
    ).hexdigest()
    return {
        "seed": seed,
        "suffix": suffix,
        "fresh": fresh,
        "perturb": perturb,
        "sha256": digest,
    }


def _transcript_replay():
    primary = tuple(fx.build_transcript(seed) for seed in fx.SEEDS)
    oracle = tuple(_oracle_transcript(seed) for seed in fx.SEEDS)
    for left, right in zip(primary, oracle, strict=True):
        _validate_transcript_hash(left.seed, left.sha256)
        if (
            left.sha256 != right["sha256"]
            or left.suffix.tobytes() != right["suffix"].tobytes()
            or left.fresh.tobytes() != right["fresh"].tobytes()
            or left.perturb.tobytes() != right["perturb"].tobytes()
        ):
            raise fx.ContractError("transcript-replay")
    observed = fx.transcript_root(primary)
    root = hashlib.sha256(
        TRANSCRIPT_DOMAIN
        + b"/root\0"
        + b"".join(item["sha256"].encode("ascii") + b"\n" for item in oracle)
    ).hexdigest()
    if observed != root or root != fx.TRANSCRIPT_ROOT_SHA256:
        raise fx.ContractError("transcript-root")
    return primary, oracle, root


def _sigmoid(values):
    values = np.asarray(values, dtype=np.float64)
    result = np.empty_like(values)
    mask = values >= 0.0
    result[mask] = 1.0 / (1.0 + np.exp(-values[mask]))
    exponential = np.exp(values[~mask])
    result[~mask] = exponential / (1.0 + exponential)
    return result


def _evaluate(world: Mapping[str, object], u) -> dict[str, object]:
    values = np.asarray(u, dtype=np.float64)
    sig = _sigmoid(values)
    x = 4.0 * sig - 2.0
    a = float(world["a"])
    b = float(world["b"])
    k = float(world["k"])
    t = float(world["t"])
    c = float(world["c"])
    threshold = float(world["threshold"])
    q = -t if world["family"] == "aligned" else t
    violation = max(0.0, threshold - float(x[0]))
    penalty = float(0.02 * violation * violation)
    sensitivity = float(
        (float(x[0]) ** 2 - a * a) ** 2
        + q * float(x[0])
        + k * (float(x[1]) - b) ** 2
        + 0.5 * (float(x[2]) - c) ** 2
    )
    loss = float(sensitivity + penalty)
    penalty_dx0 = -0.04 * violation if float(x[0]) < threshold else 0.0
    active = np.asarray(
        [
            4.0 * float(x[0]) * (float(x[0]) ** 2 - a * a)
            + q
            + penalty_dx0,
            2.0 * k * (float(x[1]) - b),
            float(x[2]) - c,
        ],
        dtype=np.float64,
    )
    raw_gradient = active * (4.0 * sig * (1.0 - sig))
    nonfinite = [bool(value) for value in ~np.isfinite(raw_gradient)]
    gradient = np.nan_to_num(raw_gradient, nan=0.0, posinf=0.0, neginf=0.0)
    return {
        "u": [float(value) for value in values],
        "x": [float(value) for value in x],
        "loss": loss,
        "gradient": [float(value) for value in gradient],
        "gradient_nonfinite": nonfinite,
        "canonical_is_feasible": bool(float(x[0]) >= threshold),
        "sensitivity": sensitivity,
        "penalty": penalty,
        "violation": float(violation),
    }


def _progress(mode: str, state: Mapping[str, object], loss: float, decision):
    if mode == "raw":
        improved = (not state["observed"]) or (
            math.isfinite(loss) and loss < float(state["first"]) - 1.0e-7
        )
        candidate = {
            "mode": "raw",
            "observed": True,
            "feasible": False,
            "first": float(loss),
            "second": 0.0,
        }
        return improved, candidate if improved else dict(state)
    candidate = (
        {
            "mode": "lex",
            "observed": True,
            "feasible": True,
            "first": float(decision["sensitivity"]),
            "second": 0.0,
        }
        if decision["is_feasible"]
        else {
            "mode": "lex",
            "observed": True,
            "feasible": False,
            "first": float(decision["penalty"]),
            "second": float(decision["sensitivity"]),
        }
    )
    if not state["observed"]:
        return True, candidate
    if decision["is_feasible"] and not state["feasible"]:
        return True, candidate
    if not decision["is_feasible"] and state["feasible"]:
        return False, dict(state)
    if decision["is_feasible"]:
        improved = float(decision["sensitivity"]) < float(state["first"]) - 1.0e-7
        return improved, candidate if improved else dict(state)
    penalty_improved = float(decision["penalty"]) < float(state["first"]) - 1.0e-7
    tied = abs(float(decision["penalty"]) - float(state["first"])) <= 1.0e-7
    sensitivity_improved = (
        float(decision["sensitivity"]) < float(state["second"]) - 1.0e-7
    )
    improved = penalty_improved or (tied and sensitivity_improved)
    return improved, candidate if improved else dict(state)


def _decision(arm: str, member: int, evaluations):
    if arm in {"protected_raw_progress", "no_restart_comparator"}:
        return "unused", -1, {"is_feasible": False, "penalty": 0.0, "sensitivity": 0.0}
    if arm == "constraint_lexicographic_progress":
        item = evaluations[member]
        return "canonical", -1, {
            "is_feasible": item["canonical_is_feasible"],
            "penalty": item["penalty"],
            "sensitivity": item["sensitivity"],
        }
    if arm == "shuffled_progress_control":
        donor = (member + 1) % 8
        item = evaluations[donor]
        return "cyclic-donor", donor, {
            "is_feasible": item["canonical_is_feasible"],
            "penalty": item["penalty"],
            "sensitivity": item["sensitivity"],
        }
    return "ablated", -1, {"is_feasible": False, "penalty": 0.0, "sensitivity": 0.0}


def _unobserved(mode: str) -> dict[str, object]:
    return {
        "mode": mode,
        "observed": False,
        "feasible": False,
        "first": 0.0,
        "second": 0.0,
    }


class _ReplayOracle:
    def __init__(self, world, transcript, roots):
        self.world = world
        self.transcript = transcript
        self.roots = roots
        self.summary = None

    def consume(self, world_record, primary_transcript, evidence) -> None:
        del primary_transcript
        world = self.world
        if world_record.to_dict() != world:
            raise fx.ContractError("oracle-world-join")
        trajectory = evidence.evaluator.trajectory
        seed = trajectory.seed
        order = trajectory.order
        arm = trajectory.arm
        observations = evidence.collector.observations
        transitions = evidence.collector.transitions
        receipts = evidence.collector.receipts
        if len(observations) != 512 or len(transitions) != 512 or len(receipts) != 64:
            raise fx.ContractError("trajectory-cardinality")

        u = np.zeros((8, 3), dtype=np.float64)
        u[1:] = self.transcript["suffix"]
        m = np.zeros_like(u)
        v = np.zeros_like(u)
        ages = np.zeros(8, dtype=np.int64)
        stalls = np.zeros(8, dtype=np.int64)
        mode = "lex" if arm in fx.ARMS[1:4] else "raw"
        progress = [_unobserved(mode) for _ in range(8)]
        incumbent_present = False
        incumbent_sensitivity = None
        incumbent_batch = -1
        incumbent_member = -1
        incumbent_center = np.zeros(3, dtype=np.float64)
        restart_round = 0
        best_feasible = None
        event = _Root("TrajectoryEvent")
        twin = _Root("TrajectoryTwin")
        feasible_observations = 0
        restart_events = 0
        schema_valid_observations = 0
        replayed_transitions = 0
        replayed_receipts = 0
        reset_checks = 0
        incumbent_tie_checks = 0
        incumbent_state_checks = 0

        for batch in range(64):
            evaluations = [_evaluate(world, u[member]) for member in range(8)]
            candidates = [
                (item["sensitivity"], member)
                for member, item in enumerate(evaluations)
                if item["canonical_is_feasible"] and math.isfinite(item["sensitivity"])
            ]
            feasible_observations += len(candidates)
            for sensitivity, _member in candidates:
                if best_feasible is None or sensitivity < best_feasible:
                    best_feasible = sensitivity
            if candidates:
                candidate_sensitivity, candidate_member = min(candidates)
                if not incumbent_present or candidate_sensitivity < incumbent_sensitivity:
                    incumbent_present = True
                    incumbent_sensitivity = float(candidate_sensitivity)
                    incumbent_batch = batch
                    incumbent_member = candidate_member
                    incumbent_center = np.asarray(
                        evaluations[candidate_member]["u"], dtype=np.float64
                    )

            decisions = []
            expected_observations = []
            for member, evaluation in enumerate(evaluations):
                source, donor, decision = _decision(arm, member, evaluations)
                decisions.append(decision)
                expected = {
                    "family": world["family"],
                    "world": world["world"],
                    "seed": seed,
                    "order": order,
                    "arm": arm,
                    "batch": batch,
                    "member": member,
                    **evaluation,
                    "decision_source": source,
                    "decision_donor_member": donor,
                    "decision_is_feasible": decision["is_feasible"],
                    "decision_penalty": decision["penalty"],
                    "decision_sensitivity": decision["sensitivity"],
                }
                actual = observations[batch * 8 + member].to_dict()
                if actual != expected:
                    raise fx.ContractError("observation-replay")
                schema_valid_observations += 1
                expected_observations.append(expected)
                self.roots["implementation_schema"].update(actual)
                self.roots["oracle_schema"].update(expected)
                event.update(expected)
                normalized = dict(expected)
                del normalized["order"]
                twin.update(normalized)

            before_hashes = []
            for member in range(8):
                before_hashes.append(
                    _state_hash(
                        {
                            "u": [float(value) for value in u[member]],
                            "m": [float(value) for value in m[member]],
                            "v": [float(value) for value in v[member]],
                            "age": int(ages[member]),
                            "stall": int(stalls[member]),
                            "progress": progress[member],
                            "incumbent_present": incumbent_present,
                            "incumbent_sensitivity": incumbent_sensitivity,
                            "incumbent_source_batch": incumbent_batch,
                            "incumbent_source_member": incumbent_member,
                            "incumbent_center": [float(value) for value in incumbent_center],
                            "restart_round": restart_round,
                        }
                    )
                )

            next_progress = []
            next_stalls = np.empty(8, dtype=np.int64)
            for member in range(8):
                improved, next_state = _progress(
                    mode, progress[member], evaluations[member]["loss"], decisions[member]
                )
                next_progress.append(next_state)
                next_stalls[member] = 0 if improved else int(stalls[member]) + 1

            gradients = np.asarray(
                [item["gradient"] for item in evaluations], dtype=np.float64
            )
            norms = np.sqrt(np.sum(gradients * gradients, axis=1))
            clipped = gradients * np.minimum(1.0, 1.0 / (norms + 1.0e-12))[:, None]
            next_ages = ages + 1
            next_m = 0.9 * m + 0.1 * clipped
            next_v = 0.999 * v + 0.001 * clipped * clipped
            mhat = next_m / (1.0 - np.power(0.9, next_ages))[:, None]
            vhat = next_v / (1.0 - np.power(0.999, next_ages))[:, None]
            rates = np.geomspace(0.03, 0.15, 8, dtype=np.float64)
            next_u = u - rates[:, None] * mhat / (np.sqrt(vhat) + 1.0e-8)
            mask_array = next_stalls >= 8
            if arm == "no_restart_comparator":
                mask_array = np.zeros(8, dtype=bool)
            mask = [bool(value) for value in mask_array]
            fresh = self.transcript["fresh"][batch]
            perturb = self.transcript["perturb"][batch]
            kinds = ["none"] * 8
            centers = ["none"] * 8
            round_after = restart_round
            if any(mask):
                fraction = (batch + 1) / 64
                scale = 0.35 * max(0.10, 1.0 - fraction)
                centered = perturb - np.mean(perturb, axis=0, dtype=np.float64)
                normalized = centered / (
                    np.std(perturb, axis=0, ddof=0, dtype=np.float64) + 1.0e-6
                )
                for member, selected in enumerate(mask):
                    if not selected:
                        continue
                    if incumbent_present and (member + restart_round) % 2 == 0:
                        next_u[member] = incumbent_center + scale * normalized[member]
                        kinds[member] = "incumbent"
                        centers[member] = "global-feasible"
                    else:
                        next_u[member] = fresh[member]
                        kinds[member] = "fresh"
                        centers[member] = "fresh"
                    next_m[member] = 0.0
                    next_v[member] = 0.0
                    next_ages[member] = 0
                    next_stalls[member] = 0
                    next_progress[member] = _unobserved(mode)
                round_after += 1

            for member in range(8):
                after_hash = _state_hash(
                    {
                        "u": [float(value) for value in next_u[member]],
                        "m": [float(value) for value in next_m[member]],
                        "v": [float(value) for value in next_v[member]],
                        "age": int(next_ages[member]),
                        "stall": int(next_stalls[member]),
                        "progress": next_progress[member],
                        "incumbent_present": incumbent_present,
                        "incumbent_sensitivity": incumbent_sensitivity,
                        "incumbent_source_batch": incumbent_batch,
                        "incumbent_source_member": incumbent_member,
                        "incumbent_center": [float(value) for value in incumbent_center],
                        "restart_round": round_after,
                    }
                )
                expected_transition = {
                    "family": world["family"],
                    "world": world["world"],
                    "seed": seed,
                    "order": order,
                    "arm": arm,
                    "batch": batch,
                    "member": member,
                    "progress_before": progress[member],
                    "progress_after": next_progress[member],
                    "stall_before": int(stalls[member]),
                    "stall_after": int(next_stalls[member]),
                    "adam_age_before": int(ages[member]),
                    "adam_age_after": int(next_ages[member]),
                    "update_applied": True,
                    "restart_triggered": mask[member],
                    "restart_kind": kinds[member],
                    "restart_round": restart_round if mask[member] else -1,
                    "center_source": centers[member],
                    "state_before_sha256": before_hashes[member],
                    "state_after_sha256": after_hash,
                }
                actual_transition = transitions[batch * 8 + member].to_dict()
                if actual_transition != expected_transition:
                    raise fx.ContractError("transition-replay")
                replayed_transitions += 1
                if mask[member]:
                    reset_checks += int(
                        next_ages[member] == 0
                        and next_stalls[member] == 0
                        and next_progress[member] == _unobserved(mode)
                        and np.all(next_m[member] == 0.0)
                        and np.all(next_v[member] == 0.0)
                    )
                self.roots["implementation_state"].update(actual_transition)
                self.roots["oracle_state"].update(expected_transition)
                event.update(expected_transition)
                normalized_transition = dict(expected_transition)
                del normalized_transition["order"]
                twin.update(normalized_transition)
                restart_events += int(mask[member])

            expected_receipt = {
                "family": world["family"],
                "world": world["world"],
                "seed": seed,
                "order": order,
                "arm": arm,
                "batch": batch,
                "incumbent_present": incumbent_present,
                "incumbent_sensitivity": incumbent_sensitivity,
                "incumbent_source_batch": incumbent_batch,
                "incumbent_source_member": incumbent_member,
                "incumbent_center_sha256": _center_hash(
                    incumbent_present, incumbent_center
                ),
                "restart_round_before": restart_round,
                "restart_round_after": round_after,
                "restart_mask": mask,
                "fresh_draw_sha256": _draw_hash("fresh", seed, batch, fresh),
                "perturb_draw_sha256": _draw_hash("perturb", seed, batch, perturb),
            }
            actual_receipt = receipts[batch].to_dict()
            if actual_receipt != expected_receipt:
                raise fx.ContractError("receipt-replay")
            replayed_receipts += 1
            incumbent_tie_checks += 1
            incumbent_state_checks += 1
            self.roots["implementation_state"].update(actual_receipt)
            self.roots["oracle_state"].update(expected_receipt)
            event.update(expected_receipt)
            normalized_receipt = dict(expected_receipt)
            del normalized_receipt["order"]
            twin.update(normalized_receipt)

            u = np.array(next_u, copy=True)
            m = np.array(next_m, copy=True)
            v = np.array(next_v, copy=True)
            ages = np.array(next_ages, copy=True)
            stalls = np.array(next_stalls, copy=True)
            progress = next_progress
            restart_round = round_after

        if world["family"] == "impossible":
            gap = 1.0
            if best_feasible is not None:
                raise fx.ContractError("false-feasible-join")
        elif best_feasible is None:
            gap = 1.0
        else:
            gap = float(
                (best_feasible - world["reference_sensitivity"])
                / world["denominator"]
            )
        expected_trajectory = {
            "family": world["family"],
            "world": world["world"],
            "seed": seed,
            "order": order,
            "arm": arm,
            "evaluations": 512,
            "transitions": 512,
            "best_feasible_sensitivity": best_feasible,
            "gap": gap,
            "normalized_twin_sha256": twin.hexdigest(),
            "event_root_sha256": event.hexdigest(),
        }
        actual_trajectory = trajectory.to_dict()
        if actual_trajectory != expected_trajectory:
            raise fx.ContractError("trajectory-replay")
        self.roots["implementation_trajectory"].update(actual_trajectory)
        self.roots["oracle_trajectory"].update(expected_trajectory)
        self.summary = {
            "gap": gap,
            "order": order,
            "normalized_twin_sha256": twin.hexdigest(),
            "event_root_sha256": event.hexdigest(),
            "observations": len(observations),
            "transitions": len(transitions),
            "batches": len(receipts),
            "trajectories": 1,
            "feasible_observations": feasible_observations,
            "restart_events": restart_events,
            "schema_valid_observations": schema_valid_observations,
            "replayed_transitions": replayed_transitions,
            "replayed_receipts": replayed_receipts,
            "reset_checks": reset_checks,
            "incumbent_tie_checks": incumbent_tie_checks,
            "incumbent_state_checks": incumbent_state_checks,
            "reference_used": int(
                world["reference_sensitivity"] is not None and best_feasible is not None
            ),
            "false_feasible_joins": int(
                world["family"] == "impossible" and best_feasible is not None
            ),
            "nonunit_gap": int(world["family"] == "impossible" and gap != 1.0),
            "transcript_sha256": self.transcript["sha256"],
        }


def _source_environment() -> dict[str, str]:
    environment = {
        name: value
        for name, value in os.environ.items()
        if name.upper() in SAFE_ENVIRONMENT
        or name.startswith("L2D_")
    }
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "JAX_PLATFORMS": "cpu",
            "LEARN2DESIGN_LOCAL_LAB_NETWORK": "disabled",
            "MKL_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "PYTHONHASHSEED": "0",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        }
    )
    return environment


class _ChildJob:
    """A kill-on-close Windows Job with an observable active-process count."""

    def __init__(self) -> None:
        self.handle = None
        if os.name != "nt":
            return
        import ctypes
        from ctypes import wintypes

        class LARGE_INTEGER(ctypes.Structure):
            _fields_ = [("QuadPart", ctypes.c_longlong)]

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_ulonglong),
                ("WriteOperationCount", ctypes.c_ulonglong),
                ("OtherOperationCount", ctypes.c_ulonglong),
                ("ReadTransferCount", ctypes.c_ulonglong),
                ("WriteTransferCount", ctypes.c_ulonglong),
                ("OtherTransferCount", ctypes.c_ulonglong),
            ]

        class BASIC_LIMIT(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", LARGE_INTEGER),
                ("PerJobUserTimeLimit", LARGE_INTEGER),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class EXTENDED_LIMIT(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", BASIC_LIMIT),
                ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        class BASIC_ACCOUNTING(ctypes.Structure):
            _fields_ = [
                ("TotalUserTime", LARGE_INTEGER),
                ("TotalKernelTime", LARGE_INTEGER),
                ("ThisPeriodTotalUserTime", LARGE_INTEGER),
                ("ThisPeriodTotalKernelTime", LARGE_INTEGER),
                ("TotalPageFaultCount", wintypes.DWORD),
                ("TotalProcesses", wintypes.DWORD),
                ("ActiveProcesses", wintypes.DWORD),
                ("TotalTerminatedProcesses", wintypes.DWORD),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel32.IsProcessInJob.argtypes = [
            wintypes.HANDLE,
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.BOOL),
        ]
        kernel32.QueryInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.c_void_p,
        ]
        self._ctypes = ctypes
        self._wintypes = wintypes
        self._kernel32 = kernel32
        self._accounting_type = BASIC_ACCOUNTING
        handle = kernel32.CreateJobObjectW(None, None)
        if not handle:
            raise fx.ContractError("child-job-create")
        limits = EXTENDED_LIMIT()
        limits.BasicLimitInformation.LimitFlags = 0x00002000
        if not kernel32.SetInformationJobObject(
            handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)
        ):
            kernel32.CloseHandle(handle)
            raise fx.ContractError("child-job-limit")
        self.handle = handle

    def assign(self, process: subprocess.Popen) -> None:
        if self.handle is None:
            return
        if not self._kernel32.AssignProcessToJobObject(
            self.handle, int(process._handle)  # type: ignore[attr-defined]
        ):
            raise fx.ContractError("child-job-assign")
        contained = self._wintypes.BOOL()
        if not self._kernel32.IsProcessInJob(
            int(process._handle),  # type: ignore[attr-defined]
            self.handle,
            self._ctypes.byref(contained),
        ) or not bool(contained.value):
            raise fx.ContractError("child-job-membership")

    def active_processes(self) -> int:
        if self.handle is None:
            return 0
        value = self._accounting_type()
        if not self._kernel32.QueryInformationJobObject(
            self.handle,
            1,
            self._ctypes.byref(value),
            self._ctypes.sizeof(value),
            None,
        ):
            raise fx.ContractError("child-job-query")
        return int(value.ActiveProcesses)

    def close(self) -> None:
        if self.handle is not None:
            self._kernel32.CloseHandle(self.handle)
            self.handle = None


def _terminate_child(process: subprocess.Popen, job: _ChildJob) -> None:
    if os.name == "nt":
        job.close()
    elif process.poll() is None:
        try:
            os.killpg(process.pid, 9)
        except ProcessLookupError:
            pass
    if process.poll() is None:
        process.kill()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _run_child(mode: str, payload: dict[str, object] | None = None):
    command = [
        sys.executable,
        "-S",
        "-P",
        str(Path(__file__).resolve()),
        "--mode",
        mode,
    ]
    encoded = None
    if payload is not None:
        encoded = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode(
            "utf-8"
        )
    if encoded is not None and len(encoded) > MAX_PACKET_BYTES:
        raise fx.ContractError("child-input-cap")
    job = _ChildJob()
    process = None
    writer = None
    writer_errors: list[str] = []
    started = time.monotonic()
    try:
        with tempfile.TemporaryDirectory(prefix="l2d-constraint-child-") as directory:
            root = Path(directory)
            stdout_path = root / "stdout"
            stderr_path = root / "stderr"
            with stdout_path.open("xb") as stdout_handle, stderr_path.open(
                "xb"
            ) as stderr_handle:
                environment = _source_environment()
                environment["L2D_CHILD_JOB_GATE"] = "required"
                process = subprocess.Popen(
                    command,
                    cwd=Path(__file__).parents[2],
                    env=environment,
                    stdin=subprocess.PIPE,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    creationflags=(
                        getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                        if os.name == "nt"
                        else 0
                    ),
                    start_new_session=os.name != "nt",
                )
                # The child blocks at its first main-path operation until this
                # exact private-Job membership check succeeds.  This closes the
                # spawn/assignment window without breakaway flags or system
                # mutation, even when the controller itself runs inside a Job.
                job.assign(process)
                packet = CHILD_JOB_GATE + (encoded or b"")

                def write_packet() -> None:
                    try:
                        assert process is not None and process.stdin is not None
                        process.stdin.write(packet)
                        process.stdin.close()
                    except (BrokenPipeError, OSError, ValueError) as error:
                        writer_errors.append(type(error).__name__)

                writer = threading.Thread(
                    target=write_packet,
                    name="constraint-child-input",
                    daemon=False,
                )
                writer.start()
                while process.poll() is None:
                    if time.monotonic() - started > CHILD_TIMEOUT_SECONDS:
                        _terminate_child(process, job)
                        raise fx.ContractError("child-timeout")
                    if stdout_path.stat().st_size + stderr_path.stat().st_size > MAX_PACKET_BYTES:
                        _terminate_child(process, job)
                        raise fx.ContractError("child-output-cap")
                    time.sleep(0.05)
                writer.join(timeout=10)
                if writer.is_alive():
                    _terminate_child(process, job)
                    raise fx.ContractError("child-input-timeout")
                if writer_errors:
                    raise fx.ContractError("child-input-write")
            stdout = stdout_path.read_bytes()
            stderr = stderr_path.read_bytes()
            if len(stdout) + len(stderr) > MAX_PACKET_BYTES:
                raise fx.ContractError("child-output-cap")
            active = job.active_processes()
            # The direct child has exited.  Any active member is therefore a
            # surviving descendant and is killed when the Job closes.
            surviving_descendants = active
            if process.returncode != 0:
                raise fx.ContractError(
                    "child-failed:"
                    + stderr[-1000:].decode("utf-8", errors="replace")
                )
            if stderr:
                raise fx.ContractError("child-stderr")
            return stdout, surviving_descendants, {
                "stdout_bytes": len(stdout),
                "stderr_bytes": len(stderr),
            }
    finally:
        if process is not None and process.poll() is None:
            _terminate_child(process, job)
        if writer is not None and writer.is_alive():
            writer.join(timeout=10)
        job.close()


def _await_child_job_gate(mode: str) -> None:
    required = mode in {
        "constraint-aware-progress-toy-v1-projection",
        "constraint-aware-progress-toy-v1-phase",
    }
    declared = os.environ.get("L2D_CHILD_JOB_GATE")
    if declared == "required":
        if sys.stdin.buffer.read(len(CHILD_JOB_GATE)) != CHILD_JOB_GATE:
            raise RuntimeError("invalid child Job gate")
        return
    if required:
        raise RuntimeError("missing child Job gate")
    if declared is not None:
        raise RuntimeError("unexpected child Job gate")


def _phase_payload() -> dict[str, object]:
    try:
        encoded = sys.stdin.buffer.read(MAX_PACKET_BYTES + 1)
        if len(encoded) > MAX_PACKET_BYTES:
            raise fx.ContractError("phase-packet-cap")
        payload = _loads_json(encoded)
    except fx.ContractError as error:
        raise fx.ContractError("phase-packet") from error
    if not isinstance(payload, dict) or tuple(payload) != (
        "family",
        "split",
        "contract_sha256",
        "transcript_root_sha256",
        "worlds",
    ):
        raise fx.ContractError("phase-packet-schema")
    return payload


class _ExplodingSourceSentinel:
    def __init__(self, forbidden_split: str) -> None:
        if forbidden_split not in {"development", "heldout"}:
            raise fx.ContractError("source-sentinel-split")
        self.forbidden_split = forbidden_split
        self.attached_source = None

    def attach(self, source) -> None:
        if self.attached_source is not None:
            raise fx.ContractError("duplicate-source-sentinel")
        self.attached_source = source

    def reject(self, attempted_split: str) -> None:
        if attempted_split != self.forbidden_split:
            raise fx.ContractError("source-sentinel-identity")
        raise fx.ContractError("forbidden-source-read")


class _WorldSource:
    """The exercised phase source boundary with an attached exploding sentinel."""

    def __init__(self, family: str, split: str, records, sentinel):
        self.family = family
        self.split = split
        self._records = {(item.family, item.world): item for item in records}
        self.attempted_reads = 0
        self.forbidden_reads = 0
        self._sentinel = sentinel
        sentinel.attach(self)

    @property
    def sentinel_connected(self) -> bool:
        return self._sentinel.attached_source is self

    def get(self, world: int):
        self.attempted_reads += 1
        key = (self.family, world)
        item = self._records.get(key)
        if item is None or item.split != self.split:
            self.forbidden_reads += 1
            attempted_split = (
                "heldout" if self.split == "development" else "development"
            )
            self._sentinel.reject(attempted_split)
        return item


def _phase() -> dict[str, object]:
    _runtime_identity()
    payload = _phase_payload()
    family = payload["family"]
    split = payload["split"]
    contract_sha256 = payload["contract_sha256"]
    transcript_root_sha256 = payload["transcript_root_sha256"]
    raw_worlds = payload["worlds"]
    if family not in fx.FAMILIES or split not in fx.SPLITS:
        raise fx.ContractError("phase-identity")
    if not isinstance(raw_worlds, list) or len(raw_worlds) != 8:
        raise fx.ContractError("phase-world-count")
    if (
        contract_sha256 != _required_environment("L2D_CONTRACT_SHA256", 64)
        or transcript_root_sha256 != fx.TRANSCRIPT_ROOT_SHA256
    ):
        raise fx.ContractError("phase-commitment-root")
    parsed_worlds = tuple(fx.world_from_mapping(item) for item in raw_worlds)
    forbidden_payload_rows = sum(
        item.family != family or item.split != split for item in parsed_worlds
    )
    if forbidden_payload_rows:
        raise fx.ContractError("phase-source-join")
    expected_keys = tuple(
        world for world in range(16) if sum(((world >> bit) & 1) for bit in range(4)) % 2 == (0 if split == "development" else 1)
    )
    if tuple(item.world for item in parsed_worlds) != expected_keys:
        raise fx.ContractError("phase-source-order")
    forbidden_split = "heldout" if split == "development" else "development"
    probe_sentinel = _ExplodingSourceSentinel(forbidden_split)
    probe_source = _WorldSource(family, split, parsed_worlds, probe_sentinel)
    forbidden_key = next(world for world in range(16) if world not in expected_keys)
    try:
        probe_source.get(forbidden_key)
    except fx.ContractError as error:
        if str(error) != "forbidden-source-read":
            raise
    else:
        raise fx.ContractError("source-sentinel-not-reached")
    sentinel = _ExplodingSourceSentinel(forbidden_split)
    source = _WorldSource(family, split, parsed_worlds, sentinel)
    worlds = tuple(source.get(world) for world in expected_keys)

    primary_transcripts, oracle_transcripts, _ = _transcript_replay()
    primary_by_seed = {item.seed: item for item in primary_transcripts}
    oracle_by_seed = {item["seed"]: item for item in oracle_transcripts}
    roots = {
        "implementation_schema": _Root("ObservationRecord"),
        "oracle_schema": _Root("ObservationRecord"),
        "implementation_state": _Root("ChronologyRecord"),
        "oracle_state": _Root("ChronologyRecord"),
        "implementation_trajectory": _Root("TrajectoryRecord"),
        "oracle_trajectory": _Root("TrajectoryRecord"),
    }
    keyed: dict[tuple[int, int, str], dict[str, object]] = {}
    order_mismatches = 0
    observations = 0
    transitions = 0
    batches = 0
    trajectories = 0
    restart_events = 0
    feasible_observations = 0
    references_used = 0
    false_feasible_joins = 0
    nonunit_gaps = 0
    schema_valid_observations = 0
    replayed_transitions = 0
    replayed_receipts = 0
    reset_checks = 0
    incumbent_tie_checks = 0
    incumbent_state_checks = 0
    arm_evaluations = {arm: 0 for arm in fx.ARMS}
    arm_transcript_receipts = {arm: 0 for arm in fx.ARMS}
    for world_record, raw_world in zip(worlds, raw_worlds, strict=True):
        for seed in fx.SEEDS:
            for order in fx.ORDERS:
                arm_order = fx.ARMS if order == "forward" else tuple(reversed(fx.ARMS))
                for arm in arm_order:
                    oracle = _ReplayOracle(raw_world, oracle_by_seed[seed], roots)
                    evidence = fx.execute_trajectory(
                        world_record,
                        primary_by_seed[seed],
                        seed=seed,
                        order=order,
                        arm=arm,
                        oracle=oracle,
                    )
                    summary = oracle.summary
                    if not isinstance(summary, dict):
                        raise fx.ContractError("oracle-summary")
                    restart_events += int(summary["restart_events"])
                    feasible_observations += int(summary["feasible_observations"])
                    observations += int(summary["observations"])
                    transitions += int(summary["transitions"])
                    batches += int(summary["batches"])
                    trajectories += int(summary["trajectories"])
                    references_used += int(summary["reference_used"])
                    false_feasible_joins += int(summary["false_feasible_joins"])
                    nonunit_gaps += int(summary["nonunit_gap"])
                    schema_valid_observations += int(
                        summary["schema_valid_observations"]
                    )
                    replayed_transitions += int(summary["replayed_transitions"])
                    replayed_receipts += int(summary["replayed_receipts"])
                    reset_checks += int(summary["reset_checks"])
                    incumbent_tie_checks += int(summary["incumbent_tie_checks"])
                    incumbent_state_checks += int(summary["incumbent_state_checks"])
                    arm_evaluations[arm] += int(summary["observations"])
                    if summary["transcript_sha256"] != fx.TRANSCRIPT_HASHES[seed]:
                        raise fx.ContractError("trajectory-transcript-join")
                    arm_transcript_receipts[arm] += 1
                    key = (world_record.world, seed, arm)
                    if key in keyed:
                        earlier = keyed[key]
                        if (
                            earlier["order"] != "forward"
                            or summary["order"] != "reverse"
                            or earlier["gap"] != summary["gap"]
                            or earlier["normalized_twin_sha256"]
                            != summary["normalized_twin_sha256"]
                            or earlier["event_root_sha256"]
                            == summary["event_root_sha256"]
                        ):
                            order_mismatches += 1
                    else:
                        keyed[key] = summary

    if order_mismatches:
        raise fx.ContractError("order-twin-mismatch")
    aggregate_rows = []
    aggregate_root = _Root("WorldAggregateRecord")
    for world_record in worlds:
        for arm in fx.ARMS:
            gaps = [float(keyed[(world_record.world, seed, arm)]["gap"]) for seed in fx.SEEDS]
            row = {
                "family": family,
                "world": world_record.world,
                "arm": arm,
                "seed_gaps": gaps,
                "mean_gap": float(sum(gaps) / len(gaps)),
            }
            aggregate_rows.append(row)
            aggregate_root.update(row)
    input_root = _Root("PhaseInput")
    for raw_world in raw_worlds:
        input_root.update(raw_world)
    output_root = aggregate_root.hexdigest()
    receipt = {
        "family": family,
        "split": split,
        "world_keys": list(expected_keys),
        "attempted_reads": source.attempted_reads,
        "forbidden_reads": source.forbidden_reads,
        "forbidden_payload_rows": int(forbidden_payload_rows),
        "sentinel_connected": source.sentinel_connected,
        "input_root_sha256": input_root.hexdigest(),
        "output_root_sha256": output_root,
    }
    return {
        "family": family,
        "split": split,
        "world_aggregates": aggregate_rows,
        "phase_receipt": receipt,
        "observations": observations,
        "transitions": transitions,
        "batches": batches,
        "trajectories": trajectories,
        "feasible_observations": feasible_observations,
        "restart_events": restart_events,
        "references_used": references_used,
        "false_feasible_joins": false_feasible_joins,
        "nonunit_gaps": nonunit_gaps,
        "schema_valid_observations": schema_valid_observations,
        "replayed_transitions": replayed_transitions,
        "replayed_receipts": replayed_receipts,
        "reset_checks": reset_checks,
        "incumbent_tie_checks": incumbent_tie_checks,
        "incumbent_state_checks": incumbent_state_checks,
        "arm_evaluations": arm_evaluations,
        "arm_transcript_receipts": arm_transcript_receipts,
        "order_mismatches": order_mismatches,
        "roots": {name: root.hexdigest() for name, root in roots.items()},
        "source_sentinel_probe_rejected": probe_source.forbidden_reads == 1,
    }


def _parse_phase(value: object) -> dict[str, object]:
    expected = (
        "family",
        "split",
        "world_aggregates",
        "phase_receipt",
        "observations",
        "transitions",
        "batches",
        "trajectories",
        "feasible_observations",
        "restart_events",
        "references_used",
        "false_feasible_joins",
        "nonunit_gaps",
        "schema_valid_observations",
        "replayed_transitions",
        "replayed_receipts",
        "reset_checks",
        "incumbent_tie_checks",
        "incumbent_state_checks",
        "arm_evaluations",
        "arm_transcript_receipts",
        "order_mismatches",
        "roots",
        "source_sentinel_probe_rejected",
    )
    if not isinstance(value, dict) or tuple(value) != expected:
        raise fx.ContractError("phase-result-schema")
    if value["family"] not in fx.FAMILIES or value["split"] not in fx.SPLITS:
        raise fx.ContractError("phase-result-identity")
    for name in (
        "observations",
        "transitions",
        "batches",
        "trajectories",
        "feasible_observations",
        "restart_events",
        "references_used",
        "false_feasible_joins",
        "nonunit_gaps",
        "schema_valid_observations",
        "replayed_transitions",
        "replayed_receipts",
        "reset_checks",
        "incumbent_tie_checks",
        "incumbent_state_checks",
        "order_mismatches",
    ):
        if type(value[name]) is not int or value[name] < 0:
            raise fx.ContractError("phase-result-count")
    if value["source_sentinel_probe_rejected"] is not True:
        raise fx.ContractError("source-sentinel-proof")
    if (
        not isinstance(value["world_aggregates"], list)
        or len(value["world_aggregates"]) != 40
        or not isinstance(value["phase_receipt"], dict)
        or tuple(value["phase_receipt"])
        != (
            "family",
            "split",
            "world_keys",
            "attempted_reads",
            "forbidden_reads",
            "forbidden_payload_rows",
            "sentinel_connected",
            "input_root_sha256",
            "output_root_sha256",
        )
    ):
        raise fx.ContractError("phase-result-evidence")
    for name in ("arm_evaluations", "arm_transcript_receipts"):
        arm_counts = value[name]
        if (
            not isinstance(arm_counts, dict)
            or tuple(arm_counts) != fx.ARMS
            or any(type(item) is not int or item < 0 for item in arm_counts.values())
        ):
            raise fx.ContractError("phase-result-arm-counts")
    roots = value["roots"]
    if (
        not isinstance(roots, dict)
        or tuple(roots)
        != (
            "implementation_schema",
            "oracle_schema",
            "implementation_state",
            "oracle_state",
            "implementation_trajectory",
            "oracle_trajectory",
        )
        or any(
            type(item) is not str
            or len(item) != 64
            or any(character not in "0123456789abcdef" for character in item)
            for item in roots.values()
        )
    ):
        raise fx.ContractError("phase-result-roots")
    return value


def _intervention_whitelist_rows():
    expected = {
        "protected_raw_progress": ("raw", "unused", True),
        "constraint_lexicographic_progress": ("lex", "canonical", True),
        "shuffled_progress_control": ("lex", "cyclic-donor", True),
        "ablated_progress_control": ("lex", "ablated", True),
        "no_restart_comparator": ("raw", "unused", False),
    }
    allowed_packet_fields = (
        "u",
        "learning_rates",
        "m",
        "v",
        "ages",
        "stalls",
        "progress",
        "losses",
        "gradients",
        "decisions",
        "incumbent_present",
        "incumbent_center",
        "restart_round",
        "budget_fraction",
    )
    if tuple(fx.OptimizerPacket.__dataclass_fields__) != allowed_packet_fields:
        raise fx.ContractError("optimizer-packet-whitelist")

    # The production adapter may branch on an arm only through the four-field
    # InterventionConfig.  Direct `self.arm` comparisons would create an
    # undeclared static difference and fail this AST audit.
    adapter_source = inspect.getsource(fx.OptimizerAdapter)
    tree = ast.parse(adapter_source)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
            and node.attr == "arm"
            and not isinstance(getattr(node, "ctx", None), ast.Store)
        ):
            raise fx.ContractError("optimizer-static-difference")
    prepare = next(
        node
        for node in tree.body[0].body
        if isinstance(node, ast.FunctionDef) and node.name == "prepare"
    )
    packet_reads = {
        node.attr
        for node in ast.walk(prepare)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "packet"
        and isinstance(node.ctx, ast.Load)
    }
    if packet_reads & {"incumbent_present", "incumbent_center", "restart_round"}:
        raise fx.ContractError("pre-receipt-center-consumption")

    implementation_rows = []
    oracle_rows = []
    for arm in fx.ARMS:
        config = fx.intervention_config(arm)
        implementation = {
            "arm_id": config.arm_id,
            "progress_comparator": config.progress_comparator,
            "tuple_adapter": config.tuple_adapter,
            "restart_enabled": config.restart_enabled,
        }
        comparator, tuple_adapter, restart_enabled = expected[arm]
        oracle = {
            "arm_id": arm,
            "progress_comparator": comparator,
            "tuple_adapter": tuple_adapter,
            "restart_enabled": restart_enabled,
        }
        if implementation != oracle:
            raise fx.ContractError("optimizer-difference-whitelist")
        implementation_rows.append(implementation)
        oracle_rows.append(oracle)
    return implementation_rows, oracle_rows


def _sentinel_and_capabilities():
    consumed_records = fx.progress_sentinel_records()
    outcomes = (
        tuple(item.canonical_decision for item in consumed_records),
        tuple(item.donor_decision for item in consumed_records),
        tuple(item.ablated_decision for item in consumed_records),
    )
    expected_outcomes = ((True, True), (False, False), (True, False))
    if outcomes != expected_outcomes:
        raise fx.ContractError("sentinel-result")
    stored = {
        "mode": "lex",
        "observed": True,
        "feasible": False,
        "first": 1.0,
        "second": 5.0,
    }
    canonical = (
        {"is_feasible": False, "penalty": 0.9, "sensitivity": 6.0},
        {"is_feasible": False, "penalty": 0.8, "sensitivity": 7.0},
    )
    donor = (
        {"is_feasible": False, "penalty": 1.1, "sensitivity": 0.0},
        {"is_feasible": False, "penalty": 1.2, "sensitivity": 0.0},
    )
    ablated = (
        {"is_feasible": False, "penalty": 0.0, "sensitivity": 0.0},
        {"is_feasible": False, "penalty": 0.0, "sensitivity": 0.0},
    )
    implementation_root = _Root("InterventionRecord")
    oracle_root = _Root("InterventionRecord")
    for ordinal in range(2):
        implementation_row = consumed_records[ordinal].to_dict()
        oracle_row = {
            "sentinel_id": "progress-consumer-v1",
            "ordinal": ordinal,
            "member": 0,
            "stored": stored,
            "canonical_tuple": canonical[ordinal],
            "donor_member": 1,
            "donor_tuple": donor[ordinal],
            "ablated_tuple": ablated[ordinal],
            "canonical_decision": expected_outcomes[0][ordinal],
            "donor_decision": expected_outcomes[1][ordinal],
            "ablated_decision": expected_outcomes[2][ordinal],
        }
        if implementation_row != oracle_row:
            raise fx.ContractError("sentinel-record-replay")
        implementation_root.update(implementation_row)
        oracle_root.update(oracle_row)
    receipts = fx.capability_attack_receipts()
    if len(receipts) != 22:
        raise fx.ContractError("capability-count")
    for receipt, attack_id in zip(receipts, fx.CAPABILITY_ATTACKS, strict=True):
        row = receipt.to_dict()
        if (
            row["attack_id"] != attack_id
            or row["injection_path"] != fx.CAPABILITY_PATHS[attack_id]
            or row["rejection_code"] != "capability-denied"
            or row["consumer_reached"] is not True
            or row["state_mutations"] != 0
        ):
            raise fx.ContractError("capability-receipt")
        implementation_root.update(row)
        oracle_root.update(
            {
                "attack_id": attack_id,
                "injection_path": fx.CAPABILITY_PATHS[attack_id],
                "rejection_code": "capability-denied",
                "consumer_reached": True,
                "state_mutations": 0,
            }
        )
    implementation_rows, oracle_rows = _intervention_whitelist_rows()
    if implementation_rows != oracle_rows:
        raise fx.ContractError("optimizer-difference-whitelist")
    return outcomes, receipts, implementation_root.hexdigest(), oracle_root.hexdigest()


def _minimal_envelope():
    # Reconstruct the valid attack envelope without any production fixture
    # constructor, evaluator, progress, optimizer-update, or state-hash helper.
    # The worker-oracle paths used here are independent; fixture functions
    # below are only the parser/consumer boundary under attack.
    world = _oracle_worlds()[0]
    transcript = _oracle_transcript(2026083001)
    u = np.zeros((8, 3), dtype=np.float64)
    u[1:] = transcript["suffix"]
    evaluations = [_evaluate(world, u[member]) for member in range(8)]
    candidates = [
        (float(item["sensitivity"]), member)
        for member, item in enumerate(evaluations)
        if item["canonical_is_feasible"]
    ]
    if not candidates:
        raise fx.ContractError("attack-envelope-incumbent")
    incumbent_sensitivity, incumbent_member = min(candidates)
    incumbent_center = [float(value) for value in u[incumbent_member]]
    observations = []
    for member, evaluation in enumerate(evaluations):
        observations.append(
            {
                "family": "canonical",
                "world": 0,
                "seed": 2026083001,
                "order": "forward",
                "arm": "constraint_lexicographic_progress",
                "batch": 0,
                "member": member,
                **evaluation,
                "decision_source": "canonical",
                "decision_donor_member": -1,
                "decision_is_feasible": evaluation["canonical_is_feasible"],
                "decision_penalty": evaluation["penalty"],
                "decision_sensitivity": evaluation["sensitivity"],
            }
        )
    progress_before = _unobserved("lex")
    decision = {
        "is_feasible": evaluations[0]["canonical_is_feasible"],
        "penalty": evaluations[0]["penalty"],
        "sensitivity": evaluations[0]["sensitivity"],
    }
    improved, progress_after = _progress(
        "lex", progress_before, float(evaluations[0]["loss"]), decision
    )
    if not improved:
        raise fx.ContractError("attack-envelope-progress")
    gradients = np.asarray(
        [item["gradient"] for item in evaluations], dtype=np.float64
    )
    norms = np.sqrt(np.sum(gradients * gradients, axis=1))
    clipped = gradients * np.minimum(1.0, 1.0 / (norms + 1.0e-12))[:, None]
    next_m = 0.1 * clipped
    next_v = 0.001 * clipped * clipped
    next_ages = np.ones(8, dtype=np.int64)
    mhat = next_m / (1.0 - np.power(0.9, next_ages))[:, None]
    vhat = next_v / (1.0 - np.power(0.999, next_ages))[:, None]
    rates = np.geomspace(0.03, 0.15, 8, dtype=np.float64)
    next_u = u - rates[:, None] * mhat / (np.sqrt(vhat) + 1.0e-8)
    state_before = _state_hash(
        {
            "u": [float(value) for value in u[0]],
            "m": [0.0, 0.0, 0.0],
            "v": [0.0, 0.0, 0.0],
            "age": 0,
            "stall": 0,
            "progress": progress_before,
            "incumbent_present": True,
            "incumbent_sensitivity": incumbent_sensitivity,
            "incumbent_source_batch": 0,
            "incumbent_source_member": incumbent_member,
            "incumbent_center": incumbent_center,
            "restart_round": 0,
        }
    )
    state_after = _state_hash(
        {
            "u": [float(value) for value in next_u[0]],
            "m": [float(value) for value in next_m[0]],
            "v": [float(value) for value in next_v[0]],
            "age": 1,
            "stall": 0,
            "progress": progress_after,
            "incumbent_present": True,
            "incumbent_sensitivity": incumbent_sensitivity,
            "incumbent_source_batch": 0,
            "incumbent_source_member": incumbent_member,
            "incumbent_center": incumbent_center,
            "restart_round": 0,
        }
    )
    transition = {
        "family": "canonical",
        "world": 0,
        "seed": 2026083001,
        "order": "forward",
        "arm": "constraint_lexicographic_progress",
        "batch": 0,
        "member": 0,
        "progress_before": progress_before,
        "progress_after": progress_after,
        "stall_before": 0,
        "stall_after": 0,
        "adam_age_before": 0,
        "adam_age_after": 1,
        "update_applied": True,
        "restart_triggered": False,
        "restart_kind": "none",
        "restart_round": -1,
        "center_source": "none",
        "state_before_sha256": state_before,
        "state_after_sha256": state_after,
    }
    for observation in observations:
        fx.validate_observation_mapping(observation)
    fx.validate_batch_identity(
        observations,
        family="canonical",
        world=0,
        seed=fx.SEEDS[0],
        order="forward",
        arm="constraint_lexicographic_progress",
        batch=0,
    )
    fx.validate_transition_mapping(
        transition, expected_seed=fx.SEEDS[0], expected_order="forward"
    )
    return observations, transition


def _validate_result_top(value: Mapping[str, object]) -> None:
    if tuple(value) != (
        "study_id",
        "plan_revision",
        "study_revision",
        "contract_sha256",
        "transcript_root_sha256",
        "status",
        "action",
        "world_aggregates",
        "cases",
    ):
        raise fx.ContractError("result-schema")


def _validate_transcript_hash(seed: int, value: str) -> None:
    if seed not in fx.SEEDS or value != fx.TRANSCRIPT_HASHES[seed]:
        raise fx.ContractError("transcript-hash")


ATTACK_MATRIX = (
    ("nan-loss", "observation.loss", "nonfinite-loss"),
    ("gradient-dtype", "observation.gradient", "gradient-dtype"),
    ("gradient-shape", "observation.gradient", "gradient-shape"),
    ("feasible-type", "observation.canonical_is_feasible", "feasible-type"),
    ("negative-penalty", "observation.decision_penalty", "negative-penalty"),
    ("duplicate-observation", "observation.identity", "duplicate-key"),
    ("missing-member", "observation.identity", "missing-member"),
    ("wrong-arm", "observation.arm", "arm-identity"),
    ("cross-seed", "transition.seed", "cross-seed-join"),
    ("cross-order", "transition.order", "cross-order-join"),
    ("extra-result-field", "result", "result-schema"),
    ("transcript-hash", "transcript.sha256", "transcript-hash"),
)


class _AttackHarness:
    """Instrumented real-consumer boundary for one reconstructed attack."""

    def __init__(self) -> None:
        observations, transition = _minimal_envelope()
        self._observations = copy.deepcopy(observations)
        self._transition = copy.deepcopy(transition)
        self._consumer_entries = 0
        self._candidate_kind: str | None = None
        self._candidate = None

    @property
    def consumer_entries(self) -> int:
        return self._consumer_entries

    def state_digest(self) -> str:
        digest = hashlib.sha256(b"L2D-constraint-progress-v1/attack-state\0")

        def update(value) -> None:
            if value is None:
                digest.update(b"N")
            elif type(value) is bool:
                digest.update(b"B1" if value else b"B0")
            elif type(value) is int:
                encoded = str(value).encode("ascii")
                digest.update(b"I" + len(encoded).to_bytes(4, "little") + encoded)
            elif type(value) is float:
                digest.update(b"F" + struct.pack("<d", value))
            elif type(value) is str:
                encoded = value.encode("utf-8")
                digest.update(b"S" + len(encoded).to_bytes(4, "little") + encoded)
            elif isinstance(value, list):
                digest.update(b"L" + len(value).to_bytes(4, "little"))
                for item in value:
                    update(item)
            elif isinstance(value, dict):
                digest.update(b"D" + len(value).to_bytes(4, "little"))
                for key, item in value.items():
                    update(key)
                    update(item)
            else:
                raise fx.ContractError("attack-state-type")

        update(
            {
                "observations": self._observations,
                "transition": self._transition,
                "candidate_kind": self._candidate_kind,
                "candidate": self._candidate,
            }
        )
        return digest.hexdigest()

    def observation(self) -> dict[str, object]:
        self._candidate_kind = "observation"
        self._candidate = copy.deepcopy(self._observations[0])
        return self._candidate

    def observations(self) -> list[dict[str, object]]:
        self._candidate_kind = "batch"
        self._candidate = copy.deepcopy(self._observations)
        return self._candidate

    def transition(self) -> dict[str, object]:
        self._candidate_kind = "transition"
        self._candidate = copy.deepcopy(self._transition)
        return self._candidate

    def result(self, value: dict[str, object]) -> None:
        self._candidate_kind = "result"
        self._candidate = copy.deepcopy(value)

    def transcript_hash(self, seed: int, value: str) -> None:
        self._candidate_kind = "transcript"
        self._candidate = {"seed": seed, "value": value}

    def _enter(self) -> None:
        self._consumer_entries += 1

    def _validate_envelope(self) -> None:
        for observation in self._observations:
            fx.validate_observation_mapping(observation)
        fx.validate_batch_identity(
            self._observations,
            family="canonical",
            world=0,
            seed=fx.SEEDS[0],
            order="forward",
            arm="constraint_lexicographic_progress",
            batch=0,
        )
        fx.validate_transition_mapping(
            self._transition,
            expected_seed=fx.SEEDS[0],
            expected_order="forward",
        )

    def _selected(self, kind: str):
        if self._candidate_kind != kind:
            raise fx.ContractError("attack-candidate-kind")
        return self._candidate

    def consume_observation(self) -> None:
        self._enter()
        fx.validate_observation_mapping(self._selected("observation"))

    def consume_batch(self) -> None:
        self._enter()
        fx.validate_batch_identity(
            self._selected("batch"),
            family="canonical",
            world=0,
            seed=fx.SEEDS[0],
            order="forward",
            arm="constraint_lexicographic_progress",
            batch=0,
        )

    def consume_transition(self) -> None:
        self._enter()
        fx.validate_transition_mapping(
            self._selected("transition"),
            expected_seed=fx.SEEDS[0],
            expected_order="forward",
        )

    def consume_result(self) -> None:
        self._enter()
        self._validate_envelope()
        _validate_result_top(self._selected("result"))

    def consume_transcript_hash(self) -> None:
        self._enter()
        self._validate_envelope()
        candidate = self._selected("transcript")
        if not isinstance(candidate, dict):
            raise fx.ContractError("attack-candidate-type")
        _validate_transcript_hash(candidate["seed"], candidate["value"])


def _malformed_attacks():
    implementation_root = _Root("AttackReceipt")
    oracle_root = _Root("AttackReceipt")
    receipts = []
    if tuple(fx.MALFORMED_ATTACKS) != ATTACK_MATRIX:
        raise fx.ContractError("attack-contract-drift")
    for attack_id, injection_path, expected_code in ATTACK_MATRIX:
        harness = _AttackHarness()
        if attack_id in {
            "nan-loss",
            "gradient-dtype",
            "gradient-shape",
            "feasible-type",
            "negative-penalty",
            "wrong-arm",
        }:
            observation = harness.observation()
            if attack_id == "nan-loss":
                observation["loss"] = float("nan")
            elif attack_id == "gradient-dtype":
                observation["gradient"] = ["0", "0", "0"]
            elif attack_id == "gradient-shape":
                observation["gradient"] = [0.0, 0.0]
            elif attack_id == "feasible-type":
                observation["canonical_is_feasible"] = 1
            elif attack_id == "negative-penalty":
                observation["decision_penalty"] = -1.0e-9
            else:
                observation["arm"] = "unknown-arm"
            consumer = harness.consume_observation
        elif attack_id in {"duplicate-observation", "missing-member"}:
            observations = harness.observations()
            if attack_id == "duplicate-observation":
                observations.insert(1, dict(observations[0]))
            else:
                observations.pop()
            consumer = harness.consume_batch
        elif attack_id in {"cross-seed", "cross-order"}:
            transition = harness.transition()
            if attack_id == "cross-seed":
                transition["seed"] = fx.SEEDS[1]
            else:
                transition["order"] = "reverse"
            consumer = harness.consume_transition
        elif attack_id == "extra-result-field":
            harness.result(
                {
                    "study_id": fx.STUDY_ID,
                    "plan_revision": fx.PLAN_REVISION,
                    "study_revision": "0" * 40,
                    "contract_sha256": "0" * 64,
                    "transcript_root_sha256": fx.TRANSCRIPT_ROOT_SHA256,
                    "status": "passed",
                    "action": "x",
                    "world_aggregates": [],
                    "cases": [],
                    "unexpected": True,
                }
            )
            consumer = harness.consume_result
        else:
            original = fx.TRANSCRIPT_HASHES[fx.SEEDS[0]]
            changed = ("0" if original[0] != "0" else "1") + original[1:]
            harness.transcript_hash(fx.SEEDS[0], changed)
            consumer = harness.consume_transcript_hash
        state_before = harness.state_digest()
        try:
            consumer()
        except fx.ContractError as error:
            observed_code = str(error)
            if observed_code != expected_code:
                raise
        else:
            raise fx.ContractError("malformed-attack-not-rejected")
        state_after = harness.state_digest()
        implementation_receipt = {
            "attack_id": attack_id,
            "injection_path": injection_path,
            "rejection_code": observed_code,
            "consumer_reached": harness.consumer_entries == 1,
            "state_mutations": int(state_before != state_after),
        }
        oracle_receipt = {
            "attack_id": attack_id,
            "injection_path": injection_path,
            "rejection_code": expected_code,
            "consumer_reached": True,
            "state_mutations": 0,
        }
        if implementation_receipt != oracle_receipt:
            raise fx.ContractError("attack-receipt-replay")
        receipts.append(implementation_receipt)
        implementation_root.update(implementation_receipt)
        oracle_root.update(oracle_receipt)
    return receipts, implementation_root.hexdigest(), oracle_root.hexdigest()


def _case(case_id: str, passed: bool, metrics: dict[str, object]) -> dict[str, object]:
    return {"case_id": case_id, "passed": bool(passed), "metrics": metrics}


def _rows_for(rows, family: str, arm: str, split: str):
    result = []
    for row in rows:
        world = int(row["world"])
        parity = sum(((world >> bit) & 1) for bit in range(4)) % 2
        expected = "development" if parity == 0 else "heldout"
        if row["family"] == family and row["arm"] == arm and expected == split:
            result.append(row)
    return result


def _comparison(rows, family: str, treatment: str, baseline: str):
    treatment_rows = _rows_for(rows, family, treatment, "heldout")
    baseline_rows = _rows_for(rows, family, baseline, "heldout")
    if len(treatment_rows) != 8 or len(baseline_rows) != 8:
        raise fx.ContractError("aggregate-subset")
    treatment_by_world = {int(row["world"]): float(row["mean_gap"]) for row in treatment_rows}
    baseline_by_world = {int(row["world"]): float(row["mean_gap"]) for row in baseline_rows}
    differences = [treatment_by_world[key] - baseline_by_world[key] for key in sorted(treatment_by_world)]
    wins = sum(value < -1.0e-12 for value in differences)
    ties = sum(abs(value) <= 1.0e-12 for value in differences)
    losses = 8 - wins - ties
    treatment_mean = float(sum(treatment_by_world.values()) / 8)
    baseline_mean = float(sum(baseline_by_world.values()) / 8)
    return treatment_mean, baseline_mean, differences, wins, ties, losses


def _projection() -> dict[str, object]:
    _runtime_identity()
    implementation_worlds, implementation_family_root, oracle_family_root = _world_replay()
    primary_transcripts, _oracle_transcripts, transcript_root = _transcript_replay()
    outcomes, capabilities, implementation_intervention_root, oracle_intervention_root = _sentinel_and_capabilities()
    attacks, implementation_attack_root, oracle_attack_root = _malformed_attacks()

    phase_results = []
    phase_child_receipts = []
    surviving_children = 0
    for family in fx.FAMILIES:
        for split in fx.SPLITS:
            worlds = [
                item
                for item in implementation_worlds
                if item["family"] == family and item["split"] == split
            ]
            stdout, survivor, child_receipt = _run_child(
                "constraint-aware-progress-toy-v1-phase",
                {
                    "family": family,
                    "split": split,
                    "contract_sha256": _required_environment(
                        "L2D_CONTRACT_SHA256", 64
                    ),
                    "transcript_root_sha256": fx.TRANSCRIPT_ROOT_SHA256,
                    "worlds": worlds,
                },
            )
            surviving_children += survivor
            phase_child_receipts.append(child_receipt)
            result = _parse_phase(_loads_json(stdout))
            phase_results.append(result)

    world_aggregates = [
        row for phase in phase_results for row in phase["world_aggregates"]
    ]
    world_aggregates.sort(
        key=lambda row: (
            fx.FAMILIES.index(row["family"]),
            int(row["world"]),
            fx.ARMS.index(row["arm"]),
        )
    )
    if len(world_aggregates) != 240:
        raise fx.ContractError("aggregate-cardinality")
    schema_impl = _root_of_roots(
        "SchemaRoots",
        [phase["roots"]["implementation_schema"] for phase in phase_results],
    )
    schema_oracle = _root_of_roots(
        "SchemaRoots",
        [phase["roots"]["oracle_schema"] for phase in phase_results],
    )
    state_impl = _root_of_roots(
        "StateRoots",
        [
            root
            for phase in phase_results
            for root in (
                phase["roots"]["implementation_state"],
                phase["roots"]["implementation_trajectory"],
            )
        ],
    )
    state_oracle = _root_of_roots(
        "StateRoots",
        [
            root
            for phase in phase_results
            for root in (
                phase["roots"]["oracle_state"],
                phase["roots"]["oracle_trajectory"],
            )
        ],
    )
    source_impl_root = _Root("PhaseReceipt")
    source_oracle_root = _Root("PhaseReceipt")
    oracle_worlds = _oracle_worlds()
    for phase in phase_results:
        implementation_receipt = phase["phase_receipt"]
        expected_worlds = [
            item
            for item in oracle_worlds
            if item["family"] == phase["family"] and item["split"] == phase["split"]
        ]
        expected_input = _Root("PhaseInput")
        for item in expected_worlds:
            expected_input.update(item)
        expected_output = _Root("WorldAggregateRecord")
        for item in phase["world_aggregates"]:
            expected_output.update(item)
        oracle_receipt = {
            "family": phase["family"],
            "split": phase["split"],
            "world_keys": [item["world"] for item in expected_worlds],
            "attempted_reads": len(expected_worlds),
            "forbidden_reads": 0,
            "forbidden_payload_rows": 0,
            "sentinel_connected": True,
            "input_root_sha256": expected_input.hexdigest(),
            "output_root_sha256": expected_output.hexdigest(),
        }
        if implementation_receipt != oracle_receipt:
            raise fx.ContractError("phase-receipt-replay")
        source_impl_root.update(implementation_receipt)
        source_oracle_root.update(oracle_receipt)

    primary_mean, baseline_mean, primary_diff, wins, ties, losses = _comparison(
        world_aggregates,
        "canonical",
        "constraint_lexicographic_progress",
        "protected_raw_progress",
    )
    primary_improvement = baseline_mean - primary_mean
    primary_mean_gate = primary_improvement >= 0.05
    primary_win_gate = wins >= 6
    primary_harm = max(primary_diff)
    primary_harm_gate = primary_harm <= 0.15

    no_restart_mean, _treatment_mean_again, no_restart_diff, *_ = _comparison(
        world_aggregates,
        "canonical",
        "no_restart_comparator",
        "constraint_lexicographic_progress",
    )
    comparator_improvement = no_restart_mean - primary_mean

    shuffled_mean, shuffled_baseline, shuffled_diff, shuffled_wins, *_ = _comparison(
        world_aggregates,
        "canonical",
        "shuffled_progress_control",
        "protected_raw_progress",
    )
    shuffled_improvement = shuffled_baseline - shuffled_mean
    shuffled_mean_gate = shuffled_improvement >= 0.05
    shuffled_win_gate = shuffled_wins >= 6
    shuffled_harm = max(shuffled_diff)
    shuffled_harm_gate = shuffled_harm <= 0.15
    shuffled_recovered = shuffled_mean_gate and shuffled_win_gate and shuffled_harm_gate

    ablated_mean, ablated_baseline, ablated_diff, ablated_wins, *_ = _comparison(
        world_aggregates,
        "canonical",
        "ablated_progress_control",
        "protected_raw_progress",
    )
    ablated_improvement = ablated_baseline - ablated_mean
    ablated_mean_gate = ablated_improvement >= 0.05
    ablated_win_gate = ablated_wins >= 6
    ablated_harm = max(ablated_diff)
    ablated_harm_gate = ablated_harm <= 0.15
    ablated_recovered = ablated_mean_gate and ablated_win_gate and ablated_harm_gate

    aligned_treatment, aligned_baseline, aligned_diff, *_ = _comparison(
        world_aggregates,
        "aligned",
        "constraint_lexicographic_progress",
        "protected_raw_progress",
    )
    aligned_abs = abs(aligned_treatment - aligned_baseline)
    aligned_harm = max(aligned_diff)

    observations = sum(int(item["observations"]) for item in phase_results)
    transitions = sum(int(item["transitions"]) for item in phase_results)
    batches = sum(int(item["batches"]) for item in phase_results)
    trajectories = sum(int(item["trajectories"]) for item in phase_results)
    schema_valid_observations = sum(
        int(item["schema_valid_observations"]) for item in phase_results
    )
    replayed_transitions = sum(
        int(item["replayed_transitions"]) for item in phase_results
    )
    replayed_receipts = sum(
        int(item["replayed_receipts"]) for item in phase_results
    )
    reset_checks = sum(int(item["reset_checks"]) for item in phase_results)
    incumbent_tie_checks = sum(
        int(item["incumbent_tie_checks"]) for item in phase_results
    )
    incumbent_state_checks = sum(
        int(item["incumbent_state_checks"]) for item in phase_results
    )
    restart_events = sum(int(item["restart_events"]) for item in phase_results)
    impossible_phases = [item for item in phase_results if item["family"] == "impossible"]
    impossible_feasible = sum(int(item["feasible_observations"]) for item in impossible_phases)
    development_receipts = [
        item["phase_receipt"]
        for item in phase_results
        if item["split"] == "development"
    ]
    heldout_receipts = [
        item["phase_receipt"] for item in phase_results if item["split"] == "heldout"
    ]
    forbidden_reads = sum(
        int(item["phase_receipt"]["forbidden_reads"]) for item in phase_results
    )
    heldout_source_in_development = sum(
        int(item["phase_receipt"]["forbidden_payload_rows"])
        + sum(
            sum((int(row["world"]) >> bit) & 1 for bit in range(4)) % 2 == 1
            for row in item["world_aggregates"]
        )
        for item in phase_results
        if item["split"] == "development"
    )
    development_outputs_in_heldout = sum(
        sum(
            sum((int(row["world"]) >> bit) & 1 for bit in range(4)) % 2 == 0
            for row in item["world_aggregates"]
        )
        for item in phase_results
        if item["split"] == "heldout"
    )

    oracle_worlds_by_key = {
        (item["family"], item["world"]): item for item in oracle_worlds
    }
    formula_fields = (
        "family",
        "world",
        "bits",
        "split",
        "a",
        "b",
        "k",
        "t",
        "c",
        "threshold",
    )
    reference_fields = ("reference_x0", "reference_sensitivity", "denominator")
    formula_mismatches = sum(
        any(item[name] != oracle_worlds_by_key[(item["family"], item["world"])][name] for name in formula_fields)
        for item in implementation_worlds
    )
    reference_mismatches = sum(
        any(item[name] != oracle_worlds_by_key[(item["family"], item["world"])][name] for name in reference_fields)
        for item in implementation_worlds
    )
    duplicate_world_keys = len(implementation_worlds) - len(
        {(item["family"], item["world"]) for item in implementation_worlds}
    )
    constrained_references = sum(
        item["reference_x0"] is not None for item in implementation_worlds
    )
    reference_exclusions = sum(
        item["family"] == "impossible"
        and all(item[name] is None for name in reference_fields)
        for item in implementation_worlds
    )
    nonpositive_denominators = sum(
        item["denominator"] is not None and item["denominator"] <= 0.0
        for item in implementation_worlds
    )
    development_counts = [
        sum(item["family"] == family and item["split"] == "development" for item in implementation_worlds)
        for family in fx.FAMILIES
    ]
    heldout_counts = [
        sum(item["family"] == family and item["split"] == "heldout" for item in implementation_worlds)
        for family in fx.FAMILIES
    ]
    arm_evaluations = {
        arm: sum(int(item["arm_evaluations"][arm]) for item in phase_results)
        for arm in fx.ARMS
    }
    arm_trajectory_counts = {
        arm: sum(int(item["arm_transcript_receipts"][arm]) for item in phase_results)
        for arm in fx.ARMS
    }
    evaluations_per_arm_trajectory = {
        arm: arm_evaluations[arm] // arm_trajectory_counts[arm]
        for arm in fx.ARMS
        if arm_trajectory_counts[arm] > 0
        and arm_evaluations[arm] % arm_trajectory_counts[arm] == 0
    }
    evaluation_parity = (
        len(evaluations_per_arm_trajectory) == len(fx.ARMS)
        and len(set(evaluations_per_arm_trajectory.values())) == 1
    )
    transcript_parity = len(set(arm_trajectory_counts.values())) == 1

    cases = [
        _case(
            "family_replay",
            implementation_family_root == oracle_family_root
            and len(implementation_worlds) == 48
            and constrained_references == 32
            and reference_exclusions == 16
            and development_counts == [8, 8, 8]
            and heldout_counts == [8, 8, 8]
            and formula_mismatches == 0
            and reference_mismatches == 0
            and nonpositive_denominators == 0
            and duplicate_world_keys == 0,
            {
                "world_records": len(implementation_worlds),
                "constrained_references": constrained_references,
                "reference_exclusions": reference_exclusions,
                "development_worlds_per_family": min(development_counts),
                "heldout_worlds_per_family": min(heldout_counts),
                "formula_mismatches": formula_mismatches,
                "reference_mismatches": reference_mismatches,
                "nonpositive_denominators": nonpositive_denominators,
                "duplicate_world_keys": duplicate_world_keys,
                "implementation_root_sha256": implementation_family_root,
                "oracle_root_sha256": oracle_family_root,
                "roots_equal": implementation_family_root == oracle_family_root,
            },
        ),
        _case(
            "transcript_commitment",
            transcript_root == fx.TRANSCRIPT_ROOT_SHA256
            and sum(int(item["order_mismatches"]) for item in phase_results) == 0
            and trajectories == 1920
            and observations == 983040
            and len(set(arm_trajectory_counts.values())) == 1,
            {
                "transcripts": len(primary_transcripts),
                "values_per_transcript": int(
                    primary_transcripts[0].suffix.size
                    + primary_transcripts[0].fresh.size
                    + primary_transcripts[0].perturb.size
                ),
                "transcript_values": sum(
                    int(item.suffix.size + item.fresh.size + item.perturb.size)
                    for item in primary_transcripts
                ),
                "trajectories": trajectories,
                "evaluations": observations,
                "unequal_arm_counts": len(set(arm_trajectory_counts.values())) - 1,
                "order_twin_mismatches": sum(int(item["order_mismatches"]) for item in phase_results),
                "committed_root_sha256": fx.TRANSCRIPT_ROOT_SHA256,
                "observed_root_sha256": transcript_root,
                "roots_equal": transcript_root == fx.TRANSCRIPT_ROOT_SHA256,
            },
        ),
        _case(
            "typed_aux_and_intervention",
            schema_impl == schema_oracle
            and implementation_intervention_root == oracle_intervention_root
            and schema_valid_observations == observations
            and all(item.state_mutations == 0 for item in capabilities)
            and all(item.consumer_reached for item in capabilities),
            {
                "observations": observations,
                "schema_valid_observations": schema_valid_observations,
                "join_failures": observations - schema_valid_observations,
                "capability_attacks": len(capabilities),
                "capability_rejected": sum(
                    item.rejection_code == "capability-denied"
                    and item.consumer_reached
                    for item in capabilities
                ),
                "capability_state_mutations": sum(
                    item.state_mutations for item in capabilities
                ),
                "canonical_decisions": list(outcomes[0]),
                "donor_decisions": list(outcomes[1]),
                "ablated_decisions": list(outcomes[2]),
                "implementation_schema_root_sha256": schema_impl,
                "oracle_schema_root_sha256": schema_oracle,
                "implementation_intervention_root_sha256": implementation_intervention_root,
                "oracle_intervention_root_sha256": oracle_intervention_root,
                "roots_equal": schema_impl == schema_oracle
                and implementation_intervention_root == oracle_intervention_root,
            },
        ),
        _case(
            "chronology_replay",
            state_impl == state_oracle
            and replayed_transitions == transitions
            and replayed_receipts == batches
            and reset_checks == restart_events
            and incumbent_tie_checks == batches
            and incumbent_state_checks == batches
            and sum(int(item["order_mismatches"]) for item in phase_results) == 0,
            {
                "batches": batches,
                "batch_receipts": replayed_receipts,
                "transitions": transitions,
                "replay_mismatches": transitions - replayed_transitions,
                "order_mismatches": sum(
                    int(item["order_mismatches"]) for item in phase_results
                ),
                "reset_mismatches": restart_events - reset_checks,
                "incumbent_tie_mismatches": batches - incumbent_tie_checks,
                "incumbent_state_mismatches": batches - incumbent_state_checks,
                "restart_events": restart_events,
                "implementation_state_root_sha256": state_impl,
                "oracle_state_root_sha256": state_oracle,
                "roots_equal": state_impl == state_oracle,
            },
        ),
        _case(
            "development_and_source_isolation",
            source_impl_root.hexdigest() == source_oracle_root.hexdigest()
            and len(development_receipts) == 3
            and len(heldout_receipts) == 3
            and forbidden_reads == 0
            and heldout_source_in_development == 0
            and development_outputs_in_heldout == 0,
            {
                "development_aggregates": sum(
                    sum((int(row["world"]) >> bit) & 1 for bit in range(4)) % 2 == 0
                    for row in world_aggregates
                ),
                "development_receipts": len(development_receipts),
                "heldout_receipts": len(heldout_receipts),
                "forbidden_reads": forbidden_reads,
                "heldout_source_in_development": heldout_source_in_development,
                "development_outputs_in_heldout": development_outputs_in_heldout,
                "implementation_source_root_sha256": source_impl_root.hexdigest(),
                "oracle_source_root_sha256": source_oracle_root.hexdigest(),
                "roots_equal": source_impl_root.hexdigest() == source_oracle_root.hexdigest(),
            },
        ),
        _case(
            "heldout_primary",
            primary_mean_gate and primary_win_gate and primary_harm_gate,
            {
                "treatment_mean_gap": primary_mean,
                "baseline_mean_gap": baseline_mean,
                "mean_improvement": primary_improvement,
                "heldout_wins": wins,
                "heldout_ties": ties,
                "heldout_losses": losses,
                "maximum_signed_world_harm": primary_harm,
                "mean_gate": primary_mean_gate,
                "win_gate": primary_win_gate,
                "harm_gate": primary_harm_gate,
            },
        ),
        _case(
            "restart_comparators",
            comparator_improvement >= 0.05
            and evaluation_parity
            and transcript_parity,
            {
                "treatment_mean_gap": primary_mean,
                "no_restart_mean_gap": no_restart_mean,
                "mean_improvement": comparator_improvement,
                "minimum_arm_evaluations": min(evaluations_per_arm_trajectory.values()),
                "maximum_arm_evaluations": max(evaluations_per_arm_trajectory.values()),
                "evaluation_parity": evaluation_parity,
                "transcript_parity": transcript_parity,
                "comparator_gate": comparator_improvement >= 0.05
                and evaluation_parity
                and transcript_parity,
            },
        ),
        _case(
            "shuffled_signal_control",
            (not shuffled_recovered) and shuffled_improvement < 0.05,
            {
                "control_mean_gap": shuffled_mean,
                "baseline_mean_gap": shuffled_baseline,
                "mean_improvement": shuffled_improvement,
                "heldout_wins": shuffled_wins,
                "maximum_signed_world_harm": shuffled_harm,
                "substituted_mean_gate": shuffled_mean_gate,
                "substituted_win_gate": shuffled_win_gate,
                "substituted_harm_gate": shuffled_harm_gate,
                "positive_gate_recovered": shuffled_recovered,
            },
        ),
        _case(
            "ablated_signal_control",
            (not ablated_recovered) and ablated_improvement < 0.05,
            {
                "control_mean_gap": ablated_mean,
                "baseline_mean_gap": ablated_baseline,
                "mean_improvement": ablated_improvement,
                "heldout_wins": ablated_wins,
                "maximum_signed_world_harm": ablated_harm,
                "substituted_mean_gate": ablated_mean_gate,
                "substituted_win_gate": ablated_win_gate,
                "substituted_harm_gate": ablated_harm_gate,
                "positive_gate_recovered": ablated_recovered,
            },
        ),
        _case(
            "aligned_control",
            aligned_abs <= 0.03 and aligned_harm <= 0.10,
            {
                "treatment_mean_gap": aligned_treatment,
                "baseline_mean_gap": aligned_baseline,
                "absolute_mean_difference": aligned_abs,
                "maximum_signed_world_harm": aligned_harm,
                "trajectories": sum(
                    int(item["trajectories"])
                    for item in phase_results
                    if item["family"] == "aligned"
                ),
                "mean_gate": aligned_abs <= 0.03,
                "harm_gate": aligned_harm <= 0.10,
            },
        ),
        _case(
            "impossible_control",
            impossible_feasible == 0
            and sum(int(item["nonunit_gaps"]) for item in impossible_phases) == 0
            and sum(int(item["references_used"]) for item in impossible_phases) == 0
            and sum(int(item["false_feasible_joins"]) for item in impossible_phases) == 0,
            {
                "trajectories": sum(int(item["trajectories"]) for item in impossible_phases),
                "observations": sum(int(item["observations"]) for item in impossible_phases),
                "feasible_observations": impossible_feasible,
                "nonunit_gaps": sum(
                    int(item["nonunit_gaps"]) for item in impossible_phases
                ),
                "references_used": sum(
                    int(item["references_used"]) for item in impossible_phases
                ),
                "false_feasible_joins": sum(
                    int(item["false_feasible_joins"]) for item in impossible_phases
                ),
            },
        ),
    ]
    return {
        "projection_version": 1,
        "world_aggregates": world_aggregates,
        "cases": cases,
        "attacks": len(attacks),
        "attacks_rejected": sum(
            item["consumer_reached"] is True
            and item["rejection_code"] == expected[2]
            for item, expected in zip(attacks, ATTACK_MATRIX, strict=True)
        ),
        "attack_state_mutations": sum(
            int(item["state_mutations"]) for item in attacks
        ),
        "implementation_attack_root_sha256": implementation_attack_root,
        "oracle_attack_root_sha256": oracle_attack_root,
        "maximum_child_stdout_bytes": max(
            int(item["stdout_bytes"]) for item in phase_child_receipts
        ),
        "stderr_bytes": sum(
            int(item["stderr_bytes"]) for item in phase_child_receipts
        ),
        "surviving_children": surviving_children,
    }


def _parse_projection(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or tuple(value) != (
        "projection_version",
        "world_aggregates",
        "cases",
        "attacks",
        "attacks_rejected",
        "attack_state_mutations",
        "implementation_attack_root_sha256",
        "oracle_attack_root_sha256",
        "maximum_child_stdout_bytes",
        "stderr_bytes",
        "surviving_children",
    ):
        raise fx.ContractError("projection-schema")
    if value["projection_version"] != 1:
        raise fx.ContractError("projection-version")
    if (
        not isinstance(value["world_aggregates"], list)
        or len(value["world_aggregates"]) != 240
        or not isinstance(value["cases"], list)
        or len(value["cases"]) != 11
    ):
        raise fx.ContractError("projection-cardinality")
    for name in (
        "attacks",
        "attacks_rejected",
        "attack_state_mutations",
        "maximum_child_stdout_bytes",
        "stderr_bytes",
        "surviving_children",
    ):
        if type(value[name]) is not int or value[name] < 0:
            raise fx.ContractError("projection-count")
    if value["maximum_child_stdout_bytes"] > MAX_PACKET_BYTES:
        raise fx.ContractError("projection-child-output-cap")
    for name in (
        "implementation_attack_root_sha256",
        "oracle_attack_root_sha256",
    ):
        item = value[name]
        if (
            type(item) is not str
            or len(item) != 64
            or any(character not in "0123456789abcdef" for character in item)
        ):
            raise fx.ContractError("projection-root")
    return value


def _required_environment(name: str, length: int) -> str:
    value = os.environ.get(name, "")
    if len(value) != length or any(character not in "0123456789abcdef" for character in value):
        raise fx.ContractError("controller-contract-environment")
    return value


def _full() -> dict[str, object]:
    _runtime_identity()
    plan_revision = _required_environment("L2D_PLAN_REVISION", 40)
    study_revision = _required_environment("L2D_STUDY_REVISION", 40)
    contract_sha256 = _required_environment("L2D_CONTRACT_SHA256", 64)
    if plan_revision != fx.PLAN_REVISION:
        raise fx.ContractError("plan-revision")
    raw_projections = []
    parsed = []
    projection_receipts = []
    surviving_children = 0
    for _ in range(2):
        stdout, survivor, child_receipt = _run_child(
            "constraint-aware-progress-toy-v1-projection"
        )
        raw_projections.append(stdout)
        surviving_children += survivor
        projection_receipts.append(child_receipt)
        parsed.append(_parse_projection(_loads_json(stdout)))
    projections_equal = raw_projections[0] == raw_projections[1]
    if not projections_equal:
        raise fx.ContractError("projection-nondeterminism")
    projection = parsed[0]
    surviving_children += sum(
        int(item["surviving_children"]) for item in parsed
    )
    case12_metrics = {
        "launches": 2,
        "projections_equal": projections_equal,
        "maximum_stdout_bytes": max(
            [len(item) for item in raw_projections]
            + [int(item["maximum_child_stdout_bytes"]) for item in parsed]
        ),
        "stderr_bytes": sum(
            int(item["stderr_bytes"]) for item in projection_receipts
        )
        + sum(int(item["stderr_bytes"]) for item in parsed),
        "surviving_children": surviving_children,
        "attacks": projection["attacks"],
        "attacks_rejected": projection["attacks_rejected"],
        "attack_state_mutations": projection["attack_state_mutations"],
        "implementation_attack_root_sha256": projection[
            "implementation_attack_root_sha256"
        ],
        "oracle_attack_root_sha256": projection["oracle_attack_root_sha256"],
        "roots_equal": projection["implementation_attack_root_sha256"]
        == projection["oracle_attack_root_sha256"],
    }
    case12_passed = (
        projections_equal
        and case12_metrics["maximum_stdout_bytes"] <= MAX_PACKET_BYTES
        and case12_metrics["stderr_bytes"] == 0
        and surviving_children == 0
        and case12_metrics["attacks"] == 12
        and case12_metrics["attacks_rejected"] == 12
        and case12_metrics["attack_state_mutations"] == 0
        and case12_metrics["roots_equal"] is True
    )
    cases = [*projection["cases"], _case("process_and_sanitizer", case12_passed, case12_metrics)]
    passed = all(case["passed"] for case in cases)
    result = {
        "study_id": fx.STUDY_ID,
        "plan_revision": plan_revision,
        "study_revision": study_revision,
        "contract_sha256": contract_sha256,
        "transcript_root_sha256": fx.TRANSCRIPT_ROOT_SHA256,
        "status": "passed" if passed else "failed",
        "action": (
            "constraint_progress_mechanics_ready_for_candidate_audit"
            if passed
            else "park_round2_constraint_progress_research"
        ),
        "world_aggregates": projection["world_aggregates"],
        "cases": cases,
    }
    _validate_result_top(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=[
            "constraint-aware-progress-toy-v1",
            "constraint-aware-progress-toy-v1-projection",
            "constraint-aware-progress-toy-v1-phase",
            "constraint-aware-progress-toy-v1-runtime-probe",
        ],
        required=True,
    )
    args = parser.parse_args()
    _await_child_job_gate(args.mode)
    _disable_network()
    _load_runtime()
    if args.mode == "constraint-aware-progress-toy-v1-runtime-probe":
        result = _runtime_identity()
    elif args.mode == "constraint-aware-progress-toy-v1":
        result = _full()
    elif args.mode == "constraint-aware-progress-toy-v1-projection":
        result = _projection()
    else:
        result = _phase()
    print(
        json.dumps(
            _normalize(result),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
