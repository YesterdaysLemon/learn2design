"""Exact replay agreement and detached-summary opening gates."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from .canonical import canonical_json_bytes, sha256_bytes


class EvidenceError(RuntimeError):
    pass


_TOKEN_SENTINEL = object()


@dataclass(frozen=True)
class ReplayAgreement:
    stage: int
    values_compared: int
    archive_sha256: str
    ordered_run_ids: tuple[str, ...]
    production_sha256: str
    reference_sha256: str
    panel_sha256: str
    split_receipt_sha256: str
    source_lock_sha256: str
    runtime_lock_sha256: str
    _sentinel: object

    def __post_init__(self) -> None:
        if self._sentinel is not _TOKEN_SENTINEL:
            raise EvidenceError("agreement tokens can only be comparator-issued")
        digests = (
            self.archive_sha256,
            self.production_sha256,
            self.reference_sha256,
            self.panel_sha256,
            self.split_receipt_sha256,
            self.source_lock_sha256,
            self.runtime_lock_sha256,
        )
        if any(
            len(value) != 64 or any(token not in "0123456789abcdef" for token in value)
            for value in digests
        ):
            raise EvidenceError("agreement contains an invalid digest")
        if not self.ordered_run_ids or len(set(self.ordered_run_ids)) != len(
            self.ordered_run_ids
        ):
            raise EvidenceError("agreement run identities are invalid")

    def receipt_payload(self) -> dict[str, object]:
        return {
            "status": "matched",
            "stage": self.stage,
            "values_compared": self.values_compared,
            "archive_sha256": self.archive_sha256,
            "ordered_run_ids": list(self.ordered_run_ids),
            "production_sha256": self.production_sha256,
            "reference_sha256": self.reference_sha256,
            "panel_sha256": self.panel_sha256,
            "split_receipt_sha256": self.split_receipt_sha256,
            "source_lock_sha256": self.source_lock_sha256,
            "runtime_lock_sha256": self.runtime_lock_sha256,
        }


def _compare(left: Any, right: Any, path: str = "$") -> int:
    if type(left) is not type(right):
        raise EvidenceError(f"replay type mismatch at {path}")
    if isinstance(left, dict):
        if list(left) != list(right):
            raise EvidenceError(f"replay key/order mismatch at {path}")
        return 1 + sum(_compare(left[key], right[key], f"{path}.{key}") for key in left)
    if isinstance(left, list):
        if len(left) != len(right):
            raise EvidenceError(f"replay sequence length mismatch at {path}")
        return 1 + sum(
            _compare(a, b, f"{path}[{index}]")
            for index, (a, b) in enumerate(zip(left, right, strict=True))
        )
    if isinstance(left, float):
        if (
            isinstance(left, bool)
            or isinstance(right, bool)
            or not math.isfinite(float(left))
            or not math.isfinite(float(right))
            or not math.isclose(
                float(left), float(right), rel_tol=1e-12, abs_tol=1e-12
            )
        ):
            raise EvidenceError(f"replay numeric mismatch at {path}")
    elif left != right:
        raise EvidenceError(f"replay value mismatch at {path}")
    return 1


def compare_replays(
    production: dict[str, object],
    reference: dict[str, object],
    *,
    stage: int,
    archive_sha256: str,
    ordered_run_ids: list[str],
    panel_sha256: str,
    split_receipt_sha256: str,
    source_lock_sha256: str,
    runtime_lock_sha256: str,
) -> ReplayAgreement:
    if stage not in (1, 2):
        raise EvidenceError("replay stage is invalid")
    values = _compare(production, reference)
    return ReplayAgreement(
        stage=stage,
        values_compared=values,
        archive_sha256=archive_sha256,
        ordered_run_ids=tuple(ordered_run_ids),
        production_sha256=sha256_bytes(canonical_json_bytes(production)),
        reference_sha256=sha256_bytes(canonical_json_bytes(reference)),
        panel_sha256=panel_sha256,
        split_receipt_sha256=split_receipt_sha256,
        source_lock_sha256=source_lock_sha256,
        runtime_lock_sha256=runtime_lock_sha256,
        _sentinel=_TOKEN_SENTINEL,
    )


def compare_detached_summary(
    production: dict[str, object],
    reference: dict[str, object],
    detached: dict[str, object],
    agreement: ReplayAgreement,
) -> dict[str, object]:
    if agreement._sentinel is not _TOKEN_SENTINEL:
        raise EvidenceError("detached summary lacks authentic replay agreement")
    _compare(production, reference)
    values = _compare(production, detached)
    return {
        "status": "matched",
        "stage": agreement.stage,
        "production_reference_values": agreement.values_compared,
        "detached_values": values,
    }
