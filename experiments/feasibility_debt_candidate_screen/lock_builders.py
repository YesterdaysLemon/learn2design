"""Write-once constructors for externally collected runtime/source locks."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .canonical import sha256_file, write_receipt
from .contract import ARM_ORDER, STUDY_ID, arm_specs
from .locks import read_runtime_lock, read_source_lock


def source_rows(
    logical_sources: dict[str, Path | bytes]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for logical_id in sorted(logical_sources):
        source = logical_sources[logical_id]
        if isinstance(source, Path):
            content_sha256 = sha256_file(source)
            size = source.stat().st_size
        else:
            content_sha256 = hashlib.sha256(source).hexdigest()
            size = len(source)
        rows.append(
            {
                "logical_id": logical_id,
                "sha256": content_sha256,
                "size_bytes": size,
            }
        )
    return rows


def write_runtime_lock(path: Path, payload: dict[str, Any]) -> str:
    digest = write_receipt(
        path,
        study_id=STUDY_ID,
        receipt_type="runtime_lock",
        payload=payload,
    )
    read_runtime_lock(path)
    return digest


def write_source_lock(
    path: Path,
    *,
    revision: str,
    runtime_lock_sha256: str,
    package_closure_sha256: str,
    panel_commitment_sha256: str,
    logical_sources: dict[str, Path | bytes],
    repository_root: Path,
) -> str:
    specs = arm_specs()
    payload = {
        "revision": revision,
        "arm_profiles": [
            specs[arm_id].lock_row(package_closure_sha256)
            for arm_id in ARM_ORDER
        ],
        "sources": source_rows(logical_sources),
        "runtime_lock_sha256": runtime_lock_sha256,
        "worker_sha256": sha256_file(
            repository_root
            / "experiments/feasibility_debt_candidate_screen/worker.py"
        ),
        "orchestrator_sha256": sha256_file(
            repository_root
            / "experiments/feasibility_debt_candidate_screen/orchestrator.py"
        ),
        "production_analyzer_sha256": sha256_file(
            repository_root
            / "experiments/feasibility_debt_candidate_screen/analysis.py"
        ),
        "reference_analyzer_sha256": sha256_file(
            repository_root
            / "experiments/feasibility_debt_candidate_screen/reference_analysis.py"
        ),
        "panel_commitment_sha256": panel_commitment_sha256,
    }
    digest = write_receipt(
        path,
        study_id=STUDY_ID,
        receipt_type="source_lock",
        payload=payload,
    )
    read_source_lock(
        path,
        runtime_lock_sha256=runtime_lock_sha256,
        logical_sources=logical_sources,
        expected_revision=revision,
    )
    return digest
