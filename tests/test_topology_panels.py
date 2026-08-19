from __future__ import annotations

import json
from pathlib import Path

import pytest


h5py = pytest.importorskip("h5py")
np = pytest.importorskip("numpy")
pytest.importorskip("dfbench")
pytest.importorskip("differometor")

from tools import build_topology_panels as panels


@pytest.mark.integration
def test_panel_builder_is_deterministic_balanced_and_disjoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed_start = 2026081900
    archived_topology = panels.topology_from_seed(seed_start)
    dataset_path = tmp_path / "dataset.h5"
    dtype = np.dtype([("topology_string", "S64")])
    entries = np.asarray([(archived_topology.encode(),)], dtype=dtype)
    with h5py.File(dataset_path, "w") as archive:
        archive.create_dataset("entries", data=entries)
    monkeypatch.setattr(
        panels, "OFFICIAL_DATASET_SHA256", panels.sha256(dataset_path)
    )

    counts = {"development-v1": 4, "confirmation-v1": 4}
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"
    first_audit = panels.build_panels(
        dataset_path,
        first_output,
        counts=counts,
        seed_start=seed_start,
        candidate_multiplier=12,
    )
    second_audit = panels.build_panels(
        dataset_path,
        second_output,
        counts=counts,
        seed_start=seed_start,
        candidate_multiplier=12,
    )

    assert first_audit == second_audit
    assert (first_output / "audit.json").read_bytes() == (
        second_output / "audit.json"
    ).read_bytes()

    observed = set()
    for panel_id, count in counts.items():
        payload = json.loads(
            (first_output / f"{panel_id}.json").read_text(encoding="utf-8")
        )
        identities = set(payload["topologies"])
        assert len(identities) == count
        assert archived_topology not in identities
        assert observed.isdisjoint(identities)
        observed.update(identities)
        readouts = [member["readout"] for member in payload["members"]]
        assert abs(readouts.count("D") - readouts.count("H")) <= 1
