from __future__ import annotations

from pathlib import Path

import pytest

h5py = pytest.importorskip("h5py")
np = pytest.importorskip("numpy")

from tools.profile_archive import profile_dataset


@pytest.mark.integration
def test_archive_profile_groups_topologies_and_lineages(tmp_path: Path) -> None:
    path = tmp_path / "dataset.h5"
    dtype = np.dtype(
        [
            ("unique_hash", "S24"),
            ("initialized_from", "S24"),
            ("topology_string", "S64"),
            ("param_length", "<u4"),
            ("size", "<u2"),
            ("loss", "<f8"),
        ]
    )
    entries = np.array(
        [
            (b"a", b"", b"TOPOLOGY-A", 190, 3, 0.2),
            (b"b", b"a", b"TOPOLOGY-A", 190, 3, 0.1),
            (b"c", b"", b"TOPOLOGY-B", 220, 4, 1.5),
        ],
        dtype=dtype,
    )
    with h5py.File(path, "w") as archive:
        archive.create_dataset("entries", data=entries)
        archive.attrs["selection"] = "synthetic fixture"

    profile = profile_dataset(path)

    assert profile["entries"] == 3
    assert profile["grid_size_counts"] == {3: 2, 4: 1}
    assert profile["topologies"]["unique"] == 2
    assert profile["topologies"]["singleton"] == 1
    assert profile["lineage"]["components"] == 2
    assert profile["lineage"]["largest_component_rows"] == 2
    assert profile["recommended_split_groups"]["groups"] == 2
