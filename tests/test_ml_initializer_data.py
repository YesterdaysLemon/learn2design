from __future__ import annotations

import pytest

pytest.importorskip("h5py")
pytest.importorskip("dfbench")
pytest.importorskip("differometor")

from experiments.ml_initializer.data import (
    canonical_pairs,
    semantic_layout,
    split_is_test,
    topology_tokens,
)


@pytest.mark.integration
def test_topology_layout_matches_known_size3_dimension() -> None:
    topology = "ABBGDGBEA-SLLLSDSLSLSL"
    keys, properties = semantic_layout(topology)

    assert len(keys) == 194
    assert len(properties) == 194
    assert set(properties) == {
        "reflectivity",
        "tuning",
        "db",
        "angle",
        "power",
        "mass",
        "length",
    }
    assert topology_tokens(topology).shape == (21,)


def test_pair_shapes_and_split_are_deterministic() -> None:
    assert canonical_pairs(["mirror", "reflectivity"]) == (
        ("mirror", "reflectivity"),
    )
    assert canonical_pairs(
        [["space_a", "length"], ["space_b", "length"]]
    ) == (("space_a", "length"), ("space_b", "length"))
    assert split_is_test("ABBGDGBEA-SLLLSDSLSLSL") == split_is_test(
        "ABBGDGBEA-SLLLSDSLSLSL"
    )
