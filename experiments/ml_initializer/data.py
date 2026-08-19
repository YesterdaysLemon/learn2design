"""Semantic Differometor-30k loading without constructing a simulator."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np
from dfbench.problems.uifo import topology_from_string
from differometor.setups import constrain_inter_grid_cell_spaces, uifo


OPTIMIZED_PROPERTIES = (
    "reflectivity",
    "tuning",
    "db",
    "angle",
    "power",
    "mass",
    "length",
)

PROPERTY_BOUNDS = {
    "reflectivity": (0.0, 1.0),
    "tuning": (-360.0, 360.0),
    "db": (0.0, 10.0),
    "angle": (-360.0, 360.0),
    "power": (0.0, 200.0),
    "mass": (0.01, 200.0),
    "length": (0.1, 4000.0),
}

TOKEN_VOCAB = "ABCDEFGHLS"
TOKEN_TO_ID = {token: index for index, token in enumerate(TOKEN_VOCAB)}


@dataclass(frozen=True)
class DesignRecord:
    topology: str
    topology_tokens: np.ndarray
    loss: float
    unit_params: np.ndarray
    semantic_keys: tuple[str, ...]
    properties: tuple[str, ...]


def topology_tokens(topology: str) -> np.ndarray:
    tokens = topology.replace("-", "")
    if len(tokens) != 21:
        raise ValueError(f"expected 21 topology tokens, got {topology!r}")
    try:
        return np.asarray([TOKEN_TO_ID[token] for token in tokens], dtype=np.uint8)
    except KeyError as error:
        raise ValueError(f"unknown topology token in {topology!r}") from error


def canonical_pairs(pair) -> tuple[tuple[str, str], ...]:
    if (
        isinstance(pair, (list, tuple))
        and len(pair) >= 2
        and isinstance(pair[0], str)
        and isinstance(pair[1], str)
    ):
        return ((pair[0], pair[1]),)
    if not isinstance(pair, (list, tuple)):
        raise TypeError(f"unsupported optimization pair: {pair!r}")
    result = tuple((str(item[0]), str(item[1])) for item in pair)
    if not result:
        raise ValueError("empty coupled optimization pair")
    return result


def semantic_key(pair) -> tuple[str, str]:
    pairs = canonical_pairs(pair)
    properties = {property_name for _, property_name in pairs}
    if len(properties) != 1:
        raise ValueError(f"coupled slot spans properties: {pairs!r}")
    property_name = next(iter(properties))
    components = "+".join(sorted(component for component, _ in pairs))
    return f"{property_name}:{components}", property_name


@lru_cache(maxsize=None)
def semantic_layout(topology: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    centers, boundaries = topology_from_string(topology, size=3)
    _, component_property_pairs = uifo(
        size=3,
        mode="space_modulation",
        random=True,
        centers=centers,
        boundaries=boundaries,
    )
    optimization_pairs = constrain_inter_grid_cell_spaces(
        component_property_pairs,
        list(OPTIMIZED_PROPERTIES),
    )
    encoded = tuple(semantic_key(pair) for pair in optimization_pairs)
    keys = tuple(item[0] for item in encoded)
    properties = tuple(item[1] for item in encoded)
    return keys, properties


def split_is_test(topology: str, test_fraction: float = 0.2) -> bool:
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must lie in (0, 1)")
    bucket = int.from_bytes(hashlib.sha256(topology.encode()).digest()[:8], "big")
    return bucket / 2**64 < test_fraction


def load_best_size3_records(dataset_path: Path) -> list[DesignRecord]:
    """Load one best-stored-loss row per size-3 topology identity."""
    with h5py.File(dataset_path, "r") as archive:
        entries = archive["entries"][:]
        size3_indices = np.flatnonzero(entries["size"] == 3)
        best_index_by_topology: dict[str, int] = {}
        for index in size3_indices:
            topology = entries[index]["topology_string"].decode("utf-8")
            loss = float(entries[index]["loss"])
            current = best_index_by_topology.get(topology)
            if current is None or loss < float(entries[current]["loss"]):
                best_index_by_topology[topology] = int(index)

        bounded_pool = archive["bounded_params"]
        records: list[DesignRecord] = []
        for topology in sorted(best_index_by_topology):
            index = best_index_by_topology[topology]
            entry = entries[index]
            offset = int(entry["param_offset"])
            length = int(entry["param_length"])
            bounded = np.asarray(bounded_pool[offset : offset + length], dtype=np.float64)
            keys, properties = semantic_layout(topology)
            if len(keys) != length:
                raise ValueError(
                    f"layout mismatch for {topology}: {len(keys)} != {length}"
                )

            lower = np.asarray([PROPERTY_BOUNDS[name][0] for name in properties])
            upper = np.asarray([PROPERTY_BOUNDS[name][1] for name in properties])
            unit = (bounded - lower) / (upper - lower)
            if not np.all(np.isfinite(unit)):
                raise ValueError(f"non-finite normalized parameters for {topology}")
            if float(np.min(unit)) < -1e-7 or float(np.max(unit)) > 1.0 + 1e-7:
                raise ValueError(f"out-of-bounds archive parameters for {topology}")

            records.append(
                DesignRecord(
                    topology=topology,
                    topology_tokens=topology_tokens(topology),
                    loss=float(entry["loss"]),
                    unit_params=np.clip(unit, 0.0, 1.0).astype(np.float32),
                    semantic_keys=keys,
                    properties=properties,
                )
            )
    return records
