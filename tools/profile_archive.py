"""Create a compact, provenance-rich profile of Differometor-30k."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np


UPSTREAM_REFERENCE = "d9b1bd7d6f2c4df335bc7725755b02aa5f6f942c"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode(values) -> np.ndarray:
    return np.char.decode(values, encoding="utf-8", errors="strict")


def quantiles(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    points = np.quantile(finite, [0.0, 0.1, 0.5, 0.9, 1.0])
    return {
        name: float(value)
        for name, value in zip(("minimum", "p10", "median", "p90", "maximum"), points)
    }


def component_sizes(
    unique_hashes: np.ndarray,
    parents: np.ndarray,
    equivalence_labels: np.ndarray | None = None,
) -> list[int]:
    nodes = set(unique_hashes.tolist()) | {parent for parent in parents if parent}
    roots = {node: node for node in nodes}

    def find(node: str) -> str:
        while roots[node] != node:
            roots[node] = roots[roots[node]]
            node = roots[node]
        return node

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            roots[right_root] = left_root

    for child, parent in zip(unique_hashes, parents):
        if parent:
            union(str(child), str(parent))

    if equivalence_labels is not None:
        first_hash_by_label: dict[str, str] = {}
        for design_hash, label in zip(unique_hashes, equivalence_labels):
            label = str(label)
            design_hash = str(design_hash)
            if label in first_hash_by_label:
                union(design_hash, first_hash_by_label[label])
            else:
                first_hash_by_label[label] = design_hash

    row_counts = Counter(find(str(node)) for node in unique_hashes)
    return sorted(row_counts.values())


def profile_dataset(dataset_path: Path) -> dict[str, object]:
    with h5py.File(dataset_path, "r") as archive:
        entries = archive["entries"][:]
        topologies = decode(entries["topology_string"])
        unique_hashes = decode(entries["unique_hash"])
        parents = decode(entries["initialized_from"])
        sizes = np.asarray(entries["size"])
        losses = np.asarray(entries["loss"])
        param_lengths = np.asarray(entries["param_length"])

        topology_counts = np.asarray(list(Counter(topologies.tolist()).values()))
        lineage_sizes = np.asarray(component_sizes(unique_hashes, parents))
        split_group_sizes = np.asarray(
            component_sizes(unique_hashes, parents, equivalence_labels=topologies)
        )
        size_counts = Counter(int(value) for value in sizes)
        length_counts = Counter(int(value) for value in param_lengths)

        return {
            "created_utc": datetime.now(UTC).isoformat(),
            "dataset": {
                "path": dataset_path.name,
                "sha256": sha256(dataset_path),
                "size_bytes": dataset_path.stat().st_size,
            },
            "entries": int(len(entries)),
            "grid_size_counts": dict(sorted(size_counts.items())),
            "lineage": {
                "components": int(len(lineage_sizes)),
                "entries_with_parent": int(np.count_nonzero(parents != "")),
                "largest_component_rows": int(lineage_sizes.max()),
                "median_component_rows": float(np.median(lineage_sizes)),
            },
            "loss_quantiles": {
                "all": quantiles(losses),
                "size_3": quantiles(losses[sizes == 3]),
            },
            "parameter_lengths": {
                "minimum": int(param_lengths.min()),
                "maximum": int(param_lengths.max()),
                "most_common": [
                    {"length": length, "entries": count}
                    for length, count in length_counts.most_common(10)
                ],
            },
            "recommended_split_groups": {
                "groups": int(len(split_group_sizes)),
                "largest_group_rows": int(split_group_sizes.max()),
                "median_group_rows": float(np.median(split_group_sizes)),
                "rule": "connected components of shared topology or initialized_from lineage",
            },
            "topologies": {
                "unique": int(len(topology_counts)),
                "singleton": int(np.count_nonzero(topology_counts == 1)),
                "largest_group_rows": int(topology_counts.max()),
                "median_group_rows": float(np.median(topology_counts)),
            },
            "upstream": {
                "git_revision": UPSTREAM_REFERENCE,
                "selection_metadata": str(archive.attrs.get("selection", "")),
            },
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="path to upstream dataset.h5")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/generated/archive-profile.json"),
    )
    args = parser.parse_args()

    profile = profile_dataset(args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"profiled {profile['entries']} entries -> {args.output}")


if __name__ == "__main__":
    main()
