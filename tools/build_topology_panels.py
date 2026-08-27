"""Build deterministic, balanced, archive-disjoint size-3 UIFO panels."""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
import hashlib
import json
import os
from pathlib import Path


OFFICIAL_DATASET_SHA256 = (
    "149f6aac17aff2e33750b4e1b6cebd3cef1c39d47ae49a3a7ed77315cb7838a7"
)
UPSTREAM_REFERENCE = "d9b1bd7d6f2c4df335bc7725755b02aa5f6f942c"
PANEL_UPSTREAM_REFERENCES = {
    "coverage-robustness-v1": "1bb7f54737dec6a08b59879a8831d125f08f8a0b",
    "coverage-triage-v1": "1bb7f54737dec6a08b59879a8831d125f08f8a0b",
}
DEFAULT_COUNTS = {
    "development-v1": 16,
    "confirmation-v1": 12,
    "submission-like-v1": 10,
    "coverage-robustness-v1": 12,
    "coverage-triage-v1": 8,
}
POSTHOC_PANEL_IDS = ("restart-mechanics-v1", "restart-screen-v1")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: dict[str, object]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def load_archive_topologies(dataset_path: Path) -> tuple[set[str], dict[str, object]]:
    import h5py

    digest = sha256(dataset_path)
    if digest != OFFICIAL_DATASET_SHA256:
        raise ValueError("official dataset SHA-256 mismatch")
    with h5py.File(dataset_path, "r") as archive:
        entries = archive["entries"]
        values = entries["topology_string"][:]
        topologies = {
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in values
        }
        metadata = {
            "source_name": dataset_path.name,
            "sha256": digest,
            "size_bytes": dataset_path.stat().st_size,
            "entries": int(len(entries)),
            "unique_topologies": len(topologies),
        }
    return topologies, metadata


def topology_from_seed(seed: int) -> str:
    """Resolve only setup topology metadata; do not construct a simulator problem."""
    from dfbench.problems.uifo import topology_to_string
    from differometor.setups import uifo

    result = uifo(
        size=3,
        mode="space_modulation",
        random=True,
        verbose=True,
        random_seed=seed,
    )
    if len(result) != 4:
        raise RuntimeError("Differometor did not return topology metadata")
    return topology_to_string(result[2], result[3], size=3)


def topology_features(topology: str) -> dict[str, object]:
    interior, boundary = topology.split("-", 1)
    readouts = [token for token in boundary if token in "DH"]
    if len(interior) != 9 or len(boundary) != 12 or len(readouts) != 1:
        raise ValueError(f"invalid explicit size-3 topology: {topology!r}")
    directional = sum(token in "EFGH" for token in interior)
    squeezers = boundary.count("S")
    return {
        "readout": readouts[0],
        "directional_interior_count": directional,
        "squeezer_count": squeezers,
        "directional_bin": _bin(directional, 3, 5),
        "squeezer_bin": _bin(squeezers, 4, 6),
    }


def _bin(value: int, low_maximum: int, middle_maximum: int) -> str:
    if value <= low_maximum:
        return "low"
    if value <= middle_maximum:
        return "middle"
    return "high"


def _stratum(features: dict[str, object]) -> tuple[str, str, str]:
    return (
        str(features["readout"]),
        str(features["squeezer_bin"]),
        str(features["directional_bin"]),
    )


def _stratum_order() -> list[tuple[str, str, str]]:
    # A Latin-style ordering spreads every prefix across both complexity axes.
    pairs = (
        ("low", "low"),
        ("middle", "middle"),
        ("high", "high"),
        ("low", "middle"),
        ("middle", "high"),
        ("high", "low"),
        ("low", "high"),
        ("middle", "low"),
        ("high", "middle"),
    )
    return [
        (readout, squeezer, directional)
        for squeezer, directional in pairs
        for readout in ("D", "H")
    ]


def candidate_pool(
    seed_start: int,
    attempts: int,
    excluded: set[str],
) -> list[dict[str, object]]:
    candidates = []
    seen = set(excluded)
    for seed in range(seed_start, seed_start + attempts):
        topology = topology_from_seed(seed)
        if topology in seen:
            continue
        seen.add(topology)
        features = topology_features(topology)
        candidates.append(
            {"topology_seed": seed, "topology": topology, **features}
        )
    return candidates


def select_balanced_panel(
    candidates: list[dict[str, object]],
    count: int,
    used: set[str],
) -> list[dict[str, object]]:
    buckets: dict[tuple[str, str, str], deque] = defaultdict(deque)
    for candidate in candidates:
        if candidate["topology"] not in used:
            buckets[_stratum(candidate)].append(candidate)

    selected = []
    order = _stratum_order()
    while len(selected) < count:
        made_progress = False
        for stratum in order:
            while buckets[stratum] and buckets[stratum][0]["topology"] in used:
                buckets[stratum].popleft()
            if buckets[stratum]:
                candidate = buckets[stratum].popleft()
                selected.append(candidate)
                used.add(str(candidate["topology"]))
                made_progress = True
                if len(selected) == count:
                    break
        if not made_progress:
            raise RuntimeError("candidate pool cannot fill the requested panel")
    return selected


def panel_payload(
    panel_id: str,
    members: list[dict[str, object]],
    seed_start: int,
) -> dict[str, object]:
    return {
        "format_version": 1,
        "panel_id": panel_id,
        "generation": {
            "method": "round-robin over readout/squeezer/directional strata",
            "seed_start": seed_start,
            "upstream_reference": PANEL_UPSTREAM_REFERENCES.get(
                panel_id, UPSTREAM_REFERENCE
            ),
        },
        "members": members,
        "topologies": [str(member["topology"]) for member in members],
    }


def panel_distribution(members: list[dict[str, object]]) -> dict[str, object]:
    def counts(field: str) -> dict[str, int]:
        result: dict[str, int] = defaultdict(int)
        for member in members:
            result[str(member[field])] += 1
        return dict(sorted(result.items()))

    return {
        "readout": counts("readout"),
        "squeezer_bin": counts("squeezer_bin"),
        "directional_bin": counts("directional_bin"),
    }


def posthoc_panel_records(
    output_dir: Path,
    archive_topologies: set[str],
    generated_panels: list[tuple[str, set[str]]],
) -> list[dict[str, object]]:
    """Record later panels without treating them as generated candidates.

    Restart panels were selected under separate, explicitly post-hoc rules.  They
    belong in the audit for provenance and exclusion checks, but must remain
    outside the generated-panel selection order.
    """
    records: list[dict[str, object]] = []
    previous = list(generated_panels)
    for panel_id in POSTHOC_PANEL_IDS:
        path = output_dir / f"{panel_id}.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        topologies = payload.get("topologies")
        if (
            not isinstance(topologies, list)
            or not topologies
            or not all(isinstance(topology, str) for topology in topologies)
            or len(set(topologies)) != len(topologies)
        ):
            raise ValueError(f"invalid post-hoc panel: {panel_id}")
        identities = {str(topology) for topology in topologies}
        features = [topology_features(topology) for topology in topologies]
        generation = payload.get("generation")
        if not isinstance(generation, dict):
            raise ValueError(f"post-hoc panel lacks generation metadata: {panel_id}")
        records.append(
            {
                "panel_id": panel_id,
                "source_name": path.name,
                "source_sha256": sha256(path),
                "topology_count": len(identities),
                "archive_overlap_count": len(identities & archive_topologies),
                "previous_panel_overlap_counts": {
                    earlier_id: len(identities & earlier_identities)
                    for earlier_id, earlier_identities in previous
                },
                "distribution": panel_distribution(features),
                "provenance": generation,
            }
        )
        previous.append((panel_id, identities))
    return records


def build_panels(
    dataset_path: Path,
    output_dir: Path,
    counts: dict[str, int] | None = None,
    seed_start: int = 2026081900,
    candidate_multiplier: int = 20,
) -> dict[str, object]:
    counts = dict(counts or DEFAULT_COUNTS)
    if not counts or any(count < 1 for count in counts.values()):
        raise ValueError("panel counts must be positive")
    archive_topologies, archive_metadata = load_archive_topologies(dataset_path)
    attempts = sum(counts.values()) * candidate_multiplier
    candidates = candidate_pool(seed_start, attempts, archive_topologies)
    used: set[str] = set()
    panel_records = []
    previous: list[tuple[str, set[str]]] = []

    for panel_id, count in counts.items():
        members = select_balanced_panel(candidates, count, used)
        payload = panel_payload(panel_id, members, seed_start)
        content = _json_bytes(payload)
        filename = f"{panel_id}.json"
        _atomic_bytes(output_dir / filename, content)
        identities = {str(member["topology"]) for member in members}
        previous_overlap = {
            earlier_id: len(identities & earlier_identities)
            for earlier_id, earlier_identities in previous
        }
        if any(previous_overlap.values()) or identities & archive_topologies:
            raise RuntimeError("panel exclusion invariant failed")
        panel_records.append(
            {
                "panel_id": panel_id,
                "source_name": filename,
                "source_sha256": hashlib.sha256(content).hexdigest(),
                "topology_count": len(identities),
                "archive_overlap_count": 0,
                "previous_panel_overlap_counts": previous_overlap,
                "distribution": panel_distribution(members),
            }
        )
        previous.append((panel_id, identities))

    audit = {
        "format_version": 1,
        "method": "exact topology-string set intersection",
        "generator": {
            "path": "tools/build_topology_panels.py",
            "seed_start": seed_start,
            "candidate_attempts": attempts,
            "upstream_reference": UPSTREAM_REFERENCE,
            "panel_upstream_reference_overrides": PANEL_UPSTREAM_REFERENCES,
        },
        "official_dataset": archive_metadata,
        "panels": panel_records,
        "posthoc_panels": posthoc_panel_records(
            output_dir, archive_topologies, previous
        ),
    }
    _atomic_bytes(output_dir / "audit.json", _json_bytes(audit))
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/uifo_paired/panels"),
    )
    parser.add_argument("--seed-start", type=int, default=2026081900)
    args = parser.parse_args()
    audit = build_panels(args.dataset, args.output, seed_start=args.seed_start)
    print(
        f"built {sum(panel['topology_count'] for panel in audit['panels'])} "
        f"audited topologies -> {args.output}"
    )


if __name__ == "__main__":
    main()
