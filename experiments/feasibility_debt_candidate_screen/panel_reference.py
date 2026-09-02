"""Independent panel/split reconstruction with no project imports.

The production generator may call :func:`reconstruct` but this module does not
import production panel code, receipt helpers, or any retired study module.
"""

from __future__ import annotations

from collections import defaultdict, deque
from itertools import combinations
import hashlib
import json
from typing import Callable, Iterable


SEED_START = 2026090100
SEED_ATTEMPTS = 4096
SMOKE_SEED = 2026095000
OFFICIAL_SHA256 = "149f6aac17aff2e33750b4e1b6cebd3cef1c39d47ae49a3a7ed77315cb7838a7"
PRIOR_SHA256 = {
    "development-v1.json": "d5f660261e413f59b179d4fadf1f157b30f117aa265fd230d1d130bd6d69246b",
    "confirmation-v1.json": "52fe189709b27e2abb7de659fae0c080faf25b89f3ce66a3b1a13025be221dba",
    "submission-like-v1.json": "d85227f216528d635e56a93094e661721f62f379808707f310bf4da60d8fa57b",
    "coverage-robustness-v1.json": "e3385a6f4939445e869d71dbf7f1bd5119aa25eebbb89e083b1ec9be336f7309",
    "coverage-triage-v1.json": "f400cdc3a947cd076ce9bd9f48a2dafcb98dfd3f9f938a74ceb11ca88c360972",
    "restart-mechanics-v1.json": "2bc42026f52c09d85625ecce8d3ce0729c1efa06d0716511ed18d9d59c9f91c6",
    "restart-screen-v1.json": "dd1404e7b260c93a141b303c1a7f88f9ef02ceba03f109523708b2a8ed54b5d3",
}


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def topology_from_seed(seed: int) -> str:
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
        raise ValueError("reference topology generator returned invalid metadata")
    return topology_to_string(result[2], result[3], size=3)


def _bin(value: int, low_maximum: int, middle_maximum: int) -> str:
    if value <= low_maximum:
        return "low"
    if value <= middle_maximum:
        return "middle"
    return "high"


def _features(topology: str) -> dict[str, object]:
    pieces = topology.split("-")
    if len(pieces) != 2:
        raise ValueError("reference topology has an invalid delimiter")
    interior, boundary = pieces
    if (
        len(interior) != 9
        or any(token not in "ABCDEFGH" for token in interior)
        or len(boundary) != 12
        or boundary[0] not in "DH"
        or any(token not in "LS" for token in boundary[1:])
    ):
        raise ValueError("reference topology is not explicit size-3 UIFO")
    directional = sum(token in "EFGH" for token in interior)
    squeezers = boundary.count("S")
    return {
        "readout": boundary[0],
        "directional_interior_count": directional,
        "squeezer_count": squeezers,
        "directional_bin": _bin(directional, 3, 5),
        "squeezer_bin": _bin(squeezers, 4, 6),
    }


def _order() -> tuple[tuple[str, str, str], ...]:
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
    return tuple(
        (readout, squeezer, directional)
        for squeezer, directional in pairs
        for readout in ("D", "H")
    )


def _candidate_rows(
    topology_from_seed: Callable[[int], str], excluded: set[str]
) -> list[dict[str, object]]:
    seen = set(excluded)
    candidates: list[dict[str, object]] = []
    for seed in range(SEED_START, SEED_START + SEED_ATTEMPTS):
        topology = topology_from_seed(seed)
        if not isinstance(topology, str) or not topology:
            raise ValueError("reference topology generator returned invalid data")
        if topology in seen:
            continue
        seen.add(topology)
        candidates.append(
            {
                "topology_seed": seed,
                "topology": topology,
                **_features(topology),
            }
        )
    return candidates


def _select(candidates: list[dict[str, object]]) -> list[dict[str, object]]:
    buckets: dict[tuple[str, str, str], deque[dict[str, object]]] = defaultdict(deque)
    for candidate in candidates:
        key = (
            str(candidate["readout"]),
            str(candidate["squeezer_bin"]),
            str(candidate["directional_bin"]),
        )
        buckets[key].append(candidate)
    selected: list[dict[str, object]] = []
    while len(selected) < 8:
        progress = False
        for key in _order():
            if buckets[key]:
                selected.append(buckets[key].popleft())
                progress = True
                if len(selected) == 8:
                    break
        if not progress:
            raise ValueError("reference candidate pool cannot fill panel")
    return selected


def _counts(rows: Iterable[dict[str, object]]) -> dict[str, dict[str, int]]:
    materialized = list(rows)
    result: dict[str, dict[str, int]] = {}
    for field in ("readout", "squeezer_bin", "directional_bin"):
        values: dict[str, int] = defaultdict(int)
        for row in materialized:
            values[str(row[field])] += 1
        result[field] = dict(sorted(values.items()))
    return result


def _split(selected: list[dict[str, object]]) -> tuple[list[dict[str, object]], dict[str, object]]:
    legal: list[dict[str, object]] = []
    universe = set(range(8))
    for subset in combinations(range(8), 4):
        stage1 = tuple(subset)
        stage2 = tuple(sorted(universe - set(stage1)))
        rows1 = [selected[index] for index in stage1]
        rows2 = [selected[index] for index in stage2]
        counts1 = _counts(rows1)
        counts2 = _counts(rows2)
        if counts1["readout"] != {"D": 2, "H": 2}:
            continue
        if counts2["readout"] != {"D": 2, "H": 2}:
            continue
        legal_bins = True
        for field in ("squeezer_bin", "directional_bin"):
            values = set(counts1[field]) | set(counts2[field])
            if any(abs(counts1[field].get(v, 0) - counts2[field].get(v, 0)) > 1 for v in values):
                legal_bins = False
                break
        if not legal_bins:
            continue
        objective = [
            abs(sum(stage1) - sum(stage2)),
            abs(
                sum(int(selected[index]["topology_seed"]) for index in stage1)
                - sum(int(selected[index]["topology_seed"]) for index in stage2)
            ),
            hashlib.sha256(_canonical({"stage1_indices": list(stage1)})).hexdigest(),
        ]
        legal.append(
            {
                "stage1_indices": list(stage1),
                "stage2_indices": list(stage2),
                "objective": objective,
            }
        )
    if not legal:
        raise ValueError("reference reconstruction found no legal split")
    chosen = min(legal, key=lambda row: tuple(row["objective"]))
    return legal, chosen


def reconstruct(
    topology_from_seed: Callable[[int], str],
    *,
    official_topologies: set[str],
    prior_topologies: set[str],
) -> dict[str, object]:
    """Independently reconstruct the selected panel and all legal splits."""
    if official_topologies & prior_topologies:
        # Overlap among exclusions is harmless, but recording this explicitly
        # prevents callers from interpreting set cardinality as a row count.
        exclusion_overlap = len(official_topologies & prior_topologies)
    else:
        exclusion_overlap = 0
    candidates = _candidate_rows(
        topology_from_seed, official_topologies | prior_topologies
    )
    selected = _select(candidates)
    smoke = topology_from_seed(SMOKE_SEED)
    if smoke in official_topologies or smoke in prior_topologies or any(
        row["topology"] == smoke for row in selected
    ):
        raise ValueError("reference smoke topology is not disjoint")
    legal, chosen = _split(selected)
    return {
        "candidate_count": len(candidates),
        "exclusion_overlap_count": exclusion_overlap,
        "selected": selected,
        "legal_splits": legal,
        "chosen_split": chosen,
        "smoke_topology_sha256": hashlib.sha256(smoke.encode("utf-8")).hexdigest(),
        "stage1_counts": _counts(
            selected[index] for index in chosen["stage1_indices"]
        ),
        "stage2_counts": _counts(
            selected[index] for index in chosen["stage2_indices"]
        ),
    }
