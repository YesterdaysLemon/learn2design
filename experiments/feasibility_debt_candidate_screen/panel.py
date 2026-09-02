"""Append-only, outcome-blind panel generation for the candidate screen."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from itertools import combinations
import hashlib
import json
from pathlib import Path
from typing import Callable, Iterable

from .canonical import (
    ReceiptError,
    canonical_json_bytes,
    exclusive_write_bytes,
    sha256_bytes,
    sha256_file,
    write_receipt,
    read_receipt,
)
from .contract import (
    OFFICIAL_ARCHIVE_SHA256,
    PANEL_ID,
    PANEL_SEED_ATTEMPTS,
    PANEL_SEED_START,
    PRIOR_PANEL_SHA256,
    SMOKE_TOPOLOGY_SEED,
    STUDY_ID,
    UPSTREAM_REFERENCE,
)
from . import panel_reference


ROOT = Path(__file__).parents[2].resolve()
PANEL_BASENAME = "panel.json"
_AUTH_SENTINEL = object()
FORBIDDEN_OFFICIAL_FIELDS = [
    "loss",
    "parameters",
    "sensitivity",
    "power",
    "complexity",
    "history",
]
OWNER_TOPOLOGY_SCOPE_TEXT = (
    "I authorize one read of only entries.topology_string from the official "
    "Learn2Design archive with SHA-256 "
    f"{OFFICIAL_ARCHIVE_SHA256} for {STUDY_ID} panel exclusion; every outcome "
    "field remains forbidden."
)
OWNER_TOPOLOGY_SCOPE_TEXT_SHA256 = sha256_bytes(
    OWNER_TOPOLOGY_SCOPE_TEXT.encode("utf-8")
)


@dataclass(frozen=True)
class OfficialTopologyScope:
    receipt_sha256: str
    approval_text_sha256: str
    _sentinel: object

    def __post_init__(self) -> None:
        if self._sentinel is not _AUTH_SENTINEL:
            raise PermissionError("official topology scope is validator-issued only")


def authorize_official_topology_scope(path: Path) -> OfficialTopologyScope:
    payload, digest = read_receipt(
        path,
        expected_study_id=STUDY_ID,
        expected_receipt_type="owner_scope_authorization",
        expected_payload_keys={
            "panel_id",
            "official_archive_sha256",
            "allowed_column",
            "allow_read",
            "forbidden_fields",
            "approval_text",
            "approval_text_sha256",
        },
    )
    approval_digest = payload["approval_text_sha256"]
    if (
        payload["panel_id"] != PANEL_ID
        or payload["official_archive_sha256"] != OFFICIAL_ARCHIVE_SHA256
        or payload["allowed_column"] != "entries.topology_string"
        or payload["allow_read"] is not True
        or payload["forbidden_fields"] != FORBIDDEN_OFFICIAL_FIELDS
        or payload["approval_text"] != OWNER_TOPOLOGY_SCOPE_TEXT
        or approval_digest != OWNER_TOPOLOGY_SCOPE_TEXT_SHA256
    ):
        raise PermissionError("owner topology-scope receipt is invalid")
    return OfficialTopologyScope(
        receipt_sha256=digest,
        approval_text_sha256=approval_digest,
        _sentinel=_AUTH_SENTINEL,
    )


def _require_private_new_directory(path: Path) -> Path:
    target = path.resolve()
    if target == ROOT or target.is_relative_to(ROOT):
        raise ReceiptError("raw panel artifacts must remain outside Git")
    if target.exists():
        raise ReceiptError("append-only panel output directory already exists")
    target.mkdir(parents=True, exist_ok=False)
    return target


def topology_features(topology: str) -> dict[str, object]:
    pieces = topology.split("-")
    if len(pieces) != 2:
        raise ValueError("topology must contain one delimiter")
    interior, boundary = pieces
    if (
        len(interior) != 9
        or any(token not in "ABCDEFGH" for token in interior)
        or len(boundary) != 12
        or boundary[0] not in "DH"
        or any(token not in "LS" for token in boundary[1:])
    ):
        raise ValueError("topology is not an explicit size-3 UIFO topology")
    directional = sum(token in "EFGH" for token in interior)
    squeezers = boundary.count("S")

    def bin_name(value: int, low: int, middle: int) -> str:
        return "low" if value <= low else "middle" if value <= middle else "high"

    return {
        "readout": boundary[0],
        "directional_interior_count": directional,
        "squeezer_count": squeezers,
        "directional_bin": bin_name(directional, 3, 5),
        "squeezer_bin": bin_name(squeezers, 4, 6),
    }


def _stratum_order() -> tuple[tuple[str, str, str], ...]:
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


def candidate_pool(
    topology_from_seed: Callable[[int], str], excluded: set[str]
) -> list[dict[str, object]]:
    seen = set(excluded)
    rows: list[dict[str, object]] = []
    for seed in range(PANEL_SEED_START, PANEL_SEED_START + PANEL_SEED_ATTEMPTS):
        topology = topology_from_seed(seed)
        if not isinstance(topology, str) or not topology:
            raise ValueError("topology generator returned invalid data")
        if topology in seen:
            continue
        seen.add(topology)
        rows.append(
            {
                "topology_seed": seed,
                "topology": topology,
                **topology_features(topology),
            }
        )
    return rows


def select_panel(candidates: list[dict[str, object]]) -> list[dict[str, object]]:
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
        for key in _stratum_order():
            if buckets[key]:
                selected.append(buckets[key].popleft())
                progress = True
                if len(selected) == 8:
                    break
        if not progress:
            raise RuntimeError("candidate pool cannot fill the frozen panel")
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


def enumerate_legal_splits(
    selected: list[dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if len(selected) != 8:
        raise ValueError("split selection requires exactly eight rows")
    legal: list[dict[str, object]] = []
    universe = set(range(8))
    for subset in combinations(range(8), 4):
        stage1 = tuple(subset)
        stage2 = tuple(sorted(universe - set(stage1)))
        counts1 = _counts(selected[index] for index in stage1)
        counts2 = _counts(selected[index] for index in stage2)
        if counts1["readout"] != {"D": 2, "H": 2}:
            continue
        if counts2["readout"] != {"D": 2, "H": 2}:
            continue
        if any(
            abs(counts1[field].get(value, 0) - counts2[field].get(value, 0)) > 1
            for field in ("squeezer_bin", "directional_bin")
            for value in set(counts1[field]) | set(counts2[field])
        ):
            continue
        objective = [
            abs(sum(stage1) - sum(stage2)),
            abs(
                sum(int(selected[index]["topology_seed"]) for index in stage1)
                - sum(int(selected[index]["topology_seed"]) for index in stage2)
            ),
            hashlib.sha256(
                canonical_json_bytes({"stage1_indices": list(stage1)})
            ).hexdigest(),
        ]
        legal.append(
            {
                "stage1_indices": list(stage1),
                "stage2_indices": list(stage2),
                "objective": objective,
            }
        )
    if not legal:
        raise RuntimeError("no legal frozen Stage-1/Stage-2 split exists")
    chosen = min(legal, key=lambda row: tuple(row["objective"]))
    return legal, chosen


def _load_prior_panels(panel_dir: Path) -> tuple[set[str], list[dict[str, object]]]:
    identities: set[str] = set()
    rows: list[dict[str, object]] = []
    for name, expected_sha256 in PRIOR_PANEL_SHA256.items():
        path = panel_dir / name
        if not path.is_file() or sha256_file(path) != expected_sha256:
            raise ReceiptError(f"prior panel identity mismatch: {name}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        topologies = payload.get("topologies") if isinstance(payload, dict) else None
        if (
            not isinstance(topologies, list)
            or not topologies
            or not all(isinstance(value, str) and value for value in topologies)
            or len(topologies) != len(set(topologies))
        ):
            raise ReceiptError(f"prior panel topology schema mismatch: {name}")
        overlap = identities & set(topologies)
        rows.append(
            {
                "logical_id": name,
                "sha256": expected_sha256,
                "topology_count": len(topologies),
                "previous_exclusion_overlap_count": len(overlap),
            }
        )
        identities.update(topologies)
    return identities, rows


def load_official_topology_column(
    dataset_path: Path, *, authorization: OfficialTopologyScope
) -> tuple[set[str], dict[str, object]]:
    """Read only the authenticated official topology-string column.

    The explicit Boolean is a programmatic tripwire, not a substitute for the
    separate owner decision required by the frozen plan.
    """
    if (
        not isinstance(authorization, OfficialTopologyScope)
        or authorization._sentinel is not _AUTH_SENTINEL
    ):
        raise PermissionError("official topology-column access is not authorized")
    if sha256_file(dataset_path) != OFFICIAL_ARCHIVE_SHA256:
        raise ReceiptError("official archive SHA-256 mismatch")
    import h5py

    with h5py.File(dataset_path, "r") as archive:
        if "entries" not in archive:
            raise ReceiptError("official archive entries table is absent")
        entries = archive["entries"]
        if "topology_string" not in entries.dtype.fields:
            raise ReceiptError("official topology-string column is absent")
        raw = entries["topology_string"][:]
        count = int(len(entries))
    topologies = {
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in raw
    }
    return topologies, {
        "sha256": OFFICIAL_ARCHIVE_SHA256,
        "entries": count,
        "unique_topologies": len(topologies),
    }


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
        raise RuntimeError("Differometor did not return topology metadata")
    return topology_to_string(result[2], result[3], size=3)


def _public_candidate_row(index: int, row: dict[str, object]) -> dict[str, object]:
    return {
        "selection_index": index,
        "topology_seed": int(row["topology_seed"]),
        "topology_sha256": sha256_bytes(str(row["topology"]).encode("utf-8")),
        "readout": str(row["readout"]),
        "squeezer_count": int(row["squeezer_count"]),
        "squeezer_bin": str(row["squeezer_bin"]),
        "directional_interior_count": int(row["directional_interior_count"]),
        "directional_bin": str(row["directional_bin"]),
    }


def build_private_panel(
    *,
    dataset_path: Path,
    prior_panel_dir: Path,
    output_dir: Path,
    authorization: OfficialTopologyScope,
) -> dict[str, str]:
    """Generate one private panel and its canonical receipts exactly once."""
    target = _require_private_new_directory(output_dir)
    try:
        official, official_row = load_official_topology_column(
            dataset_path,
            authorization=authorization,
        )
        prior, prior_rows = _load_prior_panels(prior_panel_dir)
        candidates = candidate_pool(topology_from_seed, official | prior)
        selected = select_panel(candidates)
        smoke_topology = topology_from_seed(SMOKE_TOPOLOGY_SEED)
        if smoke_topology in official or smoke_topology in prior or any(
            row["topology"] == smoke_topology for row in selected
        ):
            raise ReceiptError("frozen smoke topology is not disjoint")
        legal, chosen = enumerate_legal_splits(selected)
        reference = panel_reference.reconstruct(
            panel_reference.topology_from_seed,
            official_topologies=set(official),
            prior_topologies=set(prior),
        )
        if reference["selected"] != selected:
            raise ReceiptError("independent panel reconstruction mismatch")
        if reference["legal_splits"] != legal or reference["chosen_split"] != chosen:
            raise ReceiptError("independent split reconstruction mismatch")
        if reference["smoke_topology_sha256"] != sha256_bytes(
            smoke_topology.encode("utf-8")
        ):
            raise ReceiptError("independent smoke reconstruction mismatch")

        panel_value = {
            "format_version": 1,
            "panel_id": PANEL_ID,
            "generation": {
                "method": "first eight exact round-robin eligible candidates",
                "seed_start": PANEL_SEED_START,
                "seed_attempts": PANEL_SEED_ATTEMPTS,
                "upstream_reference": UPSTREAM_REFERENCE,
            },
            "members": selected,
            "topologies": [str(row["topology"]) for row in selected],
        }
        panel_bytes = canonical_json_bytes(panel_value)
        panel_path = target / PANEL_BASENAME
        exclusive_write_bytes(panel_path, panel_bytes)
        panel_sha256 = sha256_bytes(panel_bytes)

        commitment_sha256 = write_receipt(
            target / "panel-commitment.json",
            study_id=STUDY_ID,
            receipt_type="panel_commitment",
            payload={
                "panel_id": PANEL_ID,
                "panel_sha256": panel_sha256,
                "official_archive": official_row,
                "prior_panels": prior_rows,
                "candidate_seed_start": PANEL_SEED_START,
                "candidate_seed_attempts": PANEL_SEED_ATTEMPTS,
                "eligible_unique_candidates": len(candidates),
                "archive_overlap_count": 0,
                "prior_panel_overlap_count": 0,
                "smoke_topology_seed": SMOKE_TOPOLOGY_SEED,
                "smoke_topology_sha256": sha256_bytes(
                    smoke_topology.encode("utf-8")
                ),
                "smoke_overlap_count": 0,
                "upstream_reference": UPSTREAM_REFERENCE,
            },
        )
        candidate_rows = [
            _public_candidate_row(index, row) for index, row in enumerate(selected)
        ]
        stratum_counts = {
            "stage1": _counts(selected[index] for index in chosen["stage1_indices"]),
            "stage2": _counts(selected[index] for index in chosen["stage2_indices"]),
        }
        split_sha256 = write_receipt(
            target / "split-receipt.json",
            study_id=STUDY_ID,
            receipt_type="split_receipt",
            payload={
                "panel_sha256": panel_sha256,
                "candidate_rows": candidate_rows,
                "legal_split_rows": legal,
                "chosen_stage1_indices": chosen["stage1_indices"],
                "chosen_stage2_indices": chosen["stage2_indices"],
                "stratum_counts": stratum_counts,
                "independent_verification": {
                    "status": "matched",
                    "candidate_count": reference["candidate_count"],
                    "selected_count": len(reference["selected"]),
                    "legal_split_count": len(reference["legal_splits"]),
                },
            },
        )
        return {
            "panel_sha256": panel_sha256,
            "panel_commitment_sha256": commitment_sha256,
            "split_receipt_sha256": split_sha256,
        }
    except BaseException:
        # No partially generated directory may be reused.  We intentionally do
        # not auto-delete it: its existence is a visible collision requiring a
        # fresh operator-selected path.
        raise
