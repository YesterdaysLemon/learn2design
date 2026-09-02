from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.feasibility_debt_candidate_screen.canonical import (
    ReceiptError,
    read_receipt,
    write_receipt,
)
from experiments.feasibility_debt_candidate_screen.contract import (
    PANEL_SEED_START,
    STUDY_ID,
)
from experiments.feasibility_debt_candidate_screen.panel import (
    FORBIDDEN_OFFICIAL_FIELDS,
    OWNER_TOPOLOGY_SCOPE_TEXT,
    OWNER_TOPOLOGY_SCOPE_TEXT_SHA256,
    authorize_official_topology_scope,
    build_private_panel,
    candidate_pool,
    enumerate_legal_splits,
    select_panel,
)


ROOT = Path(__file__).parents[1]
PANEL_DIR = ROOT / "experiments/uifo_paired/panels"


def _topology(seed: int) -> str:
    offset = seed - PANEL_SEED_START
    stratum = offset % 18
    pair = stratum // 2
    readout = "D" if stratum % 2 == 0 else "H"
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
    squeezer_bin, directional_bin = pairs[pair]
    directional_count = {"low": 2, "middle": 4, "high": 6}[directional_bin]
    squeezer_count = {"low": 3, "middle": 5, "high": 7}[squeezer_bin]
    value = max(offset, 0)
    digits = []
    for _ in range(9):
        digits.append(value % 4)
        value //= 4
    directional = "EFGH"
    nondirectional = "ABCD"
    interior = "".join(
        (directional if index < directional_count else nondirectional)[digit]
        for index, digit in enumerate(digits)
    )
    boundary = readout + "S" * squeezer_count + "L" * (11 - squeezer_count)
    return f"{interior}-{boundary}"


def test_candidate_range_round_robin_and_split_are_frozen() -> None:
    calls: list[int] = []

    def resolver(seed: int) -> str:
        calls.append(seed)
        return _topology(seed)

    candidates = candidate_pool(resolver, set())
    assert calls == list(range(PANEL_SEED_START, PANEL_SEED_START + 4096))
    selected = select_panel(candidates)
    assert len(selected) == 8
    assert [row["topology_seed"] for row in selected] == [
        PANEL_SEED_START + offset for offset in range(8)
    ]
    legal, chosen = enumerate_legal_splits(selected)
    assert 1 <= len(legal) <= 70
    assert len(chosen["stage1_indices"]) == 4
    assert set(chosen["stage1_indices"]).isdisjoint(chosen["stage2_indices"])
    assert sorted(chosen["stage1_indices"] + chosen["stage2_indices"]) == list(
        range(8)
    )


def test_private_panel_bundle_is_append_only_and_independently_replayed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prior = set()
    for path in PANEL_DIR.glob("*.json"):
        if path.name == "audit.json":
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        prior.update(value["topologies"])

    monkeypatch.setattr(
        "experiments.feasibility_debt_candidate_screen.panel.load_official_topology_column",
        lambda *_args, **_kwargs: (
            set(),
            {"sha256": "149f" + "0" * 60, "entries": 0, "unique_topologies": 0},
        ),
    )
    monkeypatch.setattr(
        "experiments.feasibility_debt_candidate_screen.panel.topology_from_seed",
        _topology,
    )
    monkeypatch.setattr(
        "experiments.feasibility_debt_candidate_screen.panel_reference.topology_from_seed",
        _topology,
    )
    scope_path = tmp_path / "owner-scope.json"
    write_receipt(
        scope_path,
        study_id=STUDY_ID,
        receipt_type="owner_scope_authorization",
        payload={
            "panel_id": STUDY_ID,
            "official_archive_sha256": (
                "149f6aac17aff2e33750b4e1b6cebd3cef1c39d47ae49a3a7ed77315cb7838a7"
            ),
            "allowed_column": "entries.topology_string",
            "allow_read": True,
            "forbidden_fields": FORBIDDEN_OFFICIAL_FIELDS,
            "approval_text": OWNER_TOPOLOGY_SCOPE_TEXT,
            "approval_text_sha256": OWNER_TOPOLOGY_SCOPE_TEXT_SHA256,
        },
    )
    authorization = authorize_official_topology_scope(scope_path)
    output = tmp_path / "private-panel"
    result = build_private_panel(
        dataset_path=tmp_path / "never-opened.h5",
        prior_panel_dir=PANEL_DIR,
        output_dir=output,
        authorization=authorization,
    )
    assert set(result) == {
        "panel_sha256",
        "panel_commitment_sha256",
        "split_receipt_sha256",
    }
    panel = json.loads((output / "panel.json").read_text(encoding="utf-8"))
    assert panel["panel_id"] == STUDY_ID
    assert len(panel["members"]) == 8
    split, digest = read_receipt(
        output / "split-receipt.json",
        expected_study_id=STUDY_ID,
        expected_receipt_type="split_receipt",
        expected_payload_keys={
            "panel_sha256",
            "candidate_rows",
            "legal_split_rows",
            "chosen_stage1_indices",
            "chosen_stage2_indices",
            "stratum_counts",
            "independent_verification",
        },
    )
    assert digest == result["split_receipt_sha256"]
    assert split["independent_verification"]["status"] == "matched"
    assert all("topology" not in row for row in split["candidate_rows"])
    with pytest.raises(ReceiptError, match="already exists"):
        build_private_panel(
            dataset_path=tmp_path / "never-opened.h5",
            prior_panel_dir=PANEL_DIR,
            output_dir=output,
            authorization=authorization,
        )


def test_official_topology_access_defaults_closed(tmp_path: Path) -> None:
    from experiments.feasibility_debt_candidate_screen.panel import (
        load_official_topology_column,
    )

    with pytest.raises(PermissionError, match="not authorized"):
        load_official_topology_column(
            tmp_path / "absent.h5", authorization=None  # type: ignore[arg-type]
        )


def test_topology_grammar_rejects_non_uifo_characters() -> None:
    from experiments.feasibility_debt_candidate_screen.panel import topology_features

    with pytest.raises(ValueError, match="explicit size-3"):
        topology_features("AAAAAAAAX-DLLLLLLLLLLL")
    with pytest.raises(ValueError, match="explicit size-3"):
        topology_features("AAAAAAAAA-DLLLLLLLLLLX")
