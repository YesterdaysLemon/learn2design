from __future__ import annotations

import ast
import hashlib
import inspect
from pathlib import Path

import pytest

from experiments.feasibility_debt_candidate_screen.canonical import (
    ReceiptError,
    canonical_json_bytes,
    read_receipt,
    write_receipt,
)
from experiments.feasibility_debt_candidate_screen.contract import (
    ARM_ORDER,
    CHALLENGER_ORDER,
    OPTIMIZER_SETTINGS,
    STUDY_ID,
    arm_specs,
    stage1_order,
    stage2_order,
)
from experiments.feasibility_debt_candidate_screen.source_closure import (
    logical_source_closure,
)
from experiments.feasibility_debt_candidate_screen.evidence import compare_replays


ROOT = Path(__file__).parents[1]


def test_frozen_arm_contracts_are_complete_and_distinct() -> None:
    arms = arm_specs()
    assert tuple(arms) == ARM_ORDER
    assert tuple(arms)[1:] == CHALLENGER_ORDER
    assert arms["A_round1_control"].logical_module_id == "round1_zip::submission.py"
    assert arms["A_round1_control"].python_module_name == (
        "l2d_round1_control_submission"
    )
    assert arms["A_round1_control"].fixed_kwargs() == {
        **OPTIMIZER_SETTINGS,
        "patience": 600,
        "population_size": 8,
        "use_semantic_prior": False,
        "evaluation_chunk_size": None,
    }
    assert arms["B_round1_warmup"].fixed_kwargs()["preclock_warmup"] is True
    assert arms["C_v3_random"].fixed_kwargs()["progress_mode"] == (
        "feasibility_debt"
    )
    assert arms["D_v3_coverage"].fixed_kwargs()["initial_population_mode"] == (
        "coverage_balanced"
    )


def test_williams_and_stage2_orders_are_exact() -> None:
    assert stage1_order([0, 2, 4, 6]) == (
        (0, "A_round1_control"),
        (0, "B_round1_warmup"),
        (0, "C_v3_random"),
        (0, "D_v3_coverage"),
        (2, "B_round1_warmup"),
        (2, "A_round1_control"),
        (2, "D_v3_coverage"),
        (2, "C_v3_random"),
        (4, "C_v3_random"),
        (4, "D_v3_coverage"),
        (4, "A_round1_control"),
        (4, "B_round1_warmup"),
        (6, "D_v3_coverage"),
        (6, "C_v3_random"),
        (6, "B_round1_warmup"),
        (6, "A_round1_control"),
    )
    assert stage2_order([1, 3, 5, 7], "C_v3_random") == (
        (1, "A_round1_control"),
        (1, "C_v3_random"),
        (3, "C_v3_random"),
        (3, "A_round1_control"),
        (5, "A_round1_control"),
        (5, "C_v3_random"),
        (7, "C_v3_random"),
        (7, "A_round1_control"),
    )
    with pytest.raises(ValueError):
        stage1_order([0, 0, 1, 2])
    with pytest.raises(ValueError):
        stage2_order([1, 3, 5, 7], "A_round1_control")


def test_canonical_receipt_is_write_once_and_sidecar_bound(tmp_path: Path) -> None:
    path = tmp_path / "selection-receipt.json"
    digest = write_receipt(
        path,
        study_id=STUDY_ID,
        receipt_type="selection_receipt",
        payload={"answer": 7, "finite": True},
    )
    assert path.read_bytes() == canonical_json_bytes(
        {
            "payload": {"answer": 7, "finite": True},
            "receipt_type": "selection_receipt",
            "schema_version": 1,
            "study_id": STUDY_ID,
        }
    )
    assert (tmp_path / "selection-receipt.json.sha256").read_bytes() == (
        f"{digest}  selection-receipt.json\n".encode("ascii")
    )
    payload, observed = read_receipt(
        path,
        expected_study_id=STUDY_ID,
        expected_receipt_type="selection_receipt",
        expected_payload_keys={"answer", "finite"},
    )
    assert payload == {"answer": 7, "finite": True}
    assert observed == digest
    with pytest.raises(ReceiptError, match="already exists"):
        write_receipt(
            path,
            study_id=STUDY_ID,
            receipt_type="selection_receipt",
            payload={"answer": 8},
        )


def test_receipt_rejects_noncanonical_or_mutated_bytes(tmp_path: Path) -> None:
    path = tmp_path / "source-lock.json"
    write_receipt(
        path,
        study_id=STUDY_ID,
        receipt_type="source_lock",
        payload={"x": 1},
    )
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ReceiptError):
        read_receipt(
            path,
            expected_study_id=STUDY_ID,
            expected_receipt_type="source_lock",
        )


def test_reference_modules_have_no_project_imports() -> None:
    for relative in (
        "experiments/feasibility_debt_candidate_screen/reference_analysis.py",
        "experiments/feasibility_debt_candidate_screen/detached_analysis.py",
        "experiments/feasibility_debt_candidate_screen/panel_reference.py",
    ):
        tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
        forbidden: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module.startswith(("experiments", "submission", "tools")):
                    forbidden.append(module)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(("experiments", "submission", "tools")):
                        forbidden.append(alias.name)
        assert forbidden == []


def test_source_closure_contains_every_operational_module() -> None:
    closure = logical_source_closure(
        ROOT,
        ROOT / "artifacts/generated/submission.zip",
        ROOT / "artifacts/generated/submission.manifest.json",
    )
    package = ROOT / "experiments/feasibility_debt_candidate_screen"
    expected_package_ids = {
        path.relative_to(ROOT).as_posix() for path in package.glob("*.py")
    }
    assert expected_package_ids <= set(closure)
    assert {
        "experiments/feasibility_debt_candidate_screen/attempt.py",
        "experiments/feasibility_debt_candidate_screen/provider.py",
        "experiments/feasibility_debt_candidate_screen/hard_stop.py",
        "experiments/feasibility_debt_candidate_screen/evacuation.py",
        "experiments/feasibility_debt_candidate_screen/cleanup.py",
        "experiments/feasibility_debt_candidate_screen/host_finalizer.py",
    } <= set(closure)


def test_pod_runner_requires_external_host_finalizer_before_smoke() -> None:
    source = (
        ROOT / "experiments/feasibility_debt_candidate_screen/attempt.py"
    ).read_text(encoding="utf-8")
    host_gate = source.index('expected_receipt_type="host_finalizer"')
    smoke = source.index("controller.transition(Phase.SMOKE)")
    assert host_gate < smoke
    assert "ResourceAdapter" not in source
    assert "evidence_evacuation_receipt_sha256" in source


def test_every_compare_replays_call_uses_only_supported_keywords() -> None:
    tree = ast.parse(
        (
            ROOT / "experiments/feasibility_debt_candidate_screen/orchestrator.py"
        ).read_text(encoding="utf-8")
    )
    supported = set(inspect.signature(compare_replays).parameters)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "compare_replays"
    ]
    assert calls
    for call in calls:
        assert {keyword.arg for keyword in call.keywords} <= supported


def test_round1_archive_and_current_sources_match_frozen_hashes() -> None:
    arms = arm_specs()
    assert hashlib.sha256(
        (ROOT / "artifacts/generated/submission.zip").read_bytes()
    ).hexdigest() == (
        "4cc0dbc65a3e61ca5358c18655c432caf478fbdfc07f10512553781f8822924b"
    )
    assert hashlib.sha256(
        (ROOT / arms["B_round1_warmup"].logical_module_id).read_bytes()
    ).hexdigest() == arms["B_round1_warmup"].source_sha256
    assert hashlib.sha256(
        (ROOT / arms["C_v3_random"].logical_module_id).read_bytes()
    ).hexdigest() == arms["C_v3_random"].source_sha256


def test_every_retired_uifo_profile_and_panel_byte_remains_frozen() -> None:
    files = sorted(
        [
            *(
                ROOT / "experiments/uifo_paired"
            ).rglob("*"),
            ROOT / "tools/build_topology_panels.py",
        ],
        key=lambda path: path.relative_to(ROOT).as_posix(),
    )
    files = [
        path
        for path in files
        if path.is_file() and "__pycache__" not in path.parts
    ]
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    assert len(files) == 45
    assert digest.hexdigest() == (
        "90c29df3cc8f36cc4f73757db8152c23f188d8854ecc74d97d8e27df185c08ef"
    )
