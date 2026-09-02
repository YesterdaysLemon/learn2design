"""Frozen identities and run contracts for the candidate screen."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


STUDY_ID = "feasibility-debt-candidate-screen-v1"
PANEL_ID = STUDY_ID
PLAN_REVISION = "2a8cd4c4800485ad6564d6702c741a221507af07"
UPSTREAM_REFERENCE = "1bb7f54737dec6a08b59879a8831d125f08f8a0b"
OFFICIAL_ARCHIVE_SHA256 = (
    "149f6aac17aff2e33750b4e1b6cebd3cef1c39d47ae49a3a7ed77315cb7838a7"
)
REFERENCE_GENERATOR_SHA256 = (
    "b3a85a9f66a2f35f13f143da3c580c91fe5a642b489e4940841306cae9a91229"
)
OPTIMIZER_SETTINGS_SHA256 = (
    "bba97635abcc1599257826bb6eb775fe3cdeb6f9ddb98fb615397d353642f858"
)
PYPROJECT_SHA256 = (
    "8f412a28ddbeb284464f6c49ce2cf7c86f7e227881d8aa07f84780637a7ff5f4"
)
UV_LOCK_SHA256 = (
    "3e98259ff36b73445a93cbaeeb51d4454677677a8a2661b431add0a37996c013"
)

ROUND1_REVISION = "5ce3cdb2ddf4c505622a0aeef805936a4ea607d7"
ROUND1_ARCHIVE_SHA256 = (
    "4cc0dbc65a3e61ca5358c18655c432caf478fbdfc07f10512553781f8822924b"
)
ROUND1_MANIFEST_SHA256 = (
    "99e18299ba6dd232f203cc2e59131cea61cd83ff8b0a9bceab6c56c764ddd86a"
)
ROUND1_MEMBER_SHA256 = {
    "requirements.txt": "776cf62a1ac0727d4975a6434936449ff008540a3cfbbe5ddad579ecfd9e23d1",
    "semantic_prior.json": "c08cefb94f0285d9681ab8125c23545cc93c7231d0b5aefe849a80c74a4f4312",
    "submission.py": "45f18e17f3e9e0855629079c311a4c01318a3d2e6700db4719e2a54d61ffea76",
}
PROTECTED_SOURCE_SHA256 = (
    "0fefbaaf18d9831895d788df45c92cbaf4522da7c54d8f78646e449ffa9374c9"
)
V3_SOURCE_SHA256 = (
    "ca7abd365c5d1172dab2f47fccdf0afa3df9652e75cc2003385312cec48844d6"
)

PRIOR_PANEL_SHA256 = {
    "development-v1.json": "d5f660261e413f59b179d4fadf1f157b30f117aa265fd230d1d130bd6d69246b",
    "confirmation-v1.json": "52fe189709b27e2abb7de659fae0c080faf25b89f3ce66a3b1a13025be221dba",
    "submission-like-v1.json": "d85227f216528d635e56a93094e661721f62f379808707f310bf4da60d8fa57b",
    "coverage-robustness-v1.json": "e3385a6f4939445e869d71dbf7f1bd5119aa25eebbb89e083b1ec9be336f7309",
    "coverage-triage-v1.json": "f400cdc3a947cd076ce9bd9f48a2dafcb98dfd3f9f938a74ceb11ca88c360972",
    "restart-mechanics-v1.json": "2bc42026f52c09d85625ecce8d3ce0729c1efa06d0716511ed18d9d59c9f91c6",
    "restart-screen-v1.json": "dd1404e7b260c93a141b303c1a7f88f9ef02ceba03f109523708b2a8ed54b5d3",
}

OPTIMIZER_SETTINGS: dict[str, int | float] = {
    "learning_rate_low": 0.03,
    "learning_rate_high": 0.15,
    "minimum_improvement": 1e-7,
    "beta1": 0.9,
    "beta2": 0.999,
    "epsilon": 1e-8,
    "gradient_clip_norm": 1.0,
    "restart_noise_scale": 0.35,
    "safety_seconds": 2.0,
    "batch_time_safety_factor": 1.5,
    "batch_time_window": 8,
}

ARM_ORDER = (
    "A_round1_control",
    "B_round1_warmup",
    "C_v3_random",
    "D_v3_coverage",
)
CHALLENGER_ORDER = ARM_ORDER[1:]
STAGE1_ARM_ROWS = (
    ARM_ORDER,
    (ARM_ORDER[1], ARM_ORDER[0], ARM_ORDER[3], ARM_ORDER[2]),
    (ARM_ORDER[2], ARM_ORDER[3], ARM_ORDER[0], ARM_ORDER[1]),
    (ARM_ORDER[3], ARM_ORDER[2], ARM_ORDER[1], ARM_ORDER[0]),
)
STAGE1_OPTIMIZER_SEED = 20260901
STAGE2_OPTIMIZER_SEED = 20260902
SMOKE_TOPOLOGY_SEED = 2026095000
SMOKE_OPTIMIZER_SEED = 20260900
PANEL_SEED_START = 2026090100
PANEL_SEED_ATTEMPTS = 4096
POPULATION_SIZE = 8
N_FREQUENCIES = 50
MAX_TIME_SECONDS = 600.0
WORKER_TIMEOUT_SECONDS = 900.0
VALID_RUN_WALL_SECONDS = 720.0


@dataclass(frozen=True)
class ArmSpec:
    arm_id: str
    logical_module_id: str
    python_module_name: str
    class_name: str
    algorithm_str: str
    source_sha256: str
    patience: int
    population_mode: str
    preclock_warmup: bool
    progress_mode: str

    def fixed_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            **OPTIMIZER_SETTINGS,
            "patience": self.patience,
            "population_size": POPULATION_SIZE,
            "use_semantic_prior": False,
            "evaluation_chunk_size": None,
        }
        if self.arm_id != "A_round1_control":
            kwargs.update(
                {
                    "initial_population_mode": self.population_mode,
                    "preclock_warmup": self.preclock_warmup,
                }
            )
        if self.arm_id in {"C_v3_random", "D_v3_coverage"}:
            kwargs["progress_mode"] = self.progress_mode
        return kwargs

    def lock_row(self, package_closure_sha256: str) -> dict[str, Any]:
        return {
            **asdict(self),
            "kwargs": self.fixed_kwargs(),
            "package_closure_sha256": package_closure_sha256,
        }


_ARMS = {
    "A_round1_control": ArmSpec(
        arm_id="A_round1_control",
        logical_module_id="round1_zip::submission.py",
        python_module_name="l2d_round1_control_submission",
        class_name="BatchedRestartAdam",
        algorithm_str="batched_restart_adam",
        source_sha256=ROUND1_MEMBER_SHA256["submission.py"],
        patience=600,
        population_mode="random",
        preclock_warmup=False,
        progress_mode="protected_total_loss",
    ),
    "B_round1_warmup": ArmSpec(
        arm_id="B_round1_warmup",
        logical_module_id="submission/submission.py",
        python_module_name="submission.submission",
        class_name="BatchedRestartAdam",
        algorithm_str="batched_restart_adam",
        source_sha256=PROTECTED_SOURCE_SHA256,
        patience=600,
        population_mode="random",
        preclock_warmup=True,
        progress_mode="protected_total_loss",
    ),
    "C_v3_random": ArmSpec(
        arm_id="C_v3_random",
        logical_module_id="experiments/candidates/feasibility_debt_clock_v3.py",
        python_module_name="experiments.candidates.feasibility_debt_clock_v3",
        class_name="FeasibilityDebtBatchedRestartAdamV3",
        algorithm_str="feasibility_debt_batched_restart_adam_v3",
        source_sha256=V3_SOURCE_SHA256,
        patience=200,
        population_mode="random",
        preclock_warmup=True,
        progress_mode="feasibility_debt",
    ),
    "D_v3_coverage": ArmSpec(
        arm_id="D_v3_coverage",
        logical_module_id="experiments/candidates/feasibility_debt_clock_v3.py",
        python_module_name="experiments.candidates.feasibility_debt_clock_v3",
        class_name="FeasibilityDebtBatchedRestartAdamV3",
        algorithm_str="feasibility_debt_batched_restart_adam_v3",
        source_sha256=V3_SOURCE_SHA256,
        patience=200,
        population_mode="coverage_balanced",
        preclock_warmup=True,
        progress_mode="feasibility_debt",
    ),
}


def arm_specs() -> dict[str, ArmSpec]:
    return dict(_ARMS)


def arm_spec(arm_id: object) -> ArmSpec:
    if not isinstance(arm_id, str) or arm_id not in _ARMS:
        raise ValueError(f"unknown frozen arm: {arm_id!r}")
    return _ARMS[arm_id]


def run_id(stage: int, member_index: int, position: int, arm_id: str) -> str:
    arm_spec(arm_id)
    if stage not in (1, 2) or member_index not in range(8) or position not in range(4):
        raise ValueError("invalid run identity component")
    return f"s{stage}-m{member_index:02d}-p{position}-{arm_id}"


def stage1_order(stage1_indices: list[int] | tuple[int, ...]) -> tuple[tuple[int, str], ...]:
    indices = tuple(stage1_indices)
    if (
        len(indices) != 4
        or tuple(sorted(indices)) != indices
        or len(set(indices)) != 4
        or any(type(index) is not int or index not in range(8) for index in indices)
    ):
        raise ValueError("Stage-1 indices must be four unique ascending members")
    return tuple(
        (member_index, arm_id)
        for row, member_index in zip(STAGE1_ARM_ROWS, indices, strict=True)
        for arm_id in row
    )


def stage2_order(
    stage2_indices: list[int] | tuple[int, ...], finalist: str
) -> tuple[tuple[int, str], ...]:
    indices = tuple(stage2_indices)
    if (
        len(indices) != 4
        or tuple(sorted(indices)) != indices
        or len(set(indices)) != 4
        or any(type(index) is not int or index not in range(8) for index in indices)
    ):
        raise ValueError("Stage-2 indices must be four unique ascending members")
    if finalist not in CHALLENGER_ORDER:
        raise ValueError("Stage-2 finalist is invalid")
    return tuple(
        (member_index, arm_id)
        for offset, member_index in enumerate(indices)
        for arm_id in (
            (ARM_ORDER[0], finalist) if offset % 2 == 0 else (finalist, ARM_ORDER[0])
        )
    )
