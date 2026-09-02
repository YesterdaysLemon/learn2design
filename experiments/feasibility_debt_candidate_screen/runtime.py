"""Seven-hour provider horizon and non-transfer operation budget."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from enum import Enum
import math
from pathlib import Path

from .canonical import read_receipt, sha256_file
from .contract import STUDY_ID


class RuntimeGuardError(RuntimeError):
    pass


MAX_HORIZON_SECONDS = 25_200
CLEANUP_RESERVE_SECONDS = 1_800
MINIMUM_SUCCESS_HORIZON_SECONDS = 21_900
PROVIDER_CAP_USD = Decimal("25.00")
PRICE_CEILING_USD_PER_HOUR = Decimal("3.5714285714")
GPU_PRICE_CEILING_USD_PER_HOUR = Decimal("3.29")

OPERATION_BOUNDS = {
    "post_running_preflight": 300,
    "cold_smoke": 300,
    "scored_worker": 720,
    "stage1_seal_and_replay": 900,
    "stage2_materialization": 120,
    "stage2_seal_and_replay": 900,
    "terminal_seal": 300,
    "cleanup_evacuation": 900,
    "cleanup_deletion": 900,
}

OPERATION_LIMITS = {
    "post_running_preflight": 1,
    "cold_smoke": 1,
    "scored_worker": 24,
    "stage1_seal_and_replay": 1,
    "stage2_materialization": 1,
    "stage2_seal_and_replay": 1,
    "terminal_seal": 1,
    "cleanup_evacuation": 1,
    "cleanup_deletion": 1,
}


def parse_utc(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError) as error:
        raise RuntimeGuardError("provider timestamp is invalid") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise RuntimeGuardError("provider timestamp must be UTC")
    return parsed.astimezone(UTC)


@dataclass(frozen=True)
class BillingSemantics:
    currency: str
    gpu_hourly_rate_usd: Decimal
    combined_hourly_rate_usd: Decimal
    fixed_charge_usd: Decimal = Decimal("0")
    metering_quantum_seconds: int = 1
    round_up_each_quantum: bool = False

    def __post_init__(self) -> None:
        if (
            self.currency != "USD"
            or type(self.gpu_hourly_rate_usd) is not Decimal
            or not self.gpu_hourly_rate_usd.is_finite()
            or self.gpu_hourly_rate_usd <= 0
            or self.gpu_hourly_rate_usd > GPU_PRICE_CEILING_USD_PER_HOUR
            or self.gpu_hourly_rate_usd * Decimal("7") > Decimal("23.03")
            or type(self.combined_hourly_rate_usd) is not Decimal
            or not self.combined_hourly_rate_usd.is_finite()
            or self.combined_hourly_rate_usd <= 0
            or self.combined_hourly_rate_usd > PRICE_CEILING_USD_PER_HOUR
            or self.combined_hourly_rate_usd < self.gpu_hourly_rate_usd
        ):
            raise RuntimeGuardError("combined provider rate exceeds frozen ceiling")
        if (
            type(self.fixed_charge_usd) is not Decimal
            or not self.fixed_charge_usd.is_finite()
            or self.fixed_charge_usd < 0
            or self.fixed_charge_usd != 0
        ):
            raise RuntimeGuardError("fixed provider charges are forbidden")
        if (
            isinstance(self.metering_quantum_seconds, bool)
            or not isinstance(self.metering_quantum_seconds, int)
            or self.metering_quantum_seconds < 1
        ):
            raise RuntimeGuardError("provider metering quantum is invalid")

    def charge_at_seconds(self, seconds: int | Decimal) -> Decimal:
        elapsed = Decimal(seconds)
        if not elapsed.is_finite() or elapsed < 0:
            raise RuntimeGuardError("billing elapsed seconds are invalid")
        if self.round_up_each_quantum:
            quantum = Decimal(self.metering_quantum_seconds)
            elapsed = (elapsed / quantum).to_integral_value(
                rounding=ROUND_CEILING
            ) * quantum
        return self.fixed_charge_usd + (
            elapsed / Decimal(3600)
        ) * self.combined_hourly_rate_usd

    def conservative_cap_seconds(self) -> int:
        available = PROVIDER_CAP_USD - self.fixed_charge_usd
        if available <= 0:
            raise RuntimeGuardError("provider cap is exhausted before launch")
        continuous = available * Decimal(3600) / self.combined_hourly_rate_usd
        seconds = int(continuous.to_integral_value(rounding=ROUND_FLOOR))
        if self.round_up_each_quantum:
            quantum = self.metering_quantum_seconds
            seconds = (seconds // quantum) * quantum
            while seconds > 0 and self.charge_at_seconds(seconds) > PROVIDER_CAP_USD:
                seconds -= quantum
        return max(0, seconds)


@dataclass(frozen=True)
class ProviderLaunchReceipt:
    provider: str
    quote_sha256: str
    authenticated_response_sha256: str
    resource_request_sha256: str
    resource_manifest_sha256: str
    task_scope_sha256: str
    resource_id: str
    immutable_image_digest: str
    status: str
    create_utc: str
    running_utc: str
    billable_utc: str | None
    cloud_type: str
    gpu_model: str
    gpu_count: int
    ephemeral_disk_gib: int
    provider_running_hourly_cost_usd: str
    billing: BillingSemantics

    def __post_init__(self) -> None:
        for digest in (
            self.quote_sha256,
            self.authenticated_response_sha256,
            self.resource_request_sha256,
            self.resource_manifest_sha256,
            self.task_scope_sha256,
        ):
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(token not in "0123456789abcdef" for token in digest)
            ):
                raise RuntimeGuardError("provider receipt digest is invalid")
        if (
            self.provider != "runpod"
            or self.cloud_type != "SECURE"
            or self.gpu_model != "NVIDIA H100 80GB HBM3"
            or type(self.gpu_count) is not int
            or self.gpu_count != 1
            or type(self.ephemeral_disk_gib) is not int
            or not 0 < self.ephemeral_disk_gib <= 40
            or not isinstance(self.provider_running_hourly_cost_usd, str)
            or not isinstance(self.resource_id, str)
            or not self.resource_id
            or not isinstance(self.immutable_image_digest, str)
            or not self.immutable_image_digest.startswith("sha256:")
            or len(self.immutable_image_digest) != 71
            or any(
                token not in "0123456789abcdef"
                for token in self.immutable_image_digest.removeprefix("sha256:")
            )
            or self.status != "RUNNING"
        ):
            raise RuntimeGuardError("provider resource receipt is outside the envelope")
        try:
            running_cost = Decimal(self.provider_running_hourly_cost_usd)
        except Exception as error:
            raise RuntimeGuardError("provider running cost is invalid") from error
        if (
            str(running_cost) != self.provider_running_hourly_cost_usd
            or not running_cost.is_finite()
            or running_cost <= 0
            or running_cost > self.billing.combined_hourly_rate_usd
        ):
            raise RuntimeGuardError("provider running cost exceeds the quote")


def load_provider_launch_receipt(
    path: Path,
    *,
    expected_resource_manifest_sha256: str,
    expected_resource_request_sha256: str,
    expected_quote_sha256: str,
    authenticated_response_path: Path,
) -> tuple[ProviderLaunchReceipt, str]:
    payload, receipt_sha256 = read_receipt(
        path,
        expected_study_id=STUDY_ID,
        expected_receipt_type="provider_launch",
        expected_payload_keys={
            "provider",
            "quote_sha256",
            "authenticated_response_sha256",
            "resource_request_sha256",
            "resource_manifest_sha256",
            "task_scope_sha256",
            "resource_id",
            "immutable_image_digest",
            "status",
            "create_utc",
            "running_utc",
            "billable_utc",
            "cloud_type",
            "gpu_model",
            "gpu_count",
            "ephemeral_disk_gib",
            "provider_running_hourly_cost_usd",
            "billing",
        },
    )
    billing = payload["billing"]
    if not isinstance(billing, dict) or set(billing) != {
        "currency",
        "gpu_hourly_rate_usd",
        "combined_hourly_rate_usd",
        "fixed_charge_usd",
        "metering_quantum_seconds",
        "round_up_each_quantum",
    }:
        raise RuntimeGuardError("provider billing receipt schema mismatch")

    def decimal_field(name: str) -> Decimal:
        value = billing[name]
        if not isinstance(value, str):
            raise RuntimeGuardError("provider billing decimal must be a string")
        try:
            parsed = Decimal(value)
        except Exception as error:
            raise RuntimeGuardError("provider billing decimal is invalid") from error
        if str(parsed) != value:
            raise RuntimeGuardError("provider billing decimal is not canonical")
        return parsed

    launch = ProviderLaunchReceipt(
        provider=payload["provider"],
        quote_sha256=payload["quote_sha256"],
        authenticated_response_sha256=payload["authenticated_response_sha256"],
        resource_request_sha256=payload["resource_request_sha256"],
        resource_manifest_sha256=payload["resource_manifest_sha256"],
        task_scope_sha256=payload["task_scope_sha256"],
        resource_id=payload["resource_id"],
        immutable_image_digest=payload["immutable_image_digest"],
        status=payload["status"],
        create_utc=payload["create_utc"],
        running_utc=payload["running_utc"],
        billable_utc=payload["billable_utc"],
        cloud_type=payload["cloud_type"],
        gpu_model=payload["gpu_model"],
        gpu_count=payload["gpu_count"],
        ephemeral_disk_gib=payload["ephemeral_disk_gib"],
        provider_running_hourly_cost_usd=(
            payload["provider_running_hourly_cost_usd"]
        ),
        billing=BillingSemantics(
            currency=billing["currency"],
            gpu_hourly_rate_usd=decimal_field("gpu_hourly_rate_usd"),
            combined_hourly_rate_usd=decimal_field("combined_hourly_rate_usd"),
            fixed_charge_usd=decimal_field("fixed_charge_usd"),
            metering_quantum_seconds=billing["metering_quantum_seconds"],
            round_up_each_quantum=billing["round_up_each_quantum"],
        ),
    )
    if (
        launch.resource_manifest_sha256 != expected_resource_manifest_sha256
        or launch.resource_request_sha256 != expected_resource_request_sha256
        or launch.quote_sha256 != expected_quote_sha256
        or sha256_file(authenticated_response_path)
        != launch.authenticated_response_sha256
    ):
        raise RuntimeGuardError("provider launch evidence binding mismatch")
    return launch, receipt_sha256


@dataclass(frozen=True)
class DeadlineClock:
    t0: datetime
    b0: datetime
    hard_horizon: datetime
    dispatch_deadline: datetime
    billing: BillingSemantics

    @classmethod
    def from_authenticated_receipt(
        cls,
        receipt: ProviderLaunchReceipt,
    ) -> "DeadlineClock":
        t0 = parse_utc(receipt.running_utc)
        create = parse_utc(receipt.create_utc)
        b0 = parse_utc(receipt.billable_utc) if receipt.billable_utc is not None else create
        if create > t0 or create > b0:
            raise RuntimeGuardError("provider timestamp chronology is invalid")
        billing = receipt.billing
        cap_seconds = billing.conservative_cap_seconds()
        hard = min(
            t0 + timedelta(seconds=MAX_HORIZON_SECONDS),
            b0 + timedelta(seconds=MAX_HORIZON_SECONDS),
            b0 + timedelta(seconds=cap_seconds),
        )
        return cls(
            t0=t0,
            b0=b0,
            hard_horizon=hard,
            dispatch_deadline=hard
            - timedelta(seconds=CLEANUP_RESERVE_SECONDS),
            billing=billing,
        )

    @property
    def usable_from_t0_seconds(self) -> Decimal:
        delta = self.hard_horizon - self.t0
        return (
            Decimal(delta.days * 86_400 + delta.seconds)
            + Decimal(delta.microseconds) / Decimal(1_000_000)
        )

    def billed_elapsed_seconds(self, now: datetime) -> Decimal:
        delta = now - self.b0
        value = (
            Decimal(delta.days * 86_400 + delta.seconds)
            + Decimal(delta.microseconds) / Decimal(1_000_000)
        )
        return max(Decimal(0), value)

    def require_success_envelope(self, now: datetime) -> None:
        if now.tzinfo is None or now.utcoffset() != timedelta(0):
            raise RuntimeGuardError("deadline checks require UTC time")
        if self.usable_from_t0_seconds < Decimal(MINIMUM_SUCCESS_HORIZON_SECONDS):
            raise RuntimeGuardError("provider setup consumed the frozen success slack")
        self.admit("post_running_preflight", now)

    def admit(self, operation: str, now: datetime) -> None:
        if now.tzinfo is None or now.utcoffset() != timedelta(0):
            raise RuntimeGuardError("deadline checks require UTC time")
        if self.billing.charge_at_seconds(self.billed_elapsed_seconds(now)) > PROVIDER_CAP_USD:
            raise RuntimeGuardError("provider charge cap is exhausted")
        if operation not in OPERATION_BOUNDS:
            raise RuntimeGuardError("unknown operation bound")
        bound = OPERATION_BOUNDS[operation]
        deadline = (
            self.hard_horizon
            if operation.startswith("cleanup_")
            else self.dispatch_deadline
        )
        if now + timedelta(seconds=bound) > deadline:
            raise RuntimeGuardError(f"{operation} does not fit before its deadline")

    def snapshot(self) -> dict[str, object]:
        return {
            "t0_utc": self.t0.isoformat().replace("+00:00", "Z"),
            "b0_utc": self.b0.isoformat().replace("+00:00", "Z"),
            "hard_horizon_utc": self.hard_horizon.isoformat().replace(
                "+00:00", "Z"
            ),
            "dispatch_deadline_utc": self.dispatch_deadline.isoformat().replace(
                "+00:00", "Z"
            ),
            "combined_hourly_rate_usd": str(
                self.billing.combined_hourly_rate_usd
            ),
            "fixed_charge_usd": str(self.billing.fixed_charge_usd),
            "metering_quantum_seconds": self.billing.metering_quantum_seconds,
            "round_up_each_quantum": self.billing.round_up_each_quantum,
        }


class Phase(str, Enum):
    PREFLIGHT = "preflight"
    SMOKE = "smoke"
    STAGE1 = "stage1"
    STAGE1_SEALED = "stage1_sealed"
    STAGE2 = "stage2"
    FINALIZE = "finalize"
    CLEANUP = "cleanup"
    TERMINAL = "terminal"
    NOT_EVALUABLE = "not_evaluable"


_TRANSITIONS = {
    Phase.PREFLIGHT: {Phase.SMOKE, Phase.CLEANUP, Phase.NOT_EVALUABLE},
    Phase.SMOKE: {Phase.STAGE1, Phase.CLEANUP, Phase.NOT_EVALUABLE},
    Phase.STAGE1: {Phase.STAGE1_SEALED, Phase.CLEANUP, Phase.NOT_EVALUABLE},
    Phase.STAGE1_SEALED: {Phase.STAGE2, Phase.CLEANUP, Phase.NOT_EVALUABLE},
    Phase.STAGE2: {Phase.FINALIZE, Phase.CLEANUP, Phase.NOT_EVALUABLE},
    Phase.FINALIZE: {Phase.CLEANUP, Phase.NOT_EVALUABLE},
    Phase.NOT_EVALUABLE: {Phase.CLEANUP},
    Phase.CLEANUP: {Phase.TERMINAL},
    Phase.TERMINAL: set(),
}


@dataclass
class DeadlineController:
    clock: DeadlineClock
    provider_launch_receipt_sha256: str
    resource_manifest_sha256: str
    hard_stop_receipt_sha256: str
    hard_stop_liveness: Callable[[], None]
    phase: Phase = Phase.PREFLIGHT
    dispatched_scored_runs: int = 0
    smoke_receipt_sha256: str | None = None
    operation_admissions: dict[str, int] | None = None

    def __post_init__(self) -> None:
        for digest in (
            self.provider_launch_receipt_sha256,
            self.resource_manifest_sha256,
            self.hard_stop_receipt_sha256,
        ):
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(token not in "0123456789abcdef" for token in digest)
            ):
                raise RuntimeGuardError("deadline controller launch binding is invalid")
        if not callable(self.hard_stop_liveness):
            raise RuntimeGuardError("deadline controller lacks a live hard-stop guard")
        if self.operation_admissions is None:
            self.operation_admissions = {}

    def transition(self, next_phase: Phase) -> None:
        if (
            self.phase is Phase.SMOKE
            and next_phase is Phase.STAGE1
            and self.smoke_receipt_sha256 is None
        ):
            raise RuntimeGuardError("Stage 1 requires a verified cold-smoke receipt")
        if next_phase not in _TRANSITIONS[self.phase]:
            raise RuntimeGuardError(
                f"illegal runtime phase transition: {self.phase} -> {next_phase}"
            )
        self.phase = next_phase

    def accept_verified_smoke(self, receipt_sha256: str) -> None:
        if self.phase is not Phase.SMOKE or self.smoke_receipt_sha256 is not None:
            raise RuntimeGuardError("cold-smoke authorization is out of phase or repeated")
        if (
            not isinstance(receipt_sha256, str)
            or len(receipt_sha256) != 64
            or any(token not in "0123456789abcdef" for token in receipt_sha256)
        ):
            raise RuntimeGuardError("cold-smoke receipt digest is invalid")
        self.smoke_receipt_sha256 = receipt_sha256
        self.transition(Phase.STAGE1)

    def admit_operation(
        self,
        operation: str,
        now: datetime,
    ) -> None:
        self.hard_stop_liveness()
        if operation not in OPERATION_LIMITS:
            raise RuntimeGuardError("operation lacks a frozen admission limit")
        assert self.operation_admissions is not None
        count = self.operation_admissions.get(operation, 0)
        if count >= OPERATION_LIMITS[operation]:
            raise RuntimeGuardError(f"operation admission count exceeded: {operation}")
        self.clock.admit(operation, now)
        self.operation_admissions[operation] = count + 1

    def admit_scored_worker(self, now: datetime) -> None:
        if self.phase not in {Phase.STAGE1, Phase.STAGE2}:
            raise RuntimeGuardError("scored worker dispatch is outside a scored phase")
        if self.dispatched_scored_runs >= 24:
            raise RuntimeGuardError("scored run cap exceeded")
        self.admit_operation("scored_worker", now)
        self.dispatched_scored_runs += 1

    def watchdog_due(self, now: datetime) -> bool:
        if now.tzinfo is None or now.utcoffset() != timedelta(0):
            raise RuntimeGuardError("watchdog checks require UTC time")
        return now >= self.clock.hard_horizon

    def enforce_watchdog(self, now: datetime) -> bool:
        if not self.watchdog_due(now):
            return False
        if self.phase not in {Phase.CLEANUP, Phase.TERMINAL}:
            self.phase = Phase.CLEANUP
        return True


def validate_success_bucket_total() -> int:
    total = (
        OPERATION_BOUNDS["post_running_preflight"]
        + OPERATION_BOUNDS["cold_smoke"]
        + 16 * OPERATION_BOUNDS["scored_worker"]
        + OPERATION_BOUNDS["stage1_seal_and_replay"]
        + OPERATION_BOUNDS["stage2_materialization"]
        + 8 * OPERATION_BOUNDS["scored_worker"]
        + OPERATION_BOUNDS["stage2_seal_and_replay"]
        + OPERATION_BOUNDS["terminal_seal"]
    )
    if total != 20_100:
        raise RuntimeGuardError("frozen successful-path bucket total drifted")
    if not math.isclose(float(PROVIDER_CAP_USD), 25.0):
        raise RuntimeGuardError("provider cap drifted")
    if GPU_PRICE_CEILING_USD_PER_HOUR * Decimal("7") != Decimal("23.03"):
        raise RuntimeGuardError("GPU charge cap drifted")
    return total
