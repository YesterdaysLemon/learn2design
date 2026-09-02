"""Zero-retry Runpod REST-v2 transport and host cleanup adapter.

The adapter never discovers credentials.  A caller must inject one bearer
token in memory.  Authorization headers are never returned, logged, or
persisted.  Every successful or rejected provider response is instead written
once as a canonical transcript containing the exact response body and a
credential fingerprint, which lets later steps prove that all observations in
one attempt used the same injected capability without revealing it.

Runpod does not sign REST responses.  "Authenticated" in this module means an
HTTPS response obtained with the injected bearer capability and bound to the
source-locked request method/path/status/body.  It is deliberately not called
a provider signature.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation, ROUND_CEILING
import json
from pathlib import Path
import ssl
from typing import Callable, Protocol, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urljoin, urlparse
from urllib.request import (
    HTTPRedirectHandler,
    HTTPSHandler,
    Request,
    build_opener,
)

from .canonical import (
    canonical_json_bytes,
    exclusive_write_bytes,
    sha256_bytes,
    write_receipt,
)
from .cleanup import BillingBinding, ResourceAdapter
from .contract import STUDY_ID
from .runtime import BillingSemantics, parse_utc


class RunpodAdapterError(RuntimeError):
    """A provider request, response, or fixed binding was not admissible."""


RUNPOD_V2_BASE_URL = "https://api.runpod.io/v2/"
RUNPOD_V2_OPENAPI_SHA256 = (
    "688c7b6f1d5386f04aa6d029b1c744fa11686540350e4411e8f3abe4bbf33d38"
)
H100_GPU_TYPE_ID = "NVIDIA H100 80GB HBM3"
MAX_RESPONSE_BYTES = 1_048_576
DEFAULT_REQUEST_TIMEOUT_SECONDS = 30


def _utc_text(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise RunpodAdapterError("provider observation clock must be UTC")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _decimal(value: object, *, label: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, Decimal)):
        raise RunpodAdapterError(f"provider {label} is not an exact JSON number")
    try:
        parsed = Decimal(value)
    except (InvalidOperation, ValueError) as error:
        raise RunpodAdapterError(f"provider {label} is invalid") from error
    if not parsed.is_finite():
        raise RunpodAdapterError(f"provider {label} is not finite")
    return parsed


def _decimal_text(value: Decimal) -> str:
    if not value.is_finite():
        raise RunpodAdapterError("non-finite decimal cannot enter a receipt")
    if value == 0:
        return "0"
    return format(value.normalize(), "f")


def _strict_json(content: bytes) -> object:
    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                raise RunpodAdapterError("provider JSON contains a duplicate key")
            result[key] = value
        return result

    try:
        return json.loads(
            content.decode("utf-8"),
            parse_float=Decimal,
            parse_int=int,
            parse_constant=lambda token: (_ for _ in ()).throw(
                RunpodAdapterError(f"provider JSON constant is forbidden: {token}")
            ),
            object_pairs_hook=pairs,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RunpodAdapterError("provider response is not strict UTF-8 JSON") from error


def _object(content: bytes) -> dict[str, object]:
    value = _strict_json(content)
    if not isinstance(value, dict):
        raise RunpodAdapterError("provider response root is not an object")
    return value


@dataclass(frozen=True)
class ApiExchange:
    method: str
    path: str
    status: int
    observed_utc: str
    content_type: str | None
    request_id: str | None
    body: bytes = field(repr=False, compare=False)
    credential_fingerprint_sha256: str = field(repr=False)
    request_body_sha256: str = sha256_bytes(b"")

    def __post_init__(self) -> None:
        if self.method not in {"GET", "POST", "DELETE"}:
            raise RunpodAdapterError("provider method is outside the fixed surface")
        if not self.path.startswith("/v2/") or "#" in self.path:
            raise RunpodAdapterError("provider path is outside REST v2")
        if type(self.status) is not int or not 100 <= self.status <= 599:
            raise RunpodAdapterError("provider HTTP status is invalid")
        parse_utc(self.observed_utc)
        if len(self.body) > MAX_RESPONSE_BYTES:
            raise RunpodAdapterError("provider response exceeds the fixed cap")
        if (
            len(self.credential_fingerprint_sha256) != 64
            or any(
                token not in "0123456789abcdef"
                for token in self.credential_fingerprint_sha256
            )
        ):
            raise RunpodAdapterError("provider credential fingerprint is invalid")
        if (
            len(self.request_body_sha256) != 64
            or any(
                token not in "0123456789abcdef"
                for token in self.request_body_sha256
            )
        ):
            raise RunpodAdapterError("provider request-body digest is invalid")

    def transcript_row(self) -> dict[str, object]:
        return {
            "api_base": RUNPOD_V2_BASE_URL,
            "api_contract_sha256": RUNPOD_V2_OPENAPI_SHA256,
            "body_base64": base64.b64encode(self.body).decode("ascii"),
            "body_sha256": sha256_bytes(self.body),
            "content_type": self.content_type,
            "credential_fingerprint_sha256": (
                self.credential_fingerprint_sha256
            ),
            "method": self.method,
            "observed_utc": self.observed_utc,
            "path": self.path,
            "request_id": self.request_id,
            "request_body_sha256": self.request_body_sha256,
            "status": self.status,
            "transport": "https_tls_verified_bearer",
        }


def transcript_bytes(exchanges: Sequence[ApiExchange]) -> bytes:
    if not exchanges:
        raise RunpodAdapterError("provider transcript cannot be empty")
    fingerprint = exchanges[0].credential_fingerprint_sha256
    if any(row.credential_fingerprint_sha256 != fingerprint for row in exchanges):
        raise RunpodAdapterError("provider transcript changed credential capability")
    return canonical_json_bytes(
        {
            "exchange_count": len(exchanges),
            "exchanges": [row.transcript_row() for row in exchanges],
            "schema_version": 1,
        }
    )


class RunpodTransport(Protocol):
    def request(
        self,
        method: str,
        path: str,
        *,
        body: bytes | None,
        deadline_utc: str,
    ) -> ApiExchange: ...


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        raise RunpodAdapterError("provider redirect is forbidden")


@dataclass(frozen=True)
class BearerHttpsTransport:
    """Small stdlib HTTPS transport with one attempt and no redirects."""

    bearer_token: str = field(repr=False)
    now_utc: Callable[[], datetime] = lambda: datetime.now(UTC)
    request_timeout_seconds: int = DEFAULT_REQUEST_TIMEOUT_SECONDS
    max_response_bytes: int = MAX_RESPONSE_BYTES
    _credential_fingerprint_sha256: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.bearer_token, str)
            or not self.bearer_token
            or any(token in self.bearer_token for token in ("\r", "\n", " "))
        ):
            raise RunpodAdapterError("injected provider credential is invalid")
        if (
            type(self.request_timeout_seconds) is not int
            or not 1 <= self.request_timeout_seconds <= 60
            or type(self.max_response_bytes) is not int
            or not 1 <= self.max_response_bytes <= MAX_RESPONSE_BYTES
        ):
            raise RunpodAdapterError("provider transport bounds are invalid")
        object.__setattr__(
            self,
            "_credential_fingerprint_sha256",
            sha256_bytes(self.bearer_token.encode("utf-8")),
        )

    def request(
        self,
        method: str,
        path: str,
        *,
        body: bytes | None,
        deadline_utc: str,
    ) -> ApiExchange:
        method = method.upper()
        if method not in {"GET", "POST", "DELETE"}:
            raise RunpodAdapterError("provider method is outside the fixed surface")
        if not path.startswith("/v2/") or "#" in path:
            raise RunpodAdapterError("provider path is outside REST v2")
        absolute = urljoin(RUNPOD_V2_BASE_URL, path.removeprefix("/v2/"))
        parsed = urlparse(absolute)
        if (
            parsed.scheme != "https"
            or parsed.hostname != "api.runpod.io"
            or parsed.port not in (None, 443)
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise RunpodAdapterError("provider URL escaped the fixed TLS origin")
        now = self.now_utc()
        deadline = parse_utc(deadline_utc)
        remaining = (deadline - now).total_seconds()
        if remaining <= 0:
            raise RunpodAdapterError("provider request crossed its deadline")
        timeout = min(float(self.request_timeout_seconds), remaining)
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.bearer_token}",
            "User-Agent": "learn2design-feasibility-debt-screen/1",
        }
        if body is not None:
            headers["Content-Type"] = "application/json"
        request = Request(absolute, data=body, headers=headers, method=method)
        opener = build_opener(
            HTTPSHandler(context=ssl.create_default_context()),
            _NoRedirect(),
        )
        status: int
        response_body: bytes
        content_type: str | None
        request_id: str | None
        try:
            with opener.open(request, timeout=timeout) as response:
                status = response.status
                response_body = response.read(self.max_response_bytes + 1)
                content_type = response.headers.get("Content-Type")
                request_id = response.headers.get("X-Request-Id")
                if response.geturl() != absolute:
                    raise RunpodAdapterError("provider response changed origin")
        except HTTPError as error:
            status = error.code
            response_body = error.read(self.max_response_bytes + 1)
            content_type = error.headers.get("Content-Type")
            request_id = error.headers.get("X-Request-Id")
        except (URLError, TimeoutError, OSError) as error:
            raise RunpodAdapterError(
                f"provider transport failed: {type(error).__name__}"
            ) from error
        if len(response_body) > self.max_response_bytes:
            raise RunpodAdapterError("provider response exceeds the fixed cap")
        observed = self.now_utc()
        if observed >= deadline:
            raise RunpodAdapterError("provider response crossed its deadline")
        return ApiExchange(
            method=method,
            path=path,
            status=status,
            observed_utc=_utc_text(observed),
            content_type=content_type,
            request_id=request_id,
            body=response_body,
            credential_fingerprint_sha256=(
                self._credential_fingerprint_sha256
            ),
            request_body_sha256=sha256_bytes(body or b""),
        )


def _require_status(exchange: ApiExchange, expected: int) -> None:
    if exchange.status != expected:
        raise RunpodAdapterError(
            f"provider {exchange.method} returned HTTP {exchange.status}"
        )


def _resource_rows(
    value: object,
    *,
    root_key: str,
    kind: str,
) -> list[dict[str, str]]:
    if not isinstance(value, dict) or set(value) != {root_key}:
        raise RunpodAdapterError(f"provider {kind} inventory schema mismatch")
    rows = value[root_key]
    if not isinstance(rows, list):
        raise RunpodAdapterError(f"provider {kind} inventory is not a list")
    result: list[dict[str, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise RunpodAdapterError(f"provider {kind} row is not an object")
        resource_id = row.get("id")
        if not isinstance(resource_id, str) or not resource_id:
            raise RunpodAdapterError(f"provider {kind} row lacks an ID")
        result.append({"id": resource_id, "kind": kind})
    return sorted(result, key=lambda row: (row["kind"], row["id"]))


def _inventory_snapshot(
    transport: RunpodTransport,
    *,
    deadline_utc: str,
    authenticated_response_path: Path,
) -> tuple[dict[str, list[dict[str, str]]], list[ApiExchange]]:
    requests = (
        ("/v2/pods?includeClusterPods=true", "pods", "pod"),
        ("/v2/clusters", "clusters", "cluster"),
        ("/v2/network-volumes", "networkVolumes", "volume"),
        ("/v2/serverless", "endpoints", "endpoint"),
        ("/v2/templates", "templates", "template"),
        ("/v2/registries", "registries", "registry"),
    )
    exchanges: list[ApiExchange] = []
    values: dict[str, list[dict[str, str]]] = {}
    for index, (path, root_key, kind) in enumerate(requests):
        exchange = transport.request(
            "GET", path, body=None, deadline_utc=deadline_utc
        )
        exchanges.append(exchange)
        exclusive_write_bytes(
            authenticated_response_path.with_name(
                f"{authenticated_response_path.name}.part-{index:02d}"
            ),
            transcript_bytes([exchange]),
        )
        _require_status(exchange, 200)
        values[root_key] = _resource_rows(
            _object(exchange.body), root_key=root_key, kind=kind
        )
    return values, exchanges


@dataclass(frozen=True)
class CreatedPod:
    resource_id: str
    name: str
    create_utc: str
    image_reference: str
    manifest: dict[str, object]
    manifest_sha256: str
    create_intent_sha256: str
    create_receipt_sha256: str
    create_exchange: ApiExchange = field(repr=False, compare=False)


@dataclass
class RunpodControlPlane:
    """Pre-provision evidence and exactly-one pod creation surface."""

    transport: RunpodTransport

    def capture_quote(
        self,
        *,
        ephemeral_disk_gib: int,
        receipt_path: Path,
        authenticated_response_path: Path,
        deadline_utc: str,
    ):
        """Capture exact SKU stock plus a conservative all-in rate bound."""
        if type(ephemeral_disk_gib) is not int or not 0 < ephemeral_disk_gib <= 40:
            raise RunpodAdapterError("quote disk is outside the frozen envelope")
        path = (
            f"/v2/catalog/gpus/{quote(H100_GPU_TYPE_ID, safe='')}?"
            + urlencode(
                {
                    "include": "AVAILABILITY",
                    "product": "POD",
                    "count": 1,
                    "cloud": "SECURE",
                    "cudaVersions": "13.0",
                }
            )
        )
        exchange = self.transport.request(
            "GET", path, body=None, deadline_utc=deadline_utc
        )
        exclusive_write_bytes(
            authenticated_response_path,
            transcript_bytes([exchange]),
        )
        _require_status(exchange, 200)
        value = _object(exchange.body)
        price = value.get("price")
        max_count = value.get("maxCount")
        data_centers = value.get("dataCenters")
        availability = value.get("availability")
        cuda_versions = value.get("cudaVersions")
        if (
            value.get("id") != H100_GPU_TYPE_ID
            or value.get("manufacturer") != "NVIDIA"
            or value.get("memory") != 80
            or value.get("secure") is not True
            or not isinstance(price, dict)
            or not isinstance(max_count, dict)
            or not isinstance(data_centers, list)
            or availability not in {"LOW", "MEDIUM", "HIGH"}
            or not data_centers
            or not isinstance(cuda_versions, list)
            or cuda_versions != [{"version": "13.0", "available": True}]
        ):
            raise RunpodAdapterError("provider H100 catalog entry is unavailable")
        gpu_rate = _decimal(price.get("secure"), label="secure H100 price")
        if gpu_rate <= 0 or gpu_rate > Decimal("3.29"):
            raise RunpodAdapterError("provider H100 price exceeds the frozen ceiling")
        secure_count = max_count.get("secure")
        if type(secure_count) is not int or secure_count < 1:
            raise RunpodAdapterError("provider H100 secure capacity is unavailable")
        # Runpod documents $0.10/GB/month for running container disks but does
        # not expose a pre-create disk quote in REST v2.  A 28-day month is the
        # conservative conversion, so this bound cannot understate a calendar
        # month's hourly storage rate.
        disk_rate_bound = (
            Decimal(ephemeral_disk_gib)
            * Decimal("0.10")
            / (Decimal(28) * Decimal(24))
        )
        combined_bound = gpu_rate + disk_rate_bound
        if combined_bound > Decimal("3.5714285714"):
            raise RunpodAdapterError("provider all-in rate exceeds the frozen ceiling")
        write_receipt(
            receipt_path,
            study_id=STUDY_ID,
            receipt_type="provider_quote",
            payload={
                "provider": "runpod",
                "observed_utc": exchange.observed_utc,
                "authenticated_response_sha256": sha256_bytes(
                    authenticated_response_path.read_bytes()
                ),
                "cloud_type": "SECURE",
                "gpu_model": H100_GPU_TYPE_ID,
                "gpu_count": 1,
                "capacity_available": True,
                "max_ephemeral_disk_gib": 40,
                "billing": {
                    "currency": "USD",
                    "gpu_hourly_rate_usd": _decimal_text(gpu_rate),
                    "combined_hourly_rate_usd": _decimal_text(combined_bound),
                    "fixed_charge_usd": "0",
                    "metering_quantum_seconds": 1,
                    "round_up_each_quantum": False,
                },
            },
        )
        from .provider import authenticate_provider_quote

        return authenticate_provider_quote(
            receipt_path,
            authenticated_response_path=authenticated_response_path,
        )

    def capture_clean_inventory(
        self,
        *,
        task_scope_sha256: str,
        receipt_path: Path,
        authenticated_response_path: Path,
        deadline_utc: str,
    ):
        values, exchanges = _inventory_snapshot(
            self.transport,
            deadline_utc=deadline_utc,
            authenticated_response_path=authenticated_response_path,
        )
        exclusive_write_bytes(
            authenticated_response_path,
            transcript_bytes(exchanges),
        )
        write_receipt(
            receipt_path,
            study_id=STUDY_ID,
            receipt_type="provider_inventory",
            payload={
                "provider": "runpod",
                "observed_utc": exchanges[-1].observed_utc,
                "authenticated_response_sha256": sha256_bytes(
                    authenticated_response_path.read_bytes()
                ),
                "task_scope_sha256": task_scope_sha256,
                "pods": values["pods"],
                "clusters": values["clusters"],
                "network_volumes": values["networkVolumes"],
                "endpoints": values["endpoints"],
                "templates": values["templates"],
                "registries": values["registries"],
            },
        )
        from .provider import authenticate_clean_inventory

        return authenticate_clean_inventory(
            receipt_path,
            authenticated_response_path=authenticated_response_path,
            expected_task_scope_sha256=task_scope_sha256,
        )

    def create_pod_once(
        self,
        *,
        request,
        quote_authorization,
        provision_authorization,
        immutable_image_reference: str,
        ssh_public_key: str,
        create_intent_path: Path,
        create_receipt_path: Path,
        authenticated_response_path: Path,
        manifest_path: Path,
        deadline_utc: str,
    ) -> CreatedPod:
        """Issue exactly one POST and persist a cleanup-capable manifest."""
        from .provider import assert_provision_authorization

        provision = assert_provision_authorization(provision_authorization)
        if (
            provision.resource_request_sha256 != request.receipt_sha256
            or provision.quote_sha256 != quote_authorization.receipt_sha256
            or provision.task_scope_sha256 != request.task_scope_sha256
            or provision.immutable_image_digest != request.immutable_image_digest
        ):
            raise RunpodAdapterError("pod create authorization binding mismatch")
        suffix = f"@{request.immutable_image_digest}"
        if (
            not isinstance(immutable_image_reference, str)
            or not immutable_image_reference.endswith(suffix)
            or immutable_image_reference.startswith(("http://", "https://"))
        ):
            raise RunpodAdapterError("pod image is not the authorized immutable digest")
        if (
            not isinstance(ssh_public_key, str)
            or not ssh_public_key.startswith(("ssh-ed25519 ", "ssh-rsa "))
            or "\r" in ssh_public_key
            or "\n" in ssh_public_key
        ):
            raise RunpodAdapterError("pod SSH public key is invalid")
        name = f"l2d-fd-v1-{request.task_scope_sha256[:16]}"
        body_value = {
            "cloud": "SECURE",
            "disk": request.ephemeral_disk_gib,
            "env": {"PUBLIC_KEY": ssh_public_key},
            "globalNetworking": False,
            "gpu": {
                "allowedCudaVersions": ["13.0"],
                "count": 1,
                "id": H100_GPU_TYPE_ID,
            },
            "image": immutable_image_reference,
            "name": name,
            "ports": ["22/tcp"],
            "startJupyter": False,
            "startSsh": True,
        }
        body = canonical_json_bytes(body_value)
        create_intent_sha256 = write_receipt(
            create_intent_path,
            study_id=STUDY_ID,
            receipt_type="provider_create_intent",
            payload={
                "provider": "runpod",
                "method": "POST",
                "path": "/v2/pods",
                "name": name,
                "task_scope_sha256": request.task_scope_sha256,
                "resource_request_sha256": request.receipt_sha256,
                "request_body_sha256": sha256_bytes(body),
                "one_post_only": True,
                "retry_forbidden_after_intent": True,
            },
        )
        exchange = self.transport.request(
            "POST", "/v2/pods", body=body, deadline_utc=deadline_utc
        )
        exclusive_write_bytes(
            authenticated_response_path,
            transcript_bytes([exchange]),
        )
        if exchange.request_body_sha256 != sha256_bytes(body):
            raise RunpodAdapterError("provider create transcript changed request bytes")
        _require_status(exchange, 201)
        value = _object(exchange.body)
        resource_id = value.get("id")
        create_utc = value.get("createdAt")
        if (
            not isinstance(resource_id, str)
            or not resource_id
            or not isinstance(create_utc, str)
            or value.get("name") != name
            or value.get("cloud") != "SECURE"
            or value.get("image") != immutable_image_reference
            or value.get("disk") != request.ephemeral_disk_gib
        ):
            raise RunpodAdapterError("provider create response changed pod identity")
        parse_utc(create_utc)
        create_receipt_sha256 = write_receipt(
            create_receipt_path,
            study_id=STUDY_ID,
            receipt_type="provider_create",
            payload={
                "provider": "runpod",
                "status": "CREATED",
                "resource_id": resource_id,
                "name": name,
                "task_scope_sha256": request.task_scope_sha256,
                "resource_request_sha256": request.receipt_sha256,
                "create_intent_sha256": create_intent_sha256,
                "observed_utc": exchange.observed_utc,
                "provider_created_utc": create_utc,
                "authenticated_response_sha256": sha256_bytes(
                    authenticated_response_path.read_bytes()
                ),
                "request_body_sha256": sha256_bytes(body),
                "one_post_only": True,
            },
        )
        manifest = {
            "provider": "runpod",
            "cloud_type": "SECURE",
            "pod": {
                "id": resource_id,
                "gpu_type_id": H100_GPU_TYPE_ID,
                "gpu_count": 1,
                "immutable_image_digest": request.immutable_image_digest,
            },
            "ephemeral_disk_gib": request.ephemeral_disk_gib,
            "other_resources": [],
            "task_scope_sha256": request.task_scope_sha256,
            "resource_request_sha256": request.receipt_sha256,
        }
        manifest_bytes = canonical_json_bytes(manifest)
        exclusive_write_bytes(manifest_path, manifest_bytes)
        return CreatedPod(
            resource_id=resource_id,
            name=name,
            create_utc=create_utc,
            image_reference=immutable_image_reference,
            manifest=manifest,
            manifest_sha256=sha256_bytes(manifest_bytes),
            create_intent_sha256=create_intent_sha256,
            create_receipt_sha256=create_receipt_sha256,
            create_exchange=exchange,
        )

    def await_running_and_seal_launch(
        self,
        *,
        created: CreatedPod,
        request,
        quote_authorization,
        provision_authorization,
        launch_receipt_path: Path,
        authenticated_response_path: Path,
        deadline_utc: str,
        sleep: Callable[[float], None],
        max_observations: int = 120,
        poll_interval_seconds: int = 5,
    ):
        """Observe one created pod until RUNNING; observations are not retries."""
        if (
            type(max_observations) is not int
            or not 1 <= max_observations <= 120
            or type(poll_interval_seconds) is not int
            or not 1 <= poll_interval_seconds <= 10
        ):
            raise RunpodAdapterError("provider observation bounds are invalid")
        exchanges = [created.create_exchange]
        running: dict[str, object] | None = None
        try:
            initial = _object(created.create_exchange.body)
            if initial.get("status") == "RUNNING":
                running = initial
            for _ in range(max_observations if running is None else 0):
                sleep(float(poll_interval_seconds))
                path = f"/v2/pods/{quote(created.resource_id, safe='')}"
                exchange = self.transport.request(
                    "GET", path, body=None, deadline_utc=deadline_utc
                )
                exchanges.append(exchange)
                _require_status(exchange, 200)
                value = _object(exchange.body)
                if value.get("id") != created.resource_id:
                    raise RunpodAdapterError("provider observation changed pod ID")
                status = value.get("status")
                if status == "RUNNING":
                    running = value
                    break
                if status in {"ERROR", "EXITED", "TERMINATED"}:
                    raise RunpodAdapterError(
                        f"provider pod entered terminal state {status}"
                    )
            if running is None:
                raise RunpodAdapterError("provider pod did not reach RUNNING")
        finally:
            exclusive_write_bytes(
                authenticated_response_path,
                transcript_bytes(exchanges),
            )

        gpu = running.get("gpu")
        networking = running.get("globalNetworking")
        cost = _decimal(running.get("cost"), label="running pod cost")
        started_at = running.get("startedAt")
        cuda_version = running.get("cudaVersion")
        if (
            running.get("name") != created.name
            or running.get("image") != created.image_reference
            or running.get("disk") != request.ephemeral_disk_gib
            or running.get("cloud") != "SECURE"
            or running.get("status") != "RUNNING"
            or running.get("ports") != ["22/tcp"]
            or running.get("registry") is not None
            or running.get("template") is not None
            or running.get("mounts") != {}
            or not isinstance(gpu, dict)
            or gpu.get("id") != H100_GPU_TYPE_ID
            or gpu.get("count") != 1
            or not isinstance(networking, dict)
            or networking.get("enabled") is not False
            or not isinstance(started_at, str)
            or not isinstance(cuda_version, str)
            or not cuda_version.startswith("13.")
            or cost <= 0
            or cost > quote_authorization.billing.combined_hourly_rate_usd
        ):
            raise RunpodAdapterError("running pod differs from the frozen request")
        parse_utc(started_at)
        write_receipt(
            launch_receipt_path,
            study_id=STUDY_ID,
            receipt_type="provider_launch",
            payload={
                "provider": "runpod",
                "quote_sha256": quote_authorization.receipt_sha256,
                "authenticated_response_sha256": sha256_bytes(
                    authenticated_response_path.read_bytes()
                ),
                "resource_request_sha256": request.receipt_sha256,
                "resource_manifest_sha256": created.manifest_sha256,
                "task_scope_sha256": request.task_scope_sha256,
                "resource_id": created.resource_id,
                "immutable_image_digest": request.immutable_image_digest,
                "status": "RUNNING",
                "create_utc": created.create_utc,
                "running_utc": started_at,
                "billable_utc": created.create_utc,
                "cloud_type": "SECURE",
                "gpu_model": H100_GPU_TYPE_ID,
                "gpu_count": 1,
                "ephemeral_disk_gib": request.ephemeral_disk_gib,
                "provider_running_hourly_cost_usd": _decimal_text(cost),
                "billing": {
                    "currency": quote_authorization.billing.currency,
                    "gpu_hourly_rate_usd": _decimal_text(
                        quote_authorization.billing.gpu_hourly_rate_usd
                    ),
                    "combined_hourly_rate_usd": _decimal_text(
                        quote_authorization.billing.combined_hourly_rate_usd
                    ),
                    "fixed_charge_usd": _decimal_text(
                        quote_authorization.billing.fixed_charge_usd
                    ),
                    "metering_quantum_seconds": (
                        quote_authorization.billing.metering_quantum_seconds
                    ),
                    "round_up_each_quantum": (
                        quote_authorization.billing.round_up_each_quantum
                    ),
                },
            },
        )
        from .provider import validate_provisioned_resource
        from .runtime import load_provider_launch_receipt

        launch, digest = load_provider_launch_receipt(
            launch_receipt_path,
            expected_resource_manifest_sha256=created.manifest_sha256,
            expected_resource_request_sha256=request.receipt_sha256,
            expected_quote_sha256=quote_authorization.receipt_sha256,
            authenticated_response_path=authenticated_response_path,
        )
        validate_provisioned_resource(
            manifest=created.manifest,
            launch=launch,
            request=request,
            quote=quote_authorization,
            provision=provision_authorization,
        )
        return launch, digest


@dataclass
class RunpodResourceAdapter(ResourceAdapter):
    """Concrete host-only delete, inventory, and billing adapter."""

    transport: RunpodTransport
    binding: BillingBinding
    resource_request_sha256: str
    billable_utc: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.resource_request_sha256, str)
            or len(self.resource_request_sha256) != 64
            or any(
                token not in "0123456789abcdef"
                for token in self.resource_request_sha256
            )
        ):
            raise RunpodAdapterError("resource request digest is invalid")
        parse_utc(self.billable_utc)

    def delete(
        self,
        kind: str,
        resource_id: str,
        *,
        receipt_path: Path,
        authenticated_response_path: Path,
        deadline_utc: str,
    ) -> None:
        if kind != "pod" or resource_id != self.binding.resource_id:
            raise RunpodAdapterError("adapter may delete only the bound pod")
        path = f"/v2/pods/{quote(resource_id, safe='')}"
        exchange = self.transport.request(
            "DELETE", path, body=None, deadline_utc=deadline_utc
        )
        exclusive_write_bytes(
            authenticated_response_path,
            transcript_bytes([exchange]),
        )
        status = "DELETED" if exchange.status == 204 else "FAILED"
        write_receipt(
            receipt_path,
            study_id=STUDY_ID,
            receipt_type="provider_delete",
            payload={
                "provider": "runpod",
                "status": status,
                "kind": kind,
                "id": resource_id,
                "task_scope_sha256": self.binding.task_scope_sha256,
                "resource_request_sha256": self.resource_request_sha256,
                "quote_sha256": self.binding.quote_sha256,
                "launch_response_sha256": (
                    self.binding.launch_response_sha256
                ),
                "observed_utc": exchange.observed_utc,
                "authenticated_response_sha256": sha256_bytes(
                    authenticated_response_path.read_bytes()
                ),
            },
        )
        _require_status(exchange, 204)

    def scoped_inventory(
        self,
        *,
        receipt_path: Path,
        authenticated_response_path: Path,
        deadline_utc: str,
    ) -> None:
        values, exchanges = _inventory_snapshot(
            self.transport,
            deadline_utc=deadline_utc,
            authenticated_response_path=authenticated_response_path,
        )
        exclusive_write_bytes(
            authenticated_response_path,
            transcript_bytes(exchanges),
        )
        write_receipt(
            receipt_path,
            study_id=STUDY_ID,
            receipt_type="provider_inventory",
            payload={
                "provider": "runpod",
                "observed_utc": exchanges[-1].observed_utc,
                "authenticated_response_sha256": sha256_bytes(
                    authenticated_response_path.read_bytes()
                ),
                "task_scope_sha256": self.binding.task_scope_sha256,
                "pods": values["pods"],
                "clusters": values["clusters"],
                "network_volumes": values["networkVolumes"],
                "endpoints": values["endpoints"],
                "templates": values["templates"],
                "registries": values["registries"],
            },
        )

    def billing_receipt(
        self,
        *,
        receipt_path: Path,
        authenticated_response_path: Path,
        deadline_utc: str,
    ) -> None:
        billable = parse_utc(self.billable_utc)
        deadline = parse_utc(deadline_utc)
        window_start = billable.replace(minute=0, second=0, microsecond=0)
        window_end = deadline.replace(minute=0, second=0, microsecond=0)
        if window_end < deadline:
            window_end += timedelta(hours=1)
        if window_end <= window_start:
            raise RunpodAdapterError("provider billing window is empty")
        query_value = {
            "startTime": _utc_text(window_start),
            "endTime": _utc_text(window_end),
            "bucketSize": "hour",
            "podId": self.binding.resource_id,
        }
        query = urlencode(query_value)
        exchange = self.transport.request(
            "GET",
            f"/v2/billing/pods?{query}",
            body=None,
            deadline_utc=deadline_utc,
        )
        exclusive_write_bytes(
            authenticated_response_path,
            transcript_bytes([exchange]),
        )
        _require_status(exchange, 200)
        payload = _object(exchange.body)
        if set(payload) != {"records", "metadata"}:
            raise RunpodAdapterError("provider billing schema mismatch")
        records = payload["records"]
        metadata = payload["metadata"]
        if not isinstance(records, list) or not isinstance(metadata, dict):
            raise RunpodAdapterError("provider billing values are malformed")
        if set(metadata) != {
            "query",
            "recordCount",
            "uniquePodCount",
            "totals",
        }:
            raise RunpodAdapterError("provider billing metadata schema mismatch")
        gpu = Decimal("0")
        cpu = Decimal("0")
        disk = Decimal("0")
        total = Decimal("0")
        record_keys: set[tuple[str, str, str]] = set()
        for row in records:
            if (
                not isinstance(row, dict)
                or set(row)
                != {
                    "startTime",
                    "endTime",
                    "podId",
                    "gpuAmount",
                    "cpuAmount",
                    "diskAmount",
                    "totalAmount",
                }
                or row.get("podId") != self.binding.resource_id
            ):
                raise RunpodAdapterError("provider billing row changed pod identity")
            row_start = parse_utc(row["startTime"])
            row_end = parse_utc(row["endTime"])
            key = (row["startTime"], row["endTime"], row["podId"])
            if (
                row_start < window_start
                or row_end > window_end
                or row_end - row_start != timedelta(hours=1)
                or key in record_keys
            ):
                raise RunpodAdapterError("provider billing row changed its time bucket")
            record_keys.add(key)
            gpu += _decimal(row.get("gpuAmount"), label="GPU amount")
            cpu += _decimal(row.get("cpuAmount"), label="CPU amount")
            disk += _decimal(row.get("diskAmount"), label="disk amount")
            total += _decimal(row.get("totalAmount"), label="total amount")
        if any(value < 0 for value in (gpu, cpu, disk, total)) or cpu != 0:
            raise RunpodAdapterError("provider billing contains inadmissible charges")
        if total != gpu + disk:
            raise RunpodAdapterError("provider billing components do not sum")
        totals = metadata.get("totals")
        query_echo = metadata.get("query")
        record_count = metadata.get("recordCount")
        unique_pod_count = metadata.get("uniquePodCount")
        if (
            not isinstance(totals, dict)
            or set(totals)
            != {"gpuAmount", "cpuAmount", "diskAmount", "totalAmount"}
            or not isinstance(query_echo, dict)
            or set(query_echo) != set(query_value)
            or type(record_count) is not int
            or record_count != len(records)
            or type(unique_pod_count) is not int
            or unique_pod_count != (1 if records else 0)
        ):
            raise RunpodAdapterError("provider billing metadata is malformed")
        if query_echo != query_value:
            raise RunpodAdapterError("provider billing query echo changed pod identity")
        expected_totals = {
            "gpuAmount": gpu,
            "cpuAmount": cpu,
            "diskAmount": disk,
            "totalAmount": total,
        }
        if any(
            _decimal(totals.get(key), label=f"metadata {key}") != value
            for key, value in expected_totals.items()
        ):
            raise RunpodAdapterError("provider billing totals do not replay")

        observed = parse_utc(exchange.observed_utc)
        elapsed = Decimal(str((observed - billable).total_seconds()))
        if elapsed < 0:
            raise RunpodAdapterError("provider billing precedes billable start")
        envelope_seconds = int(elapsed.to_integral_value(rounding=ROUND_CEILING))
        if envelope_seconds > self.binding.max_provider_seconds:
            raise RunpodAdapterError("provider billing crossed the hard horizon")
        gpu_semantics = BillingSemantics(
            currency="USD",
            gpu_hourly_rate_usd=Decimal(self.binding.gpu_hourly_rate_usd),
            combined_hourly_rate_usd=Decimal(self.binding.gpu_hourly_rate_usd),
            fixed_charge_usd=Decimal("0"),
            metering_quantum_seconds=self.binding.metering_quantum_seconds,
            round_up_each_quantum=self.binding.round_up_each_quantum,
        )
        envelope = BillingSemantics(
            currency="USD",
            gpu_hourly_rate_usd=Decimal(self.binding.gpu_hourly_rate_usd),
            combined_hourly_rate_usd=Decimal(
                self.binding.combined_hourly_rate_usd
            ),
            fixed_charge_usd=Decimal(self.binding.fixed_charge_usd),
            metering_quantum_seconds=self.binding.metering_quantum_seconds,
            round_up_each_quantum=self.binding.round_up_each_quantum,
        )
        gpu_bound = gpu_semantics.charge_at_seconds(envelope_seconds)
        all_in_bound = envelope.charge_at_seconds(envelope_seconds)
        if gpu > gpu_bound or total > all_in_bound:
            raise RunpodAdapterError("provider charge exceeds the frozen envelope")
        if gpu > Decimal("23.03") or total > Decimal("25.00"):
            raise RunpodAdapterError("provider charge exceeds the owner cap")
        write_receipt(
            receipt_path,
            study_id=STUDY_ID,
            receipt_type="provider_billing",
            payload={
                "provider": "runpod",
                "resource_id": self.binding.resource_id,
                "task_scope_sha256": self.binding.task_scope_sha256,
                "quote_sha256": self.binding.quote_sha256,
                "launch_response_sha256": (
                    self.binding.launch_response_sha256
                ),
                "observed_utc": exchange.observed_utc,
                "billing_query": query_value,
                "billing_query_sha256": sha256_bytes(
                    canonical_json_bytes(query_value)
                ),
                "provider_record_count": record_count,
                "provider_unique_pod_count": unique_pod_count,
                "currency": "USD",
                "envelope_seconds": envelope_seconds,
                "gpu_hourly_rate_usd": self.binding.gpu_hourly_rate_usd,
                "combined_hourly_rate_usd": (
                    self.binding.combined_hourly_rate_usd
                ),
                "fixed_charge_usd": self.binding.fixed_charge_usd,
                "metering_quantum_seconds": (
                    self.binding.metering_quantum_seconds
                ),
                "round_up_each_quantum": (
                    self.binding.round_up_each_quantum
                ),
                "gpu_charge_bound_usd": _decimal_text(gpu_bound),
                "all_in_charge_bound_usd": _decimal_text(all_in_bound),
                "provider_gpu_charge_usd": _decimal_text(gpu),
                "provider_disk_charge_usd": _decimal_text(disk),
                "provider_total_charge_usd": _decimal_text(total),
                "provider_receipt_sha256": sha256_bytes(
                    authenticated_response_path.read_bytes()
                ),
            },
        )


__all__ = [
    "ApiExchange",
    "BearerHttpsTransport",
    "CreatedPod",
    "H100_GPU_TYPE_ID",
    "RUNPOD_V2_BASE_URL",
    "RUNPOD_V2_OPENAPI_SHA256",
    "RunpodAdapterError",
    "RunpodControlPlane",
    "RunpodResourceAdapter",
    "RunpodTransport",
    "transcript_bytes",
]
