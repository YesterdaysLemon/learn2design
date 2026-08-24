"""Record the post-cleanup Runpod cost envelope for coverage Stage A."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from tools.create_submission_like_source_lock import inside_git, sha256


PROFILE = "coverage-triage-screen-v1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--provider-hours", type=float, required=True)
    parser.add_argument("--gpu-charge", type=float, required=True)
    parser.add_argument("--total-provider-charge", type=float, required=True)
    parser.add_argument("--resources-deleted", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite billing receipt: {args.output}")
    if inside_git(args.output):
        raise ValueError("billing receipt must be written outside every Git checkout")
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    configuration = plan.get("configuration")
    if not isinstance(configuration, dict) or configuration.get(
        "study_profile"
    ) != PROFILE:
        raise ValueError("billing receipt requires the exact coverage triage plan")
    budget = configuration.get("resource_budget")
    if not isinstance(budget, dict):
        raise ValueError("coverage triage plan has no resource budget")
    values = (args.provider_hours, args.gpu_charge, args.total_provider_charge)
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("billing values must be finite and non-negative")
    if (
        args.provider_hours > float(budget["maximum_provider_hours"])
        or args.gpu_charge > 26.32 + 1e-9
        or args.gpu_charge
        > args.provider_hours * float(budget["maximum_gpu_hourly_price"]) + 0.01
        or args.total_provider_charge < args.gpu_charge
        or args.total_provider_charge > float(budget["maximum_provider_charge"])
    ):
        raise ValueError("observed provider usage exceeds the frozen cap")
    if not args.resources_deleted:
        raise ValueError("billing receipt requires verified provider cleanup")
    payload = {
        "format_version": 1,
        "study_profile": PROFILE,
        "plan_id": plan["plan_id"],
        "provider": "Runpod",
        "gpu_type_id": budget["gpu_type_id"],
        "cloud_type": budget["cloud_type"],
        "gpu_count": budget["gpu_count"],
        "maximum_gpu_hourly_price": budget["maximum_gpu_hourly_price"],
        "provider_hours": args.provider_hours,
        "gpu_charge": args.gpu_charge,
        "total_provider_charge": args.total_provider_charge,
        "resources_deleted": True,
        "captured_utc": datetime.now(UTC).isoformat(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"billing_receipt_sha256={sha256(args.output)}")


if __name__ == "__main__":
    main()
