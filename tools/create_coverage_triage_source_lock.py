"""Create a path-free source lock for the H100 coverage triage screen."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from tools.create_submission_like_source_lock import (
    build_source_lock,
    inside_git,
    sha256,
)


PROFILE = "coverage-triage-screen-v1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--checksum", type=Path, required=True)
    parser.add_argument("--package-manifest", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--terminal-attempt-receipt", type=Path, required=True)
    parser.add_argument("--provider-billing-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite source lock: {args.output}")
    if inside_git(args.output):
        raise ValueError("source lock must be written outside every Git checkout")
    payload = build_source_lock(
        archive=args.archive,
        checksum=args.checksum,
        package_manifest=args.package_manifest,
        plan=args.plan,
        terminal_attempt_receipt=args.terminal_attempt_receipt,
        study_profile=PROFILE,
    )
    files = payload["files"]
    if args.provider_billing_receipt.name in files:
        raise ValueError("provider billing receipt basename collides with another input")
    files[args.provider_billing_receipt.name] = {
        "sha256": sha256(args.provider_billing_receipt),
        "size_bytes": args.provider_billing_receipt.stat().st_size,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"source_lock_sha256={sha256(args.output)}")


if __name__ == "__main__":
    main()
