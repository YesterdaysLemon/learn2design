"""Validate and package a completed UIFO study."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from experiments.uifo_paired.package import package_study


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study_dir", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="package a stopped partial study for recovery; never marks it complete",
    )
    parser.add_argument(
        "--recover-stale-lock",
        action="store_true",
        help=(
            "prove the prior writer is dead and preserve its lock before "
            "packaging a terminal incomplete attempt"
        ),
    )
    args = parser.parse_args()
    if args.recover_stale_lock and not args.allow_incomplete:
        parser.error("--recover-stale-lock requires --allow-incomplete")
    result = package_study(
        args.study_dir,
        args.output,
        allow_incomplete=args.allow_incomplete,
        recover_stale_lock=args.recover_stale_lock,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
