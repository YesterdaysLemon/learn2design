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
    args = parser.parse_args()
    result = package_study(args.study_dir, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
