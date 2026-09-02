"""Build the frozen private candidate-screen panel after exact authorization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.feasibility_debt_candidate_screen.panel import (
    authorize_official_topology_scope,
    build_private_panel,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--prior-panel-dir",
        type=Path,
        default=Path("experiments/uifo_paired/panels"),
    )
    parser.add_argument("--owner-scope-receipt", type=Path, required=True)
    args = parser.parse_args()
    authorization = authorize_official_topology_scope(args.owner_scope_receipt)
    result = build_private_panel(
        dataset_path=args.dataset,
        prior_panel_dir=args.prior_panel_dir,
        output_dir=args.output,
        authorization=authorization,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
