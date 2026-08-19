"""CLI shim for isolated one-candidate UIFO diagnostics."""

import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from experiments.uifo_paired.candidate_probe import main


if __name__ == "__main__":
    raise SystemExit(main())
