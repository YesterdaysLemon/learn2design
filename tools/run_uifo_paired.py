"""CLI shim for the paired UIFO evaluation harness."""

import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from experiments.uifo_paired.runner import main


if __name__ == "__main__":
    main()
