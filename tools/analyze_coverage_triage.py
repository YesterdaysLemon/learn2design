"""Command alias for the sealed H100 coverage triage replay."""

import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from tools.analyze_coverage_robustness import main


if __name__ == "__main__":
    main()
