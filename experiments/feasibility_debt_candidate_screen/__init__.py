"""Isolated, pre-result harness for feasibility-debt-candidate-screen-v1.

This namespace is intentionally independent of ``experiments.uifo_paired``.
The older profiles are terminal evidence and must not acquire new dispatch or
analysis behavior as a side effect of this study.
"""

from .contract import STUDY_ID, arm_specs

__all__ = ["STUDY_ID", "arm_specs"]
