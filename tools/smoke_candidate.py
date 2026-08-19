"""Run a short candidate smoke test on the constrained Voyager problem."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import jax
from dfbench import Objective
from dfbench.problems import ConstrainedVoyagerProblem

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from submission.submission import BatchedRestartAdam


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-time", type=float, default=30.0)
    parser.add_argument("--population-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-semantic-prior", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/generated/constrained-voyager-smoke.json"),
    )
    args = parser.parse_args()

    objective = Objective(
        ConstrainedVoyagerProblem(),
        max_time=args.max_time,
        save=["batched_loss", "is_feasible", "batched_is_feasible"],
        verbose=0,
    )
    BatchedRestartAdam().optimize(
        objective,
        random_seed=args.seed,
        population_size=args.population_size,
        safety_seconds=0.0,
        use_semantic_prior=not args.no_semantic_prior,
    )

    feasible_calls = sum(
        bool(value.any())
        for value in objective.batched_is_feasible_history
        if value is not None
    )
    result = {
        "created_utc": datetime.now(UTC).isoformat(),
        "device": [str(device) for device in jax.devices()],
        "max_time_seconds": args.max_time,
        "population_size": args.population_size,
        "problem": "ConstrainedVoyagerProblem",
        "python": platform.python_version(),
        "random_seed": args.seed,
        "semantic_prior": not args.no_semantic_prior,
        "summary": objective.get_summary(),
        "logged_calls_with_feasible_member": feasible_calls,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
