"""Narrow benchmark baselines without importing dfbench's optional algorithm zoo."""

from __future__ import annotations

import optax

from dfbench import Objective, OptimizationAlgorithm


class SingleStartAdam(OptimizationAlgorithm):
    """Organizer-style Adam used only as an experimental reference arm."""

    algorithm_str = "paired_single_start_adam"

    def __init__(self) -> None:
        pass

    def optimize(
        self,
        objective: Objective,
        init_params=None,
        random_seed: int | None = None,
        learning_rate: float = 0.1,
        patience: int | None = None,
    ) -> None:
        obj = objective
        self.prepare(obj, unbounded=True, random_seed=random_seed)
        params = (
            obj.random_params_unbounded() if init_params is None else init_params
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0), optax.adam(learning_rate)
        )
        state = optimizer.init(params)

        # dfbench 0.3.3 drops asynchronous warmup outputs. Keep compilation
        # inside the timed budget until its public helper can be synchronized.
        obj.start_logging()
        while not obj.budget_exceeded:
            _, grads = obj.value_and_grad(params)
            if patience is not None and obj.evals_since_improvement > patience:
                break
            updates, state = optimizer.update(grads, state, params)
            params = optax.apply_updates(params, updates)
