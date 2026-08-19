"""Learn2Design 2026 submission entry point.

The evaluator imports this file from the root of the submitted ZIP. Keep the
runtime self-contained and limited to packages provided by the competition.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp

from dfbench import Objective, OptimizationAlgorithm


class BatchedRestartAdam(OptimizationAlgorithm):
    """A small population of Adam searches with deterministic restarts.

    Population members use different learning rates, reserve one slot for an
    official-archive semantic prior, and share only the best physically
    feasible point. Stalled members alternate between fresh random starts and
    perturbations around that point. All simulator access goes through the
    public ``Objective`` API.
    """

    algorithm_str = "batched_restart_adam"

    def __init__(self) -> None:
        pass

    @staticmethod
    def _feasibility_anchor(objective: Objective):
        """Construct a conservative low-power point in unbounded coordinates."""
        unit = jnp.full((objective.n_params,), 0.5)
        for index, pair in enumerate(objective.optimization_pairs):
            if (
                isinstance(pair, (list, tuple))
                and len(pair) >= 2
                and isinstance(pair[0], str)
                and isinstance(pair[1], str)
            ):
                pairs = [pair]
            elif isinstance(pair, (list, tuple)):
                pairs = pair
            else:
                pairs = []
            properties = {
                item[1]
                for item in pairs
                if isinstance(item, (list, tuple)) and len(item) >= 2
            }
            if "power" in properties or "db" in properties:
                unit = unit.at[index].set(1e-8)
            elif "reflectivity" in properties:
                unit = unit.at[index].set(1e-4)

        # ``prepare(..., unbounded=True)`` selects dfbench's default sigmoid
        # map, so logit(unit) is the corresponding active-space coordinate.
        return jnp.log(unit) - jnp.log1p(-unit)

    @staticmethod
    def _semantic_key(pair):
        """Return the archive key and property for one runtime parameter slot."""
        if (
            isinstance(pair, (list, tuple))
            and len(pair) >= 2
            and isinstance(pair[0], str)
            and isinstance(pair[1], str)
        ):
            pairs = [pair]
        elif isinstance(pair, (list, tuple)):
            pairs = pair
        else:
            return None, None

        decoded = [
            (str(item[0]), str(item[1]))
            for item in pairs
            if isinstance(item, (list, tuple)) and len(item) >= 2
        ]
        properties = {property_name for _, property_name in decoded}
        if not decoded or len(properties) != 1:
            return None, None
        property_name = next(iter(properties))
        components = "+".join(sorted(component for component, _ in decoded))
        return f"{property_name}:{components}", property_name

    @classmethod
    def _semantic_prior(cls, objective: Objective):
        """Load the official-archive median candidate, with safe fallbacks."""
        try:
            payload = json.loads(
                Path(__file__).with_name("semantic_prior.json").read_text(
                    encoding="utf-8"
                )
            )
            if payload.get("format_version") != 1:
                return None
            key_medians = payload["key_medians"]
            property_medians = payload["property_medians"]
        except (KeyError, OSError, TypeError, ValueError):
            return None

        unit_values = []
        for pair in objective.optimization_pairs:
            key, property_name = cls._semantic_key(pair)
            value = key_medians.get(key, property_medians.get(property_name, 0.5))
            try:
                value = float(value)
            except (TypeError, ValueError):
                value = 0.5
            if not 0.0 <= value <= 1.0:
                value = 0.5
            unit_values.append(value)
        if len(unit_values) != objective.n_params:
            return None

        unit = jnp.clip(jnp.asarray(unit_values), 1e-6, 1.0 - 1e-6)
        return jnp.log(unit) - jnp.log1p(-unit)

    @staticmethod
    def _evaluate_population(
        objective: Objective,
        params,
        evaluation_chunk_size: int | None,
    ):
        """Evaluate a population in vmap or diagnostic low-memory chunks."""
        population_size = int(params.shape[0])
        if evaluation_chunk_size is None or evaluation_chunk_size >= population_size:
            return objective.vmap_value_and_grad_aux(params)

        loss_chunks = []
        grad_chunks = []
        aux_chunks = []
        for start in range(0, population_size, evaluation_chunk_size):
            chunk = params[start : start + evaluation_chunk_size]
            if int(chunk.shape[0]) == 1:
                loss, grad, aux = objective.value_and_grad_aux(chunk[0])
                loss_chunks.append(jnp.asarray(loss)[None])
                grad_chunks.append(jnp.asarray(grad)[None, :])
                aux_chunks.append(
                    jax.tree.map(lambda value: jnp.asarray(value)[None, ...], aux)
                )
            else:
                losses, grads, aux = objective.vmap_value_and_grad_aux(chunk)
                loss_chunks.append(losses)
                grad_chunks.append(grads)
                aux_chunks.append(aux)

        combined_aux = jax.tree.map(
            lambda *values: jnp.concatenate(values, axis=0), *aux_chunks
        )
        return (
            jnp.concatenate(loss_chunks, axis=0),
            jnp.concatenate(grad_chunks, axis=0),
            combined_aux,
        )

    def optimize(
        self,
        objective: Objective,
        init_params=None,
        random_seed: int | None = None,
        population_size: int = 8,
        learning_rate_low: float = 0.03,
        learning_rate_high: float = 0.15,
        patience: int = 600,
        minimum_improvement: float = 1e-7,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        gradient_clip_norm: float = 1.0,
        restart_noise_scale: float = 0.35,
        safety_seconds: float = 2.0,
        use_semantic_prior: bool = True,
        evaluation_chunk_size: int | None = None,
        initial_population_callback=None,
        **kwargs,
    ) -> None:
        del kwargs
        obj = objective
        self.prepare(obj, unbounded=True, random_seed=random_seed)

        population_size = max(2, int(population_size))
        if evaluation_chunk_size is not None:
            evaluation_chunk_size = int(evaluation_chunk_size)
            if not 1 <= evaluation_chunk_size <= population_size:
                raise ValueError(
                    "evaluation_chunk_size must be between one and population_size"
                )
        params = obj.random_params_unbounded(population_size)
        next_index = 0

        if init_params is not None:
            supplied = jnp.asarray(init_params)
            if supplied.ndim == 1:
                supplied = supplied[None, :]
            supplied = supplied[:population_size]
            params = params.at[: supplied.shape[0]].set(supplied)
            next_index = int(supplied.shape[0])

        if next_index < population_size:
            params = params.at[next_index].set(self._feasibility_anchor(obj))
            next_index += 1
        semantic_prior = self._semantic_prior(obj) if use_semantic_prior else None
        if semantic_prior is not None and next_index < population_size:
            params = params.at[next_index].set(semantic_prior)

        learning_rates = jnp.geomspace(
            learning_rate_low, learning_rate_high, population_size
        )[:, None]
        first_moment = jnp.zeros_like(params)
        second_moment = jnp.zeros_like(params)
        member_best_loss = jnp.full((population_size,), jnp.inf)
        stalled_steps = jnp.zeros((population_size,), dtype=jnp.int32)

        global_feasible_loss = float("inf")
        global_feasible_params = params[0]
        restart_round = 0
        step = 0

        # dfbench 0.3.3's public warmup helper discards asynchronous outputs,
        # so it cannot provide a guaranteed device barrier. Count compilation
        # inside the scored wall clock until the public helper is synchronous.
        if initial_population_callback is not None:
            initial_population_callback(params)
        obj.start_logging()

        while not obj.budget_exceeded:
            if obj.time_left is not None and obj.time_left <= safety_seconds:
                break
            losses, grads, aux = self._evaluate_population(
                obj, params, evaluation_chunk_size
            )

            finite_loss = jnp.isfinite(losses)
            improved = finite_loss & (
                losses < member_best_loss - minimum_improvement
            )
            member_best_loss = jnp.where(improved, losses, member_best_loss)
            stalled_steps = jnp.where(improved, 0, stalled_steps + 1)

            feasible_losses = jnp.where(
                finite_loss & jnp.asarray(aux["is_feasible"], dtype=bool),
                losses,
                jnp.inf,
            )
            feasible_index = int(jnp.argmin(feasible_losses))
            feasible_loss = float(feasible_losses[feasible_index])
            if feasible_loss < global_feasible_loss:
                global_feasible_loss = feasible_loss
                global_feasible_params = params[feasible_index]

            # Sanitize exceptional derivatives, then clip each population member
            # independently before applying an explicit Adam update.
            grads = jnp.nan_to_num(grads, nan=0.0, posinf=0.0, neginf=0.0)
            grad_norms = jnp.linalg.norm(grads, axis=1, keepdims=True)
            grads = grads * jnp.minimum(
                1.0, gradient_clip_norm / (grad_norms + 1e-12)
            )

            step += 1
            first_moment = beta1 * first_moment + (1.0 - beta1) * grads
            second_moment = beta2 * second_moment + (1.0 - beta2) * jnp.square(grads)
            corrected_first = first_moment / (1.0 - beta1**step)
            corrected_second = second_moment / (1.0 - beta2**step)
            params = params - learning_rates * corrected_first / (
                jnp.sqrt(corrected_second) + epsilon
            )

            restart_mask = stalled_steps >= patience
            if bool(jnp.any(restart_mask)):
                fresh_params = obj.random_params_unbounded(population_size)
                if global_feasible_loss < float("inf"):
                    # A smaller radius late in the run turns restarts from broad
                    # exploration into local refinement.
                    progress = float(obj.budget_progress_fraction)
                    scale = restart_noise_scale * max(0.10, 1.0 - progress)
                    noise = obj.random_params_unbounded(population_size)
                    noise = noise - jnp.mean(noise, axis=0, keepdims=True)
                    noise = noise / (
                        jnp.std(noise, axis=0, keepdims=True) + 1e-6
                    )
                    exploit_params = global_feasible_params[None, :] + scale * noise
                    member_ids = jnp.arange(population_size)
                    exploit_mask = (member_ids + restart_round) % 2 == 0
                    restart_params = jnp.where(
                        exploit_mask[:, None], exploit_params, fresh_params
                    )
                else:
                    restart_params = fresh_params

                params = jnp.where(restart_mask[:, None], restart_params, params)
                first_moment = jnp.where(
                    restart_mask[:, None], jnp.zeros_like(first_moment), first_moment
                )
                second_moment = jnp.where(
                    restart_mask[:, None], jnp.zeros_like(second_moment), second_moment
                )
                member_best_loss = jnp.where(
                    restart_mask, jnp.inf, member_best_loss
                )
                stalled_steps = jnp.where(restart_mask, 0, stalled_steps)
                restart_round += 1
