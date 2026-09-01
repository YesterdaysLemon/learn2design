"""Experiment-owned feasibility-debt clock candidate.

This module is mechanically based on the protected submission implementation,
but it is not a submission default.  The frozen study requires callers to
select either exact protected ``total_loss`` progress or the experimental
``feasibility_debt`` progress rule explicitly.
"""

import json
import math
import time
from pathlib import Path

import jax
import jax.numpy as jnp

from dfbench import Objective, OptimizationAlgorithm


class FeasibilityDebtBatchedRestartAdam(OptimizationAlgorithm):
    """A small population of Adam searches with deterministic restarts.

    Population members use different learning rates and share only the best
    physically feasible point. The packaged candidate defaults to seeded
    random initialization after the feasibility anchor. The archived
    development experiment can still opt into its official-archive semantic
    prior explicitly. Stalled members alternate between fresh random starts
    and perturbations around the shared point. All simulator access goes
    through the public ``Objective`` API.
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

    @staticmethod
    def _coverage_balance_unbounded(params):
        """Map each random column onto midpoint Latin-hypercube levels."""
        member_count = int(params.shape[0])
        if member_count < 2:
            return params

        order = jnp.argsort(params, axis=0, stable=True)
        ranks = jnp.argsort(order, axis=0, stable=True)
        unit = (
            ranks.astype(params.dtype)
            + jnp.asarray(0.5, dtype=params.dtype)
        ) / jnp.asarray(float(member_count), dtype=params.dtype)
        return jnp.log(unit) - jnp.log1p(-unit)

    @staticmethod
    def _warmup_population_evaluation(
        objective: Objective,
        population_size: int,
        evaluation_chunk_size: int | None,
    ) -> None:
        """Dispatch public warmups for every evaluation shape used by this run.

        ``dfbench==0.3.3`` deliberately returns ``None`` from its public
        ``warmup_*`` helpers.  JAX device execution is asynchronous, so this
        is a best-effort compilation warmup rather than a completion barrier.
        Both arms in paired studies use the identical policy.
        """
        if (
            evaluation_chunk_size is None
            or evaluation_chunk_size >= population_size
        ):
            objective.warmup_vmap_value_and_grad_aux(population_size)
            return

        batch_sizes = {
            min(evaluation_chunk_size, population_size - start)
            for start in range(0, population_size, evaluation_chunk_size)
        }
        for batch_size in sorted(batch_sizes):
            if batch_size == 1:
                objective.warmup_value_and_grad_aux()
            else:
                objective.warmup_vmap_value_and_grad_aux(batch_size)

    @staticmethod
    def _adam_step(
        params,
        grads,
        first_moment,
        second_moment,
        member_steps,
        learning_rates,
        beta1: float,
        beta2: float,
        epsilon: float,
    ):
        """Apply Adam with an independent bias-correction age per member."""
        member_steps = member_steps + 1
        first_moment = beta1 * first_moment + (1.0 - beta1) * grads
        second_moment = beta2 * second_moment + (1.0 - beta2) * jnp.square(grads)
        member_ages = member_steps[:, None]
        corrected_first = first_moment / (
            1.0 - jnp.power(beta1, member_ages)
        )
        corrected_second = second_moment / (
            1.0 - jnp.power(beta2, member_ages)
        )
        params = params - learning_rates * corrected_first / (
            jnp.sqrt(corrected_second) + epsilon
        )
        return params, first_moment, second_moment, member_steps

    @staticmethod
    def _validate_feasibility_debt_aux(aux, population_size: int) -> None:
        """Fail closed on a malformed public UIFO auxiliary batch."""
        if not isinstance(aux, dict):
            raise TypeError("aux must be a dictionary")

        required = {
            "is_feasible",
            "penalty",
            "sensitivity_loss",
            "violations",
            "power_values",
        }
        missing = required.difference(aux)
        if missing:
            raise ValueError(f"missing required aux leaves: {sorted(missing)}")
        if not isinstance(aux["power_values"], dict):
            raise TypeError("aux.power_values must be a dictionary")

        power_required = {"hard", "soft", "detector"}
        power_missing = power_required.difference(aux["power_values"])
        if power_missing:
            raise ValueError(
                f"missing required power leaves: {sorted(power_missing)}"
            )

        feasible = jnp.asarray(aux["is_feasible"])
        penalty = jnp.asarray(aux["penalty"])
        sensitivity = jnp.asarray(aux["sensitivity_loss"])
        violations = jnp.asarray(aux["violations"])
        if feasible.dtype != jnp.bool_ or feasible.shape != (population_size,):
            raise TypeError("aux.is_feasible must be bool[P]")
        for name, value in (
            ("penalty", penalty),
            ("sensitivity_loss", sensitivity),
        ):
            if not jnp.issubdtype(value.dtype, jnp.floating):
                raise TypeError(f"aux.{name} must be floating")
            if value.shape != (population_size,):
                raise ValueError(f"aux.{name} must have shape [P]")
        if not jnp.issubdtype(violations.dtype, jnp.floating):
            raise TypeError("aux.violations must be floating")
        if violations.ndim < 2 or violations.shape[0] != population_size:
            raise ValueError("aux.violations must have leading dimension P")

        for name in sorted(power_required):
            value = jnp.asarray(aux["power_values"][name])
            if not jnp.issubdtype(value.dtype, jnp.floating):
                raise TypeError(f"aux.power_values.{name} must be floating")
            if value.ndim < 1 or value.shape[0] != population_size:
                raise ValueError(
                    f"aux.power_values.{name} must have leading dimension P"
                )

        negative_finite = jnp.isfinite(penalty) & (penalty < 0.0)
        if bool(jnp.any(negative_finite)):
            raise ValueError("aux.penalty must not contain negative finite values")

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
        batch_time_safety_factor: float = 1.5,
        batch_time_window: int = 8,
        use_semantic_prior: bool = False,
        evaluation_chunk_size: int | None = None,
        initial_population_mode: str = "random",
        preclock_warmup: bool = False,
        raw_initial_population_callback=None,
        initial_population_callback=None,
        optimizer_telemetry_callback=None,
        *,
        progress_mode: str,
        **kwargs,
    ) -> None:
        del kwargs
        if progress_mode not in {"total_loss", "feasibility_debt"}:
            raise ValueError(
                "progress_mode must be 'total_loss' or 'feasibility_debt'"
            )
        if initial_population_mode not in {"random", "coverage_balanced"}:
            raise ValueError(
                "initial_population_mode must be 'random' or "
                "'coverage_balanced'"
            )
        if not isinstance(preclock_warmup, bool):
            raise ValueError("preclock_warmup must be a boolean")
        obj = objective
        self.prepare(obj, unbounded=True, random_seed=random_seed)

        population_size = max(2, int(population_size))
        safety_seconds = float(safety_seconds)
        batch_time_safety_factor = float(batch_time_safety_factor)
        batch_time_window = int(batch_time_window)
        if not math.isfinite(safety_seconds) or safety_seconds < 0.0:
            raise ValueError("safety_seconds must be finite and non-negative")
        if (
            not math.isfinite(batch_time_safety_factor)
            or batch_time_safety_factor < 1.0
        ):
            raise ValueError(
                "batch_time_safety_factor must be finite and at least one"
            )
        if batch_time_window < 1:
            raise ValueError("batch_time_window must be positive")
        if evaluation_chunk_size is not None:
            evaluation_chunk_size = int(evaluation_chunk_size)
            if not 1 <= evaluation_chunk_size <= population_size:
                raise ValueError(
                    "evaluation_chunk_size must be between one and population_size"
                )
        params = obj.random_params_unbounded(population_size)
        if raw_initial_population_callback is not None:
            raw_initial_population_callback(params)
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
            next_index += 1
        if (
            initial_population_mode == "coverage_balanced"
            and next_index < population_size
        ):
            params = params.at[next_index:].set(
                self._coverage_balance_unbounded(params[next_index:])
            )

        if preclock_warmup:
            # The frozen coverage screen opts both arms into this boundary.
            # Keep the packaged/default Round-1 timing path unchanged.
            params = jax.block_until_ready(params)

        learning_rates = jnp.geomspace(
            learning_rate_low, learning_rate_high, population_size
        )[:, None]
        first_moment = jnp.zeros_like(params)
        second_moment = jnp.zeros_like(params)
        member_steps = jnp.zeros((population_size,), dtype=jnp.int32)
        member_best_loss = jnp.full((population_size,), jnp.inf)
        stalled_steps = jnp.zeros((population_size,), dtype=jnp.int32)
        ever_feasible = jnp.zeros((population_size,), dtype=bool)
        best_infeasible_debt = jnp.full((population_size,), jnp.inf)
        best_feasible_loss = jnp.full((population_size,), jnp.inf)

        global_feasible_loss = float("inf")
        global_feasible_params = params[0]
        restart_round = 0
        completed_batches = 0
        recent_batch_seconds = []
        member_generations = (
            jnp.zeros((population_size,), dtype=jnp.int32)
            if optimizer_telemetry_callback is not None
            else None
        )

        if initial_population_callback is not None:
            initial_population_callback(params)
        if preclock_warmup:
            self._warmup_population_evaluation(
                obj, population_size, evaluation_chunk_size
            )
        obj.start_logging()

        while not obj.budget_exceeded:
            required_time_margin = float(safety_seconds)
            if recent_batch_seconds:
                required_time_margin = max(
                    required_time_margin,
                    batch_time_safety_factor * max(recent_batch_seconds),
                )
            if (
                obj.time_left is not None
                and obj.time_left <= required_time_margin
            ):
                break

            active_members = population_size
            if obj.evals_left is not None:
                active_members = min(active_members, int(obj.evals_left))
                if active_members <= 0:
                    break

            batch_started = time.perf_counter()
            losses, grads, aux = self._evaluate_population(
                obj, params[:active_members], evaluation_chunk_size
            )
            jax.block_until_ready((losses, grads, aux))
            batch_seconds = time.perf_counter() - batch_started
            completed_batches += 1
            if completed_batches > 1:
                recent_batch_seconds.append(batch_seconds)
                recent_batch_seconds = recent_batch_seconds[-batch_time_window:]

            finite_loss = jnp.isfinite(losses)
            feasible_losses = jnp.where(
                finite_loss & jnp.asarray(aux["is_feasible"], dtype=bool),
                losses,
                jnp.inf,
            )
            feasible_index = int(jnp.argmin(feasible_losses))
            feasible_loss = float(feasible_losses[feasible_index])
            global_feasible_improved = feasible_loss < global_feasible_loss
            if global_feasible_improved:
                global_feasible_loss = feasible_loss
                global_feasible_params = params[feasible_index]

            # An evaluation-limited partial tail has now consumed the remaining
            # budget. Objective logged it, and no further update can be used.
            if active_members < population_size:
                if optimizer_telemetry_callback is not None:
                    assert member_generations is not None
                    partial_improved = finite_loss & (
                        losses
                        < member_best_loss[:active_members] - minimum_improvement
                    )
                    observed_member_best_loss = jnp.where(
                        partial_improved,
                        losses,
                        member_best_loss[:active_members],
                    )
                    sanitized = jnp.nan_to_num(
                        grads, nan=0.0, posinf=0.0, neginf=0.0
                    )
                    gradient_norms = jnp.linalg.norm(sanitized, axis=1)
                    clip_scales = jnp.minimum(
                        1.0, gradient_clip_norm / (gradient_norms + 1e-12)
                    )
                    member_ids = jnp.arange(active_members, dtype=jnp.int16)
                    optimizer_telemetry_callback(
                        {
                            "batch_index": jnp.full(
                                (active_members,), completed_batches - 1,
                                dtype=jnp.int32,
                            ),
                            "member_index": member_ids,
                            "eval_count_after_batch": jnp.full(
                                (active_members,), int(obj.eval_count),
                                dtype=jnp.int32,
                            ),
                            "time_seconds": jnp.full(
                                (active_members,), float(obj.time_elapsed)
                            ),
                            "evaluation_batch_seconds": jnp.full(
                                (active_members,), batch_seconds
                            ),
                            "finite_loss": finite_loss,
                            "loss_float_bits": jnp.full(
                                (active_members,),
                                int(losses.dtype.itemsize * 8),
                                dtype=jnp.int16,
                            ),
                            "feasible": jnp.asarray(
                                aux["is_feasible"], dtype=bool
                            ),
                            "observed_member_improved": partial_improved,
                            "observed_member_best_loss": observed_member_best_loss,
                            "stalled_steps_before": stalled_steps[:active_members],
                            "stalled_steps_after": stalled_steps[:active_members],
                            "adam_age_before": member_steps[:active_members],
                            "adam_age_after": member_steps[:active_members],
                            "learning_rate": learning_rates[:active_members, 0],
                            "gradient_nonfinite_count": jnp.sum(
                                ~jnp.isfinite(grads), axis=1, dtype=jnp.int32
                            ),
                            "gradient_norm": gradient_norms,
                            "gradient_clip_scale": clip_scales,
                            "global_feasible_improvement": (
                                member_ids == feasible_index
                            ) & global_feasible_improved,
                            "restart_triggered": jnp.zeros(
                                (active_members,), dtype=bool
                            ),
                            "restart_kind": jnp.full(
                                (active_members,), -1, dtype=jnp.int8
                            ),
                            "restart_round": jnp.full(
                                (active_members,), -1, dtype=jnp.int32
                            ),
                            "restart_noise_scale": jnp.full(
                                (active_members,), jnp.nan
                            ),
                            "evaluated_generation": member_generations[
                                :active_members
                            ],
                            "next_generation": member_generations[:active_members],
                            "update_applied": jnp.zeros(
                                (active_members,), dtype=bool
                            ),
                            "budget_progress_fraction": jnp.full(
                                (active_members,),
                                float(obj.budget_progress_fraction),
                            ),
                        }
                    )
                break

            stalled_steps_before = (
                stalled_steps if optimizer_telemetry_callback is not None else None
            )
            treatment_event_before = None
            if progress_mode == "total_loss":
                improved = finite_loss & (
                    losses < member_best_loss - minimum_improvement
                )
                member_best_loss = jnp.where(improved, losses, member_best_loss)
            else:
                self._validate_feasibility_debt_aux(aux, population_size)
                feasible_now = jnp.asarray(aux["is_feasible"], dtype=bool)
                debt = jnp.asarray(aux["penalty"], dtype=losses.dtype)
                valid_debt = jnp.isfinite(debt) & (debt >= 0.0)
                if optimizer_telemetry_callback is not None:
                    treatment_event_before = {
                        "progress_ever_feasible_before": ever_feasible,
                        "progress_best_infeasible_debt_before": (
                            best_infeasible_debt
                        ),
                        "progress_best_feasible_loss_before": best_feasible_loss,
                    }
                infeasible_improved = (
                    ~ever_feasible
                    & ~feasible_now
                    & finite_loss
                    & valid_debt
                    & (
                        debt
                        < best_infeasible_debt - minimum_improvement
                    )
                )
                feasible_improved = (
                    feasible_now
                    & finite_loss
                    & (losses < best_feasible_loss - minimum_improvement)
                )
                improved = infeasible_improved | feasible_improved
                ever_feasible = ever_feasible | (feasible_now & finite_loss)
                best_infeasible_debt = jnp.where(
                    infeasible_improved, debt, best_infeasible_debt
                )
                best_feasible_loss = jnp.where(
                    feasible_improved, losses, best_feasible_loss
                )
                progress_value = jnp.where(feasible_now, losses, debt)
                member_best_loss = jnp.where(
                    improved, progress_value, member_best_loss
                )
            stalled_steps = jnp.where(improved, 0, stalled_steps + 1)

            # Sanitize exceptional derivatives, then clip each population member
            # independently before applying an explicit Adam update.
            if optimizer_telemetry_callback is None:
                grads = jnp.nan_to_num(grads, nan=0.0, posinf=0.0, neginf=0.0)
                grad_norms = jnp.linalg.norm(grads, axis=1, keepdims=True)
                grads = grads * jnp.minimum(
                    1.0, gradient_clip_norm / (grad_norms + 1e-12)
                )
                gradient_nonfinite_count = None
                gradient_norms = None
                gradient_clip_scales = None
            else:
                gradient_nonfinite_count = jnp.sum(
                    ~jnp.isfinite(grads), axis=1, dtype=jnp.int32
                )
                grads = jnp.nan_to_num(grads, nan=0.0, posinf=0.0, neginf=0.0)
                gradient_norms = jnp.linalg.norm(grads, axis=1)
                gradient_clip_scales = jnp.minimum(
                    1.0, gradient_clip_norm / (gradient_norms + 1e-12)
                )
                grads = grads * gradient_clip_scales[:, None]

            member_steps_before = (
                member_steps if optimizer_telemetry_callback is not None else None
            )
            params, first_moment, second_moment, member_steps = self._adam_step(
                params,
                grads,
                first_moment,
                second_moment,
                member_steps,
                learning_rates,
                beta1,
                beta2,
                epsilon,
            )

            member_best_loss_for_event = (
                member_best_loss
                if optimizer_telemetry_callback is not None
                else None
            )
            stalled_steps_for_event = (
                stalled_steps if optimizer_telemetry_callback is not None else None
            )
            treatment_event_after = None
            if (
                progress_mode == "feasibility_debt"
                and optimizer_telemetry_callback is not None
            ):
                assert treatment_event_before is not None
                treatment_event_after = {
                    **treatment_event_before,
                    "progress_ever_feasible_after": ever_feasible,
                    "progress_best_infeasible_debt_after": best_infeasible_debt,
                    "progress_best_feasible_loss_after": best_feasible_loss,
                }
            restart_mask = stalled_steps >= patience
            restart_kind = (
                jnp.full((population_size,), -1, dtype=jnp.int8)
                if optimizer_telemetry_callback is not None
                else None
            )
            restart_scales = (
                jnp.full((population_size,), jnp.nan)
                if optimizer_telemetry_callback is not None
                else None
            )
            restart_round_for_event = (
                -1 if optimizer_telemetry_callback is not None else None
            )
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
                    if restart_kind is not None and restart_scales is not None:
                        restart_kind = jnp.where(
                            restart_mask,
                            jnp.where(exploit_mask, 1, 0),
                            restart_kind,
                        ).astype(jnp.int8)
                        restart_scales = jnp.where(
                            restart_mask & exploit_mask,
                            scale,
                            restart_scales,
                        )
                else:
                    restart_params = fresh_params
                    if restart_kind is not None:
                        restart_kind = jnp.where(
                            restart_mask, 0, restart_kind
                        ).astype(jnp.int8)

                restart_round_for_event = restart_round
                params = jnp.where(restart_mask[:, None], restart_params, params)
                first_moment = jnp.where(
                    restart_mask[:, None], jnp.zeros_like(first_moment), first_moment
                )
                second_moment = jnp.where(
                    restart_mask[:, None], jnp.zeros_like(second_moment), second_moment
                )
                member_steps = jnp.where(restart_mask, 0, member_steps)
                member_best_loss = jnp.where(
                    restart_mask, jnp.inf, member_best_loss
                )
                stalled_steps = jnp.where(restart_mask, 0, stalled_steps)
                ever_feasible = jnp.where(restart_mask, False, ever_feasible)
                best_infeasible_debt = jnp.where(
                    restart_mask, jnp.inf, best_infeasible_debt
                )
                best_feasible_loss = jnp.where(
                    restart_mask, jnp.inf, best_feasible_loss
                )
                if member_generations is not None:
                    member_generations = member_generations + restart_mask.astype(
                        jnp.int32
                    )
                restart_round += 1

            if optimizer_telemetry_callback is not None:
                assert member_generations is not None
                assert gradient_nonfinite_count is not None
                assert gradient_norms is not None
                assert gradient_clip_scales is not None
                assert member_steps_before is not None
                assert member_best_loss_for_event is not None
                assert stalled_steps_before is not None
                assert stalled_steps_for_event is not None
                assert restart_round_for_event is not None
                assert restart_kind is not None
                assert restart_scales is not None
                member_ids = jnp.arange(population_size, dtype=jnp.int16)
                telemetry_event = {
                        "batch_index": jnp.full(
                            (population_size,), completed_batches - 1,
                            dtype=jnp.int32,
                        ),
                        "member_index": member_ids,
                        "eval_count_after_batch": jnp.full(
                            (population_size,), int(obj.eval_count),
                            dtype=jnp.int32,
                        ),
                        "time_seconds": jnp.full(
                            (population_size,), float(obj.time_elapsed)
                        ),
                        "evaluation_batch_seconds": jnp.full(
                            (population_size,), batch_seconds
                        ),
                        "finite_loss": finite_loss,
                        "loss_float_bits": jnp.full(
                            (population_size,),
                            int(losses.dtype.itemsize * 8),
                            dtype=jnp.int16,
                        ),
                        "feasible": jnp.asarray(aux["is_feasible"], dtype=bool),
                        "observed_member_improved": improved,
                        "observed_member_best_loss": member_best_loss_for_event,
                        "stalled_steps_before": stalled_steps_before,
                        "stalled_steps_after": stalled_steps_for_event,
                        "adam_age_before": member_steps_before,
                        "adam_age_after": member_steps,
                        "learning_rate": learning_rates[:, 0],
                        "gradient_nonfinite_count": gradient_nonfinite_count,
                        "gradient_norm": gradient_norms,
                        "gradient_clip_scale": gradient_clip_scales,
                        "global_feasible_improvement": (
                            member_ids == feasible_index
                        ) & global_feasible_improved,
                        "restart_triggered": restart_mask,
                        "restart_kind": restart_kind,
                        "restart_round": jnp.where(
                            restart_mask,
                            restart_round_for_event,
                            jnp.full((population_size,), -1, dtype=jnp.int32),
                        ),
                        "restart_noise_scale": restart_scales,
                        "evaluated_generation": member_generations
                        - restart_mask.astype(jnp.int32),
                        "next_generation": member_generations,
                        "update_applied": jnp.ones(
                            (population_size,), dtype=bool
                        ),
                        "budget_progress_fraction": jnp.full(
                            (population_size,),
                            float(obj.budget_progress_fraction),
                        ),
                    }
                if treatment_event_after is not None:
                    telemetry_event.update(treatment_event_after)
                optimizer_telemetry_callback(telemetry_event)
