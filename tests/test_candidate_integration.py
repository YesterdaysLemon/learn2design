from __future__ import annotations

import inspect
import json
import math

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("dfbench")
np = pytest.importorskip("numpy")

from submission.submission import BatchedRestartAdam
from experiments.uifo_paired.candidate_probe import construct_candidates, host_pytree
from experiments.uifo_paired.optimizer_telemetry import (
    OptimizerTelemetryCapture,
    validate_optimizer_telemetry,
)
from experiments.uifo_paired.runner import BATCHED_SETTINGS, _parameter_hashes


class AnalyticObjective:
    """Small public-API stand-in for deterministic optimizer tests."""

    def __init__(self, n_params: int = 12, max_evals: int = 60) -> None:
        self.n_params = n_params
        self.max_evals = max_evals
        self.eval_count = 0
        self.algorithm_str = ""
        self.unbounded = False
        self._key = jax.random.PRNGKey(0)
        self._started = False
        self.warmup_calls = 0
        self.warmup_batch_sizes: list[int] = []
        self.scalar_warmup_calls = 0
        self.feasible_history: list[object] = []
        self.best_feasible_loss = math.inf
        self.first_feasible_loss = math.inf
        self.optimization_pairs = [
            ["laser", "power"],
            ["squeezer", "db"],
            ["mirror", "reflectivity"],
            *[[f"component_{index}", "tuning"] for index in range(n_params - 3)],
        ]

    def set_space_mode(self, unbounded: bool) -> None:
        self.unbounded = unbounded

    def set_seed(self, seed: int) -> None:
        self._key = jax.random.PRNGKey(seed)

    def random_params_unbounded(self, n_samples: int = 1):
        self._key, sample_key = jax.random.split(self._key)
        unit = jax.random.uniform(
            sample_key,
            shape=(n_samples, self.n_params),
            minval=1e-4,
            maxval=1.0 - 1e-4,
        )
        return jnp.log(unit) - jnp.log1p(-unit)

    @property
    def budget_exceeded(self) -> bool:
        return self.eval_count >= self.max_evals

    @property
    def evals_left(self) -> int:
        return max(0, self.max_evals - self.eval_count)

    @property
    def budget_progress_fraction(self) -> float:
        return self.eval_count / self.max_evals

    @property
    def time_left(self):
        return None

    def _value(self, params):
        unit = jax.nn.sigmoid(params)
        target = jnp.full((self.n_params,), 0.70).at[:3].set(0.10)
        sensitivity_loss = jnp.sum(jnp.square(unit - target))
        violations = jnp.maximum(unit[:3] - 0.25, 0.0)
        is_feasible = jnp.all(violations == 0.0)
        penalty = 20.0 * jnp.sum(violations)
        return sensitivity_loss + penalty, {"is_feasible": is_feasible}

    def warmup_vmap_value_and_grad_aux(self, batch_size: int) -> None:
        self.warmup_calls += 1
        self.warmup_batch_sizes.append(batch_size)
        batch = jnp.zeros((batch_size, self.n_params))
        values = jax.jit(jax.vmap(jax.value_and_grad(self._value, has_aux=True)))(batch)
        jax.block_until_ready(values)

    def warmup_value_and_grad_aux(self) -> None:
        self.warmup_calls += 1
        self.scalar_warmup_calls += 1
        values = jax.jit(jax.value_and_grad(self._value, has_aux=True))(
            jnp.zeros((self.n_params,))
        )
        jax.block_until_ready(values)

    def start_logging(self) -> None:
        self._started = True

    def vmap_value_and_grad_aux(self, params):
        if not self._started:
            raise RuntimeError("evaluation before start_logging")
        (losses, aux), grads = jax.vmap(
            jax.value_and_grad(self._value, has_aux=True)
        )(params)
        self.eval_count += int(params.shape[0])
        feasible = aux["is_feasible"]
        self.feasible_history.append(feasible)
        feasible_losses = jnp.where(feasible, losses, jnp.inf)
        batch_best = float(jnp.min(feasible_losses))
        if self.first_feasible_loss == math.inf and batch_best < math.inf:
            self.first_feasible_loss = batch_best
        self.best_feasible_loss = min(self.best_feasible_loss, batch_best)
        return losses, grads, aux

    def value_and_grad_aux(self, params):
        if not self._started:
            raise RuntimeError("evaluation before start_logging")
        (loss, aux), grad = jax.value_and_grad(self._value, has_aux=True)(params)
        self.eval_count += 1
        feasible = aux["is_feasible"]
        self.feasible_history.append(feasible)
        feasible_loss = float(jnp.where(feasible, loss, jnp.inf))
        if self.first_feasible_loss == math.inf and feasible_loss < math.inf:
            self.first_feasible_loss = feasible_loss
        self.best_feasible_loss = min(self.best_feasible_loss, feasible_loss)
        return loss, grad, aux


@pytest.mark.integration
def test_paired_harness_records_the_submission_defaults() -> None:
    parameters = inspect.signature(BatchedRestartAdam.optimize).parameters

    assert set(BATCHED_SETTINGS) <= set(parameters)
    for name, value in BATCHED_SETTINGS.items():
        assert parameters[name].default == value
    assert parameters["use_semantic_prior"].default is False
    assert parameters["initial_population_mode"].default == "random"
    assert parameters["preclock_warmup"].default is False
    assert parameters["raw_initial_population_callback"].default is None


@pytest.mark.integration
def test_packaged_default_does_not_load_the_semantic_prior(monkeypatch) -> None:
    objective = AnalyticObjective(n_params=5, max_evals=2)

    def fail_if_loaded(*_args, **_kwargs):
        raise AssertionError("the default candidate loaded the semantic prior")

    monkeypatch.setattr(BatchedRestartAdam, "_semantic_prior", fail_if_loaded)
    BatchedRestartAdam().optimize(
        objective,
        random_seed=7,
        population_size=2,
        safety_seconds=0.0,
    )


@pytest.mark.integration
def test_explicit_random_initialization_is_the_unchanged_default_path() -> None:
    class TracedObjective(AnalyticObjective):
        def __init__(self) -> None:
            super().__init__(n_params=5, max_evals=12)
            self.inputs = []

        def vmap_value_and_grad_aux(self, params):
            self.inputs.append(np.asarray(jax.device_get(params)))
            return super().vmap_value_and_grad_aux(params)

    implicit = TracedObjective()
    explicit = TracedObjective()
    settings = {
        "random_seed": 11,
        "population_size": 4,
        "patience": 10,
        "safety_seconds": 0.0,
    }

    BatchedRestartAdam().optimize(implicit, **settings)
    BatchedRestartAdam().optimize(
        explicit, initial_population_mode="random", **settings
    )

    assert len(implicit.inputs) == len(explicit.inputs) == 3
    for implicit_params, explicit_params in zip(
        implicit.inputs, explicit.inputs, strict=True
    ):
        np.testing.assert_array_equal(implicit_params, explicit_params)
    assert implicit.best_feasible_loss == explicit.best_feasible_loss


@pytest.mark.integration
def test_coverage_arms_expose_the_identical_pretransform_random_draw() -> None:
    control_draws = []
    treatment_draws = []
    common = {
        "random_seed": 17,
        "population_size": 4,
        "safety_seconds": 0.0,
    }

    BatchedRestartAdam().optimize(
        AnalyticObjective(n_params=5, max_evals=4),
        raw_initial_population_callback=lambda value: control_draws.append(
            np.asarray(jax.device_get(value))
        ),
        **common,
    )
    BatchedRestartAdam().optimize(
        AnalyticObjective(n_params=5, max_evals=4),
        initial_population_mode="coverage_balanced",
        raw_initial_population_callback=lambda value: treatment_draws.append(
            np.asarray(jax.device_get(value))
        ),
        **common,
    )

    assert len(control_draws) == len(treatment_draws) == 1
    np.testing.assert_array_equal(control_draws[0], treatment_draws[0])


@pytest.mark.integration
def test_coverage_initialization_finishes_before_callback_and_objective_clock(
    monkeypatch,
) -> None:
    events: list[str] = []
    real_block_until_ready = jax.block_until_ready

    def traced_block_until_ready(value):
        events.append("initialization_ready")
        return real_block_until_ready(value)

    monkeypatch.setattr(jax, "block_until_ready", traced_block_until_ready)

    class TracedObjective(AnalyticObjective):
        def start_logging(self) -> None:
            events.append("objective_clock_started")
            super().start_logging()

    BatchedRestartAdam().optimize(
        TracedObjective(n_params=5, max_evals=4),
        random_seed=17,
        population_size=4,
        initial_population_mode="coverage_balanced",
        preclock_warmup=True,
        initial_population_callback=lambda _value: events.append(
            "initial_population_captured"
        ),
        safety_seconds=0.0,
    )

    assert events[:2] == [
        "initialization_ready",
        "initial_population_captured",
    ]
    assert events.index("objective_clock_started") > 1


@pytest.mark.integration
def test_default_round1_path_does_not_add_a_preclock_ready_barrier(
    monkeypatch,
) -> None:
    events: list[str] = []
    real_block_until_ready = jax.block_until_ready

    def traced_block_until_ready(value):
        events.append("ready_barrier")
        return real_block_until_ready(value)

    class TracedObjective(AnalyticObjective):
        def start_logging(self) -> None:
            events.append("objective_clock_started")
            super().start_logging()

    monkeypatch.setattr(jax, "block_until_ready", traced_block_until_ready)
    BatchedRestartAdam().optimize(
        TracedObjective(n_params=5, max_evals=4),
        random_seed=17,
        population_size=4,
        safety_seconds=0.0,
    )
    assert events[0] == "objective_clock_started"
    assert "ready_barrier" in events[1:]


@pytest.mark.integration
@pytest.mark.parametrize("member_count", [6, 7])
def test_coverage_balancing_uses_every_midpoint_latin_hypercube_level(
    member_count: int,
) -> None:
    values = jnp.reshape(
        jnp.arange(member_count * 5, dtype=jnp.float32),
        (member_count, 5),
    )

    balanced = BatchedRestartAdam._coverage_balance_unbounded(values)
    unit = np.asarray(jax.device_get(jax.nn.sigmoid(balanced)))
    expected = (np.arange(member_count) + 0.5) / member_count

    assert bool(np.all(np.isfinite(unit)))
    for column in range(unit.shape[1]):
        np.testing.assert_allclose(
            np.sort(unit[:, column]), expected, rtol=1e-6, atol=1e-7
        )


@pytest.mark.integration
def test_coverage_balancing_preserves_anchor_prior_and_objective_rng_state() -> None:
    control = AnalyticObjective(n_params=5, max_evals=8)
    treatment = AnalyticObjective(n_params=5, max_evals=8)
    control_population = []
    treatment_population = []

    BatchedRestartAdam().optimize(
        control,
        random_seed=11,
        population_size=8,
        safety_seconds=0.0,
        initial_population_callback=control_population.append,
    )
    BatchedRestartAdam().optimize(
        treatment,
        random_seed=11,
        population_size=8,
        safety_seconds=0.0,
        initial_population_mode="coverage_balanced",
        initial_population_callback=treatment_population.append,
    )

    np.testing.assert_array_equal(
        np.asarray(control_population[0][0]),
        np.asarray(treatment_population[0][0]),
    )
    unit = np.asarray(jax.nn.sigmoid(treatment_population[0][1:]))
    expected = (np.arange(7) + 0.5) / 7
    for column in range(unit.shape[1]):
        np.testing.assert_allclose(
            np.sort(unit[:, column]), expected, rtol=1e-6, atol=1e-7
        )
    np.testing.assert_array_equal(
        np.asarray(control.random_params_unbounded(3)),
        np.asarray(treatment.random_params_unbounded(3)),
    )

    prior_objective = AnalyticObjective(n_params=5, max_evals=8)
    prior_population = []
    expected_prior = BatchedRestartAdam._semantic_prior(prior_objective)
    BatchedRestartAdam().optimize(
        prior_objective,
        random_seed=11,
        population_size=8,
        safety_seconds=0.0,
        use_semantic_prior=True,
        initial_population_mode="coverage_balanced",
        initial_population_callback=prior_population.append,
    )
    np.testing.assert_array_equal(
        np.asarray(prior_population[0][0]),
        np.asarray(BatchedRestartAdam._feasibility_anchor(prior_objective)),
    )
    np.testing.assert_array_equal(
        np.asarray(prior_population[0][1]), np.asarray(expected_prior)
    )
    prior_unit = np.asarray(jax.nn.sigmoid(prior_population[0][2:]))
    prior_expected = (np.arange(6) + 0.5) / 6
    for column in range(prior_unit.shape[1]):
        np.testing.assert_allclose(
            np.sort(prior_unit[:, column]),
            prior_expected,
            rtol=1e-6,
            atol=1e-7,
        )


@pytest.mark.integration
def test_invalid_initialization_mode_fails_before_random_sampling() -> None:
    class CountingObjective(AnalyticObjective):
        def __init__(self) -> None:
            super().__init__(n_params=5, max_evals=2)
            self.random_calls = 0

        def random_params_unbounded(self, n_samples: int = 1):
            self.random_calls += 1
            return super().random_params_unbounded(n_samples)

    objective = CountingObjective()
    with pytest.raises(ValueError, match="initial_population_mode"):
        BatchedRestartAdam().optimize(
            objective,
            random_seed=11,
            population_size=2,
            initial_population_mode="unknown",
        )
    assert objective.random_calls == 0


@pytest.mark.integration
def test_feasibility_anchor_understands_dfbench_pair_shapes() -> None:
    objective = AnalyticObjective(n_params=5, max_evals=5)
    objective.optimization_pairs[-1] = [
        ["space_a", "length"],
        ["space_b", "length"],
    ]

    unit = jax.nn.sigmoid(BatchedRestartAdam._feasibility_anchor(objective))

    assert float(unit[0]) < 1e-6
    assert float(unit[1]) < 1e-6
    assert float(unit[2]) == pytest.approx(1e-4, rel=1e-4)
    assert float(unit[3]) == pytest.approx(0.5)
    assert float(unit[4]) == pytest.approx(0.5)


@pytest.mark.integration
def test_semantic_prior_matches_runtime_pair_shapes() -> None:
    objective = AnalyticObjective(n_params=5, max_evals=5)
    objective.optimization_pairs[-1] = [
        ["space_a", "length"],
        ["space_b", "length"],
    ]

    unbounded = BatchedRestartAdam._semantic_prior(objective)
    assert unbounded is not None
    unit = jax.nn.sigmoid(unbounded)

    assert unit.shape == (5,)
    assert bool(jnp.all(jnp.isfinite(unit)))
    assert bool(jnp.all((unit > 0.0) & (unit < 1.0)))
    assert float(unit[0]) != pytest.approx(0.5)
    assert float(unit[4]) != pytest.approx(0.5)


@pytest.mark.integration
def test_adam_bias_correction_age_resets_per_restarted_member() -> None:
    beta1 = 0.9
    beta2 = 0.999
    late_age = 600
    params = jnp.zeros((2, 1))
    grads = jnp.ones_like(params)
    first_moment = jnp.asarray(
        [[0.0], [1.0 - beta1**late_age]], dtype=params.dtype
    )
    second_moment = jnp.asarray(
        [[0.0], [1.0 - beta2**late_age]], dtype=params.dtype
    )
    member_steps = jnp.asarray([0, late_age], dtype=jnp.int32)
    learning_rates = jnp.full((2, 1), 0.1)

    updated, _, _, updated_steps = BatchedRestartAdam._adam_step(
        params,
        grads,
        first_moment,
        second_moment,
        member_steps,
        learning_rates,
        beta1,
        beta2,
        1e-8,
    )

    assert updated_steps.tolist() == [1, late_age + 1]
    assert (-updated[:, 0]).tolist() == pytest.approx([0.1, 0.1], rel=1e-5)


@pytest.mark.integration
def test_optimizer_loop_restarts_with_a_fresh_adam_age() -> None:
    class ScriptedRestartObjective(AnalyticObjective):
        def __init__(self) -> None:
            super().__init__(n_params=1, max_evals=10)
            self.optimization_pairs = [["component", "tuning"]]
            self.inputs = []
            self.random_calls = 0
            self.losses = (
                [4.0, 3.0],
                [3.0, 3.0],
                [2.0, 3.0],
                [1.0, 2.0],
                [0.0, 1.0],
            )

        def random_params_unbounded(self, n_samples: int = 1):
            value = 0.0 if self.random_calls == 0 else 10.0
            self.random_calls += 1
            return jnp.full((n_samples, self.n_params), value)

        def vmap_value_and_grad_aux(self, params):
            if not self._started:
                raise RuntimeError("evaluation before start_logging")
            call_index = len(self.inputs)
            self.inputs.append(jnp.asarray(params))
            losses = jnp.asarray(self.losses[call_index])
            grads = jnp.ones_like(params)
            feasible = jnp.zeros((params.shape[0],), dtype=bool)
            self.eval_count += int(params.shape[0])
            self.feasible_history.append(feasible)
            return losses, grads, {"is_feasible": feasible}

    objective = ScriptedRestartObjective()
    initial = jnp.zeros((2, 1))

    BatchedRestartAdam().optimize(
        objective,
        init_params=initial,
        random_seed=11,
        population_size=2,
        learning_rate_low=0.1,
        learning_rate_high=0.1,
        patience=2,
        epsilon=0.0,
        safety_seconds=0,
    )

    assert len(objective.inputs) == 5
    assert objective.inputs[4][:, 0].tolist() == pytest.approx(
        [-0.4, 9.9], rel=1e-5
    )


@pytest.mark.integration
def test_candidate_obeys_lifecycle_budget_and_feasibility() -> None:
    objective = AnalyticObjective(max_evals=60)
    BatchedRestartAdam().optimize(
        objective,
        random_seed=11,
        population_size=6,
        patience=3,
        safety_seconds=0,
    )

    assert objective.algorithm_str == "batched_restart_adam"
    assert objective.unbounded is True
    assert objective.eval_count == objective.max_evals
    assert objective.warmup_calls == 0
    assert len(objective.feasible_history) == 10
    assert bool(objective.feasible_history[0].any())
    assert math.isfinite(objective.best_feasible_loss)
    assert objective.best_feasible_loss <= objective.first_feasible_loss


@pytest.mark.integration
def test_preclock_warmup_precedes_logging_without_spending_objective_budget() -> None:
    class LifecycleObjective(AnalyticObjective):
        def __init__(self) -> None:
            super().__init__(n_params=5, max_evals=8)
            self.events = []

        def warmup_vmap_value_and_grad_aux(self, batch_size: int) -> None:
            assert not self._started
            assert self.eval_count == 0
            assert self.feasible_history == []
            self.events.append(("warmup_vmap", batch_size))
            super().warmup_vmap_value_and_grad_aux(batch_size)

        def start_logging(self) -> None:
            assert self.eval_count == 0
            assert self.feasible_history == []
            self.events.append(("start_logging", None))
            super().start_logging()

        def vmap_value_and_grad_aux(self, params):
            self.events.append(("evaluate_vmap", int(params.shape[0])))
            return super().vmap_value_and_grad_aux(params)

    objective = LifecycleObjective()
    BatchedRestartAdam().optimize(
        objective,
        random_seed=11,
        population_size=4,
        safety_seconds=0.0,
        preclock_warmup=True,
    )

    assert objective.events[:3] == [
        ("warmup_vmap", 4),
        ("start_logging", None),
        ("evaluate_vmap", 4),
    ]
    assert objective.warmup_calls == 1
    assert objective.eval_count == 8
    assert len(objective.feasible_history) == 2


@pytest.mark.integration
def test_preclock_warmup_covers_every_chunk_evaluation_shape() -> None:
    objective = AnalyticObjective(n_params=5, max_evals=5)

    BatchedRestartAdam().optimize(
        objective,
        random_seed=11,
        population_size=5,
        safety_seconds=0.0,
        evaluation_chunk_size=2,
        preclock_warmup=True,
    )

    assert objective.warmup_calls == 2
    assert objective.scalar_warmup_calls == 1
    assert objective.warmup_batch_sizes == [2]
    assert objective.eval_count == 5
    assert [int(np.asarray(value).size) for value in objective.feasible_history] == [
        2,
        2,
        1,
    ]


@pytest.mark.integration
def test_coverage_balancing_does_not_change_forced_restart_random_samples() -> None:
    class ConstantInfeasibleObjective(AnalyticObjective):
        def __init__(self) -> None:
            super().__init__(n_params=3, max_evals=9)
            self.random_batches = []

        def random_params_unbounded(self, n_samples: int = 1):
            result = super().random_params_unbounded(n_samples)
            self.random_batches.append(np.asarray(jax.device_get(result)))
            return result

        def vmap_value_and_grad_aux(self, params):
            if not self._started:
                raise RuntimeError("evaluation before start_logging")
            member_count = int(params.shape[0])
            self.eval_count += member_count
            feasible = jnp.zeros((member_count,), dtype=bool)
            self.feasible_history.append(feasible)
            return (
                jnp.ones((member_count,)),
                jnp.zeros_like(params),
                {"is_feasible": feasible},
            )

    control = ConstantInfeasibleObjective()
    treatment = ConstantInfeasibleObjective()
    common = {
        "random_seed": 11,
        "population_size": 3,
        "patience": 1,
        "safety_seconds": 0.0,
    }

    BatchedRestartAdam().optimize(control, **common)
    BatchedRestartAdam().optimize(
        treatment, initial_population_mode="coverage_balanced", **common
    )

    assert len(control.random_batches) == len(treatment.random_batches) == 2
    for control_batch, treatment_batch in zip(
        control.random_batches, treatment.random_batches, strict=True
    ):
        np.testing.assert_array_equal(control_batch, treatment_batch)


@pytest.mark.integration
def test_candidate_logs_a_partial_final_population_without_budget_overshoot() -> None:
    objective = AnalyticObjective(n_params=5, max_evals=6)

    BatchedRestartAdam().optimize(
        objective,
        random_seed=11,
        population_size=4,
        patience=10,
        safety_seconds=0,
    )

    assert objective.eval_count == 6
    assert [len(value) for value in objective.feasible_history] == [4, 2]


@pytest.mark.integration
def test_optimizer_telemetry_preserves_default_numerical_path_and_partial_tail(
    tmp_path,
) -> None:
    class TracedObjective(AnalyticObjective):
        def __init__(self) -> None:
            super().__init__(n_params=5, max_evals=10)
            self.inputs = []

        @property
        def time_elapsed(self) -> float:
            return 0.0

        def vmap_value_and_grad_aux(self, params):
            self.inputs.append(np.asarray(jax.device_get(params)))
            return super().vmap_value_and_grad_aux(params)

    default = TracedObjective()
    instrumented = TracedObjective()
    capture = OptimizerTelemetryCapture()
    common = {
        "random_seed": 11,
        "population_size": 4,
        "patience": 10,
        "safety_seconds": 0,
    }

    BatchedRestartAdam().optimize(default, **common)
    BatchedRestartAdam().optimize(
        instrumented, optimizer_telemetry_callback=capture, **common
    )

    assert len(default.inputs) == len(instrumented.inputs) == 3
    for default_params, instrumented_params in zip(
        default.inputs, instrumented.inputs, strict=True
    ):
        np.testing.assert_array_equal(default_params, instrumented_params)
    assert default.best_feasible_loss == instrumented.best_feasible_loss

    telemetry_path = tmp_path / "telemetry.npz"
    summary = capture.write(telemetry_path)
    arrays = validate_optimizer_telemetry(
        telemetry_path,
        expected_rows=10,
        expected_population_size=4,
        expected_patience=10,
    )
    assert summary["rows"] == 10
    assert arrays["batch_index"].tolist() == [0] * 4 + [1] * 4 + [2] * 2
    assert arrays["update_applied"].tolist() == [True] * 8 + [False] * 2
    assert arrays["member_index"].tolist() == [0, 1, 2, 3] * 2 + [0, 1]


@pytest.mark.integration
def test_optimizer_telemetry_attributes_restart_mechanics(tmp_path) -> None:
    class ConstantFeasibleObjective(AnalyticObjective):
        def __init__(self) -> None:
            super().__init__(n_params=1, max_evals=6)
            self.optimization_pairs = [["component", "tuning"]]
            self.inputs = []

        @property
        def time_elapsed(self) -> float:
            return 0.0

        def vmap_value_and_grad_aux(self, params):
            if not self._started:
                raise RuntimeError("evaluation before start_logging")
            self.inputs.append(np.asarray(jax.device_get(params)))
            member_count = int(params.shape[0])
            self.eval_count += member_count
            feasible = jnp.ones((member_count,), dtype=bool)
            self.feasible_history.append(feasible)
            self.best_feasible_loss = 1.0
            if self.first_feasible_loss == math.inf:
                self.first_feasible_loss = 1.0
            return (
                jnp.ones((member_count,)),
                jnp.full_like(params, 4.0),
                {"is_feasible": feasible},
            )

    default = ConstantFeasibleObjective()
    instrumented = ConstantFeasibleObjective()
    capture = OptimizerTelemetryCapture()
    settings = {
        "random_seed": 7,
        "population_size": 2,
        "patience": 1,
        "safety_seconds": 0,
    }
    BatchedRestartAdam().optimize(default, **settings)
    BatchedRestartAdam().optimize(
        instrumented,
        optimizer_telemetry_callback=capture,
        **settings,
    )
    for default_params, instrumented_params in zip(
        default.inputs, instrumented.inputs, strict=True
    ):
        np.testing.assert_array_equal(default_params, instrumented_params)
    path = tmp_path / "restart-telemetry.npz"
    capture.write(path)
    arrays = validate_optimizer_telemetry(
        path,
        expected_rows=6,
        expected_population_size=2,
        expected_patience=1,
    )

    batch_zero = arrays["batch_index"] == 0
    assert arrays["global_feasible_improvement"][batch_zero].tolist() == [
        True,
        False,
    ]
    assert arrays["gradient_norm"][batch_zero].tolist() == pytest.approx([4, 4])
    assert arrays["gradient_clip_scale"][batch_zero].tolist() == pytest.approx(
        [0.25, 0.25]
    )

    restarted = arrays["restart_triggered"]
    assert arrays["batch_index"][restarted].tolist() == [1, 1]
    assert arrays["restart_kind"][restarted].tolist() == [1, 0]
    assert arrays["restart_round"][restarted].tolist() == [0, 0]
    assert arrays["adam_age_before"][restarted].tolist() == [1, 1]
    assert arrays["adam_age_after"][restarted].tolist() == [0, 0]
    assert arrays["evaluated_generation"][restarted].tolist() == [0, 0]
    assert arrays["next_generation"][restarted].tolist() == [1, 1]


@pytest.mark.integration
def test_candidate_uses_observed_batch_time_for_the_tail_guard() -> None:
    class TailGuardObjective(AnalyticObjective):
        @property
        def time_left(self):
            return 1.0 if self.eval_count <= 2 else 1e-12

    objective = TailGuardObjective(n_params=5, max_evals=20)

    BatchedRestartAdam().optimize(
        objective,
        random_seed=11,
        population_size=2,
        patience=10,
        safety_seconds=0,
    )

    assert objective.eval_count == 4
    assert len(objective.feasible_history) == 2


@pytest.mark.integration
def test_low_memory_chunks_preserve_the_full_initial_population() -> None:
    objective = AnalyticObjective(n_params=5, max_evals=4)
    captured = []
    BatchedRestartAdam().optimize(
        objective,
        random_seed=11,
        population_size=2,
        patience=10,
        safety_seconds=0,
        evaluation_chunk_size=1,
        initial_population_callback=captured.append,
    )

    assert objective.eval_count == 4
    assert len(objective.feasible_history) == 4
    assert captured[0].shape == (2, 5)
    assert objective.warmup_calls == 0


@pytest.mark.integration
def test_candidate_probe_reproduces_population_two_roles() -> None:
    objective = AnalyticObjective(n_params=5, max_evals=2)
    candidates, random_population = construct_candidates(
        objective, optimizer_seed=11, population_size=2
    )

    control = jnp.stack(
        [candidates["anchor"], candidates["random_member_1"]]
    )
    treatment = jnp.stack(
        [candidates["anchor"], candidates["semantic_prior"]]
    )

    assert _parameter_hashes(control)[0] == _parameter_hashes(treatment)[0]
    assert _parameter_hashes(control)[1] == _parameter_hashes(random_population)[1]
    assert _parameter_hashes(treatment)[1] == _parameter_hashes(
        candidates["semantic_prior"]
    )[0]


@pytest.mark.integration
def test_candidate_probe_recursively_hosts_nested_auxiliary_data() -> None:
    hosted = host_pytree(
        {
            "is_feasible": jnp.asarray(True),
            "power_values": {
                "laser": jnp.asarray([1.0, 2.0]),
                "nested": (jnp.asarray(3.0),),
            },
        }
    )

    assert hosted == {
        "is_feasible": True,
        "power_values": {"laser": [1.0, 2.0], "nested": (3.0,)},
    }
    json.dumps(hosted, allow_nan=False)
