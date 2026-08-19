from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from experiments.ml_initializer.model import (
    initialize_parameters,
    multiple_choice_huber_loss,
    predict,
)


@pytest.mark.integration
def test_semantic_model_shapes_and_finite_loss() -> None:
    parameters = initialize_parameters(jax.random.PRNGKey(3), n_semantic_keys=12)
    tokens = jnp.zeros((2, 21), dtype=jnp.int32)
    key_ids = jnp.asarray([[1, 2, 0], [3, 4, 5]], dtype=jnp.int32)
    base = jnp.full((2, 3), 0.5)
    targets = jnp.asarray([[0.2, 0.8, 0.0], [0.4, 0.6, 0.3]])
    weights = jnp.asarray([[0.5, 0.5, 0.0], [1 / 3, 1 / 3, 1 / 3]])

    predictions = predict(parameters, tokens, key_ids, base)
    loss = multiple_choice_huber_loss(
        parameters, tokens, key_ids, base, targets, weights
    )

    assert predictions.shape == (2, 3, 4)
    assert bool(jnp.all(jnp.isfinite(predictions)))
    assert float(loss) >= 0.0
