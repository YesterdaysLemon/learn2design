"""Small topology-conditioned, semantic-slot initializer model."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from experiments.ml_initializer.data import DesignRecord, TOKEN_VOCAB


@dataclass(frozen=True)
class ModelArrays:
    tokens: np.ndarray
    key_ids: np.ndarray
    base_values: np.ndarray
    targets: np.ndarray
    weights: np.ndarray
    train_indices: np.ndarray
    test_indices: np.ndarray
    semantic_keys: tuple[str, ...]
    key_medians: np.ndarray
    property_names: tuple[str, ...]
    property_medians: np.ndarray


def prepare_model_arrays(records: list[DesignRecord]) -> ModelArrays:
    train_indices = np.asarray(
        [index for index, record in enumerate(records) if not _is_test(record)],
        dtype=np.int32,
    )
    test_indices = np.asarray(
        [index for index, record in enumerate(records) if _is_test(record)],
        dtype=np.int32,
    )
    train_records = [records[index] for index in train_indices]

    all_keys = tuple(sorted({key for record in records for key in record.semantic_keys}))
    key_to_id = {key: index + 1 for index, key in enumerate(all_keys)}
    property_names = tuple(sorted({name for record in records for name in record.properties}))

    key_values: dict[str, list[float]] = defaultdict(list)
    property_values: dict[str, list[float]] = defaultdict(list)
    for record in train_records:
        for key, property_name, value in zip(
            record.semantic_keys, record.properties, record.unit_params
        ):
            key_values[key].append(float(value))
            property_values[property_name].append(float(value))

    property_median_map = {
        name: float(np.median(property_values[name])) for name in property_names
    }
    key_median_map = {
        key: float(np.median(values)) for key, values in key_values.items()
    }

    max_length = max(len(record.unit_params) for record in records)
    tokens = np.stack([record.topology_tokens for record in records]).astype(np.int32)
    key_ids = np.zeros((len(records), max_length), dtype=np.int32)
    base_values = np.full((len(records), max_length), 0.5, dtype=np.float32)
    targets = np.zeros((len(records), max_length), dtype=np.float32)
    weights = np.zeros((len(records), max_length), dtype=np.float32)

    for row, record in enumerate(records):
        length = len(record.unit_params)
        targets[row, :length] = record.unit_params
        key_ids[row, :length] = [key_to_id[key] for key in record.semantic_keys]
        base_values[row, :length] = [
            key_median_map.get(key, property_median_map[property_name])
            for key, property_name in zip(record.semantic_keys, record.properties)
        ]
        property_counts = Counter(record.properties)
        present_properties = len(property_counts)
        weights[row, :length] = [
            1.0 / (present_properties * property_counts[property_name])
            for property_name in record.properties
        ]

    return ModelArrays(
        tokens=tokens,
        key_ids=key_ids,
        base_values=base_values,
        targets=targets,
        weights=weights,
        train_indices=train_indices,
        test_indices=test_indices,
        semantic_keys=all_keys,
        key_medians=np.asarray(
            [key_median_map.get(key, 0.5) for key in all_keys], dtype=np.float32
        ),
        property_names=property_names,
        property_medians=np.asarray(
            [property_median_map[name] for name in property_names], dtype=np.float32
        ),
    )


def _is_test(record: DesignRecord) -> bool:
    from experiments.ml_initializer.data import split_is_test

    return split_is_test(record.topology)


def initialize_parameters(
    key: jax.Array,
    n_semantic_keys: int,
    heads: int = 4,
    token_embedding_dim: int = 8,
    key_embedding_dim: int = 16,
    topology_dim: int = 32,
    hidden_dim: int = 64,
) -> dict[str, jax.Array]:
    keys = jax.random.split(key, 5)

    def normal(shape, rng, scale):
        return jax.random.normal(rng, shape) * scale

    return {
        "token_embedding": normal(
            (len(TOKEN_VOCAB), token_embedding_dim), keys[0], 0.05
        ),
        "semantic_embedding": normal(
            (n_semantic_keys + 1, key_embedding_dim), keys[1], 0.05
        ),
        "topology_weight": normal(
            (21 * token_embedding_dim, topology_dim),
            keys[2],
            np.sqrt(2.0 / (21 * token_embedding_dim)),
        ),
        "topology_bias": jnp.zeros((topology_dim,)),
        "hidden_weight": normal(
            (topology_dim + key_embedding_dim, hidden_dim),
            keys[3],
            np.sqrt(2.0 / (topology_dim + key_embedding_dim)),
        ),
        "hidden_bias": jnp.zeros((hidden_dim,)),
        "output_weight": normal((hidden_dim, heads), keys[4], 0.01),
        "output_bias": jnp.zeros((heads,)),
    }


def predict(
    parameters: dict[str, jax.Array],
    tokens: jax.Array,
    key_ids: jax.Array,
    base_values: jax.Array,
) -> jax.Array:
    token_features = parameters["token_embedding"][tokens]
    token_features = token_features.reshape((tokens.shape[0], -1))
    topology_context = jax.nn.silu(
        token_features @ parameters["topology_weight"]
        + parameters["topology_bias"]
    )

    semantic_context = parameters["semantic_embedding"][key_ids]
    repeated_topology = jnp.broadcast_to(
        topology_context[:, None, :],
        (*semantic_context.shape[:2], topology_context.shape[-1]),
    )
    hidden = jax.nn.silu(
        jnp.concatenate([repeated_topology, semantic_context], axis=-1)
        @ parameters["hidden_weight"]
        + parameters["hidden_bias"]
    )
    residual = hidden @ parameters["output_weight"] + parameters["output_bias"]
    clipped_base = jnp.clip(base_values, 1e-5, 1.0 - 1e-5)
    base_logit = jnp.log(clipped_base) - jnp.log1p(-clipped_base)
    return jax.nn.sigmoid(base_logit[..., None] + residual)


def multiple_choice_huber_loss(
    parameters: dict[str, jax.Array],
    tokens: jax.Array,
    key_ids: jax.Array,
    base_values: jax.Array,
    targets: jax.Array,
    weights: jax.Array,
    delta: float = 0.05,
) -> jax.Array:
    predictions = predict(parameters, tokens, key_ids, base_values)
    absolute_error = jnp.abs(predictions - targets[..., None])
    huber = jnp.where(
        absolute_error <= delta,
        0.5 * jnp.square(absolute_error) / delta,
        absolute_error - 0.5 * delta,
    )
    head_losses = jnp.sum(huber * weights[..., None], axis=1)
    return jnp.mean(jnp.min(head_losses, axis=1))
