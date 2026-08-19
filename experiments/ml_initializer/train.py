"""Train and gate the topology-conditioned semantic initializer."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from experiments.ml_initializer.data import TOKEN_VOCAB, load_best_size3_records
from experiments.ml_initializer.model import (
    ModelArrays,
    initialize_parameters,
    multiple_choice_huber_loss,
    predict,
    prepare_model_arrays,
)


def train_one(
    arrays: ModelArrays,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    shuffle_topologies: bool,
):
    rng = np.random.default_rng(seed)
    parameters = initialize_parameters(
        jax.random.PRNGKey(seed), len(arrays.semantic_keys), heads=4
    )
    optimizer = optax.adam(learning_rate)
    state = optimizer.init(parameters)

    train_tokens = arrays.tokens.copy()
    if shuffle_topologies:
        shuffled = rng.permutation(arrays.train_indices)
        train_tokens[arrays.train_indices] = arrays.tokens[shuffled]

    @jax.jit
    def step(parameters, state, tokens, key_ids, base, targets, weights):
        loss, grads = jax.value_and_grad(multiple_choice_huber_loss)(
            parameters, tokens, key_ids, base, targets, weights
        )
        updates, state = optimizer.update(grads, state, parameters)
        return optax.apply_updates(parameters, updates), state, loss

    last_loss = float("nan")
    for _ in range(epochs):
        order = rng.permutation(arrays.train_indices)
        remainder = (-len(order)) % batch_size
        if remainder:
            order = np.concatenate([order, order[:remainder]])
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            parameters, state, loss = step(
                parameters,
                state,
                jnp.asarray(train_tokens[indices]),
                jnp.asarray(arrays.key_ids[indices]),
                jnp.asarray(arrays.base_values[indices]),
                jnp.asarray(arrays.targets[indices]),
                jnp.asarray(arrays.weights[indices]),
            )
            last_loss = float(loss)
    return parameters, last_loss


def evaluate(parameters, arrays: ModelArrays, batch_size: int = 128):
    all_predictions = []
    for start in range(0, len(arrays.test_indices), batch_size):
        indices = arrays.test_indices[start : start + batch_size]
        prediction = predict(
            parameters,
            jnp.asarray(arrays.tokens[indices]),
            jnp.asarray(arrays.key_ids[indices]),
            jnp.asarray(arrays.base_values[indices]),
        )
        all_predictions.append(np.asarray(prediction))
    predictions = np.concatenate(all_predictions, axis=0)
    targets = arrays.targets[arrays.test_indices]
    weights = arrays.weights[arrays.test_indices]

    absolute_error = np.abs(predictions - targets[..., None])
    head_mae = np.sum(absolute_error * weights[..., None], axis=1)
    winning_head = np.argmin(head_mae, axis=1)
    oracle = np.min(head_mae, axis=1)
    median_baseline = np.sum(
        np.abs(arrays.base_values[arrays.test_indices] - targets) * weights,
        axis=1,
    )
    utilization = np.bincount(winning_head, minlength=predictions.shape[-1]) / len(
        winning_head
    )
    return {
        "oracle_head_mae": summarize(oracle),
        "head_0_mae": summarize(head_mae[:, 0]),
        "semantic_median_mae": summarize(median_baseline),
        "head_utilization": utilization.tolist(),
    }


def summarize(values) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9)),
    }


def save_parameters(path: Path, parameters, arrays: ModelArrays) -> None:
    payload = {name: np.asarray(value) for name, value in parameters.items()}
    payload.update(
        {
            "semantic_keys": np.asarray(arrays.semantic_keys),
            "key_medians": arrays.key_medians,
            "property_names": np.asarray(arrays.property_names),
            "property_medians": arrays.property_medians,
            "token_vocab": np.asarray(TOKEN_VOCAB),
            "format_version": np.asarray(1, dtype=np.int32),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/generated/ml-initializer-training.json"),
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("artifacts/generated/ml-initializer-weights.npz"),
    )
    args = parser.parse_args()

    records = load_best_size3_records(args.dataset)
    arrays = prepare_model_arrays(records)
    parameters, train_loss = train_one(
        arrays,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        shuffle_topologies=False,
    )
    shuffled_parameters, shuffled_train_loss = train_one(
        arrays,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed + 1,
        shuffle_topologies=True,
    )

    metrics = evaluate(parameters, arrays)
    shuffled_metrics = evaluate(shuffled_parameters, arrays)
    model_mean = metrics["oracle_head_mae"]["mean"]
    median_mean = metrics["semantic_median_mae"]["mean"]
    shuffled_mean = shuffled_metrics["oracle_head_mae"]["mean"]
    utilization = metrics["head_utilization"]
    result = {
        "created_utc": datetime.now(UTC).isoformat(),
        "architecture": "topology-token encoder + semantic-key decoder + 4 heads",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "records": len(records),
        "train_topologies": len(arrays.train_indices),
        "test_topologies": len(arrays.test_indices),
        "train_loss": train_loss,
        "shuffled_train_loss": shuffled_train_loss,
        "metrics": metrics,
        "shuffled_control_metrics": shuffled_metrics,
        "gates": {
            "model_vs_median_improvement_fraction": float(
                (median_mean - model_mean) / median_mean
            ),
            "model_vs_shuffled_improvement_fraction": float(
                (shuffled_mean - model_mean) / shuffled_mean
            ),
            "all_heads_used_at_least_5_percent": bool(min(utilization) >= 0.05),
            "licenses_live_testing": bool(
                model_mean <= 0.85 * median_mean
                and model_mean <= 0.90 * shuffled_mean
                and min(utilization) >= 0.05
            ),
        },
        "limitations": [
            "Reconstruction error is not a live physics score.",
            "The four-head oracle assumes live Objective evaluation selects starts.",
            "Only one deterministic topology-group split is used in this first screen.",
        ],
    }
    save_parameters(args.weights, parameters, arrays)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
