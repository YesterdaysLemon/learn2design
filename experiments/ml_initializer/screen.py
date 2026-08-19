"""Screen simple learned-prior controls on held-out topology identities."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from experiments.ml_initializer.data import (
    DesignRecord,
    load_best_size3_records,
    split_is_test,
)


def fit_medians(records: list[DesignRecord]) -> tuple[dict[str, float], dict[str, float]]:
    by_key: dict[str, list[float]] = defaultdict(list)
    by_property: dict[str, list[float]] = defaultdict(list)
    for record in records:
        for key, property_name, value in zip(
            record.semantic_keys, record.properties, record.unit_params
        ):
            by_key[key].append(float(value))
            by_property[property_name].append(float(value))
    return (
        {key: float(np.median(values)) for key, values in by_key.items()},
        {key: float(np.median(values)) for key, values in by_property.items()},
    )


def fit_quantiles(
    records: list[DesignRecord], levels: np.ndarray
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    by_key: dict[str, list[float]] = defaultdict(list)
    by_property: dict[str, list[float]] = defaultdict(list)
    for record in records:
        for key, property_name, value in zip(
            record.semantic_keys, record.properties, record.unit_params
        ):
            by_key[key].append(float(value))
            by_property[property_name].append(float(value))
    return (
        {key: np.quantile(values, levels).astype(np.float32) for key, values in by_key.items()},
        {
            key: np.quantile(values, levels).astype(np.float32)
            for key, values in by_property.items()
        },
    )


def median_prediction(
    record: DesignRecord,
    key_medians: dict[str, float],
    property_medians: dict[str, float],
) -> np.ndarray:
    return np.asarray(
        [
            key_medians.get(key, property_medians[property_name])
            for key, property_name in zip(record.semantic_keys, record.properties)
        ],
        dtype=np.float32,
    )


def transfer_prediction(
    target: DesignRecord,
    source: DesignRecord,
    key_medians: dict[str, float],
    property_medians: dict[str, float],
) -> np.ndarray:
    source_values = dict(zip(source.semantic_keys, source.unit_params))
    return np.asarray(
        [
            source_values.get(
                key,
                key_medians.get(key, property_medians[property_name]),
            )
            for key, property_name in zip(target.semantic_keys, target.properties)
        ],
        dtype=np.float32,
    )


def property_balanced_mae(
    prediction: np.ndarray,
    target: np.ndarray,
    properties: tuple[str, ...],
) -> float:
    absolute_error = np.abs(prediction - target)
    property_scores = []
    properties_array = np.asarray(properties)
    for property_name in sorted(set(properties)):
        property_scores.append(float(np.mean(absolute_error[properties_array == property_name])))
    return float(np.mean(property_scores))


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
    }


def evaluate_screen(
    records: list[DesignRecord],
    neighbors: int = 4,
    random_seed: int = 20260819,
) -> dict[str, object]:
    train = [record for record in records if not split_is_test(record.topology)]
    test = [record for record in records if split_is_test(record.topology)]
    key_medians, property_medians = fit_medians(train)
    quantile_levels = np.linspace(0.1, 0.9, neighbors)
    key_quantiles, property_quantiles = fit_quantiles(train, quantile_levels)

    train_tokens = np.stack([record.topology_tokens for record in train])
    rng = np.random.default_rng(random_seed)
    shuffled_sources = rng.permutation(len(train))

    metric_values: dict[str, list[float]] = defaultdict(list)
    nearest_distances = []
    for target in test:
        distances = np.count_nonzero(train_tokens != target.topology_tokens[None, :], axis=1)
        nearest = np.argsort(distances, kind="stable")[:neighbors]
        nearest_distances.append(float(distances[nearest[0]]))

        median = median_prediction(target, key_medians, property_medians)
        metric_values["semantic_median"].append(
            property_balanced_mae(median, target.unit_params, target.properties)
        )

        quantile_scores = []
        for head in range(neighbors):
            prediction = np.asarray(
                [
                    key_quantiles.get(key, property_quantiles[property_name])[head]
                    for key, property_name in zip(
                        target.semantic_keys, target.properties
                    )
                ]
            )
            quantile_scores.append(
                property_balanced_mae(
                    prediction, target.unit_params, target.properties
                )
            )
        metric_values[f"semantic_quantile_oracle_{neighbors}"].append(
            min(quantile_scores)
        )

        nearest_scores = [
            property_balanced_mae(
                transfer_prediction(target, train[index], key_medians, property_medians),
                target.unit_params,
                target.properties,
            )
            for index in nearest
        ]
        metric_values["nearest_topology_1"].append(nearest_scores[0])
        metric_values[f"nearest_topology_oracle_{neighbors}"].append(min(nearest_scores))

        shuffled_scores = [
            property_balanced_mae(
                transfer_prediction(
                    target,
                    train[shuffled_sources[index]],
                    key_medians,
                    property_medians,
                ),
                target.unit_params,
                target.properties,
            )
            for index in nearest
        ]
        metric_values[f"shuffled_topology_oracle_{neighbors}"].append(
            min(shuffled_scores)
        )

        random_scores = [
            property_balanced_mae(
                rng.uniform(size=len(target.unit_params)),
                target.unit_params,
                target.properties,
            )
            for _ in range(neighbors)
        ]
        metric_values["random_1"].append(random_scores[0])
        metric_values[f"random_oracle_{neighbors}"].append(min(random_scores))

    metrics = {name: summarize(values) for name, values in metric_values.items()}
    median_mean = metrics["semantic_median"]["mean"]
    nearest_mean = metrics[f"nearest_topology_oracle_{neighbors}"]["mean"]
    shuffled_mean = metrics[f"shuffled_topology_oracle_{neighbors}"]["mean"]
    quantile_mean = metrics[f"semantic_quantile_oracle_{neighbors}"]["mean"]
    return {
        "created_utc": datetime.now(UTC).isoformat(),
        "records": len(records),
        "train_topologies": len(train),
        "test_topologies": len(test),
        "neighbors": neighbors,
        "metric": "topology-macro property-balanced unit-space MAE",
        "metrics": metrics,
        "nearest_hamming_distance": summarize(nearest_distances),
        "gates": {
            "nearest_vs_median_improvement_fraction": float(
                (median_mean - nearest_mean) / median_mean
            ),
            "nearest_vs_shuffled_improvement_fraction": float(
                (shuffled_mean - nearest_mean) / shuffled_mean
            ),
            "quantile_vs_median_improvement_fraction": float(
                (median_mean - quantile_mean) / median_mean
            ),
            "licenses_live_testing": bool(
                nearest_mean <= 0.85 * median_mean
                and nearest_mean <= 0.90 * shuffled_mean
            ),
        },
        "limitations": [
            "Stored-parameter reconstruction is not a live physics score.",
            "The oracle-k metric assumes the live Objective selects among k starts.",
            "Periodic tuning and angle distances are not yet wrapped.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--neighbors", type=int, default=4)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/generated/ml-initializer-screen.json"),
    )
    args = parser.parse_args()

    records = load_best_size3_records(args.dataset)
    result = evaluate_screen(records, neighbors=args.neighbors)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
