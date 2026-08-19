"""Export a deterministic semantic-median prior from Differometor-30k."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from experiments.ml_initializer.data import load_best_size3_records
from experiments.ml_initializer.screen import fit_medians


OFFICIAL_DATASET_SHA256 = (
    "149f6aac17aff2e33750b4e1b6cebd3cef1c39d47ae49a3a7ed77315cb7838a7"
)
UPSTREAM_REFERENCE = "d9b1bd7d6f2c4df335bc7725755b02aa5f6f942c"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_prior(dataset_path: Path, output_path: Path) -> dict[str, object]:
    dataset_sha256 = sha256(dataset_path)
    if dataset_sha256 != OFFICIAL_DATASET_SHA256:
        raise ValueError(
            "dataset checksum does not match the pinned Differometor-30k archive"
        )

    records = load_best_size3_records(dataset_path)
    key_medians, property_medians = fit_medians(records)
    key_support = Counter(key for record in records for key in record.semantic_keys)
    payload = {
        "dataset_sha256": dataset_sha256,
        "format_version": 1,
        "key_medians": dict(sorted(key_medians.items())),
        "key_support": dict(sorted(key_support.items())),
        "kind": "semantic_unit_space_median",
        "property_medians": dict(sorted(property_medians.items())),
        "record_policy": "lowest stored loss per exact size-3 topology",
        "records": len(records),
        "upstream_reference": UPSTREAM_REFERENCE,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("submission/semantic_prior.json")
    )
    args = parser.parse_args()
    payload = export_prior(args.dataset, args.output)
    print(
        f"exported {len(payload['key_medians'])} semantic medians "
        f"from {payload['records']} topology identities"
    )


if __name__ == "__main__":
    main()
