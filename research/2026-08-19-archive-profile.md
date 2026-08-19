# Differometor-30k archive profile — 2026-08-19

This profile was generated from the dataset bundled at official starter-kit revision `d9b1bd7d6f2c4df335bc7725755b02aa5f6f942c`.

- Dataset SHA-256: `149f6aac17aff2e33750b4e1b6cebd3cef1c39d47ae49a3a7ed77315cb7838a7`
- File size: 74,920,439 bytes
- Entries: 29,650
- Size-3 entries: 28,863 (97.35%)
- Size-4 entries: 787
- Unique topology strings: 12,437
- Singleton topologies: 11,797 (94.85% of topology identities)
- Largest repeated-topology group: 2,878 rows
- Rows with an `initialized_from` parent: 2,171
- Parameter-vector lengths: 174–330

Stored-loss quantiles across all rows were `-0.3990` minimum, `0.3779` p10, `0.7095` median, `4.0333` p90, and `4.8188` maximum. These values describe the archive; they are not live benchmark results.

## Consequences for experiments

The archive is highly imbalanced. Most topology identities occur once, while a small number account for many rows. A row-random train/test split would therefore put reoptimizations of the same topology on both sides and overstate generalization. The profiler constructs 12,437 connected split groups by joining rows that share a topology or an `initialized_from` lineage; in this snapshot, lineage links do not reduce the group count below the topology count.

Any initializer evaluation should therefore:

1. split by the connected group identifier, never by row;
2. report topology-macro results as well as row-weighted results;
3. filter size 3 for the primary competition study;
4. compare semantic nearest-topology transfer with property-only and random controls;
5. reevaluate candidates through the current Objective instead of trusting stored losses.

The 94.85% singleton rate also weakens the case for exact-topology lookup on hidden tasks. If archive conditioning helps, it will likely come from transferable component/property structure or topology similarity, not exact identity.

## Regeneration

```bash
uv run --group integration python tools/profile_archive.py /path/to/dataset.h5
```

The generated JSON is written to `artifacts/generated/archive-profile.json` and includes the input checksum, upstream revision, selection metadata, counts, and quantiles.
