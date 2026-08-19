# RTX 4060 UIFO deployment smoke — 2026-08-19

## Decision

Stop local batched UIFO work. WSL2 exposes the GPU correctly and scalar UIFO evaluation works, but the current batched candidate cannot fit on this 8 GB RTX 4060 even at population 2. Paired initializer evidence therefore requires a larger-memory accelerator; an A100 remains the competition-aligned target.

This is deployment evidence only. It is not an optimizer-performance comparison.

## Environment

- Git revision: `15ab6eef025981b0225b23c32ebd65a0966d148f`
- OS: WSL2 Linux 6.6.87.2, Python 3.11.15
- Device: NVIDIA GeForce RTX 4060, 8,188 MiB, not competition-aligned A100
- JAX/JAXlib: 0.9.0.1 with CUDA 13 wheels
- dfbench: 0.3.3; Differometor: 0.0.5; Optax: 0.2.8
- Allocation policy: `XLA_PYTHON_CLIENT_PREALLOCATE=false`

The CUDA environment followed the official JAX pip-wheel installation path for `jax[cuda13]`. The installed NVIDIA driver exceeded JAX's documented CUDA 13 minimum.

## Frozen problem

- Development-panel topology seed: `2026081908`
- Resolved identity: `HBHCBBCBG-LDSLSLLSLLSL`
- Topology SHA-256: `c0c1fa4342dec8536fc2ef77093518b017fccff67901c97cb597e6c4337e4ee9`
- Size: 3; frequencies: 50; active parameters: 193
- Optimizer seed: 7

## Population-2 candidate

The `no_prior` arm attempted exactly one two-member batch. It failed after 105.548 seconds full process wall time with:

```text
RESOURCE_EXHAUSTED: Out of memory while trying to allocate 5.98GiB.
```

The harness persisted the process error, traceback, device fingerprint, configuration, and log hashes. No candidate history was admitted.

- Local artifact: `artifacts/generated/uifo-gpu-deployment-smokes/population-2/70d09fcf014fed6d/`
- Manifest SHA-256: `0e53e14ba9746809625a0190a6fa94151ef7a68c818ba817cfa08bb8b618b4ac`
- Run-record SHA-256: `f8a1fbd31fb1287cdad972f593e2d7c21ca4595c6622962ede9883825b33ad9f`

## Scalar Adam control

One scalar exact-gradient evaluation completed successfully.

- First admitted evaluation: 41.130 seconds on the Objective clock
- Objective elapsed snapshot: 43.251 seconds
- Full worker wall time: 69.996 seconds
- Physical feasibility: false
- Objective loss: 6.692373689578664; not a competition score because it was infeasible

- Local artifact: `artifacts/generated/uifo-gpu-deployment-smokes/scalar-adam/29bf34b9bc1c3edd/`
- Manifest SHA-256: `82dbd846259e01f0c478387ffb741f1fec5ff0137ea3fcbf0a558b2b99ec4329`
- Run-record SHA-256: `e12d2e2a6deb1b0ce6f88353c15e04c61ae5d1594419004b55f6d86ea3a72483`
- NPZ SHA-256: `4379b0e63bbe5de5e64e527e70505e83de814e82556b52b3f8874c36e6058e04`

## Boundary

Do not infer that population 2 would fail on A100 or that scalar Adam is competitive. The OOM is specific to this device and its concurrent Windows memory use. Do not retry larger populations locally; they cannot answer the paired semantic-prior question.

Reference: [official JAX installation guide](https://docs.jax.dev/en/latest/installation.html).
