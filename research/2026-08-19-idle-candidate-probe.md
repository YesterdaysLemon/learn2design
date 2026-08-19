# Idle-GPU UIFO candidate probe — 2026-08-19

## Decision

The gross semantic-prior latency tail from the contaminated paired attempt did
not reproduce on the idle-gated RTX 4060. All six fresh workers completed, all
candidate hashes matched across forward/reverse order, and every candidate,
process, GPU-telemetry, log, and milestone integrity check passed.

This clears a local deployment-pathology concern. It does **not** establish that
the semantic prior improves optimization, because each worker evaluated only
one candidate, the random and semantic candidates were physically infeasible,
and this scalar diagnostic does not reproduce full-population vmap throughput.
The prior remains a candidate for the predeclared paired A100 screen—not a
promoted result.

## Frozen configuration

- Git revision: `84625851ab0f7725dc54108c18ecf58ee8e31bfa`
- Device: NVIDIA GeForce RTX 4060, 8,188 MiB, not competition-aligned A100
- Topology: `HBHCBBCBG-LDSLSLLSLLSL`
- Topology SHA-256: `c0c1fa4342dec8536fc2ef77093518b017fccff67901c97cb597e6c4337e4ee9`
- Active parameters: 193; frequencies: 50
- Optimizer seed: 7; reproduced population size: 2
- Order: anchor, random member 1, semantic prior; then exact reverse
- Budget: one public `Objective.value_and_grad_aux` call per fresh worker
- Host timeout: 180 seconds per worker
- Allocation policy: `XLA_PYTHON_CLIENT_PREALLOCATE=false`
- Predeclared idle gates: at most 1,950 MiB and 65% total-device utilization,
  five samples at 0.5-second spacing, with no visible compute process
- Initial idle observation: 1,816 MiB peak, 40% peak utilization, zero compute
  processes visible to `nvidia-smi`

The utilization ceiling accommodates transient WDDM/display activity; the
memory ceiling and process table exclude the earlier CUDA training workload.
The harness also requires the same idle gates after every worker.

## Results

| Order | Candidate | Call wall time | Full worker wall time | Peak observed device allocation | Physical feasibility | Raw Objective loss |
|---:|---|---:|---:|---:|:---:|---:|
| Forward 1 | Anchor | 46.268 s | 74.533 s | 7,076 MiB | yes | 12.6247038303 |
| Forward 2 | Random member 1 | 45.596 s | 72.879 s | 7,135 MiB | no | 6.6492499139 |
| Forward 3 | Semantic prior | 42.419 s | 68.790 s | 6,941 MiB | no | 9.5404988723 |
| Reverse 1 | Semantic prior | 41.966 s | 68.463 s | 7,009 MiB | no | 9.5404988723 |
| Reverse 2 | Random member 1 | 41.717 s | 67.585 s | 6,939 MiB | no | 6.6492499139 |
| Reverse 3 | Anchor | 41.698 s | 67.610 s | 6,939 MiB | yes | 12.6247038303 |

Role means were 43.983 seconds for the anchor, 43.656 seconds for the random
member, and 42.193 seconds for the semantic prior. With only two order-balanced
observations, those small differences are descriptive and not a latency claim.
The decisive observation is simply that neither semantic evaluation approached
the 180-second censoring threshold.

The lower raw losses of the infeasible random and semantic candidates are not
competition scores. Only the anchor was physically feasible in both repeats.

## Integrity result

- Expected/admissible/completed workers: 6 / 6 / 6
- Errors and timeouts: 0 / 0
- Candidate construction hashes consistent: yes
- Selected candidate hashes consistent within role: yes
- Missing or unexpected run IDs: none
- `diagnostic_complete`: true
- `performance_inference_ready`: false

Candidate SHA-256 values:

- anchor: `913f33fa21c1c3c26dea106b023c3874fd71b2afe3e42c1572928f73195a1ddb`
- random member 1: `fb7d2d66d53d87f3f11242ea17c6b073d89c0a1967b8979149bca9b25662e730`
- semantic prior: `2292e3cbbb0463a29e403d86e4353139ee6e9c33cb9fa36d4335130ddaf37b54`

## Preserved artifacts

- Local ignored artifact: `artifacts/generated/uifo-gpu-deployment-smokes/idle-candidate-probe/`
- Manifest SHA-256: `6ab38c97e1ed42b8cbc9bd38fae44a1062bda8e2a88d1e6361f211b441726cb1`
- Run ledger SHA-256: `140d84792107cb83c464c56adaeb58d8da3f4de370789b5c78c4261e399ad368`
- Summary SHA-256: `87d4f80892742fc0721819d764aa31670e8aaef415ad9bece695209239add2df`

The preserved directory also contains six hashed candidate milestones, streamed
stdout/stderr logs, raw total-device telemetry ledgers, and pre/post idle reports.

## Next gate

Return to the frozen plan: compare `semantic_prior` with `no_prior` using equal
wall-clock budgets, the default full-population vmap path, audited topologies,
and A100-class memory. Do not infer optimizer quality or a speedup from this
candidate-isolation diagnostic.
