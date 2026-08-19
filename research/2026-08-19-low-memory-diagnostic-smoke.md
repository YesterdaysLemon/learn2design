# Low-memory UIFO diagnostic smoke — 2026-08-19

## Decision

The scalar-chunk path completed one real population-2 control arm on the RTX
4060, so it is useful for bounded deployment diagnostics. The paired attempt is
**not optimizer evidence**: an unrelated Windows CUDA training process was
already using the device, and the semantic-prior arm was stopped when that
contention was discovered.

Do not resume or compare this study. Repeat it from fresh worker processes only
after confirming that the GPU has no other compute process and is near its idle
memory baseline.

## Frozen configuration

- Git revision: `ed79054be7c8e71fad8c31e53764c6b1175d7928`
- Device: NVIDIA GeForce RTX 4060, 8,188 MiB, not competition-aligned A100
- Topology seed: `2026081908`
- Resolved topology: `HBHCBBCBG-LDSLSLLSLLSL`
- Topology SHA-256: `c0c1fa4342dec8536fc2ef77093518b017fccff67901c97cb597e6c4337e4ee9`
- Active parameters: 193; frequencies: 50
- Optimizer seed: 7; population: 2; evaluation chunk: 1
- Budget: two candidate evaluations per arm
- Arms: `no_prior`, then `semantic_prior`

## Completed control arm

The `no_prior` worker completed both scalar calls through the public Objective
API.

- Full worker wall time: 83.929 seconds
- Optimizer wall time: 55.982 seconds
- Objective elapsed snapshot: 54.019 seconds
- Logged candidate evaluations: 2
- Finite physically feasible candidates: 1 of 2
- Recomputed best feasible loss: 12.624703830321241
- Time to first feasible candidate: 47.313 seconds on the Objective clock
- Initial roles: `anchor`, `random`

These timings are contaminated by shared-GPU use and must not be generalized.
The recomputed feasible loss is recorded only to verify accounting, not to make
a performance claim.

## Interrupted treatment arm

The `semantic_prior` worker had not emitted a result after more than 14 minutes
and was using the device concurrently with the unrelated CUDA job. Codex stopped
only the WSL study process group and left the other workload untouched. No
treatment record or history exists, so the harness correctly reports zero
complete pairs and `promotion_inference_ready: false`.

A parameter-dependent numerical tail remains possible, but this run cannot
distinguish it from GPU contention and allocator effects. The semantic prior
often places masses near their upper normalized range and reflectivities near
their lower range; that hypothesis needs an idle-device candidate-isolation
test before it is treated as causal.

## Preserved artifacts

- Local ignored artifact: `artifacts/generated/uifo-gpu-deployment-smokes/contaminated-paired-attempt/e19edd23fea0ef5f/`
- Manifest SHA-256: `097ab7ad4a7a2ef81c4784e89b0bb731eb44001586c63c59105a7ea1ee478515`
- Run ledger SHA-256: `c7b4d6025851ee4d073fe8048d5416a77f85f7ab5a41f78df8a1ae32a8c78850`
- Partial summary SHA-256: `fc4d945245cb772004075b984b655d7e867875bd6c5733e4ee3913dc1579e8e3`
- Control history SHA-256: `8d4786c5bd359cbfcd6794c7153a54b3381b42c0782c4de67447fabd82eb2919`

## Next clean diagnostic

On an idle GPU, evaluate the anchor, preserved random member, and semantic prior
as separate one-candidate worker processes. Reverse or randomize order, use a
short predeclared host timeout, and repeat twice. Record process wall time and
peak device memory. Only a repeated semantic-only timeout against completed
anchor and random controls would support a parameter-dependent pathology claim.

`tools/run_uifo_candidate_probe.py` implements this frozen six-worker protocol.
It requires an explicit topology and predeclared idle memory/utilization limits,
rejects compute processes visible to `nvidia-smi`, samples total-device telemetry
throughout each worker, and refuses resume so a partial run cannot disturb the
forward/reverse order. The global thresholds are still required because WSL may
not expose every host-side WDDM process.

Each worker writes the exact candidate hashes and an `evaluation_started`
milestone before its Objective call. A timeout is admitted only when that
milestone, the parent/worker/telemetry process-ID handshake, logs, telemetry,
and post-worker idle state all validate. Such a timeout is right-censored and
the planned reverse-order workers continue; any other error stops the protocol.

Competition-throughput and optimizer-promotion evidence still require the
default full-population vmap path on larger-memory hardware.
