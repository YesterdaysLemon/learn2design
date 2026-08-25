# H100 coverage-triage Stage A: validated result

Date: 2026-08-25

Status: complete and independently validated. The frozen Stage-A gate failed
and its action is `retain_random_start_candidate`. This report contains only
aggregate evidence. The source archive, histories, logs, candidate arrays,
provider receipts, and generated replay files remain private and outside Git.

## 1. Integrity and provenance

The single terminal `coverage-triage-screen-v1` attempt used clean revision
`21132483701fc5a92f91c11b3f0547cd2ec95748`, plan ID
`628fc149c117b12d`, and candidate ZIP SHA-256
`4b7384dd6d401918b9b46ace4e65c3f116c34feeeca825ae25745dfb9e7908bd`.
It compared the submitted `no_prior` random-start control with the opt-in
`coverage_balanced` midpoint Latin-hypercube suffix. The external evidence
identities are:

| Object | SHA-256 |
| --- | --- |
| Complete ZIP | `c26ec9a06d2c485895b083c37b7c955c3453e690a23f1fbc91081517908061d3` |
| ZIP checksum sidecar | `d901d4ba22365911d05212d99348ff6ffc786ff9736ab9547d61a5a83785f739` |
| Package manifest | `28f2feab79f5e1fcde845f0516c188152f808e2a7825a56702d7756fbd30ec63` |
| Frozen external plan | `dc1f9d3447722508daf79033a8c9d6ff66b4de763cd75d13e18a09722ae16ace` |
| Terminal-attempt receipt | `f9066398821460d7f7c0c9ef3b8b6396a12911879814fa357eb85d92dff6b442` |
| Detached summary release | `633878968dc27f9ff1e5ceda3021349bbe3c05e8221a07b4afba44a0ec2c529b` |
| Post-cleanup billing receipt | `3d732164d587c14dddb97aeb349fabfc3b879b4dc2c936ab06ea0e4daa65d9b1` |
| Six-file source lock | `39c9d69331f807f7a962387dac1ed2ba396ba0311ab65a94a17835afc5351d90` |

The source-lock digest was recorded separately before analysis. The exact
CUDA-13/JAX `0.9.0.1` cold smoke passed on one secure
`NVIDIA H100 80GB HBM3` with MIG disabled and no competing GPU process. The
terminal result-bearing attempt then completed 32/32 serial runs, 16/16 paired
seed comparisons, and 8/8 topology inference blocks, with zero worker errors or
interruptions. Every run was physically feasible and had a finite feasible
score.

Packaging revalidated the study and produced the exact 169-member archive. It
contained `summary.commitment.json`, not plaintext outcomes. The validator
authenticated the source lock and external hashes, checked every ZIP path,
type, size, digest, CRC, config, log, record, and pickle-free history, and
recomputed the full result while the detached summary remained unopened. The
production replay and deliberately independent no-project-import,
history-first replay matched across 32 runs, 16 seed pairs, 8 topology values,
and all 14 frozen criteria. Only then was the detached summary opened; all
three representations matched.

## 2. Frozen decision

Differences are `coverage_balanced - no_prior`, so negative values favor the
coverage-balanced treatment.

| Frozen quantity | Recomputed value |
| --- | ---: |
| Complete runs / pairs / topologies | 32 / 16 / 8 |
| Coverage wins / ties / losses | 5 / 0 / 3 |
| Required topology wins | at least 7 / 8 |
| Topology macro mean difference | `-0.17491992648617732` |
| Topology macro median difference | `-0.2051665182256992` |
| Topology p90 regret | `0.1878354888870064` |
| Maximum harmful topology difference | `0.4005582212340548` |
| 95% topology-bootstrap mean interval | `[-0.40161518941401997, 0.05658467766835468]` |
| Seed-37 / seed-41 mean differences | `-0.0860212493788427` / `-0.26381860359351195` |
| Coverage-first / random-first mean differences | `-0.31264667890786024` / `-0.037193174064494405` |
| Minimum topology evaluation ratio | `0.9979577944179714` |
| Overall median evaluation ratio | `1.0007518796992483` |
| Bootstrap seed / resamples | `20260824` / `10000` |
| Frozen status | `failed` |
| Frozen action | `retain_random_start_candidate` |

Thirteen of fourteen criteria passed. The only failure was directional
consistency: five topology wins did not meet the precommitted seven-win bar.
The favorable mean and median cannot override that gate, and the descriptive
bootstrap interval includes zero. Three topology means favored the random
control, although the worst observed harm remained inside the separate `0.5`
guard.

This result rejects escalation of this treatment under the frozen policy. It
does not establish that midpoint Latin-hypercube initialization is globally
ineffective, equivalent to random initialization, or harmful on all topology
families.

## 3. Runtime, cost, and cleanup

The loss-blind cold smoke ran from `2026-08-24T21:43:53Z` to
`2026-08-24T21:48:46Z`. The terminal main attempt ran from
`2026-08-24T21:51:34Z` to `2026-08-25T04:33:50Z`, ending before the frozen
provider evacuation guard.

The source-locked operator billing receipt authenticates equivalent provider
hours, GPU charge, all-in charge, and the cleanup assertion. A separate
post-cleanup operator capture of Runpod's resource-specific billing and account
balance supplied the detailed disk/volume breakdown below; those details
reconcile to better than one ten-millionth of a dollar but are not a provider
signature and are not separately included in the six-file source lock:

| Resource quantity | Observed value |
| --- | ---: |
| Equivalent GPU hours at `$3.29/hour` | `7.11735002356822` |
| GPU charge | `$23.416081577539444` |
| Pod disk charge | `$0.01944444654509425` |
| Standard network-volume charge | `$0.007777777500450611` |
| All-in provider charge | `$23.44330380158499` |
| Frozen all-in cap | `$30.00` |

After the local ZIP hash matched its sidecar and package manifest, the pod was
deleted, then the attached 10 GB standard network volume was deleted. The
post-cleanup operator audit queried the four in-scope resource classes and
returned zero pods, network volumes, endpoints, and templates. This is a scoped
API inventory assertion, not a provider-signed global account attestation.

## 4. Scope and limitations

- There are eight independent topology units and two repeated optimizer seeds.
- Each arm received 600 Objective seconds, far below the four-hour competition
  budget.
- The triage panel is archive- and prior-panel-disjoint but is not a complete
  complexity-factorial panel.
- Optimizer seed is confounded with forward/reverse sweep phase. Balanced arm
  order and the negative seed/order strata guard against a simple sign reversal
  but do not identify causal session drift.
- This was a treatment-selection screen, not hidden-leaderboard evidence. It
  makes no rank, speedup, non-inferiority, equivalence, or official-budget
  claim.
- Outcome-selected subgroup or topology inspection is exploratory and cannot
  rescue or retune this treatment on the observed panel.

## 5. Frozen next action

Keep the submitted patience-600/no-prior random-start candidate unchanged. Do
not run the proposed `coverage-confirmation-screen-v1`, the older
`coverage-robustness-screen-v1`, a seed top-up, a replacement panel, or a rerun
of Stage A. The untouched robustness panel remains unobserved, but this failed
gate does not unlock it.

Further work should return to local, pre-result mechanism research or wait for
official public-leaderboard feedback. Any different initializer or optimizer
change needs its own rationale, implementation audit, untouched evaluation
panel, frozen decision rule, cost envelope, and explicit owner approval before
paid compute. The favorable aggregate direction here is a research lead only;
the terminal policy decision remains to retain random initialization.
