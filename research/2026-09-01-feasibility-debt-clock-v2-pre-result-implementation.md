# Feasibility-debt restart clock v2 - pre-result implementation

Date: 2026-09-01

Study ID: `feasibility-debt-clock-v2`

Frozen plan revision:
`47efe9d7a55c6f308291b2faa12f160933dce8a5`

Implementation revision:
`29a5d265ad61ddbc0580765532b774e0bf2fbcc1`

Status: exact implementation and hostile pre-result audit complete; terminal
projection not invoked

## Implemented boundary

The V2 candidate is a self-contained experiment-owned copy of the protected
optimizer. It does not import V1. `progress_mode="total_loss"` retains the
protected transition exactly. `progress_mode="feasibility_debt"` uses only the
public finite nonnegative penalty before first feasibility, counts the first
finite feasible observation as progress, and then permanently returns that
lane to the protected finite total-loss comparison even if a later observation
is infeasible. A lane restart resets the three treatment values together with
the protected per-lane state.

The candidate adds no objective call, random draw, model, topology input,
history read, private attribute, provider call, or submission default. The
protected `submission/` tree is unchanged.

The four-lane fixture implements all ten frozen case definitions but importing
it never evaluates them. The parent transport uses exactly two children,
strict duplicate-key and canonical-JSON checks, the closed ten-case and
fourteen-transport-key schemas, byte equality, credential scrubbing, forced
CPU, disabled proxy and bytecode output, write/socket/grandchild guards, output
caps, timeouts, and a closed parent failure projection. A valid scientific
case failure exits the child successfully as data; a handled transport failure
still produces the sanitized parent schema.

## Exact source boundary

The verifier pins the complete protected and V2 source texts, all ten unified-
diff hunk payloads, the exact method set, and AST equality for every inherited
helper except `optimize` and the new auxiliary validator. Its pre-result root
is:

```text
55dcf16da8bb6c04aa7e61da1bc5e552bf51fc2944b17c353eaa79e5c27f8af3
```

## Committed raw-file hashes

| Path | SHA-256 |
| --- | --- |
| `experiments/candidates/feasibility_debt_clock_v2.py` | `a476f20af0c0058c9763a637955a89380a0a68cb7d262960e573e51909285734` |
| `experiments/candidates/feasibility_debt_clock_v2_fixture.py` | `7ffd59fc6860a43630a51ab2d868bf35530def785eb922dc4b252d6c3e939332` |
| `experiments/candidates/feasibility_debt_clock_v2_source.py` | `177ce3d7cb41057410ae7f314e81b7f4b21b08bb59803501c62a98f1f49d3278` |
| `tests/test_feasibility_debt_clock_v2.py` | `24ed33eb3124345a948cde9b313dc385efa5eb2ed1d730b673c1ec846084466f` |
| frozen plan | `fac7a4f5a0c6624685f761d95be4f13d8b4f3db1695695fbd808cf8ff8c84df9` |
| protected `submission/submission.py` | `0fefbaaf18d9831895d788df45c92cbaf4522da7c54d8f78646e449ffa9374c9` |

The source verifier uses newline-normalized text hashes internally; the table
records raw working-tree bytes at the implementation boundary.

## Verification receipt

- Python syntax compilation passed for the candidate, fixture, verifier, and
  focused tests.
- The exact source verifier passed with matching method set and inherited ASTs.
- Static case audit found exactly ten outcome assignments in frozen order.
- Static test audit confirmed that the focused suite does not call
  `_case_projection`, `--child`, or `--run`.
- Plan hash, ten case keys, fourteen transport keys, all eight frozen seeds,
  and the closed parent-failure schema matched the plan.
- A fresh-process guard smoke completed on CPU with zero stdout and stderr
  while source reads and JAX arithmetic remained available.
- The final ID-specific focused suite passed `12/12` tests.
- The final affected boundary passed `37/37` tests across
  `tests/test_candidate_integration.py` and
  `tests/test_feasibility_debt_clock_v2.py` in 16.24 seconds.
- `git diff --check` was clean before the implementation commit.

The first focused transport run found that canonical JSON sorts nested mapping
keys while the initial validator required insertion order. Before any result-
bearing execution, the validator was corrected to require the exact key set.
Later hostile review removed an import-time Git subprocess and required all
source/replay hashes on a pass. These were implementation-conformance fixes;
no frozen case, seed, threshold, row, action, or claim changed.

No full repository pass was attempted for V2 because the known broad suite
directly enters a retired terminal fixture's trace worker. The focused affected
boundary is the retained local verification surface; draft-PR CI is reported
separately and cannot be described as a V2 terminal result.

## Result guard

Neither `_case_projection`, the child entry point, nor the terminal `--run`
entry point has been called. No V2 child output, case result, score, generated
topology, private panel, official data, GPU, paid endpoint, provider resource,
portal, or submission artifact was used.

After this record commits, push the exact review surface and require green CI.
If the worktree and revision remain clean, invoke `--run` exactly once on local
CPU with bytecode writing disabled. A pass can authorize only a fresh paired-
panel plan. A failure parks V2 and cannot be retried.
