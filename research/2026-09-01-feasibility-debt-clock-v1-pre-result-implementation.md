# Feasibility-debt restart clock v1 - pre-result implementation

Date: 2026-09-01

Study ID: `feasibility-debt-clock-v1`

Frozen plan revision:
`197a7433c235ef9cf2e160e8a3bd4a8889d33029`

Clean implementation revision:
`13f53dc1ba5ea8246cd385d934e2b5c03b833fe7`

Status: implementation and affected verification complete; terminal projection
not invoked

## Implemented boundary

The candidate is an experiment-owned, self-contained copy of the protected
optimizer loop. `progress_mode="total_loss"` retains the protected progress
transition. `progress_mode="feasibility_debt"` adds exactly three per-member
progress arrays and uses public `aux["penalty"]` before first feasibility,
then public total loss after first feasibility. Restarts reset the treatment
state together with the existing member state.

The implementation adds no objective call, model, training path, topology
input, private attribute, history read, provider call, or submission default.
The protected `submission/` tree was not edited.

The exact unified source delta is fail-closed. The verifier pins the complete
protected and candidate source texts, every diff hunk payload, the exact method
set, and AST equality for every inherited helper other than the intentionally
changed `optimize` method and the new auxiliary validator. Its pre-result
boundary root is:

```text
6a312db9ba68174e1a5db70153534eb7d39b7f34bb24265b000fcc8b17f5df0d
```

## Committed raw-file hashes

| path | SHA-256 |
| --- | --- |
| `experiments/candidates/feasibility_debt_clock_v1.py` | `1259b90039f8bb216bc0daaf8c89187e0516f8de78e4768e25b78e4e654ee6d5` |
| `experiments/candidates/feasibility_debt_clock_v1_fixture.py` | `44864e2d15c67d6f41c9d5a51de5f61899ab25ba62c06885cbbccdcf4502b1b9` |
| `experiments/candidates/feasibility_debt_clock_v1_source.py` | `738da5e8d6f38b48e84bde78b9443e35d0946a35944f90c4e21c5b3bb82673f1` |
| `tests/test_feasibility_debt_clock_v1.py` | `ff07192eac978f2e2a0398a0a5a9abd662c813c19ceb9b1eda65a98eaca87cf2` |
| frozen plan | `f312663798b558dd0592aba8cb795d046529ca8e5f92a52754cae867a4c0e895` |
| protected `submission/submission.py` | `0fefbaaf18d9831895d788df45c92cbaf4522da7c54d8f78646e449ffa9374c9` |

The source verifier uses newline-normalized text hashes internally; the table
above records raw working-tree bytes. Both are pinned and intentional.

## Verification receipt

- Python syntax compilation passed for the candidate, fixture, source
  verifier, and focused tests.
- The static source-boundary verifier passed at the exact implementation
  revision.
- The initial ID-specific focused run passed `10/10` tests.
- The final affected boundary passed `35/35` tests across
  `tests/test_candidate_integration.py` and
  `tests/test_feasibility_debt_clock_v1.py` in 16.18 seconds.
- `git diff --check` was clean.
- The worktree was clean after the implementation commits.

The focused smoke sequences are deliberately distinct from the nine frozen
terminal cases. They authenticate compatibility mode, one small treatment
transition, malformed-aux refusal, CPU child-environment scrubbing, and the
source-delta boundary without invoking the terminal `--run` projection.

## Broad-suite deviation

One full repository pass was started after the focused checks. At 55% it
entered a historical integration test that directly spawned
`experiments.local_lab.multistep_td_action_prefix_v3_worker` in its trace mode.
Because the handoff excludes rerunning terminal study fixtures, the pass was
interrupted rather than allowed to continue. The worker and its parent process
tree were then confirmed absent. No stdout, result payload, private generated
file, controller action, or terminal metric from that historical worker was
inspected or retained.

This is a contained verification deviation and not a second scientific V3
claim. It also means this checkpoint makes no full-suite-pass claim. The one
allowed broad attempt will not be repeated for this study; the 35-test affected
boundary is the retained local verification surface. CI status, if a draft PR
is opened, must be reported separately and must not be described as a new V3
scientific result.

## Terminal guard

The `--run` entry point has not been called. It requires a clean Git worktree,
records the exact invocation revision, authenticates the frozen plan hash,
launches exactly two credential-scrubbed CPU child projections by module name,
requires zero stderr and byte identity, caps child output, and emits only the
sanitized contract.

The private local-laboratory controller remains parked and is not involved.
The two-hour automation remains paused. No GPU, provider, paid endpoint,
official dataset, private outcome panel, generated topology, submission build,
portal action, or score was used.

## Next gate

Push this clean review surface and obtain green candidate-specific review. If
the exact source revision and all guards remain clean, invoke the terminal
projection exactly once on local CPU. A pass can authorize only planning a
fresh generated-panel comparison. It cannot authorize accelerator spend,
candidate packaging, portal upload, or a claim that `0.444293` improved.
