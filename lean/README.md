# Formalization

Lean 4 proofs of the frozen-posterior theorem, the collapse test on the T-maze and the
conservation of posterior weight. The development builds against Mathlib `v4.34.0` with the
toolchain pinned in `lean-toolchain`. It contains no `sorry`, and every theorem listed below
depends only on the axioms `propext`, `Classical.choice` and `Quot.sound`.

```bash
lake exe cache get
lake build
```

Histories are finite lists of action-observation pairs, and likelihoods take values in
`ℝ≥0∞`. The results below are finite-horizon and combinatorial, so none of them requires a
trajectory measure.

## Contents

| File | Paper | Declarations |
|---|---|---|
| `RDP/Defs.lean` | RDP, history likelihood, policy, global and π-observational equivalence | `RDP`, `RDP.histLik`, `Policy`, `pathProb`, `ObsEquiv`, `ObsEquivOn` |
| `RDP/Frozen.lean` | Frozen posterior odds | `posteriorOdds_eq_priorOdds`, `posteriorOdds_frozen`, `posterior_eq_of_obsEquivOn`, `posterior_frozen`, `frozen_of_obsEquiv` |
| `RDP/TMaze.lean` | Collapse test | `tmaze_obsEquivOn`, `not_obsEquiv`, `collapse_test`, `waitPolicy_covers_states`, `separating_history_offPolicy`, `tmaze_posterior_frozen`, `tmaze_frozen_forever` |
| `RDP/Martingale.lean` | Conservation of posterior weight | `RDP.tsum_histLik_append`, `tsum_postWeight_append`, `tsum_evidence_append`, `posterior_le_one`, `posterior_tower` |

The odds theorems carry the hypothesis that the likelihood of the second model is nonzero.
Division in `ℝ≥0∞` is total, so without it a history to which both models assign probability
zero would give odds `0 / 0 = 0` in place of the prior odds.
