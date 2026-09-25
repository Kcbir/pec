# Experiments

Each module reproduces one result of the paper and runs from the repository root as
`python -m experiments.<group>.<module>`. Running `python -m experiments` executes every
module in the order below, and group names passed as arguments restrict the run to those
groups. Tables are written to `results/<group>/`, with rational values written exactly, for
example `245/512`.

Modules marked as checks test a claim of the paper and exit with a nonzero status when it
fails. The test suite runs all eight of them. Every environment is synthetic with known ground
truth, and all inference uses exact rational arithmetic, apart from expected information gain,
which requires a logarithm and is computed in floating point.

## Identifiability

| Module | Result | Output |
|---|---|---|
| `identifiability.tmaze` | Frozen posterior odds and the collapse test on the T-maze | check |
| `identifiability.measure_zero` | Three readings of the measure-zero characterization, compared on random and constructed pairs | check |

## Deciding π-equivalence

| Module | Result | Output |
|---|---|---|
| `equivalence.validation` | PEC against the enumeration oracle on 2500 random and 1200 constructed pairs | check |
| `equivalence.corridor` | The corridor family, whose separating horizon equals its parameter | check |
| `equivalence.scaling` | Running time of PEC on product automata of up to 1,001,000 state pairs | `scaling.csv`, `scaling_actions.csv`, `scaling_random.csv`, `scaling_witness.csv` |
| `equivalence.estimated_tables` | PEC with the tolerant test on emission tables estimated from data | `estimated_tables.csv` |

## Selecting deviations

| Module | Result | Output |
|---|---|---|
| `design.reachability` | Expected information gain against a reachability scheduler on the T-maze | check |
| `design.suite` | The frozen posterior and the same comparison on the four environments | check |
| `design.lookahead` | Greedy against exhaustive design, and the lookahead each environment requires | check, `greedy_vs_exhaustive.csv`, `lookahead_requirement.csv` |

## Baselines

| Module | Result | Output |
|---|---|---|
| `baselines.regorl` | The distinguishability parameter of RegORL, computed from its definition and by PEC | `regorl_mu0.csv`, `regorl_mu0_corridor.csv`, `regorl_mu0_cost.csv` |
| `baselines.end_to_end` | Bayes factors from corpora collected under π, under uniform exploration and under the designed experiment | `end_to_end.csv`, `end_to_end_depth.csv` |
| `baselines.active_learning` | Equivalence-testing strategies from active automata learning, scored as exploration policies | `active_learning.csv` |

`baselines.end_to_end` fits its learner with `aalpy`, and it prints a notice and exits
without writing tables when the package is not installed.

## Structure inference

| Module | Result | Output |
|---|---|---|
| `inference.tomita` | Recovery of the Tomita automata by the exact structure posterior | check |
| `inference.summary` | Bayes factors against a memoryless model, recovery of emission probabilities and separating mass for each environment | `nonmarkov_bayes_factors.csv`, `stochasticity_recovery.csv`, `identifiability.csv`, `tomita_recovery.csv` |
| `inference.structural_prior` | Posterior contraction under a structural prior, with ε-contamination and a misspecified prior | printed |
| `inference.ablations` | Seed variation, Dirichlet concentration, data scaling, deviation strength and ε-contamination | `ablation_*.csv` |
