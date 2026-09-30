<div align="center">

# Exact Distinguishability in Non-Markovian Decision Processes

*PEC, the π-equivalence certifier*

[![python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![dependencies](https://img.shields.io/badge/dependencies-standard%20library-5A7B55)](pyproject.toml)
[![proofs](https://img.shields.io/badge/proofs-Lean%204%20%2B%20Mathlib-0F4C81)](lean/)
[![licence](https://img.shields.io/badge/licence-MIT-555555)](LICENSE)

</div>

---

Reference implementation, experiments and Lean 4 formalization for *Exact Distinguishability in Non-Markovian Decision Processes* by Kabir Murjani and Nisarg Patel.

Non-Markovian environments are often modeled as Regular Decision Processes (RDPs), where dynamics depend on the interaction
history through a finite automaton. Offline methods learn an RDP from a corpus collected under a fixed behaviour policy
π, and such a corpus may never contain the experiment that separates one candidate model from
another. Two candidates are π-observationally equivalent when they assign equal likelihood to
every history that π can generate. Under π-equivalence the posterior odds between the
candidates equal the prior odds at every sample size, including when π visits every automaton
state.

PEC decides π-equivalence by reachability in the product automaton of the two candidates. It
runs in time O(|Q| |Q′| |A| |O|), independently of the horizon, provided the set of actions
that π can take does not vary with the history. When the candidates are not equivalent, PEC
returns the state pair and the action at which the corpus must be extended, which is the
missing experiment.

## Installation

The library depends only on the Python standard library and requires Python 3.10 or later.
The experiments run from the repository root without installation.

```bash
git clone https://github.com/Kcbir/pec.git
cd pec
pip install -e .                  # optional, to import pec from any directory
pip install -e ".[baselines]"     # optional, adds aalpy for the end-to-end evaluation
```

## Usage

An RDP is specified by a start state, a transition function `step(q, a, o)`, an emission
function `emit(q, a, o)` and finite sets of states, actions and observations. Probabilities
are `fractions.Fraction`, so likelihoods, posterior odds and certificates are exact, and each
emission row is checked to sum to one at construction. A `Policy` gives the probability
`prob(h, a)` of action `a` after history `h`.

```python
from fractions import Fraction

from pec import Posterior, pi_equivalent, sample_trajectories, separating_action_depth, witness_pair
from pec.envs.tmaze import blind_model, cue_model, mixed_policy, wait_policy

pi_equivalent(cue_model, blind_model, wait_policy)            # True

data = sample_trajectories(cue_model, wait_policy, n_trajectories=1000, length=10)
Posterior([cue_model, blind_model]).odds(data)                # Fraction(1, 1)

deviation = mixed_policy(Fraction(1, 2))
witness_pair(cue_model, blind_model, deviation)               # (('sawLeft', None), 'run')
separating_action_depth(cue_model, blind_model, deviation)    # 2
```

In the T-maze the behaviour policy waits and never runs the corridor, so the cue model and the
blind model are π-equivalent and the posterior odds remain exactly one after 1000
trajectories. Under a deviation that runs with probability 1/2, PEC names `run` in the state
pair where the cue model remembers a left cue, at a separating horizon of 2.

## Reproducing the results

```bash
python -m experiments                       # every experiment, in the order of the paper
python -m experiments equivalence design    # selected groups
python -m experiments.design.lookahead      # a single experiment
pytest                                      # unit tests and the experiment checks
```

Each experiment prints its tables and writes them to `results/<group>/`. Eight experiments are
checks, which exit with a nonzero status when a claim of the paper fails. A complete run takes
about ten minutes on a laptop, most of it in `inference.tomita` and `inference.summary`.
The committed tables are reproduced byte for byte, apart from the wall-clock timing columns of
`results/equivalence/scaling*.csv` and `results/baselines/regorl_mu0_{corridor,cost}.csv`.
[experiments/](experiments/) maps each module to the result it reproduces.

## Repository structure

```
pec/                 library
  rdp.py               RDPs, policies, exact posterior and trajectory sampling
  product.py           product automaton and PEC
  equivalence.py       enumeration oracle for π-equivalence at a finite horizon
  explore.py           separating mass, expected information gain and reachability
  sequential.py        greedy and exhaustive design of action sequences
  structure.py         exact structure posterior over probabilistic automata
  rdp_structure.py     structure posterior over RDPs, with emissions integrated out
  envs/                test environments
experiments/         one module per result, grouped by section of the paper
results/             tables written by the experiments
tests/               unit tests and the experiment checks
lean/                Lean 4 formalization against Mathlib
```

## Environments

Each environment consists of two candidate models, a behaviour policy under which they are
π-equivalent, and a deviation that separates them.

| Environment | Module | Experiment the behaviour policy omits |
|---|---|---|
| T-maze | `pec.envs.tmaze` | running the corridor, which reveals the remembered cue |
| Rotating MAB | `pec.envs.suite` | pulling an arm whose reward probability rotates |
| Cheat MAB | `pec.envs.suite` | playing the action sequence that unlocks reward |
| Rotating Maze | `pec.envs.suite` | moving, which rotates the orientation every third move |
| Corridor family | `pec.envs.corridor` | probing at the far end of a corridor of adjustable length |

Rotating MAB and Cheat MAB are taken from Abadi and Brafman (2020), and Rotating Maze is a
reduced version of the grid world of the same name introduced there. In the corridor family
the separating horizon equals the parameter of the construction. `pec.envs.tomita` provides
the Tomita grammars on which recovery of the automaton structure is tested.

## Formalization

The frozen-posterior theorem, the collapse test on the T-maze and the conservation of
posterior weight are proved in Lean 4 against Mathlib. The development contains no `sorry`,
and every theorem depends only on the axioms `propext`, `Classical.choice` and `Quot.sound`.
[lean/](lean/) lists the theorems and the results of the paper they formalize.

```bash
cd lean
lake exe cache get
lake build
```

## Citation

```bibtex
@misc{murjani2026pec,
  title  = {Deciding Observational Equivalence in Non-Markovian Decision Processes},
  author = {Murjani, Kabir and Patel, Nisarg},
  year   = {2026},
  url    = {https://github.com/Kcbir/pec}
}
```

Released under the [MIT licence](LICENSE).
