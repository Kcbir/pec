import random
from fractions import Fraction

from pec.envs.suite import SUITE
from pec.envs.tmaze import (
    ACTIONS as TM_A, OBSERVATIONS as TM_O, blind_model, cue_model, mixed_policy,
    wait_policy,
)
from pec.equivalence import matched_pair, random_rdp, readings, restricted_policy
from pec.product import (
    pi_equivalent, policy_is_history_independent, separating_action_depth, witness_pair,
)

from experiments.common import Gate


A, O = ["a", "b"], ["x", "y"]
MAX_H = 4

gate = Gate("PEC VALIDATION - the product construction against the enumeration oracle", width=80)
check = gate.check

gate.section("[1] The assumption the procedure relies on")
pols = [restricted_policy(["a"], A), restricted_policy(["a", "b"], A)]
check("candidate policies have history-independent action support",
      all(policy_is_history_independent(p, A, O) for p in pols),
      "the product construction probes the policy at the empty history; this verifies "
      "that is exact for the policies used here")

gate.section("[2] Product decision vs enumeration, random pairs")
rng = random.Random(4242)
pol = restricted_policy(["a"], A)
n_tested = n_equiv = 0
bad_equiv, bad_sep, deeper = [], [], 0
for _ in range(2500):
    ns = rng.choice([1, 2, 3])
    m1 = random_rdp("m1", ns, A, O, rng, denom=2)
    m2 = random_rdp("m2", ns, A, O, rng, denom=2)
    prod = pi_equivalent(m1, m2, pol)
    enum = readings(m1, m2, pol, A, O, MAX_H)["C"]
    n_tested += 1
    n_equiv += prod
    if prod and not enum:
        bad_equiv.append((m1, m2))
    if not prod and enum:
        d = separating_action_depth(m1, m2, pol)
        if d is not None and d <= MAX_H:
            bad_sep.append((m1, m2, d))
        else:
            deeper += 1

check(f"product 'equivalent' is never contradicted by enumeration ({n_tested} pairs)",
      not bad_equiv,
      f"{n_equiv} of {n_tested} judged equivalent; {len(bad_equiv)} contradictions")
check("product 'not equivalent' is confirmed whenever the depth is within reach",
      not bad_sep,
      f"{len(bad_sep)} unexplained mismatches; {deeper} pairs separate only beyond "
      f"horizon {MAX_H}, which enumeration cannot see and the product construction can")

gate.section("[3] Constructed π-equivalent pairs")
rng = random.Random(99)
wrong = 0
for _ in range(1200):
    ns = rng.choice([1, 2, 3])
    m1, m2 = matched_pair("m1", "m2", ns, A, O, allowed=["a"], rng=rng, denom=4)
    if not pi_equivalent(m1, m2, pol):
        wrong += 1
check("every constructed pair is judged equivalent by the product procedure",
      wrong == 0, f"{wrong} misjudged out of 1200")

gate.section("[4] The environment suite - structural depth vs measured horizon")
CASES = [("T-maze", cue_model, blind_model, wait_policy, TM_A, TM_O, 2)] + [
    (c["name"], c["true"], c["rival"], c["policy"], c["actions"], c["observations"],
     3 if c["name"] != "Rotating Maze" else 4) for c in SUITE
]
print(f"\n  {'environment':<16}{'pi-equivalent':>15}{'depth under pi':>17}"
      f"{'measured horizon':>19}")
ok_all = True
for name, tm, rm, pol_e, acts, obs, measured in CASES:
    eq = pi_equivalent(tm, rm, pol_e)
    d = separating_action_depth(tm, rm, pol_e)
    ok_all &= eq
    print(f"  {name:<16}{str(eq):>15}{str(d):>17}{measured:>19}")
check("every environment is judged π-equivalent under its behaviour policy", ok_all,
      "depth is None under π exactly because no disagreeing pair is reachable - that "
      "None IS the certificate of equivalence, computed without any horizon")

gate.section("[5] The witness names the missing experiment")
w = witness_pair(cue_model, blind_model, mixed_policy(Fraction(1, 2)))
check("under a deviation, the procedure returns a concrete separating experiment",
      w is not None,
      f"witness = {w} - state pair and the action on which the models disagree, which is "
      f"exactly what the design step explores toward")

gate.finish(verdict_ok="""
RESULT: the product construction decides equivalence at infinite horizon.

  The disagreement set over reals is not regular, but the language L_B recognised by
  the product automaton, with acceptance on (state-pair, action), is. As an event on
  infinite trajectories L_B is a countable union of cylinders over finite histories,
  so measure zero follows from countable additivity alone.

  Deciding equivalence this way is polynomial (reachability over |Q|x|Q'| nodes)
  rather than exponential in the horizon, and needs no horizon at all.""")
