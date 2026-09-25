from fractions import Fraction

from pec.envs.tmaze import (
    ACTIONS, OBSERVATIONS, SAW_LEFT, SAW_RIGHT, blind_model, cue_model, mixed_policy,
    wait_policy,
)
from pec.explore import (
    best_by_information_gain, best_by_reachability, expected_information_gain,
    reachability, separating_mass,
)

from experiments.common import Gate


MODELS = [cue_model, blind_model]
PRIOR = [Fraction(1, 2), Fraction(1, 2)]
TARGETS = [SAW_LEFT, SAW_RIGHT]
CANDIDATES = [wait_policy] + [mixed_policy(Fraction(k, 4)) for k in range(1, 5)]


gate = Gate("DEVIATION SELECTION - information gain vs reachability scheduling",
            width=76)
check = gate.check

print("\n[1] Expected information gain under the behaviour policy")
eigs = {n: expected_information_gain(MODELS, PRIOR, wait_policy, n, ACTIONS, OBSERVATIONS) for n in range(1, 6)}
masses = {n: separating_mass(MODELS, PRIOR, wait_policy, n, ACTIONS, OBSERVATIONS) for n in range(1, 6)}
EPS = 1e-12
check(
    "the posterior cannot move under the wait policy, at every horizon",
    all(m == 0 for m in masses.values()),
    "exact separating mass at horizons 1-5: "
    + ", ".join(str(m) for m in masses.values())
    + "  - rational arithmetic, so this is zero, not nearly zero",
)
check(
    "the EIG floats agree, to within accumulated rounding",
    all(abs(v) < EPS for v in eigs.values()),
    "EIG = " + ", ".join(f"{v:.2e}" for v in eigs.values())
    + f"  (tolerance {EPS:g}; entropy needs a log, so this path is not exact - "
    f"which is why the check above is the one that counts)",
)

print("\n[2] Expected information gain under deviations that run the corridor")
dev_eigs = [(p.name, expected_information_gain(MODELS, PRIOR, p, 3, ACTIONS, OBSERVATIONS),
             separating_mass(MODELS, PRIOR, p, 3, ACTIONS, OBSERVATIONS)) for p in CANDIDATES[1:]]
check(
    "every deviation has strictly positive EIG",
    all(e > 0 for _, e, _ in dev_eigs) and all(m > 0 for _, _, m in dev_eigs),
    "  ".join(f"{n}: EIG={e:.4f} (mass {m})" for n, e, m in dev_eigs),
)
check(
    "EIG increases with the probability of running",
    [e for _, e, _ in dev_eigs] == sorted(e for _, e, _ in dev_eigs),
    "more corridor runs, more information - the rule is not merely nonzero, it orders "
    "the deviations sensibly",
)

print("\n[3] The reachability criterion, on the same candidates")
print(f"         {'policy':<16}{'reach sawLeft':>16}{'reach sawRight':>16}")
reach_rows = []
for p in CANDIDATES:
    l = reachability(cue_model, p, SAW_LEFT, 2, ACTIONS, OBSERVATIONS)
    r = reachability(cue_model, p, SAW_RIGHT, 2, ACTIONS, OBSERVATIONS)
    reach_rows.append((p.name, l, r))
    print(f"         {p.name:<16}{str(l):>16}{str(r):>16}")

distinct = {(l, r) for _, l, r in reach_rows}
check(
    "reachability is identical across every candidate policy",
    len(distinct) == 1,
    f"all policies score {distinct.pop()} - the criterion carries no signal here, so a "
    f"reachability-driven scheduler cannot prefer the separating experiment",
)
check(
    "the cue states are already fully reached while merely waiting",
    all(v > 0 for v in reach_rows[0][1:]),
    "state coverage is complete under the behaviour policy - there is nothing left for "
    "a reachability scheduler to want",
)

print("\n[4] Head to head - what does each rule actually select?")
by_eig = best_by_information_gain(MODELS, PRIOR, CANDIDATES, 3, ACTIONS, OBSERVATIONS)
by_reach = best_by_reachability(cue_model, CANDIDATES, TARGETS, 2, ACTIONS, OBSERVATIONS)

eig_pick = by_eig[0][1]
reach_pick = by_reach[0][1]
print(f"         information gain picks : {eig_pick.name}  (EIG = {by_eig[0][0]:.4f})")
print(f"         reachability picks     : {reach_pick.name}  (score = {by_reach[0][0]})")

check(
    "information gain selects a policy that runs the corridor",
    eig_pick is not wait_policy,
    f"it picks {eig_pick.name}, the strongest available deviation",
)
reach_ties = [p.name for s, p in by_reach if s == by_reach[0][0]]
check(
    "reachability cannot distinguish, so waiting is among its optima",
    wait_policy.name in reach_ties,
    f"{len(reach_ties)} candidates tie at the top, including the frozen behaviour "
    f"policy - the scheduler has no reason to deviate",
)

gate.finish()
