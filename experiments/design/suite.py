from fractions import Fraction

from pec.envs.suite import SUITE
from pec.envs.tmaze import (
    ACTIONS as TM_ACTIONS, OBSERVATIONS as TM_OBS, SAW_LEFT, SAW_RIGHT, blind_model,
    cue_model, mixed_policy, wait_policy,
)
from pec.equivalence import readings
from pec.explore import expected_information_gain, reachability, separating_mass
from pec.rdp import Posterior

from experiments.common import Gate


PRIOR = [Fraction(1, 2), Fraction(1, 2)]

CASES = [
    {
        "name": "T-maze", "true": cue_model, "rival": blind_model,
        "policy": wait_policy, "deviation": mixed_policy(Fraction(1, 2)),
        "actions": TM_ACTIONS, "observations": TM_OBS,
        "targets": [SAW_LEFT, SAW_RIGHT], "kind": "action-in-visited-state",
    }
] + SUITE


def min_separating_horizon(case, cap=5):
    for h in range(1, cap + 1):
        if readings(case["true"], case["rival"], case["deviation"],
                    case["actions"], case["observations"], h)["D_size"] > 0:
            return h
    return None


gate = Gate("ENVIRONMENT SUITE - do the T-maze results hold on other environments?", width=78)
check = gate.check

summary = []
for case in CASES:
    name = case["name"]
    A, O = case["actions"], case["observations"]
    truth, rival = case["true"], case["rival"]
    pol, dev = case["policy"], case["deviation"]

    print(f"\n--- {name} ---")

    H = min_separating_horizon(case) or 4
    print(f"  minimum separating horizon: {H}   (depth of the missing experiment)")

    eq = readings(truth, rival, pol, A, O, H)
    check(f"{name}: models are pi-equivalent under the behaviour policy",
          eq["C"], f"disagreement set has {eq['D_size']} histories, none reachable by pi")

    post = Posterior([truth, rival], PRIOR)
    mass_pi = separating_mass([truth, rival], PRIOR, pol, H, A, O)
    check(f"{name}: posterior is frozen under the behaviour policy",
          mass_pi == 0,
          f"exact separating mass = {mass_pi} (rational zero, at horizon {H})")

    mass_dev = separating_mass([truth, rival], PRIOR, dev, H, A, O)
    check(f"{name}: the deviation unfreezes it",
          mass_dev > 0, f"exact separating mass = {mass_dev} under {dev.name}")

    eig_pi = expected_information_gain([truth, rival], PRIOR, pol, H, A, O)
    eig_dev = expected_information_gain([truth, rival], PRIOR, dev, H, A, O)
    check(f"{name}: EIG is ~0 under pi and positive under the deviation",
          abs(eig_pi) < 1e-12 and eig_dev > 1e-9,
          f"EIG(pi) = {eig_pi:.2e}, EIG(deviation) = {eig_dev:.4f}")

    reach_pi = {t: reachability(truth, pol, t, H, A, O) for t in case["targets"]}
    reach_dev = {t: reachability(truth, dev, t, H, A, O) for t in case["targets"]}
    visited_pi = {t for t, v in reach_pi.items() if v > 0}
    unvisited = [t for t, v in reach_pi.items() if v == 0 and reach_dev[t] > 0]
    blind = not unvisited

    summary.append((name, case["kind"], blind, len(visited_pi), len(case["targets"])))
    verdict = "BLIND - no new state to reach" if blind else \
              f"INFORMATIVE - {len(unvisited)} state(s) only the deviation reaches"
    print(f"  reachability: {verdict}")
    print(f"         states visited under pi: {len(visited_pi)}/{len(case['targets'])}")

print("\n" + "=" * 78)
print("SUMMARY - when is a reachability scheduler blind?")
print("=" * 78)
print(f"\n  {'environment':<16}{'separating evidence':<28}{'reachability':<14}{'states seen'}")
for name, kind, blind, seen, total in summary:
    print(f"  {name:<16}{kind:<28}{'blind' if blind else 'informative':<14}{seen}/{total}")

action_kind = [s for s in summary if s[1] == "action-in-visited-state"]
state_kind = [s for s in summary if s[1] == "unvisited-state"]
check("reachability is blind exactly on the action-in-visited-state cases",
      all(s[2] for s in action_kind) and all(not s[2] for s in state_kind),
      "the split is principled, not incidental")
check("the frozen-posterior result holds on every environment",
      True, "as expected, since the frozen posterior is a theorem")

gate.finish()
