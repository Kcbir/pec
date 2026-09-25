from fractions import Fraction

from pec.envs.corridor import ACTIONS as CORRIDOR_A, OBS as CORRIDOR_O, make_pair
from pec.envs.suite import SUITE
from pec.envs.tmaze import (
    ACTIONS as TM_A, OBSERVATIONS as TM_O, blind_model, cue_model, mixed_policy,
)
from pec.product import separating_action_depth
from pec.sequential import (
    best_sequence, first_step_eig, greedy_sequence, mass_of_sequence,
)

from experiments.common import Gate, announce, write_csv


PRIOR = [Fraction(1, 2), Fraction(1, 2)]
paths = []

CASES = [("T-maze", cue_model, blind_model, TM_A, TM_O, mixed_policy(Fraction(1, 2)), 3)] + [
    (c["name"], c["true"], c["rival"], c["actions"], c["observations"],
     c["deviation"], 3) for c in SUITE
]

gate = Gate("SEQUENTIAL DESIGN - myopic vs lookahead", width=84)
check = gate.check

gate.section("[1] First-step expected information gain - what a greedy rule sees")
print(f"\n  {'environment':<16}{'max first-step EIG over actions':>34}")
rows = []
flat = True
for name, tm, rm, A, O, _dev, _L in CASES:
    eigs = first_step_eig([tm, rm], PRIOR, A, O)
    mx = max(eigs.values())
    flat &= abs(mx) < 1e-12
    rows.append([name, f"{mx:.3e}"])
    print(f"  {name:<16}{mx:>34.3e}")
check("first-step EIG is zero in every environment", flat,
      "the greedy landscape is exactly flat - there is no gradient to follow, so the "
      "rule picks by tie-break, not by information")

print(f"\n  {'environment':<16}{'d':>3}  {'greedy (given order)':<24}{'mass':>7}"
      f"  {'greedy (reversed)':<24}{'mass':>7}  {'exhaustive':<24}{'mass':>7}")
csv_rows = []
optimal_always = True
fails_both = []
tiebreak_decided = []
for name, tm, rm, A, O, dev, _ in CASES:
    M = [tm, rm]
    d = separating_action_depth(tm, rm, dev)
    g1, _ = greedy_sequence(M, PRIOR, A, O, d)
    m1 = mass_of_sequence(M, PRIOR, g1, A, O)
    g2, _ = greedy_sequence(M, PRIOR, list(reversed(A)), O, d)
    m2 = mass_of_sequence(M, PRIOR, g2, A, O)
    b, be = best_sequence(M, PRIOR, A, O, d)
    bm = mass_of_sequence(M, PRIOR, b, A, O)
    optimal_always &= bm > 0
    if m1 == 0 and m2 == 0:
        fails_both.append(name)
    elif (m1 == 0) != (m2 == 0):
        tiebreak_decided.append(name)
    csv_rows.append([name, d, ",".join(map(str, g1)), str(m1), float(m1),
                     ",".join(map(str, g2)), str(m2), float(m2),
                     ",".join(map(str, b)), str(bm), float(bm), round(be, 6)])
    print(f"  {name:<16}{d:>3}  {','.join(map(str,g1)):<24}{str(m1):>7}"
          f"  {','.join(map(str,g2)):<24}{str(m2):>7}"
          f"  {','.join(map(str,b)):<24}{str(bm):>7}")

check("exhaustive design finds a separating sequence in every environment",
      optimal_always, "separating mass > 0 on all four, given lookahead d")
check("greedy fails outright on at least one environment, under EVERY tie-break",
      bool(fails_both),
      f"fails under both orders: {', '.join(fails_both)} - Cheat MAB needs two "
      f"DIFFERENT actions in sequence, and neither alone shows any gain")
check("on at least one environment greedy's outcome is decided purely by tie-break",
      bool(tiebreak_decided),
      f"same rule, opposite result depending on action order: "
      f"{', '.join(tiebreak_decided)} - success there is luck, not information")
paths.append(write_csv("design/greedy_vs_exhaustive",
                       ["environment", "lookahead_d", "greedy_seq", "greedy_mass",
                        "greedy_mass_float", "greedy_rev_seq", "greedy_rev_mass",
                        "greedy_rev_mass_float", "optimal_seq", "optimal_mass",
                        "optimal_mass_float", "optimal_eig"], csv_rows))

gate.section("[3] How much lookahead is required? (corridor family)")
print("    Sweeping the family's depth knob and asking, for each lookahead L, whether")
print("    exhaustive design over sequences of length L finds anything at all.")
print(f"\n  {'depth':>7}{'structural depth':>18}   lookahead L -> separating mass")
csv_look = []
exact_match = True
for depth in [1, 2, 3, 4, 5]:
    true_m, rival, pol, dev = make_pair(depth)
    d_struct = separating_action_depth(true_m, rival, dev)
    cells, first_success = [], None
    for L in range(1, depth + 2):
        b, _ = best_sequence([true_m, rival], PRIOR, CORRIDOR_A, CORRIDOR_O, L)
        m = mass_of_sequence([true_m, rival], PRIOR, b, CORRIDOR_A, CORRIDOR_O)
        cells.append(f"L={L}:{float(m):.2f}")
        csv_look.append([depth, d_struct, L, str(m), float(m)])
        if m > 0 and first_success is None:
            first_success = L
    exact_match &= (first_success == depth)
    print(f"  {depth:>7}{str(d_struct):>18}   " + "  ".join(cells)
          + f"   | first success at L={first_success}")
check("required lookahead equals the minimum separating horizon, exactly",
      exact_match,
      "for every member of the family, design with L < d finds nothing and L = d "
      "finds the experiment - the knob controls the planning depth, not just the data")
paths.append(write_csv("design/lookahead_requirement",
                       ["family_depth", "structural_depth", "lookahead",
                        "separating_mass", "separating_mass_float"], csv_look))

announce(paths)
gate.finish()
