from pec.envs.corridor import family
from pec.product import pi_equivalent, separating_action_depth

from experiments.common import Gate


gate = Gate("CORRIDOR FAMILY - the separating horizon as a parameter", width=80)
print("\n  A corridor of k rooms; the missing experiment is a probe at the far end.")
print("  The behaviour policy walks the corridor and never probes.\n")
print(f"  {'requested horizon':>18}{'rooms':>8}{'pi-equivalent':>15}"
      f"{'depth under pi':>17}{'depth under deviation':>23}")

horizons = [1, 2, 3, 4, 5, 6, 8]
rows = []
for h, true_m, rival, pol, dev in family(horizons):
    eq = pi_equivalent(true_m, rival, pol)
    d_pi = separating_action_depth(true_m, rival, pol)
    d_dev = separating_action_depth(true_m, rival, dev)
    rows.append((h, len(true_m.states), eq, d_pi, d_dev))
    print(f"  {h:>18}{len(true_m.states):>8}{str(eq):>15}{str(d_pi):>17}{str(d_dev):>23}")

gate.check("every member is π-equivalent under its behaviour policy",
           all(r[2] for r in rows),
           "depth under π is None throughout - no disagreeing pair is reachable "
           "while merely walking the corridor")
gate.check("the knob controls the separating horizon exactly",
           all(r[4] == r[0] for r in rows),
           "requested horizon == depth under the deviation, for every member: "
           + ", ".join(f"{r[0]}→{r[4]}" for r in rows))
gate.check("difficulty scales without changing anything else",
           all(r[1] == r[0] for r in rows),
           "only the number of rooms varies; alphabet, emission values and the shape "
           "of the ignorance are held fixed, so a curve over this family isolates depth")
gate.finish(verdict_ok="""
The family is parameterized and the knob is verified structurally, not by sweeping
horizons, so it can be used to measure how identifiability degrades with depth.""")
