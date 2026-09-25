from fractions import Fraction

from pec.envs.tomita import (
    ALPHABET, TOMITA, TOMITA_ACCEPTING, TOMITA_PREDICATES, TOMITA_TOO_BIG,
    distinct_emissions, generate, recovers_truth, verify_dfa_matches,
)
from pec.structure import StructurePosterior, enumerate_structures


print("=" * 72)
print("TOMITA - posterior recovery of known automata")
print("=" * 72)
print("\ninference: exact (rational marginal likelihoods, emissions integrated out)")
print("data     : sampled, seeded\n")

all_ok = True
for idx in sorted(TOMITA):
    truth = TOMITA[idx]
    emis = distinct_emissions(truth.n_states)
    data = generate(truth, emis, n_seqs=60, length=40, seed=idx)
    ok, mass, n_hyp = recovers_truth(truth, data)
    all_ok &= ok
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] Tomita {idx}  ({truth.n_states} states, "
          f"{n_hyp} hypotheses enumerated)")
    print(f"         MAP posterior mass = {float(mass):.6f}")

for idx, why in sorted(TOMITA_TOO_BIG.items()):
    print(f"  [SKIP] Tomita {idx}  - {why}, outside exact enumeration")

print("\n  5-state grammars (built via Myhill-Nerode, verified against the predicate):")
for idx in (3, 7):
    d, acc, pred = TOMITA[idx], TOMITA_ACCEPTING[idx], TOMITA_PREDICATES[idx]
    ok = verify_dfa_matches(d, acc, pred)
    print(f"    [{'PASS' if ok else 'FAIL'}] Tomita {idx} DFA has {d.n_states} states "
          f"and accepts exactly the language on all strings up to length 12")
    all_ok &= ok

print("\n  non-degeneracy checks:")
structs = list(enumerate_structures(2, ALPHABET))
flat = StructurePosterior(structs).posterior([[]])
uniform_ok = set(flat) == {Fraction(1, len(structs))}
print(f"    [{'PASS' if uniform_ok else 'FAIL'}] with no data the posterior is "
      f"exactly the uniform prior ({flat[0]})")

truth1 = TOMITA[1]
small = generate(truth1, distinct_emissions(2), n_seqs=2, length=5, seed=1)
_, small_mass, _ = recovers_truth(truth1, small)
strict_ok = 0 < small_mass < 1
print(f"    [{'PASS' if strict_ok else 'FAIL'}] with little data the MAP mass is "
      f"strictly between 0 and 1 ({small_mass})")

big = generate(truth1, distinct_emissions(2), n_seqs=60, length=40, seed=1)
_, big_mass, _ = recovers_truth(truth1, big)
never_one = big_mass < 1
print(f"    [{'PASS' if never_one else 'FAIL'}] Dirichlet smoothing keeps every "
      f"rival alive (1 - mass = {float(1 - big_mass):.3g}, never exactly 0)")
all_ok &= uniform_ok and strict_ok and never_one

print("\n" + "=" * 72)
if all_ok:
    print("PASSED - every enumerable Tomita automaton recovered exactly")
else:
    print("FAILED")
print("=" * 72)
raise SystemExit(0 if all_ok else 1)
