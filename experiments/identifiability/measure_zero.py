import random

from pec.equivalence import (
    explain_normalisation_argument, matched_pair, random_rdp, readings,
    restricted_policy,
)

from experiments.common import Gate


ACTIONS = ["a", "b"]
OBS = ["x", "y"]
MAX_LEN = 3


gate = Gate("MEASURE ZERO - three readings of the characterization", width=74)
check = gate.check
print("\nReadings of \"the disagreement set has measure zero under P^pi\":")
print("  (A)  P^pi_M(D) = 0")
print("  (B)  P^pi_M(D) = 0  and  P^pi_M'(D) = 0")
print("  (C)  every h in D has pi.histProb(h) = 0     [ = ObsEquivOn, the definition ]")

print("\n[1] Random model pairs - do the readings ever disagree?")
rng = random.Random(11)
pol = restricted_policy(["a"], ACTIONS)
mismatches = []
n_random = 3000
c_true = 0
for _ in range(n_random):
    ns = rng.choice([1, 2, 3])
    m1 = random_rdp("m1", ns, ACTIONS, OBS, rng, denom=2)
    m2 = random_rdp("m2", ns, ACTIONS, OBS, rng, denom=2)
    r = readings(m1, m2, pol, ACTIONS, OBS, MAX_LEN)
    c_true += r["C"]
    if r["A"] != r["C"] or r["B"] != r["C"]:
        mismatches.append((m1, m2, r))
check(
    f"(A) == (B) == (C) on {n_random} random pairs",
    not mismatches,
    f"{c_true} of {n_random} were pi-equivalent; {len(mismatches)} mismatches",
)

print("\n[2] Constructed pi-equivalent pairs - the readings must all hold")
rng = random.Random(23)
bad = []
n_matched = 1500
for _ in range(n_matched):
    ns = rng.choice([1, 2, 3])
    m1, m2 = matched_pair("m1", "m2", ns, ACTIONS, OBS, allowed=["a"], rng=rng, denom=4)
    r = readings(m1, m2, pol, ACTIONS, OBS, MAX_LEN)
    if not (r["A"] and r["B"] and r["C"]):
        bad.append(r)
check(
    f"all three readings hold on {n_matched} constructed pi-equivalent pairs",
    not bad,
    "agree on the policy's actions, re-randomised elsewhere" if not bad
    else f"{len(bad)} failures",
)

rng = random.Random(23)
nontrivial = 0
for _ in range(300):
    ns = rng.choice([2, 3])
    m1, m2 = matched_pair("m1", "m2", ns, ACTIONS, OBS, allowed=["a"], rng=rng, denom=4)
    open_pol = restricted_policy(["a", "b"], ACTIONS)
    if not readings(m1, m2, open_pol, ACTIONS, OBS, MAX_LEN)["C"]:
        nontrivial += 1
check(
    "those pairs are genuinely distinguishable by a policy that uses both actions",
    nontrivial > 0,
    f"{nontrivial}/300 separated once action 'b' is available - the pairs are not "
    f"secretly identical",
)

print("\n[3] Why reading (A) is not weaker - the normalisation argument")
mz, mh, witness, compensating = explain_normalisation_argument(ACTIONS, OBS)
w_lik = (mz.hist_lik(witness), mh.hist_lik(witness))
c_lik = (mz.hist_lik(compensating), mh.hist_lik(compensating))
check(
    "the intended witness is in D but has zero mass under m_zero",
    w_lik[0] == 0 and w_lik[1] != 0,
    f"{witness}: m_zero={w_lik[0]}, m_half={w_lik[1]}",
)
check(
    "but the compensating history is also in D, with positive mass under m_zero",
    c_lik[0] != c_lik[1] and c_lik[0] > 0,
    f"{compensating}: m_zero={c_lik[0]}, m_half={c_lik[1]} - the withheld mass "
    f"reappears, so (A) fails exactly when (C) does",
)
r = readings(mz, mh, pol, ACTIONS, OBS, 2)
check(
    "so all three readings agree on this pair too",
    r["A"] == r["B"] == r["C"],
    f"A={r['A']}, B={r['B']}, C={r['C']} - all three reject, in step",
)

print("\n[4] Equivalence really is policy-relative")
rng = random.Random(5)
flips = 0
for _ in range(400):
    m1, m2 = matched_pair("m1", "m2", 2, ACTIONS, OBS, allowed=["a"], rng=rng, denom=4)
    closed = readings(m1, m2, restricted_policy(["a"], ACTIONS), ACTIONS, OBS, MAX_LEN)["C"]
    opened = readings(m1, m2, restricted_policy(["a", "b"], ACTIONS), ACTIONS, OBS, MAX_LEN)["C"]
    if closed and not opened:
        flips += 1
check(
    "the same pair is equivalent under one policy and separated under another",
    flips > 0,
    f"{flips}/400 pairs flip when the policy is allowed to take action 'b' - "
    f"this is the designed deviation in miniature",
)

gate.finish()
