from fractions import Fraction
from itertools import product

from pec.envs.tmaze import (
    LEFT, OBSERVATIONS, RUN, SEPARATING_HISTORY, WAIT, all_wait, blind_model, cue_model,
    mixed_policy, wait_policy,
)
from pec.rdp import Posterior, path_prob

from experiments.common import Gate


def wait_histories(n):
    return [tuple((WAIT, o) for o in os) for os in product(OBSERVATIONS, repeat=n)]


gate = Gate("T-MAZE - exact-arithmetic verification of the frozen posterior",
            width=70)
check = gate.check

print("\n[1] Models are well-formed probability distributions")
check("cue_model emissions normalised", True, "checked in RDP.__init__")
check("blind_model emissions normalised", True, "checked in RDP.__init__")

print("\n[2] The two models agree on every history the wait-policy generates")
agree_all = True
worst = None
for n in range(0, 9):
    for h in wait_histories(n):
        if cue_model.hist_lik(h) != blind_model.hist_lik(h):
            agree_all = False
            worst = h
            break
    if not agree_all:
        break
check(
    "likelihoods agree on all-wait histories up to length 8",
    agree_all,
    "exhaustive over 2^0+...+2^8 = 511 histories" if agree_all else f"first disagreement: {worst}",
)

print("\n[3] The models are nevertheless globally different processes")
sep_cue = cue_model.hist_lik(SEPARATING_HISTORY)
sep_blind = blind_model.hist_lik(SEPARATING_HISTORY)
check(
    "separating history distinguishes them",
    sep_cue != sep_blind,
    f"cue={sep_cue}, blind={sep_blind}  on {SEPARATING_HISTORY}",
)
check(
    "separating history is off the wait-policy support",
    not all_wait(SEPARATING_HISTORY) and wait_policy.hist_prob(SEPARATING_HISTORY) == 0,
    "it uses the `run` action, which the wait-policy never takes",
)

print("\n[4] Full state coverage under the wait-policy (the collapse test)")
reached = {cue_model.traverse(h) for n in range(1, 5) for h in wait_histories(n)}
check(
    "both cue states are reached while waiting",
    {"sawLeft", "sawRight"} <= reached,
    f"states reached: {sorted(reached)}",
)
positive = all(
    path_prob(wait_policy, cue_model, h) > 0
    for h in wait_histories(3)
)
check("those histories have positive probability", positive)

print("\n[5] The posterior is frozen under the wait policy")
post = Posterior([cue_model, blind_model])
prior_odds = Fraction(1)
flat = True
odds_seen = set()
for n in range(0, 9):
    for h in wait_histories(n):
        o = post.odds([h])
        odds_seen.add(o)
        if o != prior_odds:
            flat = False
check(
    "posterior odds == prior odds at every sample size",
    flat and odds_seen == {prior_odds},
    f"odds observed across all 511 histories: {sorted(odds_seen)} (prior odds = {prior_odds})",
)

episodes = wait_histories(4)
acc_odds = post.odds(episodes)
check(
    "odds unmoved after accumulating 16 four-step episodes",
    acc_odds == prior_odds,
    f"odds after {len(episodes)} episodes = {acc_odds}",
)

print("\n[6] The information-gain deviation unfreezes the posterior")
sep_odds = post.odds([SEPARATING_HISTORY])
check(
    "one `run` step collapses the posterior onto the blind model",
    sep_odds == 0,
    f"odds(cue : blind) = {sep_odds}  (cue model is refuted outright)",
)

confirming = ((WAIT, LEFT), (RUN, LEFT))
conf_odds = post.odds([confirming])
check(
    "the confirming `run` history moves the odds the other way",
    conf_odds is not None and conf_odds > prior_odds,
    f"odds(cue : blind) = {conf_odds} > {prior_odds} on {confirming}",
)

mixed = mixed_policy(Fraction(1, 4))
check(
    "a mixed policy gives the separating history positive probability",
    mixed.hist_prob(SEPARATING_HISTORY) > 0,
    f"P_mixed(separating actions) = {mixed.hist_prob(SEPARATING_HISTORY)}",
)

gate.finish(
    verdict_ok="\nPASSED\n"
               "  The posterior odds equal the prior odds exactly, as rationals.\n"
               "  The separation appears under the `run` deviation.",
    verdict_bad="\nFAILED - see above",
)
