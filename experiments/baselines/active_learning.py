from fractions import Fraction

from pec.envs.corridor import ACTIONS, OBS, PROBE, WAIT, make_pair
from pec.envs.tmaze import (
    ACTIONS as TM_ACTIONS, OBSERVATIONS as TM_OBS, RUN, WAIT as TM_WAIT, blind_model,
    cue_model,
)
from pec.explore import separating_mass
from pec.rdp import Policy

from experiments.common import announce, write_csv


HALF = Fraction(1, 2)
UNIFORM_PRIOR = [HALF, HALF]


def random_walk(actions) -> Policy:
    return Policy(name="random-walk", actions=actions,
                  prob=lambda h, a: Fraction(1, len(actions)))


def state_prefix(access, actions) -> Policy:
    access = tuple(access)
    return Policy(
        name="state-prefix:" + ",".join(map(str, access)), actions=actions,
        prob=lambda h, a: (Fraction(1) if a == access[len(h)] else Fraction(0))
        if len(h) < len(access) else Fraction(1, len(actions)))


def designed(access, final, actions) -> Policy:
    access = tuple(access)
    seq = access + (final,)
    return Policy(name="designed", actions=actions,
                  prob=lambda h, a: (Fraction(1) if a == seq[len(h)] else Fraction(0))
                  if len(h) < len(seq) else Fraction(1, len(actions)))


def mixture_mass(models, policies, horizon, actions, observations) -> Fraction:
    return sum((separating_mass(models, UNIFORM_PRIOR, p, horizon, actions, observations)
                for p in policies), Fraction(0)) / len(policies)


if __name__ == "__main__":
    print("=" * 94)
    print("AALPY-STYLE EQUIVALENCE TESTING AS EXPLORATION POLICIES")
    print("=" * 94)
    print("\n  Scored in one currency: the exact probability of producing a history on")
    print("  which the two candidates disagree. Rationals throughout.\n")

    head = (f"  {'depth':>5} {'random walk':>16} {'state prefix':>16} "
            f"{'designed':>10} {'prefix/walk':>12}")
    print(head)
    print("  " + "-" * (len(head) - 2))

    rows = []
    for k in (2, 4, 6, 8):
        true, rival, _pi, _dev = make_pair(k)
        models, horizon = [true, rival], k
        walk = separating_mass(models, UNIFORM_PRIOR, random_walk(ACTIONS),
                               horizon, ACTIONS, OBS)
        prefix = mixture_mass(models, [state_prefix([WAIT] * j, ACTIONS)
                                       for j in range(k)], horizon, ACTIONS, OBS)
        des = separating_mass(models, UNIFORM_PRIOR,
                              designed([WAIT] * (k - 1), PROBE, ACTIONS),
                              horizon, ACTIONS, OBS)
        ratio = float(prefix / walk) if walk else float("inf")
        print(f"  {k:>5} {str(walk):>16} {str(prefix):>16} {str(des):>10} {ratio:>11.1f}x")
        rows.append([k, str(walk), str(prefix), str(des), round(ratio, 3)])

    print("\n  The random walk decays exponentially in depth, the access-sequence prefix")
    print("  decays polynomially, and the designed experiment is flat at 1. The prefix is")
    print("  a real improvement over the random walk and is still not the experiment.")

    print("\n  T-maze - where the prefix ingredient contributes nothing:\n")
    tm_models = [cue_model, blind_model]
    tm_walk = separating_mass(tm_models, UNIFORM_PRIOR, random_walk(TM_ACTIONS),
                              2, TM_ACTIONS, TM_OBS)
    tm_prefix = mixture_mass(tm_models, [state_prefix([], TM_ACTIONS),
                                         state_prefix([TM_WAIT], TM_ACTIONS)],
                             2, TM_ACTIONS, TM_OBS)
    tm_des = separating_mass(tm_models, UNIFORM_PRIOR,
                             designed([TM_WAIT], RUN, TM_ACTIONS), 2, TM_ACTIONS, TM_OBS)
    print(f"    random walk   {tm_walk}")
    print(f"    state prefix  {tm_prefix}")
    print(f"    designed      {tm_des}")
    print("\n  Every automaton state is already visited on the first step whatever the agent")
    print("  does, so no access sequence changes what the strategy sees and its yield is its")
    print("  random-walk tail. Reachability has nothing to select on here.")
    rows.append(["T-maze", str(tm_walk), str(tm_prefix), str(tm_des),
                 round(float(tm_prefix / tm_walk), 3) if tm_walk else ""])

    p = write_csv("baselines/active_learning",
                  ["environment_or_depth", "random_walk_mass", "state_prefix_mass",
                   "designed_mass", "prefix_over_walk"], rows)
    announce([p])
    print("\n" + "=" * 94)
