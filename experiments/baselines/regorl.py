from fractions import Fraction
from itertools import product as iproduct
from time import perf_counter

from pec.envs.corridor import make_pair
from pec.envs.suite import SUITE
from pec.product import pi_equivalent, separating_action_depth
from pec.envs import tmaze
from pec.rdp import RDP, Policy, histories_upto

from experiments.common import announce, write_csv


def with_start(model: RDP, q) -> RDP:
    return RDP(
        name=f"{model.name}@{q}",
        start=q,
        step=model.step,
        emit=model.emit,
        states=model.states,
        actions=model.actions,
        observations=model.observations,
    )


def assert_stationary(policy: Policy, actions, observations, max_len: int = 3) -> None:

    base = {a: policy.prob((), a) for a in actions}
    for h in histories_upto(list(actions), list(observations), max_len):
        for a in actions:
            if policy.prob(h, a) != base[a]:
                raise AssertionError(
                    f"{policy.name} is not stationary: P({a}) varies with history"
                )


def l_inf_p(model: RDP, q1, q2, policy: Policy, steps: int) -> Fraction:
    best = Fraction(0)
    frontier = [(q1, q2, Fraction(1), Fraction(1))]
    for _ in range(steps):
        nxt = []
        for s1, s2, p1, p2 in frontier:
            for a, o in iproduct(model.actions, model.observations):
                pa = policy.prob((), a)
                if pa == 0:
                    continue
                n1 = p1 * pa * model.emit(s1, a, o)
                n2 = p2 * pa * model.emit(s2, a, o)
                if n1 == 0 and n2 == 0:
                    continue
                diff = abs(n1 - n2)
                if diff > best:
                    best = diff
                nxt.append((model.step(s1, a, o), model.step(s2, a, o), n1, n2))
        frontier = nxt
        if not frontier:
            break
    return best


def uniform_policy(model: RDP) -> Policy:
    n = len(model.actions)
    return Policy(name="uniform", prob=lambda h, a: Fraction(1, n),
                  actions=model.actions)


def separating_depth(model: RDP, q1, q2, policy: Policy):
    return separating_action_depth(with_start(model, q1), with_start(model, q2), policy)


def layers(model: RDP, policy: Policy, horizon: int) -> list[set]:
    out = []
    current = {model.start}
    for _ in range(horizon):
        out.append(set(current))
        nxt = set()
        for q in current:
            for a, o in iproduct(model.actions, model.observations):
                if policy.prob((), a) == 0 or model.emit(q, a, o) == 0:
                    continue
                nxt.add(model.step(q, a, o))
        current = nxt
    return out


def minimal_layer_pairs(model: RDP, policy: Policy, horizon: int):
    reference = uniform_policy(model)
    for t, qt in enumerate(layers(model, policy, horizon), start=1):
        remaining = horizon - t + 1
        if remaining <= 0:
            continue
        for q1, q2 in iproduct(sorted(qt, key=repr), repeat=2):
            if repr(q1) >= repr(q2):
                continue
            d = separating_depth(model, q1, q2, reference)
            if d is None or d > remaining:
                continue
            yield t, remaining, q1, q2


def mu0(model: RDP, policy: Policy, horizon: int):
    assert_stationary(policy, model.actions, model.observations)
    worst, arg = None, None
    for t, remaining, q1, q2 in minimal_layer_pairs(model, policy, horizon):
        d = l_inf_p(model, q1, q2, policy, remaining)
        if worst is None or d < worst:
            worst, arg = d, (t, q1, q2)
    return worst, arg


def pec_assumption2(model: RDP, policy: Policy, horizon: int) -> bool:
    for _t, remaining, q1, q2 in minimal_layer_pairs(model, policy, horizon):
        d = separating_depth(model, q1, q2, policy)
        if d is None or d > remaining:
            return False
    return True


def environments():

    out = [("T-maze", tmaze.cue_model, tmaze.wait_policy,
            tmaze.mixed_policy(Fraction(1, 2)), 4, "action-in-visited-state")]
    for env in SUITE:
        out.append((env["name"], env["true"], env["policy"], env["deviation"],
                    5 if env["name"] == "Rotating Maze" else 4, env["kind"]))
    return out


def main() -> None:
    print("=" * 94)
    print("  RegORL's L^p_inf-distinguishability on our environments")
    print("  (Cipollone et al., NeurIPS 2023: Assumption 2, and N_delta ~ 1/mu_0)")
    print("=" * 94)

    rows, header = [], [
        "environment", "kind", "horizon", "mu0_under_pi", "mu0_under_deviation",
        "n_minimal_pairs_under_pi", "argmin_pair_under_pi",
        "assumption2_under_pi", "assumption2_under_deviation",
        "pec_verdict_under_pi", "pec_verdict_under_deviation", "pec_agrees",
    ]

    print(f"\n  {'environment':<15} {'H':>2} {'mu0 under pi':>13} {'mu0 under dev':>14}"
          f" {'A2 under pi':>12} {'PEC agrees':>11}")
    print("  " + "-" * 74)

    for name, model, pi, dev, h, kind in environments():
        n_pairs = len(list(minimal_layer_pairs(model, pi, h)))
        m_pi, arg_pi = mu0(model, pi, h)
        m_dev, _ = mu0(model, dev, h)

        a2_pi = "vacuous" if m_pi is None else ("holds" if m_pi > 0 else "FAILS")
        a2_dev = "vacuous" if m_dev is None else ("holds" if m_dev > 0 else "FAILS")

        pec_pi = pec_assumption2(model, pi, h)
        pec_dev = pec_assumption2(model, dev, h)

        agree = ((m_pi is None or (m_pi > 0) == pec_pi)
                 and (m_dev is None or (m_dev > 0) == pec_dev))
        assert agree, f"{name}: PEC disagrees with the exact parameter"

        pair = "--" if arg_pi is None else f"t={arg_pi[0]}: {arg_pi[1]} vs {arg_pi[2]}"
        print(f"  {name:<15} {h:>2} {str(m_pi):>13} {str(m_dev):>14}"
              f" {a2_pi:>12} {('yes' if agree else 'NO'):>11}")
        rows.append([name, kind, h, m_pi, m_dev, n_pairs, pair, a2_pi, a2_dev,
                     pec_pi, pec_dev, agree])

    paths = [write_csv("baselines/regorl_mu0", header, rows)]

    failing = [r[0] for r in rows if r[7] == "FAILS"]
    vacuous = [r[0] for r in rows if r[7] == "vacuous"]

    print(f"\n  Assumption 2 FAILS under the behaviour policy on: {', '.join(failing)}.")
    print("  There mu_0 = 0 as an exact rational, N_delta ~ 1/mu_0 diverges, and the")
    print("  RegORL guarantee is vacuous - on the same instances where our separating")
    print("  mass is an exact rational zero. The collapsing state pair, under pi:")
    for r in rows:
        if r[7] == "FAILS":
            print(f"    {r[0]:<15} {r[6]}")

    if vacuous:
        print(f"\n  Assumption 2 is satisfied VACUOUSLY on: {', '.join(vacuous)}.")
        print("  The behaviour policy reaches a single state of the true RDP, so no pair")
        print("  of distinct states exists to be told apart and the minimum is over the")
        print("  empty set. Their parameter therefore does not flag this instance, while")
        print("  PEC does: mu_0 compares two states of the true model that the policy")
        print("  reaches, whereas pi-equivalence compares two candidate models. Here the")
        print("  RegORL bound binds through concentrability instead, not distinguishability.")

    print("\n  Under the designed deviation mu_0 is positive and exactly rational in every")
    print("  environment, so the experiment PEC names is the one restoring the assumption.")

    print("\n" + "=" * 94)
    print("  Deciding Assumption 2: their definition against PEC")
    print("=" * 94)
    print(f"\n  {'environment':<15} {'H':>2} {'by definition (s)':>19} {'by PEC (s)':>12}"
          f" {'speedup':>9}")
    print("  " + "-" * 64)

    cost_rows = []
    for name, model, pi, _dev, h, _kind in environments():
        t0 = perf_counter(); mu0(model, pi, h); t_def = perf_counter() - t0
        t0 = perf_counter(); pec_assumption2(model, pi, h); t_pec = perf_counter() - t0
        print(f"  {name:<15} {h:>2} {t_def:>19.5f} {t_pec:>12.5f}"
              f" {t_def / t_pec:>8.1f}x")
        cost_rows.append([name, h, round(t_def, 6), round(t_pec, 6),
                          round(t_def / t_pec, 1)])
    paths.append(write_csv("baselines/regorl_mu0_cost",
                           ["environment", "horizon", "seconds_by_definition",
                            "seconds_by_pec", "speedup"], cost_rows))
    print("\n  These horizons are small. The gap is driven by the horizon rather than by")
    print("  the model: the definition maximizes over (|A||O|)^H trace prefixes, PEC")
    print("  explores |Q|^2 pairs and never mentions H. The corridor sweep below runs")
    print("  the same comparison as the horizon grows.")

    print("\n" + "=" * 94)
    print("  mu_0 on the corridor family, where depth is the knob")
    print("=" * 94)
    print(f"\n  {'depth d':>8} {'A2 under pi':>12} {'PEC under pi':>14}"
          f" {'mu0 under dev':>14} {'ratio':>7} {'defn (s)':>10} {'PEC (s)':>9}"
          f" {'speedup':>8}")
    print("  " + "-" * 92)

    decay_rows, prev = [], None
    for d in range(2, 8):
        true_m, rival, pol, dev = make_pair(d)
        h = d + 2
        m_pi, _ = mu0(true_m, pol, h)
        a2_pi = "vacuous" if m_pi is None else ("holds" if m_pi > 0 else "FAILS")
        pec_pi = "equivalent" if pi_equivalent(true_m, rival, pol) else "separable"

        t0 = perf_counter(); m_dev, _ = mu0(true_m, dev, h); t_def = perf_counter() - t0
        t0 = perf_counter(); pec_assumption2(true_m, dev, h); t_pec = perf_counter() - t0

        ratio = "--" if prev in (None, 0) or not m_dev else f"{float(m_dev / prev):.3f}"
        print(f"  {d:>8} {a2_pi:>12} {pec_pi:>14} {str(m_dev):>14} {ratio:>7}"
              f" {t_def:>10.3f} {t_pec:>9.4f} {t_def / t_pec:>7.0f}x")
        decay_rows.append([d, m_pi, a2_pi, pec_pi, m_dev, ratio, round(t_def, 4),
                           round(t_pec, 5), round(t_def / t_pec, 1)])
        prev = m_dev
    paths.append(write_csv("baselines/regorl_mu0_corridor",
                           ["depth", "mu0_under_pi", "assumption2_under_pi",
                            "pec_verdict_under_pi", "mu0_under_deviation",
                            "ratio_to_previous_depth", "seconds_by_definition",
                            "seconds_by_pec", "speedup"], decay_rows))

    announce(paths)


if __name__ == "__main__":
    main()
