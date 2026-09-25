from fractions import Fraction

from pec.envs.corridor import ACTIONS, PROBE, make_pair
from pec.product import _policy_can_take, pi_equivalent, reachable_pairs
from pec.rdp import RDP, Policy, sample_trajectories

from experiments.common import announce, write_csv


N_TRAJ, SEEDS = 600, 5
GRID_POINTS = 8


def covering(p_probe=Fraction(1, 4)) -> Policy:
    return Policy(name=f"probe-w.p.-{p_probe}", actions=ACTIONS,
                  prob=lambda h, a: p_probe if a == PROBE else 1 - p_probe)


def estimate(model: RDP, policy: Policy, n_traj: int, length: int, seed: int,
             alpha: int = 1) -> RDP:
    counts = {(q, a, o): alpha for q in model.states
              for a in model.actions for o in model.observations}
    for hist in sample_trajectories(model, policy, n_traj, length, seed=seed):
        q = model.start
        for (a, o) in hist:
            counts[(q, a, o)] += 1
            q = model.step(q, a, o)

    table = {}
    for q in model.states:
        for a in model.actions:
            total = sum(counts[(q, a, o)] for o in model.observations)
            for o in model.observations:
                table[(q, a, o)] = Fraction(counts[(q, a, o)], total)

    return RDP(name=f"{model.name}-hat", start=model.start, step=model.step,
               emit=lambda q, a, o: table[(q, a, o)], states=model.states,
               actions=model.actions, observations=model.observations)


def consulted_error(m1, m2, h1, h2, policy) -> Fraction:
    worst = Fraction(0)
    for (q1, q2) in reachable_pairs(m1, m2, policy):
        for a in m1.actions:
            if not _policy_can_take(policy, a):
                continue
            for o in m1.observations:
                worst = max(worst,
                            abs(h1.emit(q1, a, o) - m1.emit(q1, a, o)),
                            abs(h2.emit(q2, a, o) - m2.emit(q2, a, o)))
    return worst


def separation_gap(m1: RDP, m2: RDP) -> Fraction:
    gaps = []
    for q1 in m1.states:
        for q2 in m2.states:
            for a in m1.actions:
                d = max(abs(m1.emit(q1, a, o) - m2.emit(q2, a, o)) for o in m1.observations)
                if d > 0:
                    gaps.append(d)
    return min(gaps) if gaps else Fraction(0)


def window_grid(eta: Fraction, gap: Fraction, points: int = GRID_POINTS):
    lo, hi = 2 * eta, gap - 2 * eta
    if lo >= hi:
        return []
    return [lo + (hi - lo) * Fraction(i, points + 1) for i in range(1, points + 1)]


if __name__ == "__main__":
    print("=" * 100)
    print("PEC UNDER ESTIMATED EMISSIONS - the exact test never certifies; the tolerant one does")
    print("=" * 100)

    print(f"\n  {N_TRAJ} trajectories per candidate under the covering policy,"
          f" {SEEDS} seeds, Laplace-smoothed MLE.")
    print("  eta is measured over the rows PEC consults; the window is (2*eta, gap - 2*eta).\n")
    head = (f"  {'depth':>5} {'gap':>6} {'eta':>8} {'window':>18} {'eps=0':>11} "
            f"{'in window':>11} {'eps>gap':>10}")
    print(head)
    print("  " + "-" * (len(head) - 2))

    rows, all_ok = [], True
    for k in (2, 4, 6, 8):
        true, rival, pi, dev = make_pair(k)
        length, gap = 2 * k, separation_gap(true, rival)

        etas, exact_verdicts, in_window, over_eps = [], [], [], []
        for seed in range(SEEDS):
            h1 = estimate(true, covering(), N_TRAJ, length, seed)
            h2 = estimate(rival, covering(), N_TRAJ, length, 1000 + seed)
            eta = max(consulted_error(true, rival, h1, h2, pi),
                      consulted_error(true, rival, h1, h2, dev))
            etas.append(eta)
            exact_verdicts.append(pi_equivalent(h1, h2, pi, Fraction(0)))
            grid = window_grid(eta, gap)
            in_window.append(bool(grid) and all(
                pi_equivalent(h1, h2, pi, e) and not pi_equivalent(h1, h2, dev, e)
                for e in grid))
            over_eps.append(pi_equivalent(h1, h2, dev, gap + 2 * eta))

        eta = max(etas)
        lo, hi = 2 * eta, gap - 2 * eta
        w = f"({float(lo):.3f}, {float(hi):.3f})" if lo < hi else "empty"
        eps0 = "certifies" if any(exact_verdicts) else "never"
        ok_in, ok_over = all(in_window), all(over_eps)
        all_ok &= ok_in and ok_over

        print(f"  {k:>5} {str(gap):>6} {float(eta):>8.4f} {w:>18} {eps0:>11} "
              f"{('all correct' if ok_in else 'FAILS'):>11} "
              f"{('over-certifies' if ok_over else 'FAILS'):>10}")
        rows.append([k, str(gap), round(float(eta), 6), round(float(lo), 6),
                     round(float(hi), 6), eps0, ok_in, ok_over])

    p = write_csv("equivalence/estimated_tables",
                  ["separating_depth", "gap", "eta_consulted", "window_lo", "window_hi",
                   "exact_test_behaviour", "window_verdicts_correct", "over_eps_certifies"],
                  rows)
    announce([p])

    assert all_ok, "epsilon-tolerant PEC did not behave as the window predicts"
    print("\n" + "=" * 100)
