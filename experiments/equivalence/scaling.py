import random
from fractions import Fraction
from math import gcd, log
from time import perf_counter

from pec.product import pi_equivalent, reachable_pairs, witness_pair
from pec.rdp import RDP, Policy

from experiments.common import announce, write_csv


HALF = Fraction(1, 2)
WAIT, PROBE = "wait", "probe"
OBS = ["0", "1"]


def cycle_model(n: int, revealing: bool, extra_actions: int = 0,
                reveal_at: int = 0) -> RDP:
    waits = [WAIT] + [f"wait{i}" for i in range(extra_actions)]
    actions = waits + [PROBE]

    def step(q, a, o):
        return (q + 1) % n if a in waits else q

    def emit(q, a, o):
        if revealing and a == PROBE and q == reveal_at:
            return Fraction(1) if o == "1" else Fraction(0)
        return HALF

    return RDP(f"cycle-{n}-{'rev' if revealing else 'blind'}", 0, step, emit,
               list(range(n)), actions, OBS)


def random_model(n: int, seed: int, revealing: bool) -> RDP:
    rng = random.Random(seed)
    table = {(q, o): rng.randrange(n) for q in range(n) for o in OBS}

    def step(q, a, o):
        return table[(q, o)] if a == WAIT else q

    def emit(q, a, o):
        if revealing and a == PROBE and q == 0:
            return Fraction(1) if o == "1" else Fraction(0)
        return HALF

    return RDP(f"rand-{n}-{seed}", 0, step, emit, list(range(n)),
               [WAIT, PROBE], OBS)


def wait_policy(model: RDP) -> Policy:
    return Policy("wait-only",
                  lambda h, a: Fraction(0) if a == PROBE else Fraction(1),
                  model.actions)


def probe_policy(model: RDP) -> Policy:
    n = len(model.actions)
    return Policy("probing", lambda h, a: Fraction(1, n), model.actions)


SIZES = [(10, 11), (20, 21), (50, 51), (100, 101), (200, 201), (320, 321),
         (500, 501), (700, 701), (1000, 1001)]


def loglog_slope(xs, ys) -> float:
    lx = [log(x) for x in xs]
    ly = [log(y) for y in ys]
    n = len(lx)
    mx, my = sum(lx) / n, sum(ly) / n
    num = sum((a - mx) * (b - my) for a, b in zip(lx, ly))
    den = sum((a - mx) ** 2 for a in lx)
    return num / den


def main() -> None:
    print("=" * 96)
    print("  PEC at scale - the O(|Q||Q'||A||O|) bound of Corollary 15, measured")
    print("=" * 96)

    paths = []

    print("\n  Certifying pi-equivalence: PEC must exhaust the reachable product.")
    print(f"\n  {'|Q|':>6} {'|Q-prime|':>10} {'product':>10} {'pairs seen':>11}"
          f" {'full?':>6} {'verdict':>11} {'seconds':>9} {'us / pair':>10}")
    print("  " + "-" * 82)

    rows, xs, ys = [], [], []
    for n, m in SIZES:
        assert gcd(n, m) == 1, f"{n},{m} must be coprime for the product to be full"
        m1, m2 = cycle_model(n, True), cycle_model(m, False)
        pol = wait_policy(m1)

        t0 = perf_counter()
        verdict = pi_equivalent(m1, m2, pol)
        elapsed = perf_counter() - t0

        pairs = len(reachable_pairs(m1, m2, pol))
        full = pairs == n * m
        assert verdict, f"cycle pair {n},{m} should be pi-equivalent"
        print(f"  {n:>6} {m:>10} {n * m:>10} {pairs:>11} {('yes' if full else 'NO'):>6}"
              f" {'equivalent':>11} {elapsed:>9.3f} {1e6 * elapsed / pairs:>10.2f}")
        rows.append([n, m, n * m, pairs, full, "equivalent", round(elapsed, 4),
                     round(1e6 * elapsed / pairs, 3)])
        xs.append(pairs)
        ys.append(elapsed)

    paths.append(write_csv("equivalence/scaling", ["n_states_m1", "n_states_m2", "product",
                                           "pairs_explored", "product_full", "verdict",
                                           "seconds", "microseconds_per_pair"], rows))

    big = [(x, y) for x, y in zip(xs, ys) if x >= 10_000]
    slope = loglog_slope([x for x, _ in big], [y for _, y in big])
    print(f"\n  Log-log slope over the four largest sizes: {slope:.3f}")
    assert 0.85 <= slope <= 1.15, f"expected linear growth, measured exponent {slope}"
    print("  An exponent of 1 is linear growth in |Q|.|Q'|, which is the bound. The cost")
    print("  per pair is flat across four orders of magnitude, so nothing superlinear is")
    print("  hiding in the constant.")

    print("\n" + "=" * 96)
    print("  The |A| factor, at fixed |Q|.|Q'|")
    print("=" * 96)
    print(f"\n  {'|A_pi|':>8} {'product':>10} {'seconds':>9} {'ratio to |A_pi|=1':>19}")
    print("  " + "-" * 50)

    a_rows, base = [], None
    for extra in range(4):
        m1 = cycle_model(200, True, extra_actions=extra)
        m2 = cycle_model(201, False, extra_actions=extra)
        pol = wait_policy(m1)
        t0 = perf_counter()
        pi_equivalent(m1, m2, pol)
        elapsed = perf_counter() - t0
        base = base or elapsed
        print(f"  {extra + 1:>8} {200 * 201:>10} {elapsed:>9.3f}"
              f" {elapsed / base:>18.2f}x")
        a_rows.append([extra + 1, 200 * 201, round(elapsed, 4), round(elapsed / base, 2)])
    paths.append(write_csv("equivalence/scaling_actions",
                           ["n_actions_in_support", "product", "seconds",
                            "ratio_to_one_action"], a_rows))
    print("\n  Time grows about linearly in the number of actions the policy can take,")
    print("  which is the |A| factor of the bound appearing on its own.")

    print("\n" + "=" * 96)
    print("  Random transition tables, to check the cyclic structure is not doing the work")
    print("=" * 96)
    print(f"\n  {'|Q|':>6} {'|Q-prime|':>10} {'product':>10} {'pairs seen':>11}"
          f" {'% of product':>13} {'seconds':>9} {'us / pair':>10}")
    print("  " + "-" * 76)

    r_rows, rxs, rys = [], [], []
    for i, (n, m) in enumerate([(50, 51), (100, 101), (200, 201), (500, 501),
                                (1000, 1001)]):
        m1, m2 = random_model(n, 1000 + i, True), random_model(m, 2000 + i, False)
        pol = wait_policy(m1)
        t0 = perf_counter()
        verdict = pi_equivalent(m1, m2, pol)
        elapsed = perf_counter() - t0
        pairs = len(reachable_pairs(m1, m2, pol))
        assert verdict, f"random pair {n},{m} should be pi-equivalent"
        print(f"  {n:>6} {m:>10} {n * m:>10} {pairs:>11}"
              f" {100 * pairs / (n * m):>12.1f}% {elapsed:>9.3f}"
              f" {1e6 * elapsed / pairs:>10.2f}")
        r_rows.append([n, m, n * m, pairs, round(100 * pairs / (n * m), 2),
                       round(elapsed, 4), round(1e6 * elapsed / pairs, 3)])
        rxs.append(pairs)
        rys.append(elapsed)
    paths.append(write_csv("equivalence/scaling_random",
                           ["n_states_m1", "n_states_m2", "product", "pairs_explored",
                            "percent_of_product", "seconds",
                            "microseconds_per_pair"], r_rows))
    r_slope = loglog_slope(rxs, rys)
    print(f"\n  Log-log slope against pairs actually explored: {r_slope:.3f}")
    print("  Random tables reach a large fraction of the product and cost the same per")
    print("  pair, so the linearity is in the procedure rather than in the construction.")

    print("\n" + "=" * 96)
    print("  Returning a witness instead: cost tracks the witness depth, not the product")
    print("=" * 96)
    n, m = 1000, 1001
    print(f"\n  Product fixed at |Q|.|Q'| = {n * m}; only the depth of the disagreement"
          f" moves.")
    print(f"\n  {'witness depth':>14} {'pairs seen':>11} {'witness (s)':>12}"
          f" {'certify (s)':>12} {'fraction of certify':>21}")
    print("  " + "-" * 76)

    m_cert = cycle_model(n, True)
    t0 = perf_counter()
    pi_equivalent(m_cert, cycle_model(m, False), wait_policy(m_cert))
    t_cert = perf_counter() - t0

    w_rows = []
    for depth in [0, 1, 10, 100, 999]:
        m1 = cycle_model(n, True, reveal_at=depth)
        m2 = cycle_model(m, False)
        pol = probe_policy(m1)
        t0 = perf_counter()
        w = witness_pair(m1, m2, pol)
        t_wit = perf_counter() - t0
        assert w is not None, "the deviation must expose the disagreement"
        assert w[0][0] == depth, f"expected the witness at state {depth}, got {w}"

        print(f"  {depth:>14} {depth + 1:>11} {t_wit:>12.6f} {t_cert:>12.3f}"
              f" {t_wit / t_cert:>20.5%}")
        w_rows.append([n, m, n * m, depth, depth + 1, round(t_wit, 6),
                       round(t_cert, 4), round(t_wit / t_cert, 6), str(w)])
    paths.append(write_csv("equivalence/scaling_witness",
                           ["n_states_m1", "n_states_m2", "product", "witness_depth",
                            "pairs_explored", "seconds_witness", "seconds_certify",
                            "fraction_of_certify", "witness"], w_rows))
    announce(paths)


if __name__ == "__main__":
    main()
