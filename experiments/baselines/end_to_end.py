import math
from fractions import Fraction

from pec.envs.corridor import ACTIONS as CORRIDOR_ACTIONS, PROBE, WAIT, make_pair
from pec.envs.suite import SUITE
from pec.envs.tmaze import (
    ACTIONS as TM_ACTIONS, OBSERVATIONS as TM_OBS, blind_model, cue_model, mixed_policy,
    wait_policy,
)
from pec.product import witness_pair
from pec.rdp import Policy, Posterior, sample_trajectories

from experiments.common import announce, write_csv


N_TRAIN, N_TEST, LENGTH, EPS = 400, 120, 6, 0.05
N_SEEDS = 10
START = "@"


def uniform_policy(actions):
    p = Fraction(1, len(actions))
    return Policy("uniform", lambda h, a: p, actions)


def to_alergia(histories):
    return [[START] + [(a, o) for (a, o) in h] for h in histories]


def learn(histories, eps=EPS):
    from aalpy.learning_algs import run_Alergia
    return run_Alergia(to_alergia(histories), "mdp", eps=eps)


def score(model, test, n_obs):
    floor = math.log(1.0 / n_obs)
    total_ll, backoff_ll, scorable, steps = 0.0, 0.0, 0, 0
    for h in test:
        state = model.initial_state
        dead = False
        for (a, o) in h:
            steps += 1
            if dead:
                backoff_ll += floor
                continue
            trans = state.transitions.get(a)
            p = sum(pr for t, pr in trans if t.output == o) if trans else 0.0
            nxt = next((t for t, pr in trans if t.output == o and pr > 0), None) if trans else None
            if nxt is None or p <= 0.0:
                dead = True
                backoff_ll += floor
                continue
            total_ll += math.log(p)
            backoff_ll += math.log(p)
            scorable += 1
            state = nxt
    return (scorable / steps,
            (total_ll / scorable if scorable else float("-inf")),
            backoff_ll / steps)


def targeted(k):
    def prob(h, a):
        want = WAIT if len(h) < k else PROBE
        return Fraction(1) if a == want else Fraction(0)
    return Policy(f"targeted-{k}", prob, CORRIDOR_ACTIONS)


def log10_bf(true_model, rival, corpus):
    post = Posterior([true_model, rival])
    w = post.weights(corpus)
    if w[1] == 0:
        return float("inf")
    return math.log10(float(w[0] / w[1]))


def true_ll(model, test):
    total, n = 0.0, 0
    for h in test:
        q = model.start
        for (a, o) in h:
            total += math.log(float(model.emit(q, a, o)))
            q = model.step(q, a, o)
            n += 1
    return total / n


def cases():
    yield dict(name="T-maze", true=cue_model, rival=blind_model,
               policy=wait_policy, deviation=mixed_policy(Fraction(1, 2)),
               actions=TM_ACTIONS, observations=TM_OBS)
    for c in SUITE:
        yield dict(name=c["name"], true=c["true"], rival=c["rival"],
                   policy=c["policy"], deviation=c["deviation"],
                   actions=c["actions"], observations=c["observations"])


if __name__ == "__main__":
    print("=" * 92)
    print("THE METHOD - detect, design, then learn")
    print("=" * 92)

    try:
        import importlib
        importlib.import_module("aalpy")
    except ImportError:
        print("\n  aalpy not installed - skipping.\n  pip install aalpy\n")
        raise SystemExit(0)

    rows = []
    print("\n[1] Step 1 - DETECT. Is the behaviour policy's corpus identifying at all?\n")
    print(f"  {'environment':<16} {'pi-equivalent?':>15}   witness (the missing experiment)")
    print("  " + "-" * 78)
    for c in cases():
        w = witness_pair(c["true"], c["rival"], c["policy"])
        verdict = "YES - blind" if w is None else "no"
        print(f"  {c['name']:<16} {verdict:>15}   "
              f"{'- none under pi' if w is None else w}")
    print("\n  Every environment is π-equivalent under its behaviour policy, so every")
    print("  corpus below is consistent with two different environments. That is decided")
    print("  in polynomial time, before any data is collected.")

    print("\n[2] Steps 2-4 - DESIGN, then learn. Same learner, same budget, two corpora.\n")
    print(f"  {'environment':<16} {'corpus':<10} {'log10 BF':>8} {'':<10} "
          f"{'coverage':>7} {'':<8} {'LL/step*':>8}")
    print(f"  mean +/- half-range over {N_SEEDS} seeds. The 0.00 under pi is an exact")
    print("  rational zero, not an estimate, so its spread is exactly zero by Theorem 8.")
    print("  " + "-" * 82)

    def spread(xs):
        return sum(xs) / len(xs), (max(xs) - min(xs)) / 2

    for c in cases():
        corpora = (("under pi", c["policy"]),
                   ("random", uniform_policy(c["actions"])),
                   ("designed", c["deviation"]))
        for label, pol in corpora:
            bfs, covs, bos, true_lls = [], [], [], []
            for seed in range(N_SEEDS):
                test = sample_trajectories(c["true"], uniform_policy(c["actions"]),
                                           N_TEST, LENGTH, seed=99 + seed)
                train = sample_trajectories(c["true"], pol, N_TRAIN, LENGTH, seed=7 + seed)
                bf = log10_bf(c["true"], c["rival"], train)
                cov, _ll, bo = score(learn(train), test, len(c["observations"]))
                bfs.append(1e9 if bf == float("inf") else bf)
                covs.append(cov)
                bos.append(bo)
                true_lls.append(true_ll(c["true"], test))
            (mbf, sbf), (mcov, scov) = spread(bfs), spread(covs)
            (mbo, sbo), (mtrue, _) = spread(bos), spread(true_lls)
            rows.append([c["name"], label, N_SEEDS,
                         round(mcov, 4), round(scov, 4),
                         round(mbo, 4), round(sbo, 4), round(mtrue, 4),
                         round(mbf, 2), round(sbf, 2)])
            print(f"  {c['name']:<16} {label:<10} {mbf:>8.2f} +/-{sbf:<7.2f} "
                  f"{mcov:>7.1%} +/-{scov:<6.1%} {mbo:>8.4f} +/-{sbo:<6.4f}")
        print()

    print("\n[3] Designed against uniform exploration on the corridor family, by separating depth\n")
    print(f"  {'depth':>5} {'budget':>7} {'random':>10} {'designed':>10} {'ratio':>8}")
    print("  " + "-" * 44)
    depth_rows = []
    for k in (2, 4, 6, 8):
        true, rival, _pi, _dev = make_pair(k)
        length = k + 2
        for n in (5, 25):
            rs, ds = [], []
            for seed in range(10):
                r = sample_trajectories(true, uniform_policy(CORRIDOR_ACTIONS),
                                        n, length, seed=seed)
                d = sample_trajectories(true, targeted(k), n, length, seed=seed)
                for acc, corpus in ((rs, r), (ds, d)):
                    v = log10_bf(true, rival, corpus)
                    acc.append(999.0 if v == float("inf") else v)
            mr, md = sum(rs) / len(rs), sum(ds) / len(ds)
            ratio = float("inf") if mr <= 0 else md / mr
            depth_rows.append([k, n, round(mr, 3), round(md, 3),
                               "inf" if ratio == float("inf") else round(ratio, 2)])
            rs_txt = "  inf" if ratio == float("inf") else f"{ratio:>7.1f}x"
            print(f"  {k:>5} {n:>7} {mr:>10.2f} {md:>10.2f} {rs_txt}")
        print()

    p2 = write_csv("baselines/end_to_end_depth",
                   ["separating_depth", "n_trajectories", "random_log10_bf",
                    "designed_log10_bf", "ratio"], depth_rows)

    p = write_csv("baselines/end_to_end",
                  ["environment", "corpus", "n_seeds",
                   "coverage_mean", "coverage_halfrange",
                   "loglik_per_step_backoff_mean", "loglik_per_step_backoff_halfrange",
                   "true_loglik_mean",
                   "log10_bayes_factor_mean", "log10_bayes_factor_halfrange"], rows)
    announce([p, p2])
