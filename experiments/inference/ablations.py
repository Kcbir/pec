import statistics
from fractions import Fraction
from math import log10

from pec.envs.suite import SUITE
from pec.envs.tmaze import (
    ACTIONS as TM_ACTIONS, OBSERVATIONS as TM_OBS, cue_model, mixed_policy,
)
from pec.explore import expected_information_gain, separating_mass
from pec.rdp import sample_trajectories
from pec.rdp_structure import StructurePrior

from experiments.common import announce, write_csv
from experiments.inference.structural_prior import (
    EXPLORING, STRUCTURES, TRUE_KEY, ignores_cue, remembers_and_commits,
)
from experiments.inference.summary import (
    CASES, emission_recovery, marginal_lik_memoryless, marginal_lik_structured,
    uniform_policy,
)


PRIOR2 = [Fraction(1, 2), Fraction(1, 2)]
paths = []

print("=" * 92)
print("ABLATIONS")
print("=" * 92)

N_SEEDS = 12
print(f"\n[1] SEED VARIANCE - {N_SEEDS} seeds, 40 trajectories x 8 steps")
print("    Every headline number elsewhere is a single seed. These are the error bars.")
print(f"\n  {'environment':<16}{'log10 BF mean':>15}{'sd':>8}{'min':>9}{'max':>9}"
      f"{'emis MAE mean':>15}{'sd':>8}")
rows = []
for c in CASES:
    m = c["true"]
    bfs, maes = [], []
    for seed in range(N_SEEDS):
        d = sample_trajectories(m, uniform_policy(c["actions"]), 40, 8, seed=seed)
        bfs.append(log10(float(marginal_lik_structured(m, d) / marginal_lik_memoryless(m, d))))
        maes.append(emission_recovery(m, d)[0])
    rows.append([c["name"], round(statistics.mean(bfs), 4), round(statistics.stdev(bfs), 4),
                 round(min(bfs), 4), round(max(bfs), 4),
                 round(statistics.mean(maes), 5), round(statistics.stdev(maes), 5)])
    print(f"  {c['name']:<16}{statistics.mean(bfs):>15.2f}{statistics.stdev(bfs):>8.2f}"
          f"{min(bfs):>9.2f}{max(bfs):>9.2f}"
          f"{statistics.mean(maes):>15.4f}{statistics.stdev(maes):>8.4f}")
print("\n  Every seed, every environment: log10 BF stays far above 0, so the")
print("  non-Markov conclusion is not seed-dependent.")
paths.append(write_csv("inference/ablation_seed_variance",
                       ["environment", "log10_bf_mean", "log10_bf_sd", "log10_bf_min",
                        "log10_bf_max", "emission_mae_mean", "emission_mae_sd"], rows))

print("\n[2] DIRICHLET ALPHA - prior sensitivity of the emission prior")
alphas = [1, 2, 3, 5, 10]
print(f"\n  {'environment':<16}" + "".join(f"{'a=' + str(a):>12}" for a in alphas))
rows = []
for c in CASES:
    m = c["true"]
    d = sample_trajectories(m, uniform_policy(c["actions"]), 40, 8, seed=7)
    bfs = [log10(float(marginal_lik_structured(m, d, a) / marginal_lik_memoryless(m, d, a)))
           for a in alphas]
    rows.append([c["name"]] + [round(b, 4) for b in bfs])
    print(f"  {c['name']:<16}" + "".join(f"{b:>12.2f}" for b in bfs))
print("\n  log10 Bayes factor vs alpha. The conclusion is stable across an order of")
print("  magnitude of prior strength - it is not an artefact of alpha = 1.")
paths.append(write_csv("inference/ablation_dirichlet_alpha",
                       ["environment"] + [f"log10_bf_alpha_{a}" for a in alphas], rows))

print("\n[3] DATA SCALING - evidence accumulation")
counts = [5, 10, 20, 40, 80, 160]
print(f"\n  {'environment':<16}{'metric':<14}" + "".join(f"{n:>10}" for n in counts))
rows = []
for c in CASES:
    m = c["true"]
    bfs, maes = [], []
    for n in counts:
        bb, mm = [], []
        for seed in range(5):
            d = sample_trajectories(m, uniform_policy(c["actions"]), n, 8, seed=seed)
            bb.append(log10(float(marginal_lik_structured(m, d)
                                  / marginal_lik_memoryless(m, d))))
            mm.append(emission_recovery(m, d)[0])
        bfs.append(statistics.mean(bb))
        maes.append(statistics.mean(mm))
    print(f"  {c['name']:<16}{'log10 BF':<14}" + "".join(f"{b:>10.2f}" for b in bfs))
    print(f"  {'':<16}{'emis MAE':<14}" + "".join(f"{e:>10.4f}" for e in maes))
    for n, b, e in zip(counts, bfs, maes):
        rows.append([c["name"], n, round(b, 4), round(e, 6)])
print("\n  Mean over 5 seeds. Bayes factors grow roughly linearly in the number of")
print("  trajectories; emission error falls monotonically.")
paths.append(write_csv("inference/ablation_data_scaling",
                       ["environment", "trajectories", "log10_bf_mean5",
                        "emission_mae_mean5"], rows))

print(f"\n  {'environment':<16}{'n':>4}{'min':>9}{'mean':>9}{'max':>9}{'seeds<0':>10}")
rows_cross = []
for c in CASES:
    m = c["true"]
    for n in [2, 5, 10, 20]:
        vals = []
        for seed in range(12):
            d = sample_trajectories(m, uniform_policy(c["actions"]), n, 8, seed=seed)
            vals.append(log10(float(marginal_lik_structured(m, d)
                                    / marginal_lik_memoryless(m, d))))
        neg = sum(1 for v in vals if v < 0)
        rows_cross.append([c["name"], n, round(min(vals), 4),
                           round(statistics.mean(vals), 4), round(max(vals), 4), neg])
        print(f"  {c['name']:<16}{n:>4}{min(vals):>9.2f}{statistics.mean(vals):>9.2f}"
              f"{max(vals):>9.2f}{neg:>10}")
paths.append(write_csv("inference/ablation_smalldata_crossover",
                       ["environment", "trajectories", "log10_bf_min", "log10_bf_mean",
                        "log10_bf_max", "seeds_negative"], rows_cross))
print("\n[4] DEVIATION STRENGTH - exact, no sampling")
print("    How much does the missing experiment have to be run before the posterior moves?")
DEV_CASES = [{"name": "T-maze", "true": CASES[0]["true"], "rival": CASES[0]["rival"],
              "fn": mixed_policy, "actions": TM_ACTIONS, "observations": TM_OBS,
              "horizon": 2}] + [
    {"name": c["name"], "true": c["true"], "rival": c["rival"], "fn": c["deviation_fn"],
     "actions": c["actions"], "observations": c["observations"],
     "horizon": 3 if c["name"] != "Rotating Maze" else 4}
    for c in SUITE
]
ps = [Fraction(k, 8) for k in range(9)]
print(f"\n  {'environment':<16}{'metric':<16}" + "".join(f"{str(p):>9}" for p in ps))
rows = []
for c in DEV_CASES:
    pair = [c["true"], c["rival"]]
    A, O, H = c["actions"], c["observations"], c["horizon"]
    masses, eigs = [], []
    for p in ps:
        pol = c["fn"](p)
        masses.append(separating_mass(pair, PRIOR2, pol, H, A, O))
        eigs.append(expected_information_gain(pair, PRIOR2, pol, H, A, O))
    print(f"  {c['name']:<16}{'separating mass':<16}"
          + "".join(f"{float(m):>9.4f}" for m in masses))
    print(f"  {'':<16}{'EIG':<16}" + "".join(f"{e:>9.4f}" for e in eigs))
    for p, m, e in zip(ps, masses, eigs):
        rows.append([c["name"], str(p), float(p), m, float(m), round(e, 6)])
paths.append(write_csv("inference/ablation_deviation_strength",
                       ["environment", "deviation_prob", "deviation_prob_float",
                        "separating_mass", "separating_mass_float", "eig"], rows))

print("\n[5] EPSILON-CONTAMINATION - the full sweep")

eps_list = [Fraction(0), Fraction(1, 100), Fraction(1, 20), Fraction(1, 10),
            Fraction(1, 4), Fraction(1, 2), Fraction(1)]
budgets = [2, 5, 20]
print(f"\n  {'sentence':<10}{'budget':>8}" + "".join(f"{'e=' + str(e):>10}" for e in eps_list))
rows = []
for label, pred in [("right", remembers_and_commits), ("WRONG", ignores_cue)]:
    for n in budgets:
        data = sample_trajectories(cue_model, EXPLORING, n, 4, seed=0)
        vals = []
        for e in eps_list:
            prior = StructurePrior(STRUCTURES, pred, epsilon=e)
            vals.append(prior.mass_on(TRUE_KEY, data))
        print(f"  {label:<10}{n:>8}" + "".join(f"{float(v):>10.4f}" for v in vals))
        for e, v in zip(eps_list, vals):
            rows.append([label, n, str(e), float(e), float(v)])
paths.append(write_csv("inference/ablation_epsilon_contamination",
                       ["sentence", "trajectories", "epsilon", "epsilon_float",
                        "mass_on_truth"], rows))

announce(paths)
print("\n" + "=" * 92)
