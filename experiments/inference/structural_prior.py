from fractions import Fraction

from pec.envs.tmaze import ACTIONS, LEFT, OBSERVATIONS, RIGHT, cue_model, mixed_policy
from pec.rdp import sample_trajectories
from pec.rdp_structure import RDPStructure, StructurePrior, enumerate_structures


MAX_STATES = 3
EXPLORING = mixed_policy(Fraction(1, 2))
TRUE_STRUCTURE = RDPStructure(
    3, ACTIONS, OBSERVATIONS,
    {(0, LEFT): 1, (0, RIGHT): 2,
     (1, LEFT): 1, (1, RIGHT): 1,
     (2, LEFT): 2, (2, RIGHT): 2},
)
TRUE_KEY = (TRUE_STRUCTURE.n_states, TRUE_STRUCTURE.canonical())

STRUCTURES = list(enumerate_structures(MAX_STATES, ACTIONS, OBSERVATIONS))


def remembers_cue(s: RDPStructure) -> bool:
    return s.distinguishes_first_observation()


def _successor_absorbing(s: RDPStructure, o) -> bool:
    q = s.delta[(0, o)]
    return all(s.delta[(q, o2)] == q for o2 in s.observations)


def remembers_and_commits(s: RDPStructure) -> bool:
    return (s.distinguishes_first_observation()
            and any(_successor_absorbing(s, o) for o in s.observations))


def remembers_and_never_revises(s: RDPStructure) -> bool:
    return (s.distinguishes_first_observation()
            and all(_successor_absorbing(s, o) for o in s.observations))


def exact_structure(s: RDPStructure) -> bool:
    return (s.n_states, s.canonical()) == TRUE_KEY


def ignores_cue(s: RDPStructure) -> bool:
    return not s.distinguishes_first_observation()


def contraction(prior: StructurePrior, counts, length=4, seed=0):
    out = []
    for n in counts:
        data = sample_trajectories(cue_model, EXPLORING, n, length, seed=seed)
        out.append(prior.mass_on(TRUE_KEY, data))
    return out


if __name__ == "__main__":
    print("=" * 78)
    print("VALUE OF A SENTENCE, AND ROBUSTNESS - T-maze, exact posterior")
    print("=" * 78)
    n_match = sum(1 for s in STRUCTURES if remembers_cue(s))
    print(f"\nhypothesis space: {len(STRUCTURES)} structures with <= {MAX_STATES} states")
    print("  (predicate breakdown below)")
    print(f"  so it removes {len(STRUCTURES) - n_match} structures a priori "
          f"({100 * (len(STRUCTURES) - n_match) / len(STRUCTURES):.0f}% of the space)")
    print(f"  truth is in the selected set: {remembers_cue(TRUE_STRUCTURE)}")

    counts = [1, 2, 3, 5, 8, 12, 20]

    uninformative = StructurePrior(STRUCTURES)
    language = StructurePrior(STRUCTURES, remembers_cue, epsilon=Fraction(0))
    contaminated = StructurePrior(STRUCTURES, remembers_and_commits,
                                  epsilon=Fraction(1, 10))
    wrong = StructurePrior(STRUCTURES, ignores_cue, epsilon=Fraction(0))
    wrong_contam = StructurePrior(STRUCTURES, ignores_cue, epsilon=Fraction(1, 10))

    partial = StructurePrior(STRUCTURES, remembers_and_commits, epsilon=Fraction(0))
    sharp = StructurePrior(STRUCTURES, remembers_and_never_revises, epsilon=Fraction(0))

    rows = [
        ("uninformative π₀ (no sentence)", uninformative),
        ("vague: 'history matters'", language),
        ("partial: 'remembers, commits'", partial),
        ("saturated: pins it uniquely", sharp),
        ("partial, contaminated (ε=1/10)", contaminated),
        ("WRONG sentence (ε=0)", wrong),
        ("WRONG sentence (ε=1/10)", wrong_contam),
    ]

    print("\n  how much each sentence constrains the space:")
    for label, pred in [("vague: 'history matters'", remembers_cue),
                        ("partial: 'remembers, commits'", remembers_and_commits),
                        ("saturated: pins it uniquely", remembers_and_never_revises)]:
        k = sum(1 for s in STRUCTURES if pred(s))
        print(f"    {label:<36} {k:>4}/{len(STRUCTURES)} structures "
              f"({100 * k / len(STRUCTURES):>5.1f}% of the space)")

    print("\nposterior mass on the TRUE structure, by trajectory budget "
          "(each trajectory = 4 steps):\n")
    header = f"  {'prior':<32}" + "".join(f"{n:>9}" for n in counts)
    print(header)
    print("  " + "-" * (len(header) - 2))
    results = {}
    for label, prior in rows:
        vals = contraction(prior, counts)
        results[label] = vals
        print(f"  {label:<32}" + "".join(f"{float(v):>9.4f}" for v in vals))

    print("\n--- what the sentence is worth ---")
    u, l = results["uninformative π₀ (no sentence)"], results["partial: 'remembers, commits'"]
    for i, n in enumerate(counts):
        if u[i] > 0 and l[i] > 0:
            print(f"  at {n:>2} trajectories: language prior gives "
                  f"{float(l[i] / u[i]):.2f}x the mass on the truth")

    print("\n--- ε-contamination ---")
    print("  cost of insurance when the sentence is RIGHT, at the largest budget:")
    right = results["partial: 'remembers, commits'"][-1]
    print(f"    ε=0     -> {float(right):.6f}")
    print(f"    ε=1/10  -> {float(results['partial, contaminated (ε=1/10)'][-1]):.6f}")
    ratio = (results['partial, contaminated (ε=1/10)'][-1]
             / results["partial: 'remembers, commits'"][-1])
    print(f"    ratio    = {float(ratio):.4f}   (small - insurance is cheap)")

    print("\n--- misspecification ---")
    w0 = results["WRONG sentence (ε=0)"]
    wc = results["WRONG sentence (ε=1/10)"]
