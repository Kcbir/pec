from fractions import Fraction
from itertools import product as iproduct

from .explore import expected_information_gain, separating_mass
from .rdp import Policy


def sequence_policy(seq, actions) -> Policy:
    seq = tuple(seq)

    def prob(h, a):
        t = len(h)
        if t < len(seq):
            return Fraction(1) if a == seq[t] else Fraction(0)
        return Fraction(1, len(actions))

    return Policy(name="seq:" + ",".join(str(s) for s in seq), prob=prob, actions=actions)


def eig_of_sequence(models, prior, seq, actions, observations) -> float:
    return expected_information_gain(models, prior, sequence_policy(seq, actions),
                                     len(seq), actions, observations)


def mass_of_sequence(models, prior, seq, actions, observations) -> Fraction:
    return separating_mass(models, prior, sequence_policy(seq, actions),
                           len(seq), actions, observations)


def greedy_sequence(models, prior, actions, observations, length):
    seq, gains = [], []
    for _ in range(length):
        best, best_eig = None, None
        for a in actions:
            cand = seq + [a]
            e = eig_of_sequence(models, prior, cand, actions, observations)
            if best_eig is None or e > best_eig:
                best, best_eig = a, e
        seq.append(best)
        gains.append(best_eig)
    return seq, gains


def best_sequence(models, prior, actions, observations, length):
    best, best_eig = None, None
    for seq in iproduct(actions, repeat=length):
        e = eig_of_sequence(models, prior, list(seq), actions, observations)
        if best_eig is None or e > best_eig:
            best, best_eig = list(seq), e
    return best, best_eig


def first_step_eig(models, prior, actions, observations):
    return {a: eig_of_sequence(models, prior, [a], actions, observations) for a in actions}
