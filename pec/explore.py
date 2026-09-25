import math
from fractions import Fraction
from typing import Sequence

from .rdp import RDP, Policy, histories


def entropy(dist: Sequence[Fraction]) -> float:
    h = 0.0
    for p in dist:
        if p > 0:
            f = float(p)
            h -= f * math.log(f)
    return h


def _normalise(weights: Sequence[Fraction]) -> list[Fraction] | None:
    total = sum(weights, Fraction(0))
    if total == 0:
        return None
    return [w / total for w in weights]


def expected_information_gain(models: Sequence[RDP], prior: Sequence[Fraction],
                              policy: Policy, horizon: int,
                              actions: Sequence, observations: Sequence) -> float:
    h_prior = entropy(prior)
    expected_post = 0.0
    for h in histories(actions, observations, horizon):
        joint = [w * policy.hist_prob(h) * m.hist_lik(h) for w, m in zip(prior, models)]
        p_h = sum(joint, Fraction(0))
        if p_h == 0:
            continue
        post = _normalise(joint)
        expected_post += float(p_h) * entropy(post)
    return h_prior - expected_post


def separating_mass(models: Sequence[RDP], prior: Sequence[Fraction], policy: Policy,
                    horizon: int, actions: Sequence,
                    observations: Sequence) -> Fraction:
    total = Fraction(0)
    for h in histories(actions, observations, horizon):
        liks = [m.hist_lik(h) for m in models]
        if len(set(liks)) == 1:
            continue
        p_h = sum((w * policy.hist_prob(h) * lk for w, lk in zip(prior, liks)), Fraction(0))
        total += p_h
    return total


def reachability(model: RDP, policy: Policy, target, horizon: int,
                 actions: Sequence, observations: Sequence) -> Fraction:
    total = Fraction(0)
    for h in histories(actions, observations, horizon):
        if model.traverse(h) != target:
            continue
        total += policy.hist_prob(h) * model.hist_lik(h)
    return total


def best_by_information_gain(models, prior, candidates, horizon, actions, observations):
    scored = [(expected_information_gain(models, prior, p, horizon, actions, observations), p)
              for p in candidates]
    scored.sort(key=lambda t: -t[0])
    return scored


def best_by_reachability(model, candidates, targets, horizon, actions, observations):
    scored = []
    for p in candidates:
        worst = min(reachability(model, p, t, horizon, actions, observations)
                    for t in targets)
        scored.append((worst, p))
    scored.sort(key=lambda t: -t[0])
    return scored
