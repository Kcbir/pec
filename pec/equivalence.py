import random
from fractions import Fraction
from itertools import product

from .rdp import RDP, Policy, histories_upto


def disagreement_set(m1: RDP, m2: RDP, actions, observations, max_len: int):
    return [
        h for h in histories_upto(actions, observations, max_len)
        if m1.hist_lik(h) != m2.hist_lik(h)
    ]


def obs_equiv_on(m1: RDP, m2: RDP, policy: Policy, actions, observations,
                 max_len: int) -> bool:
    for h in histories_upto(actions, observations, max_len):
        if policy.hist_prob(h) != 0 and m1.hist_lik(h) != m2.hist_lik(h):
            return False
    return True


def measure_under(histories, policy: Policy, model: RDP) -> Fraction:
    return sum(
        (policy.hist_prob(h) * model.hist_lik(h) for h in histories),
        Fraction(0),
    )


def readings(m1: RDP, m2: RDP, policy: Policy, actions, observations, max_len: int):
    D = disagreement_set(m1, m2, actions, observations, max_len)
    mu1 = measure_under(D, policy, m1)
    mu2 = measure_under(D, policy, m2)
    return {
        "D_size": len(D),
        "A": mu1 == 0,
        "B": mu1 == 0 and mu2 == 0,
        "C": obs_equiv_on(m1, m2, policy, actions, observations, max_len),
        "mu_M": mu1,
        "mu_Mprime": mu2,
        "D": D,
    }


def random_rdp(name: str, n_states: int, actions, observations, rng: random.Random,
               denom: int = 4, allow_zeros: bool = True) -> RDP:
    states = list(range(n_states))
    emit_table = {}
    for q, a in product(states, actions):
        k = len(observations)
        while True:
            cuts = sorted(rng.randint(0, denom) for _ in range(k - 1))
            parts = []
            prev = 0
            for c in cuts:
                parts.append(c - prev)
                prev = c
            parts.append(denom - prev)
            if allow_zeros or all(p > 0 for p in parts):
                break
        for o, p in zip(observations, parts):
            emit_table[(q, a, o)] = Fraction(p, denom)

    step_table = {
        (q, a, o): rng.randrange(n_states)
        for q, a, o in product(states, actions, observations)
    }
    return RDP(
        name=name,
        start=0,
        step=lambda q, a, o: step_table[(q, a, o)],
        emit=lambda q, a, o: emit_table[(q, a, o)],
        states=states,
        actions=actions,
        observations=observations,
    )


def restricted_policy(allowed, actions) -> Policy:
    allowed = list(allowed)
    p = Fraction(1, len(allowed))
    return Policy(
        name=f"uniform-on-{allowed}",
        prob=lambda h, a: p if a in allowed else Fraction(0),
        actions=actions,
    )


def matched_pair(name1: str, name2: str, n_states: int, actions, observations,
                 allowed, rng: random.Random, denom: int = 4) -> tuple[RDP, RDP]:
    states = list(range(n_states))
    allowed = set(allowed)

    def random_emit_row():
        k = len(observations)
        cuts = sorted(rng.randint(0, denom) for _ in range(k - 1))
        parts, prev = [], 0
        for c in cuts:
            parts.append(c - prev)
            prev = c
        parts.append(denom - prev)
        return parts

    shared_emit, shared_step = {}, {}
    alt_emit, alt_step = {}, {}
    for q, a in product(states, actions):
        parts = random_emit_row()
        for o, p in zip(observations, parts):
            shared_emit[(q, a, o)] = Fraction(p, denom)
        if a in allowed:
            for o in observations:
                alt_emit[(q, a, o)] = shared_emit[(q, a, o)]
        else:
            parts2 = random_emit_row()
            for o, p in zip(observations, parts2):
                alt_emit[(q, a, o)] = Fraction(p, denom)

    for q, a, o in product(states, actions, observations):
        shared_step[(q, a, o)] = rng.randrange(n_states)
        alt_step[(q, a, o)] = (
            shared_step[(q, a, o)] if a in allowed else rng.randrange(n_states)
        )

    def build(nm, et, st):
        return RDP(
            name=nm, start=0,
            step=lambda q, a, o: st[(q, a, o)],
            emit=lambda q, a, o: et[(q, a, o)],
            states=states, actions=actions, observations=observations,
        )

    return build(name1, shared_emit, shared_step), build(name2, alt_emit, alt_step)


def explain_normalisation_argument(actions, observations):
    a = actions[0]
    x, y = observations[0], observations[1]

    def mk(name, py):
        table = {}
        for act in actions:
            for o in observations:
                table[(act, o)] = py if o == y else 1 - py
        return RDP(
            name=name, start=0,
            step=lambda q, act, o: 0,
            emit=lambda q, act, o: table[(act, o)],
            states=[0], actions=actions, observations=observations,
        )

    m_zero, m_half = mk("m_zero", Fraction(0)), mk("m_half", Fraction(1, 2))
    return m_zero, m_half, ((a, y),), ((a, x),)
