from fractions import Fraction

from ..rdp import RDP, Policy


WAIT, PROBE = "wait", "probe"
ACTIONS = [WAIT, PROBE]
OBS = ["0", "1"]
HALF = Fraction(1, 2)


def _corridor(n_rooms: int, revealing: bool) -> RDP:
    end = n_rooms - 1

    def step(q, a, o):
        return min(q + 1, end) if a == WAIT else q

    def emit(q, a, o):
        if revealing and a == PROBE and q == end:
            return Fraction(1) if o == "1" else Fraction(0)
        return HALF

    return RDP(
        name=f"corridor-{n_rooms}-{'revealing' if revealing else 'blind'}",
        start=0, step=step, emit=emit,
        states=list(range(n_rooms)), actions=ACTIONS, observations=OBS,
    )


def wait_only() -> Policy:
    return Policy(name="wait-only",
                  prob=lambda h, a: Fraction(1) if a == WAIT else Fraction(0),
                  actions=ACTIONS)


def probing(p: Fraction) -> Policy:
    p = Fraction(p)
    return Policy(name=f"probe-w.p.-{p}",
                  prob=lambda h, a: p if a == PROBE else 1 - p,
                  actions=ACTIONS)


def make_pair(separating_horizon: int):
    if separating_horizon < 1:
        raise ValueError("separating_horizon must be at least 1")
    n_rooms = separating_horizon
    return (_corridor(n_rooms, revealing=True),
            _corridor(n_rooms, revealing=False),
            wait_only(),
            probing(HALF))


def family(horizons):
    return [(h, *make_pair(h)) for h in horizons]
