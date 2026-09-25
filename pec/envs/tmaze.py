from fractions import Fraction

from ..rdp import RDP, Policy


HALF = Fraction(1, 2)
WAIT, RUN = "wait", "run"
LEFT, RIGHT = "left", "right"
INIT, SAW_LEFT, SAW_RIGHT = "init", "sawLeft", "sawRight"

ACTIONS = [WAIT, RUN]
OBSERVATIONS = [LEFT, RIGHT]


def _cue_step(q, a, o):
    if q == INIT:
        return SAW_LEFT if o == LEFT else SAW_RIGHT
    return q


def _cue_emit(q, a, o):
    if q == INIT or a == WAIT:
        return HALF
    remembered = LEFT if q == SAW_LEFT else RIGHT
    return Fraction(1) if o == remembered else Fraction(0)


cue_model = RDP(
    name="cue",
    start=INIT,
    step=_cue_step,
    emit=_cue_emit,
    states=[INIT, SAW_LEFT, SAW_RIGHT],
    actions=ACTIONS,
    observations=OBSERVATIONS,
)

blind_model = RDP(
    name="blind",
    start=None,
    step=lambda q, a, o: None,
    emit=lambda q, a, o: HALF,
    states=[None],
    actions=ACTIONS,
    observations=OBSERVATIONS,
)

wait_policy = Policy(
    name="always-wait",
    prob=lambda h, a: Fraction(1) if a == WAIT else Fraction(0),
    actions=ACTIONS,
)

run_policy = Policy(
    name="always-run",
    prob=lambda h, a: Fraction(1) if a == RUN else Fraction(0),
    actions=ACTIONS,
)


def mixed_policy(p_run: Fraction) -> Policy:
    p_run = Fraction(p_run)
    if not (0 <= p_run <= 1):
        raise ValueError("p_run must be in [0, 1]")
    return Policy(
        name=f"run-w.p.-{p_run}",
        prob=lambda h, a: p_run if a == RUN else 1 - p_run,
        actions=ACTIONS,
    )


def all_wait(history) -> bool:
    return all(a == WAIT for a, _ in history)


SEPARATING_HISTORY = ((WAIT, LEFT), (RUN, RIGHT))
