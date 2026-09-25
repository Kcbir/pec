from fractions import Fraction

from ..rdp import RDP, Policy


HALF = Fraction(1, 2)
NO_REWARD, REWARD = 0, 1
REWARDS = [NO_REWARD, REWARD]


def _bernoulli(p: Fraction, o) -> Fraction:
    return p if o == REWARD else 1 - p


ROT_ARMS = ["arm0", "arm1", "arm2"]


def _rot_prob(q: int, a: str) -> Fraction:
    if a == "arm0":
        return HALF
    if a == "arm1":
        return Fraction(3, 4) if q == 0 else Fraction(1, 4)
    return Fraction(1, 4) if q == 0 else Fraction(3, 4)


rotating_mab = RDP(
    name="rotating-mab",
    start=0,
    step=lambda q, a, o: (1 - q) if o == REWARD else q,
    emit=lambda q, a, o: _bernoulli(_rot_prob(q, a), o),
    states=[0, 1],
    actions=ROT_ARMS,
    observations=REWARDS,
)

static_mab = RDP(
    name="static-mab",
    start=0,
    step=lambda q, a, o: 0,
    emit=lambda q, a, o: _bernoulli(_rot_prob(0, a), o),
    states=[0],
    actions=ROT_ARMS,
    observations=REWARDS,
)

arm0_policy = Policy(
    name="pull-arm0-only",
    prob=lambda h, a: Fraction(1) if a == "arm0" else Fraction(0),
    actions=ROT_ARMS,
)


def rotating_mab_deviation(p: Fraction) -> Policy:
    p = Fraction(p)
    return Policy(
        name=f"arm1-w.p.-{p}",
        prob=lambda h, a: p if a == "arm1" else (1 - p if a == "arm0" else Fraction(0)),
        actions=ROT_ARMS,
    )


CHEAT_ARMS = ["arm0", "arm1", "arm2"]
CHEAT_BASE = Fraction(1, 4)
CHEAT_UNLOCKED = Fraction(3, 4)
CHEAT_SEQUENCE = ("arm1", "arm2")


def _cheat_step(q: int, a: str, o) -> int:
    if q == 2:
        return 2
    if q == 0:
        return 1 if a == CHEAT_SEQUENCE[0] else 0
    if a == CHEAT_SEQUENCE[1]:
        return 2
    return 1 if a == CHEAT_SEQUENCE[0] else 0


cheat_mab = RDP(
    name="cheat-mab",
    start=0,
    step=_cheat_step,
    emit=lambda q, a, o: _bernoulli(CHEAT_UNLOCKED if q == 2 else CHEAT_BASE, o),
    states=[0, 1, 2],
    actions=CHEAT_ARMS,
    observations=REWARDS,
)

plain_mab = RDP(
    name="plain-mab",
    start=0,
    step=lambda q, a, o: 0,
    emit=lambda q, a, o: _bernoulli(CHEAT_BASE, o),
    states=[0],
    actions=CHEAT_ARMS,
    observations=REWARDS,
)

no_cheat_policy = Policy(
    name="never-play-arm2",
    prob=lambda h, a: HALF if a in ("arm0", "arm1") else Fraction(0),
    actions=CHEAT_ARMS,
)


def cheat_deviation(p: Fraction) -> Policy:
    p = Fraction(p)
    rest = (1 - p) / 2
    return Policy(
        name=f"arm2-w.p.-{p}",
        prob=lambda h, a: p if a == "arm2" else rest,
        actions=CHEAT_ARMS,
    )


MAZE_ACTIONS = ["stay", "move"]
MAZE_OBS = ["L", "R"]


def _maze_emit(orientation: int, o) -> Fraction:
    p_left = Fraction(3, 4) if orientation == 0 else Fraction(1, 4)
    return p_left if o == "L" else 1 - p_left


def _maze_step(q: int, a: str, o) -> int:
    if a != "move":
        return q
    orientation, counter = divmod(q, 3)
    counter = (counter + 1) % 3
    if counter == 0:
        orientation = 1 - orientation
    return orientation * 3 + counter


rotating_maze = RDP(
    name="rotating-maze",
    start=0,
    step=_maze_step,
    emit=lambda q, a, o: _maze_emit(q // 3, o),
    states=list(range(6)),
    actions=MAZE_ACTIONS,
    observations=MAZE_OBS,
)

fixed_maze = RDP(
    name="fixed-maze",
    start=0,
    step=lambda q, a, o: (q + 1) % 3 if a == "move" else q,
    emit=lambda q, a, o: _maze_emit(0, o),
    states=[0, 1, 2],
    actions=MAZE_ACTIONS,
    observations=MAZE_OBS,
)

stay_policy = Policy(
    name="never-move",
    prob=lambda h, a: Fraction(1) if a == "stay" else Fraction(0),
    actions=MAZE_ACTIONS,
)


def maze_deviation(p: Fraction) -> Policy:
    p = Fraction(p)
    return Policy(
        name=f"move-w.p.-{p}",
        prob=lambda h, a: p if a == "move" else 1 - p,
        actions=MAZE_ACTIONS,
    )


SUITE = [
    {
        "name": "Rotating MAB",
        "true": rotating_mab, "rival": static_mab,
        "policy": arm0_policy, "deviation": rotating_mab_deviation(Fraction(1, 2)),
        "actions": ROT_ARMS, "observations": REWARDS,
        "deviation_fn": rotating_mab_deviation,
        "targets": [0, 1],
        "kind": "action-in-visited-state",
    },
    {
        "name": "Cheat MAB",
        "true": cheat_mab, "rival": plain_mab,
        "policy": no_cheat_policy, "deviation": cheat_deviation(Fraction(1, 3)),
        "actions": CHEAT_ARMS, "observations": REWARDS,
        "deviation_fn": cheat_deviation,
        "targets": [0, 1, 2],
        "kind": "unvisited-state",
    },
    {
        "name": "Rotating Maze",
        "true": rotating_maze, "rival": fixed_maze,
        "policy": stay_policy, "deviation": maze_deviation(Fraction(1, 2)),
        "actions": MAZE_ACTIONS, "observations": MAZE_OBS,
        "deviation_fn": maze_deviation,
        "targets": [0, 1, 2, 3, 4, 5],
        "kind": "unvisited-state",
    },
]
