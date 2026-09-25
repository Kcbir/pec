import random
from fractions import Fraction
from itertools import product
from typing import Callable, Hashable, Iterable, Sequence


Action = Hashable
Obs = Hashable
State = Hashable
History = tuple[tuple[Action, Obs], ...]


class RDP:
    def __init__(
        self,
        name: str,
        start: State,
        step: Callable[[State, Action, Obs], State],
        emit: Callable[[State, Action, Obs], Fraction],
        states: Iterable[State],
        actions: Iterable[Action],
        observations: Iterable[Obs],
    ) -> None:
        self.name = name
        self.start = start
        self.step = step
        self.emit = emit
        self.states = list(states)
        self.actions = list(actions)
        self.observations = list(observations)
        self._check_normalised()

    def _check_normalised(self) -> None:
        for q, a in product(self.states, self.actions):
            total = sum((self.emit(q, a, o) for o in self.observations), Fraction(0))
            if total != 1:
                raise ValueError(
                    f"{self.name}: emissions from ({q!r}, {a!r}) sum to {total}, not 1"
                )

    def traverse(self, history: History) -> State:
        q = self.start
        for a, o in history:
            q = self.step(q, a, o)
        return q

    def hist_lik(self, history: History) -> Fraction:
        q = self.start
        p = Fraction(1)
        for a, o in history:
            p *= self.emit(q, a, o)
            if p == 0:
                return Fraction(0)
            q = self.step(q, a, o)
        return p


class Policy:
    def __init__(
        self,
        name: str,
        prob: Callable[[History, Action], Fraction],
        actions: Iterable[Action],
    ) -> None:
        self.name = name
        self.prob = prob
        self.actions = list(actions)

    def hist_prob(self, history: History) -> Fraction:
        p = Fraction(1)
        for i, (a, _) in enumerate(history):
            p *= self.prob(history[:i], a)
            if p == 0:
                return Fraction(0)
        return p

    def support(self, history: History) -> bool:
        return self.hist_prob(history) != 0


def path_prob(policy: Policy, model: RDP, history: History) -> Fraction:
    return policy.hist_prob(history) * model.hist_lik(history)


def histories(
    actions: Sequence[Action], observations: Sequence[Obs], length: int
) -> Iterable[History]:
    steps = list(product(actions, observations))
    return product(steps, repeat=length)


def histories_upto(
    actions: Sequence[Action], observations: Sequence[Obs], max_len: int
) -> Iterable[History]:
    for n in range(max_len + 1):
        yield from histories(actions, observations, n)


def sample_trajectories(model: "RDP", policy: "Policy", n_trajectories: int,
                        length: int, seed: int = 0) -> list[History]:
    rng = random.Random(seed)

    def pick(items, probs):
        r = Fraction(rng.randrange(10 ** 9), 10 ** 9)
        acc = Fraction(0)
        for it, p in zip(items, probs):
            acc += p
            if r < acc:
                return it
        return items[-1]

    out = []
    for _ in range(n_trajectories):
        hist: History = ()
        q = model.start
        for _ in range(length):
            a = pick(model.actions, [policy.prob(hist, act) for act in model.actions])
            o = pick(model.observations, [model.emit(q, a, ob) for ob in model.observations])
            hist = hist + ((a, o),)
            q = model.step(q, a, o)
        out.append(hist)
    return out


class Posterior:
    def __init__(self, models: Sequence[RDP], prior: Sequence[Fraction] | None = None):
        if prior is None:
            prior = [Fraction(1, len(models))] * len(models)
        if sum(prior) != 1:
            raise ValueError(f"prior sums to {sum(prior)}, not 1")
        self.models = list(models)
        self.prior = [Fraction(p) for p in prior]

    def weights(self, data: Sequence[History]) -> list[Fraction]:
        w = list(self.prior)
        for h in data:
            w = [wi * m.hist_lik(h) for wi, m in zip(w, self.models)]
        return w

    def posterior(self, data: Sequence[History]) -> list[Fraction]:
        w = self.weights(data)
        total = sum(w, Fraction(0))
        if total == 0:
            raise ValueError("all hypotheses assign this data probability zero")
        return [wi / total for wi in w]

    def odds(self, data: Sequence[History], i: int = 0, j: int = 1) -> Fraction | None:
        w = self.weights(data)
        if w[j] == 0:
            return None
        return w[i] / w[j]
