from fractions import Fraction
from itertools import product
from typing import Iterable, Sequence

from .structure import _beta_ratio


class RDPStructure:
    __slots__ = ("n_states", "actions", "observations", "delta")

    def __init__(self, n_states: int, actions: Sequence, observations: Sequence,
                 delta: dict):
        self.n_states = n_states
        self.actions = tuple(actions)
        self.observations = tuple(observations)
        self.delta = dict(delta)

    def traverse(self, history) -> int:
        q = 0
        for _, o in history:
            q = self.delta[(q, o)]
        return q

    def counts(self, data: Iterable) -> dict:
        c = {(q, a): {o: 0 for o in self.observations}
             for q in range(self.n_states) for a in self.actions}
        for hist in data:
            q = 0
            for a, o in hist:
                c[(q, a)][o] += 1
                q = self.delta[(q, o)]
        return c

    def reachable(self) -> set:
        seen, frontier = {0}, [0]
        while frontier:
            q = frontier.pop()
            for o in self.observations:
                r = self.delta[(q, o)]
                if r not in seen:
                    seen.add(r)
                    frontier.append(r)
        return seen

    def distinguishes_first_observation(self) -> bool:
        targets = {self.delta[(0, o)] for o in self.observations}
        return len(targets) > 1

    def canonical(self) -> tuple:
        order, queue, nxt = {0: 0}, [0], 1
        while queue:
            q = queue.pop(0)
            for o in self.observations:
                r = self.delta[(q, o)]
                if r not in order:
                    order[r] = nxt
                    nxt += 1
                    queue.append(r)
        return tuple(
            (order[q], o, order[self.delta[(q, o)]])
            for q in sorted(order, key=lambda x: order[x])
            for o in self.observations
        )

    def __repr__(self) -> str:
        body = " | ".join(
            f"{q}:" + ",".join(f"{o}->{self.delta[(q, o)]}" for o in self.observations)
            for q in range(self.n_states)
        )
        return f"RDPStructure({self.n_states}st; {body})"


def marginal_likelihood(struct: RDPStructure, data, alpha: int = 1) -> Fraction:
    k = len(struct.observations)
    total = Fraction(1)
    for c in struct.counts(data).values():
        total *= _beta_ratio(c, alpha, k)
    return total


def enumerate_structures(max_states: int, actions: Sequence, observations: Sequence):
    seen = set()
    for n in range(1, max_states + 1):
        slots = [(q, o) for q in range(n) for o in observations]
        for targets in product(range(n), repeat=len(slots)):
            s = RDPStructure(n, actions, observations, dict(zip(slots, targets)))
            if len(s.reachable()) != n:
                continue
            key = (n, s.canonical())
            if key in seen:
                continue
            seen.add(key)
            yield s


class StructurePrior:
    def __init__(self, structures: Sequence[RDPStructure], predicate=None,
                 epsilon: Fraction = Fraction(0)):
        self.structures = list(structures)
        n = len(self.structures)
        uniform = [Fraction(1, n)] * n

        if predicate is None:
            self.weights = uniform
            return

        flagged = [i for i, s in enumerate(self.structures) if predicate(s)]
        if not flagged:
            raise ValueError("the language predicate matches no structure")
        lang = [Fraction(0)] * n
        for i in flagged:
            lang[i] = Fraction(1, len(flagged))
        eps = Fraction(epsilon)
        self.weights = [(1 - eps) * l + eps * u for l, u in zip(lang, uniform)]

    def posterior(self, data, alpha: int = 1) -> list[Fraction]:
        w = [p * marginal_likelihood(s, data, alpha)
             for p, s in zip(self.weights, self.structures)]
        total = sum(w, Fraction(0))
        if total == 0:
            raise ValueError("every structure assigns this data probability zero")
        return [wi / total for wi in w]

    def mass_on(self, key: tuple, data, alpha: int = 1) -> Fraction:
        post = self.posterior(data, alpha)
        return sum(
            (p for p, s in zip(post, self.structures)
             if (s.n_states, s.canonical()) == key),
            Fraction(0),
        )
