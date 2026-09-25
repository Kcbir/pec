from fractions import Fraction
from itertools import product
from math import factorial
from typing import Iterable, Sequence


class Structure:
    __slots__ = ("n_states", "alphabet", "delta")

    def __init__(self, n_states: int, alphabet: Sequence, delta: dict):
        self.n_states = n_states
        self.alphabet = tuple(alphabet)
        self.delta = dict(delta)

    def step(self, q: int, sym) -> int:
        return self.delta[(q, sym)]

    def run(self, seq: Sequence) -> int:
        q = 0
        for sym in seq:
            q = self.delta[(q, sym)]
        return q

    def counts(self, data: Iterable[Sequence]) -> dict:
        c = {q: {a: 0 for a in self.alphabet} for q in range(self.n_states)}
        for seq in data:
            q = 0
            for sym in seq:
                c[q][sym] += 1
                q = self.delta[(q, sym)]
        return c

    def reachable(self) -> set:
        seen, frontier = {0}, [0]
        while frontier:
            q = frontier.pop()
            for a in self.alphabet:
                r = self.delta[(q, a)]
                if r not in seen:
                    seen.add(r)
                    frontier.append(r)
        return seen

    def key(self) -> tuple:
        return (self.n_states, tuple(sorted(self.delta.items(), key=lambda kv: (kv[0][0], str(kv[0][1])))))

    def __repr__(self) -> str:
        rows = []
        for q in range(self.n_states):
            rows.append(f"{q}:" + ",".join(f"{a}->{self.delta[(q, a)]}" for a in self.alphabet))
        return f"Structure({self.n_states} states; " + " | ".join(rows) + ")"


def _beta_ratio(counts: dict, alpha: int, k: int) -> Fraction:
    num = 1
    for c in counts.values():
        num *= factorial(c + alpha - 1)
    num *= factorial(k * alpha - 1)
    den = (factorial(alpha - 1) ** k) * factorial(sum(counts.values()) + k * alpha - 1)
    return Fraction(num, den)


def marginal_likelihood(struct: Structure, data: Iterable[Sequence], alpha: int = 1) -> Fraction:
    if alpha < 1:
        raise ValueError("alpha must be a positive integer for exact arithmetic")
    k = len(struct.alphabet)
    total = Fraction(1)
    for q, c in struct.counts(data).items():
        total *= _beta_ratio(c, alpha, k)
    return total


def canonical_form(struct: Structure) -> tuple:
    order = {0: 0}
    queue = [0]
    nxt = 1
    while queue:
        q = queue.pop(0)
        for a in struct.alphabet:
            r = struct.delta[(q, a)]
            if r not in order:
                order[r] = nxt
                nxt += 1
                queue.append(r)
    return tuple(
        (order[q], a, order[struct.delta[(q, a)]])
        for q in sorted(order, key=lambda x: order[x])
        for a in struct.alphabet
    )


def enumerate_structures(n_states: int, alphabet: Sequence, connected: bool = True):
    alphabet = tuple(alphabet)
    slots = [(q, a) for q in range(n_states) for a in alphabet]
    seen = set()
    for targets in product(range(n_states), repeat=len(slots)):
        delta = dict(zip(slots, targets))
        s = Structure(n_states, alphabet, delta)
        if connected and len(s.reachable()) != n_states:
            continue
        c = canonical_form(s)
        if c in seen:
            continue
        seen.add(c)
        yield s


def enumerate_canonical(n_states: int, alphabet: Sequence):
    alphabet = tuple(alphabet)
    slots = [(q, a) for q in range(n_states) for a in alphabet]

    def rec(i, m, delta):
        if i == len(slots):
            if m == n_states - 1:
                yield Structure(n_states, alphabet, dict(delta))
            return
        q, a = slots[i]
        if q > m:
            return
        top = min(m + 1, n_states - 1)
        for t in range(top + 1):
            delta[(q, a)] = t
            yield from rec(i + 1, max(m, t), delta)
        delta.pop((q, a), None)

    yield from rec(0, 0, {})


class StructurePosterior:
    def __init__(self, structures: Sequence[Structure], alpha: int = 1,
                 prior: Sequence[Fraction] | None = None):
        self.structures = list(structures)
        self.alpha = alpha
        n = len(self.structures)
        if prior is None:
            prior = [Fraction(1, n)] * n
        if sum(prior) != 1:
            raise ValueError(f"prior sums to {sum(prior)}, not 1")
        self.prior = [Fraction(p) for p in prior]

    def weights(self, data: Sequence[Sequence]) -> list[Fraction]:
        return [
            p * marginal_likelihood(s, data, self.alpha)
            for p, s in zip(self.prior, self.structures)
        ]

    def posterior(self, data: Sequence[Sequence]) -> list[Fraction]:
        w = self.weights(data)
        total = sum(w, Fraction(0))
        if total == 0:
            raise ValueError("every structure assigns this data probability zero")
        return [wi / total for wi in w]

    def map_structure(self, data: Sequence[Sequence]) -> tuple[Structure, Fraction]:
        post = self.posterior(data)
        i = max(range(len(post)), key=lambda j: post[j])
        return self.structures[i], post[i]
