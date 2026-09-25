import random
from fractions import Fraction
from itertools import product as iproduct
from typing import Sequence

from ..structure import (
    Structure, StructurePosterior, canonical_form, enumerate_canonical,
)


ALPHABET = ("0", "1")


def _mk(n_states: int, table: dict) -> Structure:
    delta = {}
    for q, (t0, t1) in table.items():
        delta[(q, "0")] = t0
        delta[(q, "1")] = t1
    return Structure(n_states, ALPHABET, delta)


TOMITA = {
    1: _mk(2, {0: (1, 0), 1: (1, 1)}),
    2: _mk(3, {0: (2, 1), 1: (0, 2), 2: (2, 2)}),
    4: _mk(4, {0: (1, 0), 1: (2, 0), 2: (3, 0), 3: (3, 3)}),
    5: _mk(4, {0: (2, 1), 1: (3, 0), 2: (0, 3), 3: (1, 2)}),
    6: _mk(3, {0: (1, 2), 1: (2, 0), 2: (0, 1)}),
}


def _tomita3_ok(w: str) -> bool:
    i, n = 0, len(w)
    while i < n:
        if w[i] == "1":
            j = i
            while j < n and w[j] == "1":
                j += 1
            ones = j - i
            k = j
            while k < n and w[k] == "0":
                k += 1
            zeros = k - j
            if ones % 2 == 1 and zeros % 2 == 1:
                return False
            i = k if k > j else j
        else:
            i += 1
    return True


def _tomita7_ok(w: str) -> bool:
    phase = 0
    want = "0101"
    for ch in w:
        while phase < 4 and ch != want[phase]:
            phase += 1
        if phase == 4:
            return False
    return True


def dfa_from_predicate(pred, alphabet=ALPHABET, prefix_len: int = 9,
                       suffix_len: int = 9) -> Structure:

    def residual(u):
        sig = []
        for m in range(suffix_len + 1):
            for suf in iproduct(alphabet, repeat=m):
                sig.append(pred(u + "".join(suf)))
        return tuple(sig)

    seen, order = {residual(""): 0}, [""]
    m = 0
    while m < len(order):
        u = order[m]
        for a in alphabet:
            v = u + a
            if len(v) > prefix_len:
                continue
            r = residual(v)
            if r not in seen:
                seen[r] = len(order)
                order.append(v)
        m += 1

    idx = {residual(u): i for i, u in enumerate(order)}
    delta = {}
    for i, u in enumerate(order):
        for a in alphabet:
            delta[(i, a)] = idx[residual(u + a)]
    return Structure(len(order), alphabet, delta), [pred(u) for u in order]


def verify_dfa_matches(struct: Structure, accepting, pred, alphabet=ALPHABET,
                       max_len: int = 12) -> bool:
    for n in range(max_len + 1):
        for w in iproduct(alphabet, repeat=n):
            s = "".join(w)
            if accepting[struct.run(s)] != pred(s):
                return False
    return True


_t3, _t3_acc = dfa_from_predicate(_tomita3_ok)
_t7, _t7_acc = dfa_from_predicate(_tomita7_ok)
TOMITA[3] = _t3
TOMITA[7] = _t7
TOMITA_ACCEPTING = {3: _t3_acc, 7: _t7_acc}
TOMITA_PREDICATES = {3: _tomita3_ok, 7: _tomita7_ok}

TOMITA_TOO_BIG = {}


def distinct_emissions(n_states: int) -> dict:
    return {
        q: {"0": Fraction(q + 1, n_states + 1), "1": Fraction(n_states - q, n_states + 1)}
        for q in range(n_states)
    }


def generate(struct: Structure, emissions: dict, n_seqs: int, length: int,
             seed: int = 0) -> list[list[str]]:
    rng = random.Random(seed)
    data = []
    for _ in range(n_seqs):
        q, seq = 0, []
        for _ in range(length):
            p0 = emissions[q]["0"]
            sym = "0" if rng.random() < float(p0) else "1"
            seq.append(sym)
            q = struct.step(q, sym)
        data.append(seq)
    return data


def recover(n_states: int, data: Sequence[Sequence[str]], alpha: int = 1):
    structs = list(enumerate_canonical(n_states, ALPHABET))
    post = StructurePosterior(structs, alpha=alpha)
    best, mass = post.map_structure(data)
    return best, mass, len(structs)


def recovers_truth(truth: Structure, data: Sequence[Sequence[str]], alpha: int = 1):
    best, mass, n = recover(truth.n_states, data, alpha)
    return canonical_form(best) == canonical_form(truth), mass, n
