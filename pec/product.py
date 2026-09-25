from .rdp import RDP, Policy, histories_upto


def disagreeing(m1: RDP, m2: RDP, q1, q2, a, eps=0) -> bool:
    return any(abs(m1.emit(q1, a, o) - m2.emit(q2, a, o)) > eps for o in m1.observations)


def reachable_pairs(m1: RDP, m2: RDP, policy: Policy | None = None):
    start = (m1.start, m2.start)
    seen, frontier = {start}, [start]
    while frontier:
        q1, q2 = frontier.pop()
        for a in m1.actions:
            if policy is not None and _policy_can_take(policy, a):
                pass
            elif policy is not None:
                continue
            for o in m1.observations:
                if policy is not None and m1.emit(q1, a, o) == 0:
                    continue
                nxt = (m1.step(q1, a, o), m2.step(q2, a, o))
                if nxt not in seen:
                    seen.add(nxt)
                    frontier.append(nxt)
    return seen


def _policy_can_take(policy: Policy, a) -> bool:
    return policy.prob((), a) != 0


def policy_is_history_independent(policy: Policy, actions, observations,
                                  max_len: int = 3) -> bool:
    base = {a for a in actions if policy.prob((), a) != 0}
    for h in histories_upto(actions, observations, max_len):
        if {a for a in actions if policy.prob(h, a) != 0} != base:
            return False
    return True


def witness_pair(m1: RDP, m2: RDP, policy: Policy, eps=0):
    start = (m1.start, m2.start)
    seen, frontier, head = {start}, [start], 0
    while head < len(frontier):
        q1, q2 = frontier[head]
        head += 1
        for a in m1.actions:
            if _policy_can_take(policy, a) and disagreeing(m1, m2, q1, q2, a, eps):
                return ((q1, q2), a)
        for a in m1.actions:
            if not _policy_can_take(policy, a):
                continue
            for o in m1.observations:
                if m1.emit(q1, a, o) == 0:
                    continue
                nxt = (m1.step(q1, a, o), m2.step(q2, a, o))
                if nxt not in seen:
                    seen.add(nxt)
                    frontier.append(nxt)
    return None


def pi_equivalent(m1: RDP, m2: RDP, policy: Policy, eps=0) -> bool:
    return witness_pair(m1, m2, policy, eps) is None


def separating_action_depth(m1: RDP, m2: RDP, policy: Policy) -> int | None:
    start = (m1.start, m2.start)
    seen = {start}
    frontier = [start]
    depth = 0
    while frontier:
        for (q1, q2) in frontier:
            for a in m1.actions:
                if _policy_can_take(policy, a) and disagreeing(m1, m2, q1, q2, a):
                    return depth + 1
        nxt_frontier = []
        for (q1, q2) in frontier:
            for a in m1.actions:
                if not _policy_can_take(policy, a):
                    continue
                for o in m1.observations:
                    if m1.emit(q1, a, o) == 0:
                        continue
                    nxt = (m1.step(q1, a, o), m2.step(q2, a, o))
                    if nxt not in seen:
                        seen.add(nxt)
                        nxt_frontier.append(nxt)
        frontier = nxt_frontier
        depth += 1
    return None
