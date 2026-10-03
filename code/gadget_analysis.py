"""Find Toffoli pairs that are genuine clean compute/uncompute pairs, replace them by Gidney AND
gadgets, and verify the result -- both by exhaustive basis-state simulation and, in every
measurement branch, by statevector simulation.

Nothing here is decided by inspection: a pair is only called gadget-eligible if the simulation
over ALL inputs shows (i) the target is |0> before the compute, (ii) the two controls keep their
values until the uncompute, (iii) the target is untouched as a target in between, and
(iv) the target returns to |0>.
"""
import itertools
import os
import re

import numpy as np

import qcircuit as qc

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_PARENT = os.path.join(os.path.dirname(HERE), 'revkit')      # RevKit baseline circuits


def parse_qasm(path):
    gates = []
    for line in open(path):
        m = re.match(r'(x|cx|ccx)\s+(.*);', line.strip())
        if m:
            qs = tuple(int(a) for a in re.findall(r'q\[(\d+)\]', m.group(2)))
            gates.append((m.group(1),) + qs)
    return gates


def load_revkit():
    out = {}
    for k in range(4):
        p = os.path.join(PROJECT_PARENT, 'qarma_sigma%d.qasm' % k)
        if os.path.exists(p):
            out[k] = parse_qasm(p)
    return out


def targets_of(g):
    return {g[-1]} if g[0] in ('cx', 'ccx', 'c3x', 'and') else ({g[1]} if g[0] == 'x' else set())


def trace_states(n, gates, n_data):
    """states[i][x] = tuple of wire values just before gate i, for data input x."""
    states = []
    cur = []
    for x in range(2 ** n_data):
        b = bytearray(n)
        for i in range(n_data):
            b[i] = (x >> i) & 1
        cur.append(b)
    for g in gates:
        states.append([bytes(b) for b in cur])
        for b in cur:
            k = g[0]
            if k == 'x':
                b[g[1]] ^= 1
            elif k == 'cx':
                b[g[2]] ^= b[g[1]]
            elif k == 'ccx':
                b[g[3]] ^= b[g[1]] & b[g[2]]
            else:
                raise ValueError(k)
    states.append([bytes(b) for b in cur])
    return states


def find_and_pairs(n, gates, n_data):
    """Return (pairs, rejected) where pairs = [(i, j, a, b, t)] are verified clean AND pairs."""
    st = trace_states(n, gates, n_data)
    pairs, rejected, used = [], [], set()
    for i, g in enumerate(gates):
        if g[0] != 'ccx' or i in used:
            continue
        a, b, t = g[1], g[2], g[3]
        if any(s[t] != 0 for s in st[i]):
            rejected.append((i, None, 'target not |0> before compute'))
            continue
        j = None
        for jj in range(i + 1, len(gates)):
            if t in targets_of(gates[jj]):
                j = jj
                break
        if j is None:
            rejected.append((i, None, 'no gate ever clears the target'))
            continue
        if j in used or gates[j][0] != 'ccx' or {gates[j][1], gates[j][2]} != {a, b} or gates[j][3] != t:
            rejected.append((i, j, 'next gate touching the target is not the matching uncompute'))
            continue
        if any(s[a] != s2[a] or s[b] != s2[b] for s, s2 in zip(st[i], st[j])):
            rejected.append((i, j, 'a control changes value between compute and uncompute'))
            continue
        if any(s[t] != (s[a] & s[b]) for s in st[j]):
            rejected.append((i, j, 'target does not hold a AND b at the uncompute'))
            continue
        if any(s[t] != 0 for s in st[j + 1]):
            rejected.append((i, j, 'target not |0> after uncompute'))
            continue
        pairs.append((i, j, a, b, t))
        used.add(i)
        used.add(j)
    return pairs, rejected


def apply_pairs(gates, pairs):
    out = list(gates)
    for (i, j, a, b, t) in pairs:
        out[i] = ('and', a, b, t)
        out[j] = ('and_dg', a, b, t)
    return out


def gadgetise_inplace(gates, anc=None):
    """Replace EVERY remaining plain Toffoli by  AND(a,b->anc); CX(anc->t); AND_dg(a,b->anc),
    using one borrowed clean ancilla (reused sequentially).  T-count 7 -> 4 per Toffoli.

    The borrow wire must be a FRESH qubit: a wire that the base circuit already uses may hold
    garbage (or be an AND target inside an open compute/uncompute window), which would make the
    gadget incorrect.  By default we therefore take one wire above everything the circuit touches.
    """
    if anc is None:
        anc = 1 + max((max(g[1:]) for g in gates), default=3)
    assert all(anc not in g[1:] for g in gates), 'borrow ancilla must not be used by the circuit'
    out = []
    for g in gates:
        if g[0] == 'ccx':
            a, b, t = g[1], g[2], g[3]
            out += [('and', a, b, anc), ('cx', anc, t), ('and_dg', a, b, anc)]
        else:
            out.append(g)
    return out


def verify_classical(n, gates, table, n_data=4, ancillas=()):
    """Check the circuit computes `table` on all inputs and returns every ancilla to 0."""
    bad = []
    for x in range(2 ** n_data):
        init = {i: (x >> i) & 1 for i in range(n_data)}
        b = qc.simulate(n, gates, init)
        y = sum(b[i] << i for i in range(n_data))
        if y != table[x] or any(b[q] for q in ancillas):
            bad.append((x, y, table[x], [b[q] for q in ancillas]))
    return bad


def verify_statevector_all_branches(n, gates, table, n_data=4, max_branches=64, seed=5):
    """For every combination of measurement outcomes, check that the circuit acts on a random
    superposition exactly as the permutation `table` (ancillas returned to |0>)."""
    meas_idx = [i for i, g in enumerate(gates) if g[0] == 'and_dg']
    if len(meas_idx) > 6:
        combos = [tuple(int(x) for x in np.random.default_rng(seed).integers(0, 2, len(meas_idx)))
                  for _ in range(max_branches)]
        combos = list(dict.fromkeys(combos + [(0,) * len(meas_idx), (1,) * len(meas_idx)]))
    else:
        combos = list(itertools.product((0, 1), repeat=len(meas_idx)))
    rng = np.random.default_rng(seed)
    psi = np.zeros(2 ** n, dtype=complex)
    for x in range(2 ** n_data):
        flat = sum(((x >> i) & 1) << (n - 1 - i) for i in range(n_data))     # ancillas = |0>
        psi[flat] = rng.normal() + 1j * rng.normal()
    psi /= np.linalg.norm(psi)
    ideal = np.zeros(2 ** n, dtype=complex)
    for x in range(2 ** n_data):
        src = sum(((x >> i) & 1) << (n - 1 - i) for i in range(n_data))
        dst = sum(((table[x] >> i) & 1) << (n - 1 - i) for i in range(n_data))
        ideal[dst] = psi[src]
    worst = 0.0
    for combo in combos:
        outcomes = dict(zip(meas_idx, combo))
        st, p, _ = qc.run_statevector(n, gates, psi.copy(), outcomes=outcomes)
        worst = max(worst, float(np.abs(st - ideal).max()))
    return worst, len(combos)


def summarise(name, n, gates, table, n_data=4, ancillas=()):
    m_plain = qc.metrics(n, gates, gadget=True)
    bad = verify_classical(n, gates, table, n_data, ancillas)
    worst, nbranch = verify_statevector_all_branches(n, gates, table, n_data)
    return dict(name=name, qubits=n, metrics=m_plain, classical_ok=not bad,
                statevector_dev=worst, branches=nbranch, ok=(not bad and worst < 1e-10))
