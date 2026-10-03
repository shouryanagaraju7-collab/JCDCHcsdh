"""Grover key-search oracle and diffusion operator for QARMA-64, with Gidney AND gadgets.

Oracle:  s parallel encryptions sharing one 128-qubit key register -> comparator against the known
ciphertexts -> phase kickback -> reverse encryptions (uncomputation).
Diffusion: H^128 X^128 . C^127 Z . X^128 H^128 on the key register.

Every multi-controlled AND is built as a logarithmic-depth tree of AND gadgets: n-1 AND computes
(4 T each) and n-1 measurement-based uncomputes (0 T).
"""
import math

import numpy as np

import cipher
import qarma_ref as ref
import qcircuit as qc


def and_tree(controls, target, pool, gadget=True):
    """AND of all `controls` into `target` (via CNOT), then uncompute.  Uses len(controls)-1 ancillas."""
    comp = []
    layer = list(controls)
    used = 0
    while len(layer) > 1:
        nxt = []
        for i in range(0, len(layer) - 1, 2):
            a = pool[used]
            used += 1
            comp.append(('and', layer[i], layer[i + 1], a) if gadget else ('ccx', layer[i], layer[i + 1], a))
            nxt.append(a)
        if len(layer) % 2:
            nxt.append(layer[-1])
        layer = nxt
    root = layer[0]
    un = [(('and_dg',) + g[1:]) if gadget else g for g in reversed(comp)]
    return comp + [('cx', root, target)] + un, used


def build_iteration(sb=1, r=7, kind='min', gadget='full', pairs=3, mark='phase',
                    key_value=None, plaintexts=None, with_diffusion=True):
    """One full Grover iteration (oracle + diffusion).

    mark='phase' : standard phase oracle (phase qubit in |->)
    mark='flip'  : the marking bit is written into an ordinary qubit, which makes the oracle
                   classically simulable and therefore testable on basis states.
    """
    use_gadget = gadget in ('pairs', 'full')
    kv = key_value or (ref.TV['w0'], ref.TV['k0'])
    pts = plaintexts or [((ref.TV['P'] + i) & ref.MASK64, ref.TV['T']) for i in range(pairs)]
    c, inst, (w0, k0) = cipher.build_encryption(sb, r, kind, gadget, instances=pts, restore_key=True)
    enc_gates = list(c.gates)
    n_ctrl = 64 * pairs
    pool = c.reg('tree', max(n_ctrl - 1, 127))
    markq = c.reg('mark', 1)[0]
    ciphertexts = [ref.encrypt(P, T, kv[0], kv[1], r, sb) for P, T in pts]

    c.tag('oracle_compare')
    ctrls = []
    for j, I in enumerate(inst):
        C = ciphertexts[j]
        for i in range(16):
            for b in range(4):
                if not (C >> cipher.pos(i, b)) & 1:
                    c.x(I.map[i][b])
                ctrls.append(I.map[i][b])
    if mark == 'phase':
        c.x(markq)
        c.h(markq)
    tree, used = and_tree(ctrls, markq, pool, use_gadget)
    for g in tree:
        c.apply(g)
    if mark == 'phase':
        c.h(markq)
        c.x(markq)
    for j, I in enumerate(inst):
        C = ciphertexts[j]
        for i in range(16):
            for b in range(4):
                if not (C >> cipher.pos(i, b)) & 1:
                    c.x(I.map[i][b])
    c.flush_all()

    c.tag('oracle_uncompute')
    for g in reversed(enc_gates):
        c.emit(invert_gate(g))

    if not with_diffusion:
        return c, inst, (w0, k0), markq, ciphertexts

    c.tag('diffusion')
    keyq = list(w0) + list(k0)
    for q in keyq:
        c.h(q)
    for q in keyq:
        c.x(q)
    c.h(keyq[-1])
    tree2, _ = and_tree(keyq[:-1], keyq[-1], pool, use_gadget)
    for g in tree2:
        c.apply(g)
    c.h(keyq[-1])
    for q in keyq:
        c.x(q)
    for q in keyq:
        c.h(q)
    c.flush_all()
    return c, inst, (w0, k0), markq, ciphertexts


def invert_gate(g):
    """All gates used inside the encryption are self-inverse except the AND gadgets."""
    if g[0] == 'and':
        return ('and_dg',) + g[1:]
    if g[0] == 'and_dg':
        return ('and',) + g[1:]
    if g[0] in ('s', 'sdg', 't', 'tdg'):
        return ({'s': 'sdg', 'sdg': 's', 't': 'tdg', 'tdg': 't'}[g[0]],) + g[1:]
    return g


# --------------------------------------------------------------------------------- verification
def verify_oracle_classical(sb=1, r=7, kind='min', gadget='full', pairs=2, n_wrong=6, seed=1):
    """With mark='flip' the oracle is a permutation of basis states, so it can be checked exactly:
    the marking qubit must flip for the correct key and for no other key, all work registers must
    return to |0>, and the key register must come back unchanged."""
    kv = (ref.TV['w0'], ref.TV['k0'])
    c, inst, (w0, k0), markq, cts = build_iteration(sb, r, kind, gadget, pairs, mark='flip',
                                                    key_value=kv, with_diffusion=False)
    rng = np.random.default_rng(seed)
    results = []
    keys = [kv] + [(int(rng.integers(0, 1 << 63)) * 2 + 1, int(rng.integers(0, 1 << 63)))
                   for _ in range(n_wrong)]
    for idx, (tw0, tk0) in enumerate(keys):
        init = {}
        for k in range(64):
            init[w0[k]] = (tw0 >> k) & 1
            init[k0[k]] = (tk0 >> k) & 1
        bits = qc.simulate(c.n, c.gates, init)
        key_back = all(bits[w0[k]] == (tw0 >> k) & 1 and bits[k0[k]] == (tk0 >> k) & 1 for k in range(64))
        work_clean = all(bits[q] == 0 for I in inst for q in list(I.map[0]) + [])
        states_clean = all(bits[q] == 0 for I in inst for row in I.map for q in row)
        anc_clean = all(bits[q] == 0 for I in inst for q in I.anc) and \
                    all(bits[q] == 0 for q in c.regs['tree'])
        results.append(dict(correct_key=(idx == 0), marked=bool(bits[markq]),
                            key_restored=key_back, states_clean=states_clean, anc_clean=anc_clean))
    ok = (results[0]['marked'] and not any(x['marked'] for x in results[1:])
          and all(x['key_restored'] and x['states_clean'] and x['anc_clean'] for x in results))
    return ok, results, c


def toy_grover_test(nkey=6, marked=37, seed=3, gadget=True):
    """End-to-end statevector Grover on a toy search problem, using exactly the same AND-tree and
    diffusion code paths (including measurement-based uncomputation) as the QARMA oracle."""
    n_tree = max(nkey - 1, nkey - 1)
    c = qc.Circuit()
    key = c.reg('key', nkey)
    pool = c.reg('tree', n_tree)
    markq = c.reg('mark', 1)[0]
    # oracle: phase flip on |marked>
    def oracle():
        for i, q in enumerate(key):
            if not (marked >> i) & 1:
                c.emit(('x', q))
        c.emit(('x', markq))
        c.emit(('h', markq))
        tree, _ = and_tree(key, markq, pool, gadget)
        for g in tree:
            c.emit(g)
        c.emit(('h', markq))
        c.emit(('x', markq))
        for i, q in enumerate(key):
            if not (marked >> i) & 1:
                c.emit(('x', q))

    def diffusion():
        for q in key:
            c.emit(('h', q))
        for q in key:
            c.emit(('x', q))
        c.emit(('h', key[-1]))
        tree, _ = and_tree(key[:-1], key[-1], pool, gadget)
        for g in tree:
            c.emit(g)
        c.emit(('h', key[-1]))
        for q in key:
            c.emit(('x', q))
        for q in key:
            c.emit(('h', q))

    for q in key:
        c.emit(('h', q))
    iters = int(round(math.pi / 4 * math.sqrt(2 ** nkey)))
    for _ in range(iters):
        oracle()
        diffusion()
    rng = np.random.default_rng(seed)
    meas_idx = [i for i, g in enumerate(c.gates) if g[0] == 'and_dg']
    outcomes = {i: int(rng.integers(0, 2)) for i in meas_idx}
    st, _, _ = qc.run_statevector(c.n, c.gates, outcomes=outcomes)
    probs = np.abs(st.reshape([2] * c.n)) ** 2
    # marginal probability of each key value (qubit 0 is the most significant axis)
    axes = tuple(range(nkey, c.n))
    marg = probs.sum(axis=axes)
    flat = marg.reshape(-1)
    want = sum(((marked >> i) & 1) << (nkey - 1 - i) for i in range(nkey))
    return float(flat[want]), iters, len(meas_idx), c.n
