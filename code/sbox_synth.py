"""Turn the minimum-Toffoli decompositions into concrete X/CNOT/Toffoli gate sequences.

The Toffoli count is fixed by the search (and proven minimal there); this step only chooses the
affine layers, which contribute CNOT and NOT gates.  Each generalised Toffoli G = B CCX B^{-1}
may use any conjugator B from its stabiliser coset, so a dynamic program over those choices
minimises (#CNOT, #X) lexicographically.  Linear layers are synthesised with a provably optimal
number of CNOT gates by breadth-first search over GL(4,2).
"""
import itertools
import json
import os
import random
import time
from functools import lru_cache

from qarma_ref import SBOX
from sbox_search import (DATA, ID, apply_lin, build_generators, canon, comp, gl4, inv, parity)

TOF = tuple(x ^ (4 if (x & 1 and x & 2) else 0) for x in range(16))
C3X = tuple(x ^ (8 if (x & 7) == 7 else 0) for x in range(16))


# ------------------------------------------------------- optimal CNOT synthesis over GL(4,2)
def cols2rows(cols):
    return tuple(sum(((cols[j] >> i) & 1) << j for j in range(4)) for i in range(4))


def build_cnot_table():
    I = (1, 2, 4, 8)
    dist = {I: 0}
    par = {I: None}
    fr = [I]
    while fr:
        nf = []
        for R in fr:
            for c in range(4):
                for t in range(4):
                    if c != t:
                        L = list(R)
                        L[t] ^= L[c]
                        L = tuple(L)
                        if L not in dist:
                            dist[L] = dist[R] + 1
                            par[L] = (R, (c, t))
                            nf.append(L)
        fr = nf
    return dist, par


DIST, PAR = build_cnot_table()


def cnot_seq(R):
    out = []
    while PAR[R]:
        R, g = PAR[R]
        out.append(g)
    return out[::-1]


@lru_cache(maxsize=None)
def inv_cols(cols):
    m = {apply_lin(cols, x): x for x in range(16)}
    return tuple(m[1 << j] for j in range(4))


def aff_compose(a, b):
    ca, xa = a
    cb, xb = b
    return (tuple(apply_lin(ca, cb[j]) for j in range(4)), apply_lin(ca, xb) ^ xa)


def aff_inv(a):
    ci = inv_cols(a[0])
    return (ci, apply_lin(ci, a[1]))


def perm_of(a):
    return tuple(apply_lin(a[0], x) ^ a[1] for x in range(16))


@lru_cache(maxsize=None)
def layer_cost(a):
    cols, c = a
    ci = inv_cols(cols)
    return (DIST[cols2rows(cols)], min(bin(c).count('1'), bin(apply_lin(ci, c)).count('1')))


def emit_aff(a):
    cols, c = a
    R = cols2rows(cols)
    ci = inv_cols(cols)
    pre = apply_lin(ci, c)
    xa, xb = bin(c).count('1'), bin(pre).count('1')
    g = []
    if xb < xa:
        g += [('x', i) for i in range(4) if pre >> i & 1]
    g += [('cx', cc, tt) for cc, tt in cnot_seq(R)]
    if xb >= xa:
        g += [('x', i) for i in range(4) if c >> i & 1]
    return g


def c3x_conjugators(tr, gl):
    a, b = tr
    out = []
    for cols in gl:
        for c in range(16):
            if {apply_lin(cols, 7) ^ c, apply_lin(cols, 15) ^ c} == {a, b}:
                out.append((cols, c))
    return out


def build_circuit(sb, seq, tr, gens, G, gl, cap, seed):
    """Emit a concrete gate list for sigma_sb = A o G_seq0 o ... o G_seqk o theta."""
    rng = random.Random(seed)
    P = ID
    for gi in seq:
        P = comp(P, G[gi])
    if tr:
        t = list(range(16))
        t[tr[0]], t[tr[1]] = t[tr[1]], t[tr[0]]
        P = comp(P, tuple(t))
    A = comp(tuple(SBOX[sb]), inv(P))
    c = A[0]
    Aaff = (tuple(A[1 << j] ^ c for j in range(4)), c)
    assert perm_of(Aaff) == A, 'residual map is not affine'
    blocks = []
    if tr:
        Ds = c3x_conjugators(tr, gl)
        rng.shuffle(Ds)
        blocks.append(('c3x', Ds[:cap]))
    for gi in reversed(seq):
        Bs = list(gens[G[gi]])
        rng.shuffle(Bs)
        blocks.append(('ccx', Bs[:cap]))
    costs = [layer_cost(aff_inv(X)) for X in blocks[0][1]]
    ptr = [None]
    for bi in range(1, len(blocks)):
        Xp = blocks[bi - 1][1]
        row, prow = [], []
        for X in blocks[bi][1]:
            Xi = aff_inv(X)
            best, arg = None, None
            for j, Y in enumerate(Xp):
                lc = layer_cost(aff_compose(Xi, Y))
                v = (costs[j][0] + lc[0], costs[j][1] + lc[1])
                if best is None or v < best:
                    best, arg = v, j
            row.append(best)
            prow.append(arg)
        costs = row
        ptr.append(prow)
    last = blocks[-1][1]
    fin = []
    for j, X in enumerate(last):
        lc = layer_cost(aff_compose(Aaff, X))
        fin.append((costs[j][0] + lc[0], costs[j][1] + lc[1]))
    j = min(range(len(fin)), key=lambda k: fin[k])
    ch = [0] * len(blocks)
    ch[-1] = j
    for bi in range(len(blocks) - 1, 0, -1):
        ch[bi - 1] = ptr[bi][ch[bi]]
    X = [blocks[b][1][ch[b]] for b in range(len(blocks))]
    kinds = [b[0] for b in blocks]
    layers = ([aff_inv(X[0])]
              + [aff_compose(aff_inv(X[b]), X[b - 1]) for b in range(1, len(blocks))]
              + [aff_compose(Aaff, X[-1])])
    gates = []
    for li, a in enumerate(layers):
        gates += emit_aff(a)
        if li < len(layers) - 1:
            gates.append(('c3x', 0, 1, 2, 3) if kinds[li] == 'c3x' else ('ccx', 0, 1, 2))
    return fin[j], gates


def sim4(gates, x):
    b = [(x >> i) & 1 for i in range(4)]
    for g in gates:
        if g[0] == 'x':
            b[g[1]] ^= 1
        elif g[0] == 'cx':
            b[g[2]] ^= b[g[1]]
        elif g[0] == 'ccx':
            b[g[3]] ^= b[g[1]] & b[g[2]]
        elif g[0] == 'c3x':
            b[g[4]] ^= b[g[1]] & b[g[2]] & b[g[3]]
    return sum(v << i for i, v in enumerate(b))


def main(nsol=40, cap=256):
    sols = json.load(open(os.path.join(DATA, 'sbox_solutions.json')))
    gens = build_generators()
    G = list(gens)
    gl = gl4()
    out = {}
    t0 = time.time()
    for sb in (0, 1, 2):
        entry = sols['solutions'][str(sb)]
        cand = [(s[0], tuple(s[1]) if s[1] else None) for s in entry['solutions']]
        random.Random(sb).shuffle(cand)
        best = None
        for k, (seq, tr) in enumerate(cand[:nsol]):
            cost, gates = build_circuit(sb, seq, tr, gens, G, gl, cap, k)
            for x in range(16):
                assert sim4(gates, x) == SBOX[sb][x], 'circuit does not implement sigma_%d' % sb
            if best is None or cost < best[0]:
                best = (cost, gates)
        cnot, xg = best[0]
        n_tof = sum(1 for g in best[1] if g[0] == 'ccx')
        n_c3x = sum(1 for g in best[1] if g[0] == 'c3x')
        print('  sigma_%d: %d Toffoli + %d C^3X, %d CNOT, %d X   (%.0fs)'
              % (sb, n_tof, n_c3x, cnot, xg, time.time() - t0), flush=True)
        out[str(sb)] = [list(g) for g in best[1]]
    with open(os.path.join(DATA, 'sbox_circuits.json'), 'w') as f:
        json.dump(out, f, indent=1)
    return out


if __name__ == '__main__':
    print('=== S-box gate-level synthesis ===')
    main()
