"""In-place CNOT circuit for one MixColumns column of QARMA-64.

The 16x16 binary matrix is not written down by hand: it is PROBED from the verified reference
implementation (qarma_ref.mixcolumns) on unit inputs, so the circuit cannot drift from the spec.
Synthesis reduces M to a permutation matrix by randomised greedy row reduction; the leftover
permutation is free, because it is absorbed into the logical-to-physical qubit map.
"""
import json
import os
import random
import time

from qarma_ref import cells, mixcolumns, uncells
from sbox_search import DATA

N = 16          # wires of one column: w = 4*row + bit


def probe_matrix():
    """rows[w] = bitmask of input wires that XOR into output wire w."""
    rows = [0] * N
    for w_in in range(N):
        row, bit = divmod(w_in, 4)
        st = [0] * 16
        st[4 * row + 0] = 1 << bit                      # column 0, cell `row`, bit `bit`
        out = cells(mixcolumns(uncells(st)))
        for w_out in range(N):
            r2, b2 = divmod(w_out, 4)
            if (out[4 * r2 + 0] >> b2) & 1:
                rows[w_out] |= 1 << w_in
    return rows


def is_perm(A):
    return all(bin(a).count('1') == 1 for a in A) and len(set(A)) == N


def synth(M, rng, temp):
    A = list(M)
    ops = []
    while not is_perm(A):
        cands = []
        for c in range(N):
            for t in range(N):
                if c != t:
                    d = bin(A[t] ^ A[c]).count('1') - bin(A[t]).count('1')
                    cands.append((d, c, t))
        md = min(x[0] for x in cands)
        pool = ([x for x in cands if x[0] == md] if (md < 0 or rng.random() > temp)
                else [x for x in cands if x[0] <= md + 1])
        d, c, t = rng.choice(pool)
        A[t] ^= A[c]
        ops.append((c, t))
        if len(ops) > 120:
            return None
    return ops, A


def verify(cnots, relabel, trials=500, seed=1):
    """Check the circuit against the reference MixColumns on random columns."""
    rng = random.Random(seed)
    for _ in range(trials):
        col = [rng.randrange(16) for _ in range(4)]
        bits = [(col[w // 4] >> (w % 4)) & 1 for w in range(N)]
        v = [bits[relabel[i]] for i in range(N)]
        for c, t in cnots:
            v[t] ^= v[c]
        got = [sum(v[4 * r + b] << b for b in range(4)) for r in range(4)]
        st = [0] * 16
        for r in range(4):
            st[4 * r] = col[r]
        want = cells(mixcolumns(uncells(st)))
        want = [want[4 * r] for r in range(4)]
        if got != want:
            return False, (col, got, want)
    return True, None


def main(iters=40000, seed=1):
    M = probe_matrix()
    assert all(bin(r).count('1') == 3 for r in M), 'each output bit must be a XOR of 3 input bits'
    # M must be involutory
    prod = [0] * N
    for i in range(N):
        acc = 0
        for j in range(N):
            if M[i] >> j & 1:
                acc ^= M[j]
        prod[i] = acc
    assert prod == [1 << i for i in range(N)], 'M^2 must be the identity'
    rng = random.Random(seed)
    best = None
    t0 = time.time()
    for it in range(iters):
        res = synth(M, rng, 0.02 if it % 2 else 0.0)
        if res and (best is None or len(res[0]) < len(best[0])):
            best = res
    ops, A = best
    cnots = [list(o) for o in reversed(ops)]            # M = E_1 ... E_k P  =>  apply P then E_k..E_1
    relabel = [a.bit_length() - 1 for a in A]
    ok, info = verify(cnots, relabel)
    assert ok, info
    out = dict(cnots=cnots, relabel=relabel, matrix=M)
    with open(os.path.join(DATA, 'mixcolumns.json'), 'w') as f:
        json.dump(out, f)
    print('  MixColumns column: %d CNOT, verified on 500 random columns (%.0fs, %d restarts)'
          % (len(cnots), time.time() - t0, iters))
    return out


if __name__ == '__main__':
    print('=== MixColumns synthesis ===')
    main()
