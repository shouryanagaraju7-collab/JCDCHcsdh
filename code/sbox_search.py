"""Exhaustive minimum-Toffoli synthesis for the QARMA-64 S-boxes.

Method.  An ancilla-free NCT circuit with t Toffoli gates on four wires is
    A_t CCX A_{t-1} ... A_1 CCX A_0         (A_i affine, i.e. CNOT/NOT only).
Inserting B B^{-1} rewrites it as   A o G_1 o ... o G_t   where each
G_i = B CCX B^{-1} is a *generalised Toffoli* and A is affine.  Hence the Toffoli count of a
permutation pi is the least t such that the LEFT COSET AGL(4,2) o pi contains a product of t
generalised Toffoli gates.  Cosets are canonicalised, BFS runs from the identity coset, and a
second search from the target meets it in the middle, which makes the result exhaustive.

Odd permutations (sigma_0, sigma_2) have no ancilla-free NCT circuit at all (Lemma: every NCT
gate on 4 wires is an even permutation), so they are searched as  pi = A o G_1 ... G_t o theta
with theta an affine conjugate of C^3X, i.e. an arbitrary transposition.

Run:  python sbox_search.py            (writes ../data/sbox_solutions.json, prints a report)
"""
import itertools
import json
import os
import random
import sys
import time

from qarma_ref import SBOX

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), 'data')

ID = tuple(range(16))


def comp(p, q):
    """p after q."""
    return tuple(p[q[i]] for i in range(16))


def inv(p):
    o = [0] * 16
    for i, v in enumerate(p):
        o[v] = i
    return tuple(o)


def parity(p):
    seen = [0] * 16
    par = 0
    for i in range(16):
        if not seen[i]:
            j, l = i, 0
            while not seen[j]:
                seen[j] = 1
                j = p[j]
                l += 1
            par ^= (l - 1) & 1
    return par


def apply_lin(cols, x):
    v = 0
    for i in range(4):
        if x >> i & 1:
            v ^= cols[i]
    return v


def gl4():
    out = []
    for c in itertools.product(range(1, 16), repeat=4):
        span = {0}
        ok = True
        for col in c:
            if col in span:
                ok = False
                break
            span |= {s ^ col for s in span}
        if ok:
            out.append(c)
    return out


def affine_perm(cols, c):
    return tuple(apply_lin(cols, x) ^ c for x in range(16))


def canon(p):
    """Canonical representative of the left coset AGL(4,2) o p."""
    c0 = p[0]
    q = [v ^ c0 for v in p]
    imgs = []
    span = {0}
    for x in range(1, 16):
        if q[x] not in span:
            imgs.append(q[x])
            span |= {s ^ q[x] for s in span}
            if len(imgs) == 4:
                break
    lut = {}
    for m in range(16):
        v = 0
        for j in range(4):
            if m >> j & 1:
                v ^= imgs[j]
        lut[v] = m
    r = 0
    for i in range(16):
        r |= lut[q[i]] << (4 * i)
    return r


def unpack(k):
    return tuple((k >> (4 * i)) & 15 for i in range(16))


TOF = tuple(x ^ (4 if (x & 1 and x & 2) else 0) for x in range(16))     # controls 0,1 -> target 2
C3X = tuple(x ^ (8 if (x & 7) == 7 else 0) for x in range(16))


def build_generators():
    """All B o CCX o B^{-1}, together with every conjugator B that produces them."""
    gens = {}
    for cols in gl4():
        lin_inv = {apply_lin(cols, x): x for x in range(16)}
        for c in range(16):
            B = tuple(apply_lin(cols, x) ^ c for x in range(16))
            Binv = tuple(lin_inv[y ^ c] for y in range(16))
            gens.setdefault(comp(B, comp(TOF, Binv)), []).append((cols, c))
    return gens


# ------------------------------------------------------------------------------- unit tests
def unit_tests(gens):
    rnd = random.Random(2024)
    G = list(gens)
    checks = []
    checks.append(('|AGL(4,2)| = 322560', len(gl4()) * 16 == 322560))
    checks.append(('number of generalised Toffoli gates = 420', len(G) == 420))
    checks.append(('each has a stabiliser of order 768', all(len(v) == 768 for v in gens.values())))
    checks.append(('each generalised Toffoli is an involution', all(comp(g, g) == ID for g in G)))
    checks.append(('each is an even permutation', all(parity(g) == 0 for g in G)))
    # canonical form must be invariant under LEFT multiplication by an affine map, and only that
    cols_list = gl4()
    ok_inv = True
    for _ in range(300):
        p = tuple(rnd.sample(range(16), 16))
        A = affine_perm(rnd.choice(cols_list), rnd.randrange(16))
        if canon(comp(A, p)) != canon(p):
            ok_inv = False
            break
    checks.append(('canon() is invariant under left affine multiplication (300 random tests)', ok_inv))
    ok_sep = True
    for _ in range(300):
        p = tuple(rnd.sample(range(16), 16))
        q = tuple(rnd.sample(range(16), 16))
        same_coset = canon(p) == canon(q)
        # p, q in the same left coset  <=>  p o q^{-1} affine
        r = comp(p, inv(q))
        c = r[0]
        is_aff = affine_perm(tuple(r[1 << j] ^ c for j in range(4)), c) == r if len(
            {r[1 << j] ^ c for j in range(4)}) == 4 else False
        if same_coset != is_aff:
            ok_sep = False
            break
    checks.append(('canon() separates distinct cosets (300 random pairs)', ok_sep))
    checks.append(('sigma_1 is even; sigma_0 and sigma_2 are odd',
                   parity(tuple(SBOX[1])) == 0 and parity(tuple(SBOX[0])) == 1 and parity(tuple(SBOX[2])) == 1))
    return checks


# ------------------------------------------------------------------------------- the search
def build_levels(G, max_level=3, verbose=True):
    L = {canon(ID): (0, None, None)}
    frontier = [canon(ID)]
    sizes = []
    t0 = time.time()
    for lev in range(1, max_level + 1):
        nf = []
        for kr in frontier:
            r = unpack(kr)
            for gi, T in enumerate(G):
                k = canon(comp(r, T))
                if k not in L:
                    L[k] = (lev, kr, gi)
                    nf.append(k)
        frontier = nf
        sizes.append(len(nf))
        if verbose:
            print('    level %d: %d new cosets (%.1fs)' % (lev, len(nf), time.time() - t0), flush=True)
    return L, sizes


def chain(L, k):
    out = []
    while L[k][0] > 0:
        lev, kp, gi = L[k]
        out.append(gi)
        k = kp
    return out[::-1]


def search_sbox(sb, G, L, max_level=3, verbose=True):
    """Return (min_count, solutions, certified_bound).  Each solution is (gate_indices, transposition)."""
    target = tuple(SBOX[sb])
    odd = parity(target) == 1
    if odd:
        starts = []
        for a in range(16):
            for b in range(a + 1, 16):
                tr = list(range(16))
                tr[a], tr[b] = tr[b], tr[a]
                starts.append((comp(target, tuple(tr)), (a, b)))
    else:
        starts = [(target, None)]
    layer = [(p, [], tr) for p, tr in starts]
    seen = set()
    best = None
    for d in range(0, 4):
        hits = []
        for p, gl, tr in layer:
            k = canon(p)
            if k in L:
                hits.append((L[k][0] + d, p, gl, tr, k))
        if hits:
            m = min(h[0] for h in hits)
            if best is None or m < best[0][0]:
                best = [h for h in hits if h[0] == m]
        # after finishing depth d, every solution with total <= d + max_level has been seen
        if best is not None and best[0][0] <= d + max_level + 1:
            break
        nl = []
        for p, gl, tr in layer:
            for gi, T in enumerate(G):
                q = comp(p, T)
                k = canon(q)
                if k in seen:
                    continue
                seen.add(k)
                nl.append((q, gl + [gi], tr))
        layer = nl
        if verbose:
            print('    sigma_%d: target-side depth %d -> %d cosets' % (sb, d + 1, len(layer)), flush=True)
    sols = []
    for tot, p, gl, tr, k in best:
        sols.append((chain(L, k) + gl[::-1], tr))
    return best[0][0], sols


def known_answer_test(G, L, max_level=3):
    """Random products of k generalised Toffoli gates must be found with count <= k."""
    rnd = random.Random(99)
    results = []
    for k in (1, 2, 3, 4):
        p = ID
        idx = [rnd.randrange(len(G)) for _ in range(k)]
        for gi in idx:
            p = comp(p, G[gi])
        found = None
        layer = [(p, [])]
        seen = set()
        for d in range(0, 3):
            for q, gl in layer:
                kk = canon(q)
                if kk in L:
                    found = L[kk][0] + d
                    break
            if found is not None:
                break
            nl = []
            for q, gl in layer:
                for gi, T in enumerate(G):
                    r = comp(q, T)
                    kk = canon(r)
                    if kk in seen:
                        continue
                    seen.add(kk)
                    nl.append((r, gl + [gi]))
            layer = nl
        results.append((k, found, found is not None and found <= k))
    return results


def main():
    os.makedirs(DATA, exist_ok=True)
    print('=== S-box minimum-Toffoli search ===')
    t0 = time.time()
    gens = build_generators()
    G = list(gens)
    print('  unit tests:')
    for name, ok in unit_tests(gens):
        print('   ', 'PASS' if ok else 'FAIL', name)
        assert ok, name
    print('  building coset levels from the identity:')
    L, sizes = build_levels(G, 3)
    print('  known-answer tests (random products of k generalised Toffoli gates):')
    for k, found, ok in known_answer_test(G, L):
        print('   ', 'PASS' if ok else 'FAIL', 'k=%d -> found count %s' % (k, found))
        assert ok
    out = {'levels': sizes, 'generators': len(G), 'solutions': {}}
    for sb in (0, 1, 2):
        cnt, sols = search_sbox(sb, G, L)
        kind = 'Toffoli only' if parity(tuple(SBOX[sb])) == 0 else 'Toffoli + one C^3X'
        print('  sigma_%d: minimum count = %d (%s), %d optimal decompositions' % (sb, cnt, kind, len(sols)))
        out['solutions'][str(sb)] = dict(count=cnt, odd=parity(tuple(SBOX[sb])) == 1,
                                         n_solutions=len(sols),
                                         solutions=[[s[0], list(s[1]) if s[1] else None] for s in sols[:200]])
    out['generalised_toffolis'] = [list(g) for g in G]
    with open(os.path.join(DATA, 'sbox_solutions.json'), 'w') as f:
        json.dump(out, f)
    print('  total time %.1fs' % (time.time() - t0))
    return out


if __name__ == '__main__':
    main()
