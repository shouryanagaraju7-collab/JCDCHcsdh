"""Gate-level quantum circuit for QARMA-64 encryption.

Three T-cost modes (`gadget`):
    'none'   every Toffoli via the textbook 7-T decomposition        (no extra ancillas)
    'pairs'  only *verified* clean compute/uncompute pairs become AND gadgets
    'full'   every Toffoli uses an AND gadget, borrowing one clean ancilla per cell (4 T each)

Wire permutations (ShuffleCells tau, the tweak permutation h, the rotation inside the tweak LFSR,
the rotation in the key specialisation o, and the MixColumns output permutation) are realised by
relabelling the logical-to-physical qubit map and cost no gates.
"""
import json
import os

import qarma_ref as ref
import qcircuit as qc
from gadget_analysis import apply_pairs, find_and_pairs, gadgetise_inplace, load_revkit
from sbox_search import DATA


def pos(cell, bit):
    """Integer bit position of (cell, bit) inside the 64-bit state."""
    return 60 - 4 * cell + bit


def load_sbox_circuits():
    with open(os.path.join(DATA, 'sbox_circuits.json')) as f:
        return {int(k): [tuple(g) for g in v] for k, v in json.load(f).items()}


def load_mix():
    with open(os.path.join(DATA, 'mixcolumns.json')) as f:
        return json.load(f)


SB_MIN = load_sbox_circuits()
MIX = load_mix()
SB_REV = load_revkit()


def expand_c3x(gates, anc, gadget):
    out = []
    for g in gates:
        if g[0] == 'c3x':
            a, b, c, t = g[1:]
            out += (qc.c3x_gadget(a, b, c, t, anc) if gadget in ('pairs', 'full')
                    else qc.c3x_toffoli(a, b, c, t, anc))
        else:
            out.append(g)
    return out


def sbox_gates(kind, sb, gadget):
    """Return (forward, inverse, n_anc) on wires 0..3 with wire 4 = borrowed/clean ancilla."""
    if kind == 'min':
        base = SB_MIN[sb]
        fwd = expand_c3x(base, 4, gadget)            # wire 4 = C^3X ancilla (sigma_0, sigma_2 only)
        if gadget == 'full':
            fwd = gadgetise_inplace(fwd)             # borrow wire = one above everything used
        inv = fwd if sb in (0, 1) else list(reversed(fwd))
        if sb == 2 and gadget in ('pairs', 'full'):
            # reversing turns AND/AND_dg around: swap them back
            inv = [(('and' if g[0] == 'and_dg' else 'and_dg'),) + g[1:] if g[0] in ('and', 'and_dg')
                   else g for g in inv]
        n_anc = max(0, max(max(g[1:]) for g in fwd + inv) - 3)
    else:
        base = SB_REV[sb]
        if gadget == 'none':
            fwd = list(base)
        else:
            pairs, _ = find_and_pairs(5, base, 4)
            fwd = apply_pairs(base, pairs)
            if gadget == 'full':
                fwd = gadgetise_inplace(fwd)             # fresh borrow wire above wire 4
        if sb in (0, 1):
            inv = fwd
        else:
            rbase = SB_REV[3]
            if gadget == 'none':
                inv = list(rbase)
            else:
                pairs, _ = find_and_pairs(5, rbase, 4)
                inv = apply_pairs(rbase, pairs)
                if gadget == 'full':
                    inv = gadgetise_inplace(inv)
        n_anc = max(0, max(max(g[1:]) for g in fwd + inv) - 3)
    return fwd, inv, n_anc


class Instance:
    def __init__(self, circ, idx, P, T, quantum_tweak, n_anc):
        st = circ.reg('state%d' % idx, 64)
        self.map = [[st[pos(i, j)] for j in range(4)] for i in range(16)]
        self.P, self.T = P, T
        self.anc = circ.reg('anc%d' % idx, 16 * n_anc) if n_anc else []
        self.n_anc = n_anc
        self.tw = None
        if quantum_tweak:
            tq = circ.reg('tweak%d' % idx, 64)
            self.tw = [[tq[pos(i, j)] for j in range(4)] for i in range(16)]

    def cell_wires(self, i):
        return self.map[i] + [self.anc[self.n_anc * i + k] for k in range(self.n_anc)]


def build_encryption(sb=1, r=7, kind='min', gadget='full', instances=None, quantum_tweak=False,
                     restore_key=True, circ=None, key=None):
    if instances is None:
        instances = [(ref.TV['P'], ref.TV['T'])]
    fwd_s, inv_s, n_anc = sbox_gates(kind, sb, gadget)
    c = circ or qc.Circuit()
    if key is None:
        w0 = c.reg('w0', 64)
        k0 = c.reg('k0', 64)
    else:
        w0, k0 = key
    inst = [Instance(c, j, P, T, quantum_tweak, n_anc) for j, (P, T) in enumerate(instances)]
    wmap = list(w0)

    def add_key(I, keymap):
        for i in range(16):
            for j in range(4):
                c.cx(keymap[pos(i, j)], I.map[i][j])

    def add_const(I, const):
        for i in range(16):
            for j in range(4):
                if (const >> pos(i, j)) & 1:
                    c.x(I.map[i][j])

    def add_tweak(I, Tclass, const):
        if I.tw is None:
            add_const(I, Tclass ^ const)
        else:
            for i in range(16):
                for j in range(4):
                    c.cx(I.tw[i][j], I.map[i][j])
            add_const(I, const)

    def shuffle_cells(I, p):
        I.map = [I.map[p[i]] for i in range(16)]

    def mix(I):
        for col in range(4):
            wires = [I.map[4 * row + col][b] for row in range(4) for b in range(4)]
            v = [wires[MIX['relabel'][w]] for w in range(16)]
            for cc, tt in MIX['cnots']:
                c.cx(v[cc], v[tt])
            for row in range(4):
                I.map[4 * row + col] = [v[4 * row + b] for b in range(4)]

    def sub(I, gates):
        for i in range(16):
            w = I.cell_wires(i)
            for g in gates:
                c.apply((g[0],) + tuple(w[q] for q in g[1:]))

    def tweak_update(I, forward):
        if I.tw is None:
            return
        if forward:
            I.tw = [I.tw[ref.H_PERM[i]] for i in range(16)]
            for i in ref.LFSR_CELLS:
                b = I.tw[i]
                c.cx(b[1], b[0])
                I.tw[i] = [b[1], b[2], b[3], b[0]]
        else:
            for i in ref.LFSR_CELLS:
                b = I.tw[i]
                I.tw[i] = [b[3], b[0], b[1], b[2]]
                c.cx(I.tw[i][1], I.tw[i][0])
            I.tw = [I.tw[ref.H_INV[i]] for i in range(16)]

    def ortho(forward):
        nonlocal wmap
        if forward:
            c.cx(wmap[63], wmap[1])
            wmap = [wmap[(k + 1) % 64] for k in range(64)]
        else:
            wmap = [wmap[(k - 1) % 64] for k in range(64)]
            c.cx(wmap[63], wmap[1])

    Ts = []
    for I in inst:
        seq = [I.T]
        for _ in range(r):
            seq.append(ref.tweak_fwd(seq[-1]))
        Ts.append(seq)

    c.tag('whitening_in')
    for I in inst:
        add_const(I, I.P)
        add_key(I, wmap)
    for i in range(r):
        c.tag('F%d' % i)
        for j, I in enumerate(inst):
            add_key(I, k0)
            add_tweak(I, Ts[j][i], ref.RC[i])
            if i:
                shuffle_cells(I, ref.TAU)
                mix(I)
            sub(I, fwd_s)
        for I in inst:
            tweak_update(I, True)
    c.tag('central_fwd')
    ortho(True)
    for j, I in enumerate(inst):
        add_key(I, wmap)
        add_tweak(I, Ts[j][r], 0)
        shuffle_cells(I, ref.TAU)
        mix(I)
        sub(I, fwd_s)
    c.tag('reflector')
    for I in inst:
        shuffle_cells(I, ref.TAU)
        mix(I)
        add_key(I, k0)
        shuffle_cells(I, ref.TAU_INV)
    c.tag('central_bwd')
    ortho(False)
    for j, I in enumerate(inst):
        sub(I, inv_s)
        mix(I)
        shuffle_cells(I, ref.TAU_INV)
        add_key(I, wmap)
        add_tweak(I, Ts[j][r], 0)
    for i in reversed(range(r)):
        c.tag('B%d' % i)
        for I in inst:
            tweak_update(I, False)
        for j, I in enumerate(inst):
            sub(I, inv_s)
            if i:
                mix(I)
                shuffle_cells(I, ref.TAU_INV)
            add_key(I, k0)
            add_tweak(I, Ts[j][i], ref.RC[i] ^ ref.ALPHA)
    c.tag('whitening_out')
    ortho(True)
    for I in inst:
        add_key(I, wmap)
    if restore_key:
        ortho(False)
    c.flush_all()
    return c, inst, (w0, k0)


def read_state(bits, I):
    return sum(bits[I.map[i][j]] << pos(i, j) for i in range(16) for j in range(4))


def verify_encryption(sb, r, kind, gadget, quantum_tweak=False):
    """End-to-end check against the published test vector."""
    c, inst, (w0, k0) = build_encryption(sb, r, kind, gadget, quantum_tweak=quantum_tweak)
    init = {}
    for k in range(64):
        init[w0[k]] = (ref.TV['w0'] >> k) & 1
        init[k0[k]] = (ref.TV['k0'] >> k) & 1
    I = inst[0]
    if I.tw:
        for i in range(16):
            for j in range(4):
                init[I.tw[i][j]] = (ref.TV['T'] >> pos(i, j)) & 1
    bits = qc.simulate(c.n, c.gates, init)
    out = read_state(bits, I)
    want = ref.TV_CIPHER[(sb, r)]
    key_ok = all(bits[w0[k]] == (ref.TV['w0'] >> k) & 1 and bits[k0[k]] == (ref.TV['k0'] >> k) & 1
                 for k in range(64))
    anc_ok = all(bits[q] == 0 for q in I.anc)
    tw_ok = True
    if I.tw:
        tw_ok = sum(bits[I.tw[i][j]] << pos(i, j) for i in range(16) for j in range(4)) == ref.TV['T']
    return dict(circuit='%016x' % out, published='%016x' % want,
                ok=(out == want and key_ok and anc_ok and tw_ok),
                key_restored=key_ok, ancilla_clean=anc_ok, tweak_restored=tw_ok,
                metrics=qc.metrics(c.n, c.gates))
