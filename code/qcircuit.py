"""Circuit container, simulators, gate decompositions (incl. Gidney's AND gadget) and metrics.

Gate encoding: tuples.
    ('x', q) ('h', q) ('s', q) ('sdg', q) ('t', q) ('tdg', q) ('z', q)
    ('cx', c, t) ('cz', a, b) ('ccx', a, b, t) ('c3x', a, b, c, t)
    ('and', a, b, t)      Gidney AND compute   : |a,b,0> -> |a,b,ab>,  T-count 4
    ('and_dg', a, b, t)   Gidney AND uncompute : measurement based,    T-count 0
"""
import math
from collections import Counter

import numpy as np

# ----------------------------------------------------------------------------------- gate sets
SINGLE = {'x', 'h', 's', 'sdg', 't', 'tdg', 'z'}
CLASSICAL = {'x', 'cx', 'ccx', 'c3x', 'and', 'and_dg'}     # act as permutations on basis states


def qubits(g):
    return g[1:]


# ------------------------------------------------------------------- Gidney / Qualtran AND gadget
def and_compute_gates(a, b, t):
    """Qualtran `qualtran.bloqs.mcmt.and_bloq.And` (Gidney 2018) compute decomposition.

    Layers: [H,T] [CX,CX] [CX,CX] [Tdg,Tdg,T] [CX,CX] [H,S].  13 gates, T-count 4.
    """
    return [('h', t), ('t', t),
            ('cx', a, t), ('cx', b, t),
            ('cx', t, a), ('cx', t, b),
            ('tdg', a), ('tdg', b), ('t', t),
            ('cx', t, a), ('cx', t, b),
            ('h', t), ('s', t)]


def and_uncompute_gates(a, b, t):
    """Measurement-based uncompute: H, measure t, classically controlled CZ(a,b).  T-count 0."""
    return [('h', t), ('measure', t), ('cz_if', a, b, t)]


# ---------------------------------------------------------------------- Toffoli decompositions
def toffoli_ct(a, b, t):
    """Textbook 7-T Toffoli (Nielsen & Chuang Fig. 4.9); verified exactly equal to CCX."""
    return [('h', t), ('cx', b, t), ('tdg', t), ('cx', a, t), ('t', t), ('cx', b, t), ('tdg', t),
            ('cx', a, t), ('t', b), ('t', t), ('h', t), ('cx', a, b), ('t', a), ('tdg', b), ('cx', a, b)]


def c3x_toffoli(a, b, c, t, anc):
    """C^3X via one clean ancilla: 3 Toffoli gates."""
    return [('ccx', a, b, anc), ('ccx', anc, c, t), ('ccx', a, b, anc)]


def c3x_gadget(a, b, c, t, anc):
    """C^3X with Gidney AND gadget on the ancilla: T-count 4 + 7 + 0 = 11."""
    return [('and', a, b, anc), ('ccx', anc, c, t), ('and_dg', a, b, anc)]


# ------------------------------------------------------------------------------ circuit object
class Circuit:
    def __init__(self):
        self.n = 0
        self.gates = []
        self.tags = []
        self.regs = {}
        self.cur_tag = 'init'
        self.pending_x = set()

    def reg(self, name, size):
        self.regs[name] = list(range(self.n, self.n + size))
        self.n += size
        return self.regs[name]

    def tag(self, t):
        self.flush_all()
        self.cur_tag = t

    def emit(self, g):
        self.gates.append(g)
        self.tags.append(self.cur_tag)

    # X gates commute through CNOT/Toffoli targets, so they are folded until the qubit is a control
    def flush(self, q):
        if q in self.pending_x:
            self.pending_x.discard(q)
            self.emit(('x', q))

    def flush_all(self):
        for q in sorted(self.pending_x):
            self.emit(('x', q))
        self.pending_x.clear()

    def x(self, q):
        self.pending_x ^= {q}

    def apply(self, g):
        if g[0] == 'x':
            self.x(g[1])
            return
        for q in g[1:-1] if g[0] in ('cx', 'ccx', 'c3x', 'and', 'and_dg') else g[1:]:
            self.flush(q)
        if g[0] in ('and', 'and_dg', 'cz', 'h', 's', 'sdg', 't', 'tdg', 'z'):
            for q in g[1:]:
                self.flush(q)
        self.emit(g)

    def cx(self, c, t):
        self.flush(c)
        self.emit(('cx', c, t))

    def ccx(self, a, b, t):
        self.flush(a)
        self.flush(b)
        self.emit(('ccx', a, b, t))

    def h(self, q):
        self.flush(q)
        self.emit(('h', q))


# --------------------------------------------------------------------------- classical simulator
def simulate(n, gates, init=None, rng=None):
    """Basis-state simulation.  AND gadgets are simulated by their ideal (permutation) action;
    `verify_statevector` is what checks that the gadget really implements that action."""
    b = bytearray(n)
    if init:
        for q, v in init.items():
            b[q] = v
    for g in gates:
        k = g[0]
        if k == 'x':
            b[g[1]] ^= 1
        elif k == 'cx':
            b[g[2]] ^= b[g[1]]
        elif k in ('ccx', 'and'):
            if k == 'and' and b[g[3]]:
                raise ValueError('AND gadget target not |0> at %s' % (g,))
            b[g[3]] ^= b[g[1]] & b[g[2]]
        elif k == 'and_dg':
            if b[g[3]] != (b[g[1]] & b[g[2]]):
                raise ValueError('AND uncompute on inconsistent target at %s' % (g,))
            b[g[3]] = 0
        elif k == 'c3x':
            b[g[4]] ^= b[g[1]] & b[g[2]] & b[g[3]]
        elif k in ('z', 'cz', 'cz_if', 's', 'sdg', 'measure'):
            pass                                   # phase-only / measurement: no basis-state change
        else:
            raise ValueError('non-classical gate in basis-state simulation: %s' % (g,))
    return b


# ------------------------------------------------------------------------- statevector simulator
H_M = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)
T_M = np.diag([1, np.exp(1j * math.pi / 4)])
S_M = np.diag([1, 1j])
X_M = np.array([[0, 1], [1, 0]], dtype=complex)


def _apply1(state, n, q, M):
    state = state.reshape([2] * n)
    state = np.moveaxis(state, q, 0)
    shape = state.shape
    state = state.reshape(2, -1)
    state = M @ state
    state = state.reshape(shape)
    state = np.moveaxis(state, 0, q)
    return state.reshape(-1)


def _apply_controlled(state, n, ctrls, targ, M):
    st = state.reshape([2] * n)
    idx = [slice(None)] * n
    for c in ctrls:
        idx[c] = 1
    sub = st[tuple(idx)]
    sub_moved = np.moveaxis(sub, targ - sum(1 for c in ctrls if c < targ), 0)
    shp = sub_moved.shape
    sub_moved = (M @ sub_moved.reshape(2, -1)).reshape(shp)
    st[tuple(idx)] = np.moveaxis(sub_moved, 0, targ - sum(1 for c in ctrls if c < targ))
    return st.reshape(-1)


def run_statevector(n, gates, state=None, outcomes=None):
    """Full statevector simulation.  Measurements are handled by projecting onto the outcome given
    in `outcomes` (a dict gate-index -> 0/1); returns (state, probability, outcomes_used)."""
    if state is None:
        state = np.zeros(2 ** n, dtype=complex)
        state[0] = 1.0
    prob = 1.0
    used = {}
    for i, g in enumerate(gates):
        k = g[0]
        if k in SINGLE:
            M = {'x': X_M, 'h': H_M, 't': T_M, 'tdg': T_M.conj().T, 's': S_M,
                 'sdg': S_M.conj().T, 'z': np.diag([1, -1]).astype(complex)}[k]
            state = _apply1(state, n, g[1], M)
        elif k == 'cx':
            state = _apply_controlled(state, n, [g[1]], g[2], X_M)
        elif k == 'ccx':
            state = _apply_controlled(state, n, [g[1], g[2]], g[3], X_M)
        elif k == 'c3x':
            state = _apply_controlled(state, n, [g[1], g[2], g[3]], g[4], X_M)
        elif k == 'cz':
            state = _apply_controlled(state, n, [g[1]], g[2], np.diag([1, -1]).astype(complex))
        elif k == 'and':
            for h in and_compute_gates(*g[1:]):
                state = run_statevector(n, [h], state)[0]
        elif k == 'and_dg':
            sub = and_uncompute_gates(*g[1:])
            for h in sub:
                if h[0] == 'measure':
                    q = h[1]
                    st = state.reshape([2] * n)
                    o = outcomes.get(i, 0) if outcomes else 0
                    idx = [slice(None)] * n
                    idx[q] = 1 - o
                    st = st.copy()
                    st[tuple(idx)] = 0
                    p = float(np.vdot(st, st).real)
                    prob *= p
                    used[i] = o
                    state = (st / math.sqrt(p)).reshape(-1) if p > 1e-15 else st.reshape(-1)
                elif h[0] == 'cz_if':
                    if used.get(i, 0) == 1:
                        state = _apply_controlled(state, n, [h[1]], h[2], np.diag([1, -1]).astype(complex))
                        state = _apply1(state, n, h[3], X_M)      # reset: ancilla back to |0>
                else:
                    state = run_statevector(n, [h], state)[0]
        elif k == 'measure':
            q = g[1]
            st = state.reshape([2] * n).copy()
            o = outcomes.get(i, 0) if outcomes else 0
            idx = [slice(None)] * n
            idx[q] = 1 - o
            st[tuple(idx)] = 0
            p = float(np.vdot(st, st).real)
            prob *= p
            used[i] = o
            state = (st / math.sqrt(p)).reshape(-1) if p > 1e-15 else st.reshape(-1)
        elif k == 'cz_if':
            if used.get(i - 1, 0) == 1 or outcomes and outcomes.get(i - 1) == 1:
                state = _apply_controlled(state, n, [g[1]], g[2], np.diag([1, -1]).astype(complex))
        else:
            raise ValueError('unknown gate %s' % (g,))
    return state, prob, used


def unitary(n, gates):
    """Dense unitary of a measurement-free circuit."""
    U = np.zeros((2 ** n, 2 ** n), dtype=complex)
    for j in range(2 ** n):
        v = np.zeros(2 ** n, dtype=complex)
        v[j] = 1
        U[:, j] = run_statevector(n, gates, v)[0]
    return U


# ------------------------------------------------------------------------------------- metrics
def expand_clifford_t(gates, gadget=True):
    """Expand to Clifford+T (+measurement).  With gadget=False, AND gadgets become plain Toffoli."""
    out = []
    for g in gates:
        if g[0] == 'ccx':
            out += toffoli_ct(*g[1:])
        elif g[0] == 'and':
            out += and_compute_gates(*g[1:]) if gadget else toffoli_ct(*g[1:])
        elif g[0] == 'and_dg':
            out += and_uncompute_gates(*g[1:]) if gadget else toffoli_ct(*g[1:])
        elif g[0] == 'c3x':
            raise ValueError('expand c3x before counting')
        else:
            out.append(g)
    return out


def metrics(n, gates, gadget=True):
    """Resource metrics.  Depths come from ASAP scheduling of the explicit gate list."""
    raw = Counter(g[0] for g in gates)
    lvl = [0] * n
    tl = [0] * n
    for g in gates:
        qs = g[1:]
        m = max(lvl[q] for q in qs) + 1
        for q in qs:
            lvl[q] = m
        nl = 1 if g[0] in ('ccx', 'and', 'and_dg') else 0
        m2 = max(tl[q] for q in qs) + nl
        for q in qs:
            tl[q] = m2
    nct_depth = max(lvl) if gates else 0
    and_depth = max(tl) if gates else 0
    ct = expand_clifford_t(gates, gadget=gadget)
    lvl = [0] * n
    tl = [0] * n
    c = Counter()
    for g in ct:
        qs = g[1:]
        m = max(lvl[q] for q in qs) + 1
        for q in qs:
            lvl[q] = m
        m2 = max(tl[q] for q in qs) + (1 if g[0] in ('t', 'tdg') else 0)
        for q in qs:
            tl[q] = m2
        c[g[0]] += 1
    T = c['t'] + c['tdg']
    clifford = c['cx'] + c['h'] + c['x'] + c['s'] + c['sdg'] + c['z'] + c['cz'] + c['cz_if']
    return dict(qubits=n,
                X=raw['x'], CNOT=raw['cx'], Toffoli=raw['ccx'], AND=raw['and'], ANDdg=raw['and_dg'],
                NCT_total=sum(raw[k] for k in ('x', 'cx', 'ccx', 'and', 'and_dg', 'h', 'cz')),
                NCT_depth=nct_depth, Toffoli_depth=and_depth,
                T=T, Clifford=clifford, measurements=c['measure'],
                CT_total=T + clifford + c['measure'],
                CT_depth=max(lvl) if ct else 0, T_depth=max(tl) if ct else 0)


def metrics_by_tag(c, gadget=True):
    groups = {}
    order = []
    for g, t in zip(c.gates, c.tags):
        if t not in groups:
            groups[t] = []
            order.append(t)
        groups[t].append(g)
    return {t: metrics(c.n, groups[t], gadget) for t in order}


def to_qasm(n, gates, path):
    """OpenQASM 2.0.  AND gadgets are written out as their Clifford+T decomposition;
    measurement-based uncompute uses a creg and an `if` on it."""
    lines = ['OPENQASM 2.0;', 'include "qelib1.inc";', 'qreg q[%d];' % n]
    meas = sum(1 for g in gates if g[0] == 'and_dg')
    if meas:
        lines.append('creg m[%d];' % meas)
    mi = 0
    for g in gates:
        k = g[0]
        if k == 'and':
            for h in and_compute_gates(*g[1:]):
                lines.append('%s q[%d],q[%d];' % (h[0], h[1], h[2]) if h[0] == 'cx'
                             else '%s q[%d];' % (h[0], h[1]))
        elif k == 'and_dg':
            a, b, t = g[1:]
            lines += ['h q[%d];' % t, 'measure q[%d] -> m[%d];' % (t, mi),
                      'if(m[%d]==1) cz q[%d],q[%d];' % (mi, a, b),
                      'if(m[%d]==1) x q[%d];' % (mi, t)]
            mi += 1
        elif k in SINGLE:
            lines.append('%s q[%d];' % ({'tdg': 'tdg', 'sdg': 'sdg'}.get(k, k), g[1]))
        elif k == 'cx':
            lines.append('cx q[%d],q[%d];' % (g[1], g[2]))
        elif k == 'cz':
            lines.append('cz q[%d],q[%d];' % (g[1], g[2]))
        elif k == 'ccx':
            lines.append('ccx q[%d],q[%d],q[%d];' % g[1:])
        elif k == 'c3x':
            raise ValueError('expand c3x first')
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
