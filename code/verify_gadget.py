"""Numerical verification of the Gidney/Qualtran AND gadget and of the Toffoli decomposition."""
import itertools
import math

import numpy as np

import qcircuit as qc


def report(name, ok, detail=''):
    print(('PASS' if ok else 'FAIL'), name, detail)
    assert ok, name
    return ok


def verify_toffoli_ct():
    U = qc.unitary(3, qc.toffoli_ct(0, 1, 2))
    TOF = np.eye(8, dtype=complex)
    TOF[[6, 7]] = TOF[[7, 6]]                       # |110> <-> |111| with qubit 0 as MSB
    return report('Toffoli 7-T decomposition == CCX exactly (no global phase)',
                  np.allclose(U, TOF), 'max|U-CCX| = %.2e' % np.abs(U - TOF).max())


def verify_and_compute_exact():
    """|a,b,0> -> |a,b,ab> with amplitude exactly +1 (exact, not relative-phase)."""
    U = qc.unitary(3, qc.and_compute_gates(0, 1, 2))
    worst = 0.0
    for a, b in itertools.product((0, 1), repeat=2):
        col = (a << 2) | (b << 1)                   # target = 0
        want = (a << 2) | (b << 1) | (a & b)
        v = U[:, col]
        amp = v[want]
        worst = max(worst, abs(amp - 1.0), float(np.linalg.norm(v) ** 2 - abs(amp) ** 2))
    ok = worst < 1e-12
    report('AND compute is exact on all 4 |a,b,0> inputs (amplitude +1, no relative phase)', ok,
           'max deviation = %.2e' % worst)
    # count gates / T
    gs = qc.and_compute_gates(0, 1, 2)
    from collections import Counter
    c = Counter(g[0] for g in gs)
    report('AND compute gate profile: 13 gates, 2H 2T 2Tdg 6CX 1S, T-count 4',
           len(gs) == 13 and c['h'] == 2 and c['t'] == 2 and c['tdg'] == 2 and c['cx'] == 6 and c['s'] == 1,
           str(dict(c)))
    return True


def verify_and_uncompute():
    """For an arbitrary two-control superposition, both measurement branches must restore
    the exact input state on (a,b) with the ancilla back in |0>."""
    rng = np.random.default_rng(7)
    worst = 0.0
    probs = []
    for trial in range(6):
        amp = rng.normal(size=4) + 1j * rng.normal(size=4)
        amp /= np.linalg.norm(amp)
        psi = np.zeros(8, dtype=complex)
        for i, (a, b) in enumerate(itertools.product((0, 1), repeat=2)):
            psi[(a << 2) | (b << 1)] = amp[i]
        after, _, _ = qc.run_statevector(3, [('and', 0, 1, 2)], psi.copy())
        # check compute produced |a,b,ab>
        want = np.zeros(8, dtype=complex)
        for i, (a, b) in enumerate(itertools.product((0, 1), repeat=2)):
            want[(a << 2) | (b << 1) | (a & b)] = amp[i]
        worst = max(worst, float(np.abs(after - want).max()))
        for outcome in (0, 1):
            st, p, _ = qc.run_statevector(3, [('and_dg', 0, 1, 2)], after.copy(), outcomes={0: outcome})
            probs.append(p)
            worst = max(worst, float(np.abs(st - psi).max()))
    ok = worst < 1e-12
    report('AND uncompute restores the exact input state in BOTH measurement branches', ok,
           'max deviation = %.2e' % worst)
    report('measurement outcomes are 50/50 (probabilities all 0.5)',
           all(abs(p - 0.5) < 1e-12 for p in probs), 'p in [%.3f, %.3f]' % (min(probs), max(probs)))
    gs = qc.and_uncompute_gates(0, 1, 2)
    report('AND uncompute has T-count 0', not any(g[0] in ('t', 'tdg') for g in gs), str(gs))
    return True


def verify_c3x():
    for kind, gates in (('3 Toffoli', qc.c3x_toffoli(0, 1, 2, 3, 4)), ('AND gadget', qc.c3x_gadget(0, 1, 2, 3, 4))):
        bad = []
        for x in range(16):
            init = {i: (x >> i) & 1 for i in range(4)}
            b = qc.simulate(5, gates, init)
            want = ((x >> 3) & 1) ^ (((x >> 0) & 1) & ((x >> 1) & 1) & ((x >> 2) & 1))
            if b[3] != want or b[4] != 0 or any(b[i] != (x >> i) & 1 for i in range(3)):
                bad.append(x)
        report('C^3X via %s: correct on all 16 inputs, ancilla clean' % kind, not bad, str(bad))
    # statevector check of the gadget version on a superposition
    n = 5
    rng = np.random.default_rng(3)
    # qubit 0 is the most significant index; qubit 4 (the ancilla) must start in |0>
    psi = np.zeros(2 ** n, dtype=complex)
    for x in range(16):
        flat = sum(((x >> (3 - i)) & 1) << (n - 1 - i) for i in range(4))   # ancilla bit = 0
        psi[flat] = rng.normal() + 1j * rng.normal()
    psi /= np.linalg.norm(psi)
    ideal, _, _ = qc.run_statevector(n, [('c3x', 0, 1, 2, 3)], psi.copy())
    worst = 0.0
    for outcome in (0, 1):
        st, _, _ = qc.run_statevector(n, qc.c3x_gadget(0, 1, 2, 3, 4), psi.copy(), outcomes={2: outcome})
        worst = max(worst, float(np.abs(st - ideal).max()))
    report('C^3X AND-gadget equals C^3X on a random superposition, both branches',
           worst < 1e-12, 'max deviation = %.2e' % worst)
    return True


def verify_inplace_toffoli_via_and():
    """An in-place Toffoli CCX(a,b -> t), where t is NOT a clean ancilla, can still use the AND
    gadget by borrowing one clean ancilla:  AND(a,b->anc); CNOT(anc->t); AND_dg(a,b->anc).
    T-count 4 instead of 7.  Verified on random superpositions of all three input qubits."""
    n = 4                                            # a, b, t, ancilla
    gadget = [('and', 0, 1, 3), ('cx', 3, 2), ('and_dg', 0, 1, 3)]
    rng = np.random.default_rng(11)
    worst = 0.0
    for trial in range(5):
        psi = np.zeros(2 ** n, dtype=complex)
        for x in range(8):                           # a,b,t arbitrary; ancilla (qubit 3) = |0>
            psi[x << 1] = rng.normal() + 1j * rng.normal()
        psi /= np.linalg.norm(psi)
        ideal, _, _ = qc.run_statevector(n, [('ccx', 0, 1, 2)], psi.copy())
        for outcome in (0, 1):
            st, p, _ = qc.run_statevector(n, gadget, psi.copy(), outcomes={2: outcome})
            worst = max(worst, float(np.abs(st - ideal).max()))
    ok = worst < 1e-12
    report('in-place Toffoli via AND gadget + 1 clean ancilla == CCX exactly (both branches)',
           ok, 'max deviation = %.2e' % worst)
    m_tof = qc.metrics(3, [('ccx', 0, 1, 2)])
    m_and = qc.metrics(4, gadget)
    report('T-count 7 -> 4 and T-depth 4 -> 2 for one Toffoli',
           m_tof['T'] == 7 and m_and['T'] == 4 and m_tof['T_depth'] == 4 and m_and['T_depth'] == 2,
           'textbook: T=%d T-depth=%d | gadget: T=%d T-depth=%d (+1 ancilla, 1 measurement)'
           % (m_tof['T'], m_tof['T_depth'], m_and['T'], m_and['T_depth']))
    return True


def main():
    print('=== Gadget / decomposition verification ===')
    verify_toffoli_ct()
    verify_and_compute_exact()
    verify_and_uncompute()
    verify_c3x()
    verify_inplace_toffoli_via_and()
    print('all gadget checks passed')


if __name__ == '__main__':
    main()
