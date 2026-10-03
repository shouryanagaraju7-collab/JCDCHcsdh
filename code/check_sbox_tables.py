"""Check the S-box gate tables printed in the paper.

1. Every LIGHTER-R circuit in ../lighter_r/*.c implements its QARMA-64 S-box
   (including the free input/output wire relabelling written in the C file).
2. Every minimum-Toffoli circuit in ../data/sbox_circuits.json implements its S-box, and the
   reversed sigma_2 circuit implements sigma_2^{-1}.
3. The sigma_2^{-1} rows of the S-box table (three compilation modes) are recomputed from the
   reversed circuit and checked on all inputs and in every measurement branch.

Run:  python check_sbox_tables.py
"""
import json
import os
import re

import qcircuit as qc
from gadget_analysis import gadgetise_inplace, verify_classical, verify_statevector_all_branches
from qarma_ref import SBOX

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
NAMES = {0: 'sigma_0', 1: 'sigma_1', 2: 'sigma_2', 3: 'sigma_2^-1'}
FILES = {0: 'implementation_sigma0_3.c', 1: 'implementation_sigma1_0.c',
         2: 'implementation_sigma2_1.c', 3: 'implementation_sigma3_0.c'}


def check_lighter_r():
    for sb, fn in FILES.items():
        src = open(os.path.join(ROOT, 'lighter_r', fn)).read()
        inmap = {int(a): int(b) for a, b in re.findall(r'F\[(\d)\] = X\[(\d)\];', src)}
        outmap = {int(a): int(b) for a, b in re.findall(r'X\[(\d)\] = F\[(\d)\];', src)}
        ops = re.findall(r'F\[(\d)\] = (\w+)\(([^;]*)\);', src)
        ok = True
        for v in range(16):
            X = [(v >> (3 - i)) & 1 for i in range(4)]        # X[0] is the most significant bit
            F = [X[inmap[i]] for i in range(4)]
            for tgt, op, args in ops:
                a = [int(t) for t in re.findall(r'F\[(\d)\]', args)]
                t = int(tgt)
                if op == 'CNOT1':
                    F[t] = F[a[0]] ^ F[a[1]]
                elif op == 'CCNOT2':
                    F[t] = F[a[2]] ^ (F[a[0]] & F[a[1]])
                elif op == 'CCCNOT2':
                    F[t] = F[a[3]] ^ (F[a[0]] & F[a[1]] & F[a[2]])
                elif op == 'RNOT1':
                    F[t] = F[a[0]] ^ 1
                else:
                    raise ValueError(op)
            Y = [F[outmap[i]] for i in range(4)]
            ok &= sum(Y[i] << (3 - i) for i in range(4)) == SBOX[sb][v]
        print('[%s] LIGHTER-R %-27s implements %s' % ('PASS' if ok else 'FAIL', fn, NAMES[sb]))
        assert ok


def sim4(gates, v):
    b = [(v >> i) & 1 for i in range(4)]                     # x_0 is the least significant bit
    for g in gates:
        if g[0] == 'x':
            b[g[1]] ^= 1
        elif g[0] == 'cx':
            b[g[2]] ^= b[g[1]]
        elif g[0] == 'ccx':
            b[g[3]] ^= b[g[1]] & b[g[2]]
        elif g[0] == 'c3x':
            b[g[4]] ^= b[g[1]] & b[g[2]] & b[g[3]]
    return sum(b[i] << i for i in range(4))


def expand_c3x(gates, gadget):
    out = []
    for g in gates:
        if g[0] == 'c3x':
            a, b, c, t = g[1:]
            out += qc.c3x_gadget(a, b, c, t, 4) if gadget in ('pairs', 'full') else qc.c3x_toffoli(a, b, c, t, 4)
        else:
            out.append(g)
    return out


def check_min_toffoli():
    circ = {int(k): [tuple(g) for g in v] for k, v in
            json.load(open(os.path.join(ROOT, 'data', 'sbox_circuits.json'))).items()}
    circ[3] = list(reversed(circ[2]))
    for sb in range(4):
        ok = all(sim4(circ[sb], v) == SBOX[sb][v] for v in range(16))
        cnt = {k: sum(1 for g in circ[sb] if g[0] == k) for k in ('ccx', 'c3x', 'cx', 'x')}
        print('[%s] minimum-Toffoli circuit for %-11s %s' % ('PASS' if ok else 'FAIL', NAMES[sb], cnt))
        assert ok
    return circ


def sigma2_inverse_rows(circ):
    for mode, label in (('none', 'Nielsen and Chuang'), ('pairs', 'AND pairs'), ('full', 'AND everywhere')):
        fwd = expand_c3x(circ[2], mode)
        if mode == 'full':
            fwd = gadgetise_inplace(fwd)
        inv = list(reversed(fwd))
        if mode in ('pairs', 'full'):
            inv = [(('and' if g[0] == 'and_dg' else 'and_dg'),) + g[1:] if g[0] in ('and', 'and_dg') else g
                   for g in inv]
        n = max(max(g[1:]) for g in inv) + 1
        m = qc.metrics(n, inv, gadget=True)
        bad = verify_classical(n, inv, SBOX[3], 4, tuple(range(4, n)))
        dev, nb = verify_statevector_all_branches(n, inv, SBOX[3], 4)
        ok = not bad and dev < 1e-10
        print('[%s] sigma_2^-1, %-19s qubits %d, Toffoli %d, AND %d, CNOT %d, NOT %d, T-count %d, '
              'T-depth %d (depth %d), %d branches'
              % ('PASS' if ok else 'FAIL', label, n, m['Toffoli'], m['AND'], m['CNOT'], m['X'], m['T'],
                 m['T_depth'], m['CT_depth'], nb))
        assert ok


if __name__ == '__main__':
    check_lighter_r()
    sigma2_inverse_rows(check_min_toffoli())
    print('all S-box table checks passed')
