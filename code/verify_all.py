"""Run the complete verification suite and write results.json + VERIFICATION.md.

Every number quoted in the paper comes from this run.
"""
import io
import json
import math
import os
import sys
import time
from contextlib import redirect_stdout

import numpy as np

import cipher
import gadget_analysis as ga
import grover
import qarma_ref as ref
import qcircuit as qc
import verify_gadget
from sbox_search import DATA, parity

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = {}
LOG = []


def log(line=''):
    print(line, flush=True)
    LOG.append(line)


def check(name, ok, detail=''):
    log('  [%s] %s%s' % ('PASS' if ok else 'FAIL', name, (' -- ' + detail) if detail else ''))
    R.setdefault('checks', []).append(dict(name=name, ok=bool(ok), detail=detail))
    assert ok, name
    return ok


SB_NAMES = {0: 'sigma_0', 1: 'sigma_1', 2: 'sigma_2', 3: 'sigma_2^{-1}'}


def section(title):
    log()
    log('== %s ==' % title)


def main():
    t0 = time.time()
    log('QARMA-64 quantum resource estimation -- verification report')
    log('run: %s' % time.strftime('%Y-%m-%d %H:%M:%S'))

    # ---------------------------------------------------------------- 1. reference cipher
    section('1. Classical reference implementation')
    check('structural self-tests (involutions, M^2=I, tweak update invertible)', ref.self_test())
    bad = ref.check_test_vectors()
    check('all 9 published test vectors reproduced (sigma_0/1/2 x r=5,6,7)', not bad, str(bad))
    R['test_vectors'] = {'%d_%d' % k: '%016x' % v for k, v in ref.TV_CIPHER.items()}

    # ---------------------------------------------------------------- 2. gadget verification
    section('2. Gidney AND gadget and Toffoli decompositions')
    buf = io.StringIO()
    with redirect_stdout(buf):
        verify_gadget.main()
    for line in buf.getvalue().strip().splitlines():
        log('  ' + line)
    R['gadget_report'] = buf.getvalue()

    # ---------------------------------------------------------------- 3. S-box search results
    section('3. Minimum-Toffoli S-box search')
    sols = json.load(open(os.path.join(DATA, 'sbox_solutions.json')))
    R['search'] = dict(levels=sols['levels'], generators=sols['generators'],
                       counts={k: v['count'] for k, v in sols['solutions'].items()},
                       n_solutions={k: v['n_solutions'] for k, v in sols['solutions'].items()})
    log('  generalised Toffoli gates: %d ; coset levels: %s' % (sols['generators'], sols['levels']))
    for k, v in sols['solutions'].items():
        log('  %s: minimum count %d %s, %d optimal decompositions'
            % (SB_NAMES[int(k)], v['count'], '(+ one C^3X)' if v['odd'] else '(Toffoli only)', v['n_solutions']))
    check('sigma_1 minimum ancilla-free Toffoli count == 5 (draft claim)',
          sols['solutions']['1']['count'] == 5 and not sols['solutions']['1']['odd'])
    check('sigma_0 and sigma_2 need one C^3X and then 4 Toffoli gates',
          sols['solutions']['0']['count'] == 4 and sols['solutions']['2']['count'] == 4)

    # ---------------------------------------------------------------- 4. S-box circuits
    section('4. S-box circuits: verification and AND-gadget application')
    R['sbox'] = {}
    for sb in (0, 1, 2, 3):
        for kind in ('revkit', 'min'):
            base_sb = min(sb, 2)
            if kind == 'min' and sb == 3:
                continue
            for gadget in ('none', 'pairs', 'full'):
                fwd, inv, n_anc = cipher.sbox_gates(kind, base_sb, gadget)
                gates = inv if sb == 3 else fwd
                table = ref.SBOX[3] if sb == 3 else ref.SBOX[sb]
                n = 4 + (2 if (kind == 'revkit' and gadget == 'full') else (1 if n_anc else 0))
                n = max(n, 1 + max((max(g[1:]) for g in gates), default=3))
                res = ga.summarise('%s/%s/%s' % (SB_NAMES[sb], kind, gadget), n, gates, table,
                                   4, tuple(range(4, n)))
                R['sbox']['%d_%s_%s' % (sb, kind, gadget)] = res
                check('%-12s %-6s %-5s : correct on 16 inputs, ancillas clean, %d measurement branches'
                      % (SB_NAMES[sb], kind, gadget, res['branches']), res['ok'],
                      'T=%d T-depth=%d qubits=%d statevector dev=%.1e'
                      % (res['metrics']['T'], res['metrics']['T_depth'], n, res['statevector_dev']))
    # verified clean pairs in the RevKit circuits
    R['revkit_pairs'] = {}
    for sb, g in ga.load_revkit().items():
        pairs, rej = ga.find_and_pairs(5, g, 4)
        R['revkit_pairs'][str(sb)] = dict(n_pairs=len(pairs), n_rejected=len(rej),
                                          reasons=sorted(set(r[2] for r in rej)))
        log('  %s: %d Toffoli, %d verified clean AND pairs, %d candidates rejected (%s)'
            % (SB_NAMES[sb], sum(1 for x in g if x[0] == 'ccx'), len(pairs), len(rej),
               '; '.join(sorted(set(r[2] for r in rej)))))

    # ---------------------------------------------------------------- 5. MixColumns
    section('5. MixColumns')
    import mixcolumns as mixmod
    M = mixmod.probe_matrix()
    mix = cipher.MIX
    ok, info = mixmod.verify(mix['cnots'], mix['relabel'], trials=500, seed=99)
    check('MixColumns circuit matches the reference on 500 random columns', ok, str(info))
    check('probed matrix equals the stored one', M == mix['matrix'])
    m_mix = qc.metrics(16, [('cx', c, t) for c, t in mix['cnots']])
    R['mixcolumns'] = dict(cnots=len(mix['cnots']), depth=m_mix['NCT_depth'])
    log('  one column: %d CNOT, depth %d; full layer: %d CNOT'
        % (len(mix['cnots']), m_mix['NCT_depth'], 4 * len(mix['cnots'])))

    # ---------------------------------------------------------------- 6. full cipher
    section('6. Full QARMA-64 encryption circuit')
    R['encryption'] = {}
    for kind in ('min', 'revkit'):
        for gadget in ('none', 'pairs', 'full'):
            for sb in (0, 1, 2):
                for r in (5, 6, 7):
                    if kind == 'revkit' and (sb != 1 or r != 7) and gadget != 'full':
                        continue
                    if kind == 'revkit' and gadget == 'full' and (sb != 1 or r != 7):
                        continue
                    v = cipher.verify_encryption(sb, r, kind, gadget)
                    R['encryption']['%s_%s_s%d_r%d' % (kind, gadget, sb, r)] = dict(
                        ok=v['ok'], cipher=v['circuit'], published=v['published'], **v['metrics'])
                    assert v['ok'], (kind, gadget, sb, r, v)
    for sb in (0, 1, 2):
        for r in (5, 6, 7):
            v = cipher.verify_encryption(sb, r, 'min', 'full', quantum_tweak=True)
            R['encryption']['min_full_qtweak_s%d_r%d' % (sb, r)] = dict(
                ok=v['ok'], cipher=v['circuit'], published=v['published'], **v['metrics'])
            assert v['ok']
    n_ok = sum(1 for k, v in R['encryption'].items() if v['ok'])
    check('every encryption circuit reproduces the published ciphertext '
          '(key, tweak and ancillas restored)', n_ok == len(R['encryption']),
          '%d configurations' % n_ok)
    e0 = R['encryption']['min_none_s1_r7']
    e1 = R['encryption']['min_full_s1_r7']
    ev = R['encryption']['revkit_none_s1_r7']
    log('  sigma_1, r=7:  ancilla-free  %d qubits, T=%d, T-depth=%d, depth=%d'
        % (e0['qubits'], e0['T'], e0['T_depth'], e0['CT_depth']))
    log('                 AND gadget    %d qubits, T=%d, T-depth=%d, depth=%d'
        % (e1['qubits'], e1['T'], e1['T_depth'], e1['CT_depth']))
    log('                 RevKit base   %d qubits, T=%d, T-depth=%d, depth=%d'
        % (ev['qubits'], ev['T'], ev['T_depth'], ev['CT_depth']))
    check('draft claim reproduced: full-cipher T-count 8960 for sigma_1/r=7 ancilla-free',
          e0['T'] == 8960, 'got %d' % e0['T'])

    # ---------------------------------------------------------------- 7. Grover
    section('7. Grover oracle and diffusion')
    p, iters, nmeas, nq = grover.toy_grover_test(nkey=6, marked=37, seed=3, gadget=True)
    p_plain, _, _, _ = grover.toy_grover_test(nkey=6, marked=37, seed=3, gadget=False)
    ps = [grover.toy_grover_test(nkey=6, marked=37, seed=s)[0] for s in (1, 2, 5)]
    R['toy_grover'] = dict(prob=p, iterations=iters, measurements=nmeas, qubits=nq,
                           prob_plain=p_plain, prob_random_outcomes=ps)
    check('toy Grover (same AND-tree/diffusion code) reaches the theoretical success probability',
          p > 0.99 and abs(p - p_plain) < 1e-9, 'p=%.4f after %d iterations (plain Toffoli: %.4f)'
          % (p, iters, p_plain))
    check('success probability independent of the random measurement outcomes',
          max(abs(x - p) for x in ps) < 1e-9, 'p in {%s}' % ', '.join('%.4f' % x for x in ps))
    ok, res, _ = grover.verify_oracle_classical(sb=1, r=7, kind='min', gadget='full', pairs=2, n_wrong=6)
    R['oracle_check'] = res
    check('QARMA-64 oracle marks the correct key and no wrong key (6 random wrong keys)', ok)
    check('oracle uncomputation restores state registers, ancillas and key register',
          all(x['key_restored'] and x['states_clean'] and x['anc_clean'] for x in res))

    R['grover'] = {}
    for gadget in ('none', 'full'):
        for pairs in (1, 2, 3):
            c, inst, key, markq, cts = grover.build_iteration(1, 7, 'min', gadget, pairs, mark='phase')
            m = qc.metrics(c.n, c.gates)
            R['grover']['%s_p%d' % (gadget, pairs)] = m
            log('  iteration (%-4s, %d pairs): %d qubits, T=%d, T-depth=%d, depth=%d'
                % (gadget, pairs, m['qubits'], m['T'], m['T_depth'], m['CT_depth']))

    # ---------------------------------------------------------------- 8. attack cost
    section('8. Grover key-search cost')
    from decimal import Decimal, getcontext
    getcontext().prec = 60
    I = int(Decimal('3.14159265358979323846264338327950288419716939937510') / 4 * 2 ** 64)
    R['iterations'] = I
    R['attack'] = {}
    for key, m in R['grover'].items():
        G = I * m['CT_total']
        D = I * m['CT_depth']
        R['attack'][key] = dict(logG=math.log2(G), logT=math.log2(I * m['T']), logD=math.log2(D),
                                logDW=math.log2(D * m['qubits']),
                                logGD=math.log2(G) + math.log2(D))
    for md in (40, 64, 96):
        for key in ('none_p3', 'full_p3'):
            m = R['grover'][key]
            d_tot = math.log2(I) + math.log2(m['CT_depth'])
            if d_tot <= md:
                logS, logG = 0.0, math.log2(I) + math.log2(m['CT_total'])
            else:
                logS = 2 * (d_tot - md)
                logG = 2 * math.log2(I) + math.log2(m['CT_total']) + math.log2(m['CT_depth']) - md
            R['attack']['md%d_%s' % (md, key)] = dict(logS=logS, logG=logG,
                                                      logW=logS + math.log2(m['qubits']),
                                                      nist_aes128=170 - md)
    a = R['attack']['full_p3']
    log('  3 pairs, AND gadgets: G-cost 2^%.2f, T-count 2^%.2f, depth 2^%.2f, DW 2^%.2f'
        % (a['logG'], a['logT'], a['logD'], a['logDW']))

    R['runtime_s'] = time.time() - t0
    with open(os.path.join(ROOT, 'results.json'), 'w') as f:
        json.dump(R, f, indent=1, default=str)
    with open(os.path.join(ROOT, 'VERIFICATION.md'), 'w', encoding='utf-8') as f:
        f.write('# Verification report\n\n```\n' + '\n'.join(LOG) + '\n```\n')
    log()
    log('all checks passed in %.0f s' % R['runtime_s'])
    return R


if __name__ == '__main__':
    main()
