#!/usr/bin/env python3
"""Noiseless verification for GeneralDiamondCircuit, any (dx,dz)."""
import sys
from pathlib import Path

import numpy as np
from qiskit_aer import AerSimulator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from heavyhex_circuits.heavyhex_general import GeneralDiamondCircuit  # noqa: E402

SHOTS = 300
sim = AerSimulator()
CJ = "coupling_ibm_boston.json"


def run(gc, num_cycles=3, initial_state=0, inject=None):
    gc.num_cycles = num_cycles
    qc = gc.build_circuit(initial_state, inject)
    counts = sim.run(qc, shots=SHOTS).result().get_counts()
    syn, dat, w = [], [], []
    for bs, c in counts.items():
        d, s = bs.split()
        syn.append([int(b) for b in s[::-1]])
        dat.append([int(b) for b in d[::-1]])
        w.append(c)
    return np.array(syn), np.array(dat), np.array(w), qc


def p1(bits, w):
    return float((bits * w).sum() / w.sum())


def verify(dx, dz, n_cyc=3):
    gc = GeneralDiamondCircuit(dx, dz, CJ, num_cycles=n_cyc)
    data_sorted = sorted(gc.code.data)
    didx = {tuple(d): i for i, d in enumerate(data_sorted)}
    Zs = {nm: p['supp'] for nm, p in gc.plan.items() if p['t'] == 'Z'}
    Xs = {nm: p['supp'] for nm, p in gc.plan.items() if p['t'] == 'X'}
    zl = [tuple(s) for s in gc.sol['zl']]

    syn, dat, w, qc = run(gc, n_cyc)
    ops = qc.count_ops()
    print(f"=== ({dx},{dz}) {gc.nq}q, {len(gc.order)} checks/cyc ===")
    print(f"depth={qc.depth()/n_cyc:.0f}/cyc cx={ops.get('cx')/n_cyc:.0f}/cyc "
          f"h={ops.get('h', 0)/n_cyc:.0f}/cyc reset={ops.get('reset', 0)}")
    V = gc.check_values(syn, n_cyc)
    ok = True

    bad = [f"c{c}.{nm}" for nm in Zs for c in range(n_cyc)
           if p1(V[nm][:, c], w) > 0.003]
    print(f"[A] Z det-0: {'PASS' if not bad else 'FAIL ' + str(bad[:6])}")
    ok &= not bad
    bad = [f"d{c}.{nm}" for nm in Xs for c in range(n_cyc - 1)
           if p1(V[nm][:, c] ^ V[nm][:, c + 1], w) > 0.003]
    print(f"[B] X XOR det-0: {'PASS' if not bad else 'FAIL ' + str(bad[:6])}")
    ok &= not bad
    bad = []
    for nm, s in Zs.items():
        par = np.bitwise_xor.reduce(dat[:, [didx[q] for q in s]], axis=1)
        if p1(par ^ V[nm][:, n_cyc - 1], w) > 0.003:
            bad.append(nm)
    print(f"[C] final data vs Z: {'PASS' if not bad else 'FAIL ' + str(bad)}")
    ok &= not bad
    p0 = p1(np.bitwise_xor.reduce(dat[:, [didx[q] for q in zl]], axis=1), w)
    syn1, dat1, w1, _ = run(gc, 2, initial_state=1)
    pv = p1(np.bitwise_xor.reduce(dat1[:, [didx[q] for q in zl]], axis=1), w1)
    V1 = gc.check_values(syn1, 2)
    zbad = [nm for nm in Zs for c in range(2) if p1(V1[nm][:, c], w1) > 0.003]
    d_ok = p0 < 0.003 and pv > 0.997 and not zbad
    print(f"[D] logicals: |0>={p0:.3f} |1>={pv:.3f} "
          f"{'PASS' if d_ok else 'FAIL ' + str(zbad[:4])}")
    ok &= d_ok

    # [E] sample error signatures incl. relay-check supports
    samples = []
    relay_qs = [q for nm, p in gc.plan.items() if p['relay'] for q in p['supp']]
    data_pool = list(dict.fromkeys(relay_qs))[:2] + \
        [tuple(d) for d in data_sorted[::max(1, len(data_sorted) // 3)]][:3]
    for pauli in 'XZ':
        for dq in data_pool:
            samples.append((pauli, dq))
    print("[E] error signatures:")
    for pauli, dq in samples:
        syn_e, _, w_e, _ = run(gc, 3, inject=(pauli, dq, 0))
        Ve = gc.check_values(syn_e, 3)
        fired = []
        for nm in gc.order:
            if nm in Zs:
                p = p1(Ve[nm][:, 1], w_e)
            else:
                p = p1(Ve[nm][:, 1] ^ Ve[nm][:, 0], w_e)
            if p > 0.997:
                fired.append(nm)
            elif p > 0.003:
                fired.append(f"{nm}?{p:.2f}")
        stabs = Zs if pauli == 'X' else Xs
        exp = sorted(nm for nm, s in stabs.items() if dq in s)
        good = sorted(fired) == exp
        ok &= good
        print(f"  {pauli}{dq}: {'PASS' if good else f'FAIL fired={sorted(fired)} exp={exp}'}")
    print(f"OVERALL ({dx},{dz}): {'ALL PASS' if ok else 'FAILURES'}\n")
    return ok


if __name__ == '__main__':
    args = sys.argv[1:] or ['35']
    for a in args:
        verify(int(a[0]), int(a[1]))
