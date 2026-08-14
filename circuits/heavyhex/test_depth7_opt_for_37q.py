#!/usr/bin/env python3
"""V5 noiseless verification (37-qubit paper code)."""
import sys
from pathlib import Path

import numpy as np
from qiskit_aer import AerSimulator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from circuits.heavyhex.heavyhex_depth7_opt_for_37q import (  # noqa: E402
    HeavyHex37QDepthOpt as HeavyHex37Q, check_values, CYCLE_ORDER,
    X_INRUN_SUPPORT, N_CHECKS)
from circuits.heavyhex.heavyhex_37q import (  # noqa: E402
    Z_STABS, X_STABS, DATA_PHYS, LOGICAL_Z)

SHOTS = 400
sim = AerSimulator()
DIDX = {p: i for i, p in enumerate(DATA_PHYS)}


def run(num_cycles=3, initial_state=0, inject=None):
    qc = HeavyHex37Q(num_cycles).build_circuit(initial_state, inject)
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


def main():
    n_cyc, ok = 3, True
    syn, dat, w, qc = run(n_cyc)
    ops = qc.count_ops()
    print(f"=== V6 (depth-opt) |0>_L {n_cyc} cycles ===")
    print(f"depth={qc.depth()/n_cyc:.0f}/cyc cx={ops.get('cx')/n_cyc:.0f}/cyc "
          f"h={ops.get('h')/n_cyc:.0f}/cyc reset={ops.get('reset', 0)} "
          f"meas={ops.get('measure')/n_cyc:.1f}/cyc")
    V = check_values(syn, n_cyc)

    fails = []
    for name in Z_STABS:
        for c in range(n_cyc):
            p = p1(V[name][:, c], w)
            if p > 0.002:
                fails.append(f"A:c{c}.{name}={p:.3f}")
    print(f"[A] Z det-0 all cycles: {'PASS' if not fails else 'FAIL ' + str(fails[:6])}")
    ok &= not fails

    fails = []
    for name in X_STABS:
        for c in range(n_cyc - 1):
            p = p1(V[name][:, c] ^ V[name][:, c + 1], w)
            if p > 0.002:
                fails.append(f"B:d{c}.{name}={p:.3f}")
    print(f"[B] X temporal XOR det-0: {'PASS' if not fails else 'FAIL ' + str(fails[:6])}")
    ok &= not fails

    fails = []
    for name, supp in Z_STABS.items():
        par = np.bitwise_xor.reduce(dat[:, [DIDX[p] for p in supp]], axis=1)
        p = p1(par ^ V[name][:, n_cyc - 1], w)
        if p > 0.002:
            fails.append(f"C:{name}={p:.3f}")
    print(f"[C] final data vs last Z: {'PASS' if not fails else 'FAIL ' + str(fails)}")
    ok &= not fails

    zl = np.bitwise_xor.reduce(dat[:, [DIDX[p] for p in LOGICAL_Z]], axis=1)
    p0 = p1(zl, w)
    syn1, dat1, w1, _ = run(2, initial_state=1)
    V1 = check_values(syn1, 2)
    zl1 = np.bitwise_xor.reduce(dat1[:, [DIDX[p] for p in LOGICAL_Z]], axis=1)
    p1v = p1(zl1, w1)
    zbad = [f"c{c}.{n}" for n in Z_STABS for c in range(2)
            if p1(V1[n][:, c], w1) > 0.002]
    d_ok = p0 < 0.002 and p1v > 0.998 and not zbad
    print(f"[D] logicals: |0>_L Z_L={p0:.3f}, |1>_L Z_L={p1v:.3f}, "
          f"|1>_L Z-checks {'clean' if not zbad else zbad[:4]} "
          f"{'PASS' if d_ok else 'FAIL'}")
    ok &= d_ok

    print("[E] single-error signatures (inject after c0, read c1):")
    cases = [('X', 65), ('X', 25), ('X', 61), ('X', 89), ('X', 105),
             ('Z', 65), ('Z', 43), ('Z', 69), ('Z', 107), ('Z', 81)]
    for pauli, dq in cases:
        syn_e, _, w_e, _ = run(3, inject=(pauli, dq, 0))
        Ve = check_values(syn_e, 3)
        fired = []
        for name in CYCLE_ORDER:
            if name in Z_STABS:
                p = p1(Ve[name][:, 1], w_e)
            else:
                p = p1(Ve[name][:, 1] ^ Ve[name][:, 0], w_e)
            if p > 0.998:
                fired.append(name)
            elif p > 0.002:
                fired.append(f"{name}?{p:.2f}")
        stabs = Z_STABS if pauli == 'X' else X_STABS
        exp = sorted(n for n, s in stabs.items() if dq in s)
        good = sorted(fired) == exp
        ok &= good
        print(f"  {pauli} q{dq}: fired={sorted(fired)} exp={exp} "
              f"{'PASS' if good else 'FAIL'}")

    print(f"\nOVERALL: {'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return ok


if __name__ == '__main__':
    main()
