#!/usr/bin/env python3
"""
Heavy-Hex Surface Code (3,3) -- v6: depth-optimized (paper Fig 1b scheme)
=========================================================================
Same 37-qubit patch and 16 stabilizers as v5. Implements the Benito/Vezvaee
shared-fold protocol:

  Abstract fold (Benito Eq. 1): one CX(outer->rep) per shared pair folds
  Z-info onto rep AND X-info onto outer simultaneously. Rung ancillas then
  measure the weight-2 remnants directly (Z: CX(rep->anc); X: anc-as-control).

  Heavy-hex mediation: each abstract CX is a 2-CX parity fold through the
  bridge (CX(outer->br), CX(br->rep)); the unfold is the exact 2-CX inverse.
  Bridges are never measured or reset; ancillas are never reset (XOR chains).
  X-check outcomes correspond to persistent JOINT operators (data (x) bridge);
  all joints mutually commute, so cycle-to-cycle differences are deterministic.
  Effective in-run X-check data support is the rung pair; full abstract
  support is recovered by final-readout detectors. For memory-Z experiments
  the protection of Z_L rests on the Z-checks, whose joints carry the FULL
  abstract support, so d_z = 3 is preserved in-run.

Round structure (all folds within a round are disjoint -> single CX layer):
  R1 checks: X37 Z56 X77 Z96 | Z57 X78 | X76 Z97
  R1 folds : (45>44>43) (65>64>63) (85>84>83) (105>104>103) (69>68>67) (89>88>87)
  R2 checks: Z37 X57 Z77 X97 | X56 Z76 X96 | Z78
  R2 folds : (47>46>45) (67>66>65) (87>86>85) (107>106>105) (63>62>61) (83>82>81)

Bridges used: R1 {44,64,84,104,68,88}, R2 {46,66,86,106,62,82} -- disjoint.
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from circuits.heavyhex.heavyhex_37q import (validate_backend,  # noqa: F401 (re-export)
                             ALL_PHYS, DATA_PHYS, ANC_PHYS, L, br,
                             Z_STABS, X_STABS, CHECK_DEFS,
                             LOGICAL_Z, LOGICAL_X)

RUNG = {37: (25, 45), 56: (43, 63), 57: (47, 67), 76: (61, 81),
        77: (65, 85), 78: (69, 89), 96: (83, 103), 97: (87, 107)}

ROUND1 = ['X37', 'Z56', 'X77', 'Z96', 'Z57', 'X78', 'X76', 'Z97']
ROUND2 = ['Z37', 'X57', 'Z77', 'X97', 'X56', 'Z76', 'X96', 'Z78']
CYCLE_ORDER = ROUND1 + ROUND2
N_CHECKS = 16
ANC_OF = {n: CHECK_DEFS[n][2] for n in CYCLE_ORDER}

FOLDS = {  # round -> [(outer, rep)]; each is a TRUE CX via 3-CX bridge relay
    1: [(45, 43), (65, 63), (85, 83), (105, 103), (69, 67), (89, 87),
        (105, 107)],                                   # (105,107): Z97's arm
    2: [(47, 45), (67, 65), (87, 85), (107, 105), (63, 61), (83, 81)],
}


class HeavyHex37QDepthOpt:
    def __init__(self, num_cycles=3):
        self.num_cycles = num_cycles

    def _relay_layers(self, qc, rnd):
        # true CX(outer->rep) via bridge: CX(o->b), CX(b->r), CX(o->b).
        # Palindrome -> self-inverse; bridge returns exactly to |0>, and the
        # conjugation action equals a genuine CX on both Z and X sectors,
        # so all measured joints are PURE abstract stabilizers.
        for outer, rep in FOLDS[rnd]:
            qc.cx(L[outer], L[br(outer, rep)])
        for outer, rep in FOLDS[rnd]:
            qc.cx(L[br(outer, rep)], L[rep])
        for outer, rep in FOLDS[rnd]:
            qc.cx(L[outer], L[br(outer, rep)])

    def _fold(self, qc, rnd):
        self._relay_layers(qc, rnd)

    def _unfold(self, qc, rnd):
        self._relay_layers(qc, rnd)

    def _round(self, qc, rnd, names, creg, bits):
        self._fold(qc, rnd)
        # ancilla couplings (in-frame weight-2 reads on rung pairs)
        for name in names:
            ctype, support, anc, _ = CHECK_DEFS[name]
            u, v = RUNG[anc]
            reps = [q for q in (u, v) if q in support]
            if ctype == 'Z':
                for q in reps:
                    qc.cx(L[q], L[anc])
            else:
                qc.h(L[anc])
                for q in reps:
                    qc.cx(L[anc], L[q])
                qc.h(L[anc])
        for name in names:
            qc.measure(L[ANC_OF[name]], creg[bits[name]])   # no reset
        self._unfold(qc, rnd)

    def build_circuit(self, initial_state=0, inject=None):
        q = QuantumRegister(37, 'q')
        syn = ClassicalRegister(N_CHECKS * self.num_cycles, 'syn')
        dat = ClassicalRegister(17, 'data')
        qc = QuantumCircuit(q, syn, dat)
        if initial_state == 1:
            for p in LOGICAL_X:
                qc.x(L[p])
        for cyc in range(self.num_cycles):
            base = cyc * N_CHECKS
            bits = {n: base + i for i, n in enumerate(CYCLE_ORDER)}
            self._round(qc, 1, ROUND1, syn, bits)
            self._round(qc, 2, ROUND2, syn, bits)
            qc.barrier()
            if inject is not None and inject[2] == cyc:
                pauli, dq, _ = inject
                getattr(qc, pauli.lower())(L[dq])
                qc.barrier()
        for i, p in enumerate(DATA_PHYS):
            qc.measure(L[p], dat[i])
        return qc


def check_values(syn_rows, num_cycles):
    n = syn_rows.shape[0]
    vals = {name: np.zeros((n, num_cycles), dtype=int) for name in CYCLE_ORDER}
    prev = {a: np.zeros(n, dtype=int) for a in ANC_PHYS}
    for cyc in range(num_cycles):
        for j, name in enumerate(CYCLE_ORDER):
            raw = syn_rows[:, cyc * N_CHECKS + j]
            a = ANC_OF[name]
            vals[name][:, cyc] = raw ^ prev[a]
            prev[a] = raw.copy()
    return vals


# in-run detection supports under the joint scheme:
#   X errors  -> flip Z-checks containing the qubit (FULL abstract support)
#   Z errors  -> flip X-checks whose RUNG PAIR contains the qubit
X_INRUN_SUPPORT = {n: tuple(q for q in RUNG[CHECK_DEFS[n][2]]
                            if q in CHECK_DEFS[n][1])
                   for n in X_STABS}

if __name__ == '__main__':
    sc = HeavyHex37QDepthOpt(2)
    qc = sc.build_circuit()
    ops = qc.count_ops()
    print(f"2-cycle: depth={qc.depth()/2:.0f}/cyc cx={ops.get('cx')/2:.0f}/cyc "
          f"h={ops.get('h')/2:.0f}/cyc reset={ops.get('reset', 0)} "
          f"meas={ops.get('measure')/2:.1f}/cyc")
