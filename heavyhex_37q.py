#!/usr/bin/env python3
"""
Heavy-Hex Surface Code (3,3) -- v5: paper-faithful 37-qubit embedding
=====================================================================
Vezvaee et al. layout (Fig. 1a / Fig. 13): 17 data + 8 dedicated dual-use
rung ancillas + 12 bridges = 37 physical qubits (diamond patch).

  row1:            25
  row2:      43 44 45 46 47          rung 37 (25-45)
  row3: 61 62 63 64 65 66 67 68 69   rungs 56 (43-63), 57 (47-67)
  row4: 81 82 83 84 85 86 87 88 89   rungs 76 (61-81), 77 (65-85), 78 (69-89)
  row5:    103 104 105 106 107       rungs 96 (83-103), 97 (87-107)

16 stabilizers (verified: all commute, rank 16, dz = dx = 3), one Z and one
X per ancilla (bicolor dual-use, measured in alternate rounds):

  anc   Z-check                X-check
  37    Z{25,45,47}            X{25,43,45}
  56    Z{43,45,63,65}         X{43,61,63}
  57    Z{47,67,69}            X{45,47,65,67}
  76    Z{61,63,81,83}         X{61,81}
  77    Z{65,67,85,87}         X{63,65,83,85}
  78    Z{69,89}               X{67,69,87,89}
  96    Z{83,85,103,105}       X{81,83,103}
  97    Z{105,107}             X{85,87,105,107}

  Z_L = Z{69,87,105}   X_L = X{43,65,87}   (chirality: completion 1)

v5 stage-1 design: CORRECTNESS FIRST (proven v2/v4 method)
  - every check isolated: fold -> CX(rep->anc) -> M -> unfold
  - X-checks in the H-conjugated frame (all-classical XOR network; bridges
    provably return to |0>, zero cross-check contamination)
  - dedicated ancillas -> no-reset from the start; values decoded by
    per-ancilla XOR chain (raw_j ^ raw_{j-1}, valid across Z/X alternation)
  - depth-7 fold-sharing optimization is stage 2, gated on these tests.
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

DATA_PHYS = [25, 43, 45, 47, 61, 63, 65, 67, 69,
             81, 83, 85, 87, 89, 103, 105, 107]
BRIDGE_PHYS = [44, 46, 62, 64, 66, 68, 82, 84, 86, 88, 104, 106]
ANC_PHYS = [37, 56, 57, 76, 77, 78, 96, 97]
ALL_PHYS = DATA_PHYS + BRIDGE_PHYS + ANC_PHYS          # 17 + 12 + 8 = 37
L = {p: i for i, p in enumerate(ALL_PHYS)}             # phys -> local

# horizontal bridge between two row neighbours
HBR = {(25, 45): None,  # rung edge, no bridge
       (43, 45): 44, (45, 47): 46, (61, 63): 62, (63, 65): 64,
       (65, 67): 66, (67, 69): 68, (81, 83): 82, (83, 85): 84,
       (85, 87): 86, (87, 89): 88, (103, 105): 104, (105, 107): 106}
def br(a, b):
    return HBR[tuple(sorted((a, b)))]

LOGICAL_Z = [69, 87, 105]
LOGICAL_X = [43, 65, 87]

# name -> (type, support, anc, folds [(outer, rep), ...]); all phys labels.
# rep qubits are the ancilla's two rung neighbours (or one, for Z97).
CHECK_DEFS = {
    'Z37': ('Z', (25, 45, 47),      37, [(47, 45)]),
    'X37': ('X', (25, 43, 45),      37, [(43, 45)]),
    'Z56': ('Z', (43, 45, 63, 65),  56, [(45, 43), (65, 63)]),
    'X56': ('X', (43, 61, 63),      56, [(61, 63)]),
    'Z57': ('Z', (47, 67, 69),      57, [(69, 67)]),
    'X57': ('X', (45, 47, 65, 67),  57, [(45, 47), (65, 67)]),
    'Z76': ('Z', (61, 63, 81, 83),  76, [(63, 61), (83, 81)]),
    'X76': ('X', (61, 81),          76, []),
    'Z77': ('Z', (65, 67, 85, 87),  77, [(67, 65), (87, 85)]),
    'X77': ('X', (63, 65, 83, 85),  77, [(63, 65), (83, 85)]),
    'Z78': ('Z', (69, 89),          78, []),
    'X78': ('X', (67, 69, 87, 89),  78, [(67, 69), (87, 89)]),
    'Z96': ('Z', (83, 85, 103, 105), 96, [(85, 83), (105, 103)]),
    'X96': ('X', (81, 83, 103),     96, [(81, 83)]),
    'Z97': ('Z', (105, 107),        97, [(105, 107)]),
    'X97': ('X', (85, 87, 105, 107), 97, [(85, 87), (105, 107)]),
}
Z_STABS = {n: d[1] for n, d in CHECK_DEFS.items() if d[0] == 'Z'}
X_STABS = {n: d[1] for n, d in CHECK_DEFS.items() if d[0] == 'X'}

# per-cycle measurement order: round 1 = Z-checks, round 2 = X-checks
ROUND1 = ['Z37', 'Z56', 'Z57', 'Z76', 'Z77', 'Z78', 'Z96', 'Z97']
ROUND2 = ['X37', 'X56', 'X57', 'X76', 'X77', 'X78', 'X96', 'X97']
CYCLE_ORDER = ROUND1 + ROUND2
N_CHECKS = 16
ANC_OF = {n: CHECK_DEFS[n][2] for n in CYCLE_ORDER}


class HeavyHex37Q:
    """Stage-1 correctness-first circuit on the paper's 37-qubit patch."""

    def __init__(self, num_cycles=3):
        self.num_cycles = num_cycles

    def _fold(self, qc, folds):
        for outer, rep in folds:
            b = br(outer, rep)
            qc.cx(L[outer], L[b])
            qc.cx(L[b], L[rep])

    def _unfold(self, qc, folds):
        for outer, rep in reversed(folds):
            b = br(outer, rep)
            qc.cx(L[b], L[rep])
            qc.cx(L[outer], L[b])

    def _measure_check(self, qc, name, creg, cbit):
        ctype, support, anc, folds = CHECK_DEFS[name]
        # rep qubits actually wired to the ancilla: rung neighbours in support
        rung_u, rung_v = {37: (25, 45), 56: (43, 63), 57: (47, 67),
                          76: (61, 81), 77: (65, 85), 78: (69, 89),
                          96: (83, 103), 97: (87, 107)}[anc]
        reps = [q for q in (rung_u, rung_v) if q in support]
        if ctype == 'X':
            for q in support:
                qc.h(L[q])
            self._fold(qc, folds)
            for q in reps:
                qc.cx(L[q], L[anc])
            qc.measure(L[anc], creg[cbit])
            self._unfold(qc, folds)
            for q in support:
                qc.h(L[q])
        else:
            self._fold(qc, folds)
            for q in reps:
                qc.cx(L[q], L[anc])
            qc.measure(L[anc], creg[cbit])
            self._unfold(qc, folds)

    def build_circuit(self, initial_state=0, inject=None):
        """inject: (pauli, data_phys, after_cycle)"""
        q = QuantumRegister(37, 'q')
        syn = ClassicalRegister(N_CHECKS * self.num_cycles, 'syn')
        dat = ClassicalRegister(17, 'data')
        qc = QuantumCircuit(q, syn, dat)
        if initial_state == 1:
            for p in LOGICAL_X:
                qc.x(L[p])
        bit = 0
        for cyc in range(self.num_cycles):
            for name in CYCLE_ORDER:
                self._measure_check(qc, name, syn, bit)
                bit += 1
            qc.barrier()
            if inject is not None and inject[2] == cyc:
                pauli, dq, _ = inject
                getattr(qc, pauli.lower())(L[dq])
                qc.barrier()
        for i, p in enumerate(DATA_PHYS):
            qc.measure(L[p], dat[i])
        return qc


def check_values(syn_rows, num_cycles):
    """raw bits -> per-cycle check values via per-ancilla XOR chains
    (all 8 ancillas are no-reset; each measures Z then X per cycle)."""
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


if __name__ == '__main__':
    sc = HeavyHex37Q(num_cycles=2)
    qc = sc.build_circuit()
    ops = qc.count_ops()
    print(f"2-cycle: depth={qc.depth()} ({qc.depth()/2:.0f}/cyc), "
          f"cx={ops.get('cx')/2:.0f}/cyc, h={ops.get('h')/2:.0f}/cyc, "
          f"reset={ops.get('reset', 0)}, meas={ops.get('measure')/2:.1f}/cyc")
