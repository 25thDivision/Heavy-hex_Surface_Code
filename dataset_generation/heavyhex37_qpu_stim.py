#!/usr/bin/env python3
"""
Hardware-shaped 37-qubit Stim circuit with QPU-averaged calibration noise
=========================================================================
Mirrors heavyhex_depth7_opt_for_37q.HeavyHex37QDepthOpt gate-for-gate on
all 37 physical qubits (17 data + 12 bridges + 8 ancillas): the same
round structure (R1/R2), the same 3-CX bridge fold/unfold relays, the
same H-conjugated X-check reads, and NO ancilla reset — raw ancilla bits
accumulate exactly like on hardware and check values are recovered by the
per-ancilla XOR chain (heavyhex_37q.check_values).

Noise attachment (profile mode "qpu_avg_v1", built by
make_qpu_avg_profile.py; every rate is the run-averaged calibration value
of THAT physical qubit / edge):
  * each CX               -> DEPOLARIZE2(edge 2Q error) on its qubits
  * each H                -> DEPOLARIZE1(qubit 1Q error, sx proxy)
  * right before each M   -> X_ERROR(qubit readout error)
  * injected data error   -> X_ERROR/Z_ERROR(p) on data right after the
                             initial reset (same Error_Type/Error_Rate
                             axis as the abstract datasets)

NOT modeled (documented limitation, also in README.md):
  * calibration drift between/within runs (values are averages)
  * T1/T2 idle decoherence and delay errors (incl. DD behavior)
  * measurement crosstalk / correlated readout errors
  * coherent (non-Pauli) errors; the transpiled 1Q-gate structure is
    approximated by one sx-proxy depolarizing event per H

Detectors are defined on the RAW measurement record via XOR expansion of
the per-ancilla chains (raw_j ^ raw_{j-1} = check value), in EXACTLY the
detector order of heavyhex33_stim._append_detectors — so the tensor /
MWPM reconstruction logic is unchanged and the DEM can feed PyMatching.

The check-value stream equals the abstract 25q circuit's at the
check-value level (same contract as hardware vs Stim, README §4);
verification/verify_equivalence.py section [E] enforces it together with
detector determinism and single-error signature agreement vs Aer.
"""
import sys
from pathlib import Path

import numpy as np
import stim

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from circuits.heavyhex.heavyhex_37q import (  # noqa: E402
    DATA_PHYS, ANC_PHYS, ALL_PHYS, L, br, CHECK_DEFS, Z_STABS,
    LOGICAL_Z, check_values)
from circuits.heavyhex.heavyhex_depth7_opt_for_37q import (  # noqa: E402
    RUNG, ROUND1, ROUND2, CYCLE_ORDER, N_CHECKS, FOLDS, ANC_OF)
from dataset_generation.heavyhex33_stim import (  # noqa: E402
    CHECK_AT, Z_POS, DIDX, num_detectors, split_stim_sample,
    check_matrix_from_dict)

NUM_DATA = len(DATA_PHYS)                     # 17
LOGICAL_Z_IDX = [DATA_PHYS.index(p) for p in LOGICAL_Z]

_ZERO = {"readout": {}, "error_1q": {}, "error_2q": {}}


def _rates(profile):
    """Profile dict -> lookup helpers (0.0 where absent / noiseless)."""
    prof = profile or _ZERO
    ro = {int(k): v for k, v in prof.get("readout", {}).items()}
    e1 = {int(k): v for k, v in prof.get("error_1q", {}).items()}
    e2 = {}
    for k, v in prof.get("error_2q", {}).items():
        u, w = (int(x) for x in k.split("-"))
        e2[tuple(sorted((u, w)))] = v
    return (lambda q: ro.get(q, 0.0),
            lambda q: e1.get(q, 0.0),
            lambda a, b: e2.get(tuple(sorted((a, b))), 0.0))


class _Builder:
    """Accumulates the circuit and the raw-measurement bookkeeping."""

    def __init__(self, profile):
        self.c = stim.Circuit()
        self.read_err, self.h_err, self.cx_err = _rates(profile)
        self.n_meas = 0
        self.meas_idx = {}          # (name, cyc) -> raw syn meas index
        self.prev_meas = {}         # anc phys -> previous meas index
        self.chain = {}             # (name, cyc) -> [raw indices] whose
        #                             XOR is the check VALUE

    def cx(self, a_phys, b_phys):
        self.c.append("CX", [L[a_phys], L[b_phys]])
        p = self.cx_err(a_phys, b_phys)
        if p > 0:
            self.c.append("DEPOLARIZE2", [L[a_phys], L[b_phys]], p)

    def h(self, q_phys):
        self.c.append("H", [L[q_phys]])
        p = self.h_err(q_phys)
        if p > 0:
            self.c.append("DEPOLARIZE1", [L[q_phys]], p)

    def measure(self, q_phys):
        p = self.read_err(q_phys)
        if p > 0:
            self.c.append("X_ERROR", [L[q_phys]], p)
        self.c.append("M", [L[q_phys]])          # no reset
        self.n_meas += 1
        return self.n_meas - 1


def _relay_layers(b, rnd):
    """True CX(outer->rep) via bridge: CX(o->b), CX(b->r), CX(o->b)."""
    for outer, rep in FOLDS[rnd]:
        b.cx(outer, br(outer, rep))
    for outer, rep in FOLDS[rnd]:
        b.cx(br(outer, rep), rep)
    for outer, rep in FOLDS[rnd]:
        b.cx(outer, br(outer, rep))


def _round(b, rnd, names, cyc):
    _relay_layers(b, rnd)                        # fold
    for name in names:
        ctype, support, anc, _ = CHECK_DEFS[name]
        u, v = RUNG[anc]
        reps = [q for q in (u, v) if q in support]
        if ctype == "Z":
            for q in reps:
                b.cx(q, anc)
        else:
            b.h(anc)
            for q in reps:
                b.cx(anc, q)
            b.h(anc)
    for name in names:                           # measure (no reset)
        anc = ANC_OF[name]
        m = b.measure(anc)
        b.meas_idx[(name, cyc)] = m
        prev = b.prev_meas.get(anc)
        b.chain[(name, cyc)] = [m] if prev is None else [m, prev]
        b.prev_meas[anc] = m
    _relay_layers(b, rnd)                        # unfold


def build_qpu_stim_circuit(num_cycles=3, noise_type="X", p=0.0,
                           profile=None, inject=None):
    """Hardware-shaped 37q Stim circuit (initial state |0>_L only).

    Args:
        num_cycles: QEC cycles (default 3, same as the HW experiment)
        noise_type: injected error type "X"|"Z" (Error_Type)
        p:          injected error probability (Error_Rate)
        profile:    qpu_avg_v1 profile dict (None -> noiseless circuit,
                    used by the verification gate)
        inject:     (pauli, data_phys, after_cycle) — deterministic single
                    error for verification (same meaning as the Aer
                    circuit's inject argument)
    """
    b = _Builder(profile)
    b.c.append("R", list(range(len(ALL_PHYS))))
    if p > 0:
        b.c.append("X_ERROR" if noise_type == "X" else "Z_ERROR",
                   [L[q] for q in DATA_PHYS], p)

    for cyc in range(num_cycles):
        _round(b, 1, ROUND1, cyc)
        _round(b, 2, ROUND2, cyc)
        if inject is not None and inject[2] == cyc:
            pauli, dq, _ = inject
            b.c.append(pauli.upper(), [L[dq]])

    fin = {}
    for q in DATA_PHYS:                          # final data readout
        fin[q] = b.measure(q)

    # ---- detectors: XOR expansion of the raw record, in the exact
    # ---- order of heavyhex33_stim._append_detectors -------------------
    M = b.n_meas
    assert M == N_CHECKS * num_cycles + NUM_DATA
    rec = lambda k: stim.target_rec(k - M)  # noqa: E731

    def value_set(name, cyc):
        return set(b.chain[(name, cyc)])

    for cyc in range(num_cycles):
        for name in CHECK_AT:
            if name in Z_STABS:
                s = value_set(name, cyc)
                if cyc >= 1:
                    s = s ^ value_set(name, cyc - 1)
                b.c.append("DETECTOR", [rec(k) for k in sorted(s)])
            elif cyc >= 1:
                s = value_set(name, cyc) ^ value_set(name, cyc - 1)
                b.c.append("DETECTOR", [rec(k) for k in sorted(s)])
    for j in Z_POS:
        name = CHECK_AT[j]
        s = {fin[qp] for qp in CHECK_DEFS[name][1]}
        s ^= value_set(name, num_cycles - 1)
        b.c.append("DETECTOR", [rec(k) for k in sorted(s)])
    b.c.append("OBSERVABLE_INCLUDE",
               [rec(fin[DATA_PHYS[i]]) for i in LOGICAL_Z_IDX], 0)
    assert b.c.num_detectors == num_detectors(num_cycles)
    return b.c


def sample_qpu_flips(circuit, shots, num_cycles, seed=None):
    """FlipSimulator flips of the RAW record -> (check-value flip matrix
    (shots, 16C), final-data flips (shots, 17)).

    The raw syn flips pass through the same per-ancilla XOR chain as
    hardware raw bits (check_values), which is linear, so the result is
    the flip of each CHECK VALUE — directly comparable to the abstract
    circuit's MR bits and consumable by syndrome_tensor / the detector
    reconstructions (same justification as heavyhex33_stim.sample_flips)."""
    fs = stim.FlipSimulator(batch_size=shots, seed=seed,
                            disable_stabilizer_randomization=True)
    fs.do(circuit)
    mf = fs.get_measurement_flips().T.astype(np.uint8)
    raw_syn, dat = split_stim_sample(mf, num_cycles)
    vals = check_values(raw_syn, num_cycles)
    return check_matrix_from_dict(vals, num_cycles), dat


if __name__ == "__main__":
    c = build_qpu_stim_circuit(3)
    print(f"noiseless: qubits={c.num_qubits} meas={c.num_measurements} "
          f"detectors={c.num_detectors} (expected {num_detectors(3)})")
    det = c.compile_detector_sampler().sample(shots=256,
                                              append_observables=True)
    print(f"noiseless detectors+observable all-zero: "
          f"{np.asarray(det).sum() == 0}")
