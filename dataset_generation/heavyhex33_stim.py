#!/usr/bin/env python3
"""
Heavy-Hex (3,3) abstract Stim circuit generator
===============================================
Builds the abstract-level circuit with 17 data + 8 ancilla qubits (no bridges).

Bit-level agreement contract with the hardware circuit
(heavyhex_depth7_opt_for_37q.HeavyHex37QDepthOpt):
  * per-cycle measurement order of the 16 checks
      = heavyhex_depth7_opt_for_37q.CYCLE_ORDER (imported)
  * stabilizer support of each check = heavyhex_37q.CHECK_DEFS (imported)
  * syn bit index = cyc * 16 + j  (j = position in CYCLE_ORDER) — same as HW

MR (measure+reset) vs no-reset equivalence (top-level contract, README §2):
  Stim treats every ancilla with MR, so each raw measurement bit IS the check
  value. The hardware never resets its ancillas, so a raw bit is the XOR
  accumulation with the same ancilla's previous measurement; the per-ancilla
  XOR chain in heavyhex_37q.check_values() (raw_j ^ raw_{j-1}) recovers the
  check value. Therefore the two streams are **identical at the check-value
  level**:
      Stim:      value[name][cyc] = raw_stim[cyc*16 + j]
      Hardware:  value[name][cyc] = check_values(raw_hw)[name][:, cyc]
  This equivalence is enforced by verification/verify_equivalence.py.

Detector definitions:
  * Z-check: deterministically 0 for the initial state |0>_L, so detectors
    exist from cycle 0 on: cycle 0 compares the value against the
    deterministic 0 (i.e. the value itself), and cycles >= 1 compare
    consecutive cycles (temporal XOR, the standard memory-experiment
    definition).
  * X-check: the first measurement is random -> only cycle-to-cycle XOR is a
    detector (from cycle 1 on).
  * Final: 8 Z-detectors based on the final data measurement
    (support parity ^ last Z-check value).
  * Observable: logical Z = parity of data [69, 87, 105]
    (converted to DATA_PHYS indices).

Noise model:
    - inject Error_Type ("X"/"Z") errors on data qubits with probability
      p (= Error_Rate) right after the initial reset
      (a deterministic per-shot Bernoulli(p) mask is statistically
       identical to the X_ERROR(p) probabilistic channel)
    - background noise profile (noise_profiles):
        data_depol: DEPOLARIZE1 on data at the start of each cycle
        gate_depol: DEPOLARIZE2 after each CX, DEPOLARIZE1 after each H
        meas_flip : X_ERROR right before measurement
        reset_flip: X_ERROR right after reset
"""

import json
import sys
from pathlib import Path

import numpy as np
import stim

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from heavyhex_circuits.heavyhex_37q import (  # noqa: E402
    DATA_PHYS, ANC_PHYS, CHECK_DEFS, Z_STABS, X_STABS, LOGICAL_Z, LOGICAL_X)
from heavyhex_circuits.heavyhex_depth7_opt_for_37q import (  # noqa: E402
    CYCLE_ORDER, N_CHECKS, RUNG)

# ------------------------------------------------------------------
# Stim qubit indices: data = order of DATA_PHYS (0..16), ancilla = 17..24
# ------------------------------------------------------------------
DIDX = {p: i for i, p in enumerate(DATA_PHYS)}          # phys -> stim data idx
AIDX = {a: 17 + i for i, a in enumerate(ANC_PHYS)}      # phys -> stim anc idx
NUM_DATA = len(DATA_PHYS)                               # 17
NUM_ANC = len(ANC_PHYS)                                 # 8
LOGICAL_Z_IDX = [DIDX[p] for p in LOGICAL_Z]            # final-data bit indices

# single distance number used in file names (d3): the larger of dz/dx,
# derived from the logical operator weights. For the (3,3) patch dz = dx = 3;
# a future (3,5)/(5,3) patch would yield d5, so names stay unambiguous.
DISTANCE = max(len(LOGICAL_Z), len(LOGICAL_X))

# position j in CYCLE_ORDER -> check name (syn bit = cyc*16 + j)
CHECK_AT = list(CYCLE_ORDER)
Z_POS = [j for j, n in enumerate(CHECK_AT) if n in Z_STABS]   # 8 Z-check slots
X_POS = [j for j, n in enumerate(CHECK_AT) if n in X_STABS]   # 8 X-check slots

# ------------------------------------------------------------------
# Fixed Error_Type x Error_Rate grid and noise_profiles
# ------------------------------------------------------------------
ERROR_TYPES = ["X"]
ERROR_RATES = [0.005, 0.01, 0.05]

# noise_profiles.json (repo root) defines every runnable profile; all of them
# are generated/trained/evaluated by default. The noiseless profile is a
# fixture for verification/verify_equivalence.py only, so it stays out of the
# JSON (and out of ALL_NOISE).
with open(_ROOT / "noise_profiles.json") as _f:
    _JSON_PROFILES = json.load(_f)
ALL_NOISE = list(_JSON_PROFILES)
NOISE_PROFILES = {
    "ideal/dp0_mf0_rf0_gd0": {
        "data_depol": 0.0, "meas_flip": 0.0, "reset_flip": 0.0, "gate_depol": 0.0},
    **_JSON_PROFILES,
}


def noise_tag(noise_profile):
    """Short tag encoding the noise parameters, used in file/folder names.

    'realistic/dp0.001_mf0.01_rf0.01_gd0.008' -> 'dp0.001_mf0.01_rf0.01_gd0.008'
    A parameter dict is formatted the same way."""
    if isinstance(noise_profile, str):
        return noise_profile.split("/")[-1]
    p = noise_profile
    return (f"dp{p['data_depol']}_mf{p['meas_flip']}"
            f"_rf{p['reset_flip']}_gd{p['gate_depol']}")

# ------------------------------------------------------------------
# 2D diamond embedding: ancilla -> (row-pair, col) on a 4x5 grid,
# derived from heavyhex_depth7_opt_for_37q.RUNG (= the rung definition of
# heavyhex_37q). A rung (u, v) is a vertical link between two adjacent data
# rows, so
#   row-pair = index of the data row containing u (0..3)
#   col      = horizontal lattice position of u // 2  (0..4)
# ------------------------------------------------------------------
GRID_SHAPE = (4, 5)


def _lattice_rows():
    """Recover the 5 data rows by splitting the sorted DATA_PHYS at gaps > 2."""
    rows, cur = [], [DATA_PHYS[0]]
    for q in DATA_PHYS[1:]:
        if q - cur[-1] <= 2:
            cur.append(q)
        else:
            rows.append(cur)
            cur = [q]
    rows.append(cur)
    assert len(rows) == 5, rows
    return rows


def _grid_col(q, rows):
    """Horizontal position (0..8) of data qubit q on the 9-wide grid,
    with each row centered."""
    for r in rows:
        if q in r:
            offset = (9 - (2 * len(r) - 1)) // 2
            return offset + 2 * r.index(q)
    raise KeyError(q)


def _anc_coords():
    rows = _lattice_rows()
    row_of = {q: i for i, r in enumerate(rows) for q in r}
    coords = {}
    for anc, (u, v) in RUNG.items():
        cu, cv = _grid_col(u, rows), _grid_col(v, rows)
        assert cu == cv, f"rung {anc}: ({u},{v}) not vertically aligned"
        coords[anc] = (min(row_of[u], row_of[v]), cu // 2)
    assert len(set(coords.values())) == NUM_ANC, coords
    return coords


ANC_COORD = _anc_coords()
# {37:(0,2), 56:(1,1), 57:(1,3), 76:(2,0), 77:(2,2), 78:(2,4), 96:(3,1), 97:(3,3)}


# ==================================================================
# Stim circuit construction
# ==================================================================
def build_stim_circuit(num_cycles=3, noise_type="X", p=0.0,
                       noise_profile="ideal/dp0_mf0_rf0_gd0", inject=None):
    """Abstract (3,3) Stim circuit. Only the initial state |0>_L is supported.

    Args:
        num_cycles: number of QEC cycles (default 3, same as the HW experiment)
        noise_type: injected error type "X"|"Z" (Error_Type)
        p:          injected error probability (Error_Rate)
        noise_profile: a NOISE_PROFILES key or a parameter dict
        inject:     (pauli, data_phys, after_cycle) — deterministic single
                    error for verification (same meaning as the inject arg of
                    heavyhex_37q.HeavyHex37Q.build_circuit)
    """
    prof = (NOISE_PROFILES[noise_profile] if isinstance(noise_profile, str)
            else dict(noise_profile))
    dp, mf = prof["data_depol"], prof["meas_flip"]
    rf, gd = prof["reset_flip"], prof["gate_depol"]

    data = list(range(NUM_DATA))
    ancs = [AIDX[a] for a in ANC_PHYS]
    c = stim.Circuit()

    # initial reset (+ reset noise), then inject the data error right after
    # the reset
    c.append("R", data + ancs)
    if rf > 0:
        c.append("X_ERROR", data + ancs, rf)
    if p > 0:
        c.append("X_ERROR" if noise_type == "X" else "Z_ERROR", data, p)

    for cyc in range(num_cycles):
        if dp > 0:
            c.append("DEPOLARIZE1", data, dp)
        for name in CYCLE_ORDER:
            ctype, support, anc_phys, _ = CHECK_DEFS[name]
            a = AIDX[anc_phys]
            if ctype == "Z":
                for qp in support:
                    c.append("CX", [DIDX[qp], a])
                    if gd > 0:
                        c.append("DEPOLARIZE2", [DIDX[qp], a], gd)
            else:
                c.append("H", [a])
                if gd > 0:
                    c.append("DEPOLARIZE1", [a], gd)
                for qp in support:
                    c.append("CX", [a, DIDX[qp]])
                    if gd > 0:
                        c.append("DEPOLARIZE2", [a, DIDX[qp]], gd)
                c.append("H", [a])
                if gd > 0:
                    c.append("DEPOLARIZE1", [a], gd)
            if mf > 0:
                c.append("X_ERROR", [a], mf)
            c.append("MR", [a])
            if rf > 0:
                c.append("X_ERROR", [a], rf)
        if inject is not None and inject[2] == cyc:
            pauli, dq, _ = inject
            c.append(pauli.upper(), [DIDX[dq]])

    # final data measurement
    if mf > 0:
        c.append("X_ERROR", data, mf)
    c.append("M", data)

    _append_detectors(c, num_cycles)
    return c


def _append_detectors(c, num_cycles):
    """Attach detectors/observable at the end of the circuit via rec lookback.

    Measurement record layout: syn[cyc*16 + j] (j = CYCLE_ORDER position),
    followed by the 17 final data bits (DATA_PHYS order).
    Total M = 16*num_cycles + 17.
    Detector order (must match the reconstruction in baseline/mwpm.py):
      1) per cycle, iterate CYCLE_ORDER: Z-checks at every cycle
         (cycle 0: the value itself vs the deterministic 0; cycle >= 1:
         temporal XOR with the previous cycle), X-checks from cycle 1 on
         (temporal XOR)
      2) 8 final Z-detectors (order of Z-checks within CYCLE_ORDER)
    """
    M = N_CHECKS * num_cycles + NUM_DATA
    rec = lambda k: stim.target_rec(k - M)  # noqa: E731
    syn = lambda cyc, j: cyc * N_CHECKS + j  # noqa: E731
    fin = lambda i: N_CHECKS * num_cycles + i  # noqa: E731

    for cyc in range(num_cycles):
        for j, name in enumerate(CHECK_AT):
            if name in Z_STABS:
                if cyc == 0:
                    c.append("DETECTOR", [rec(syn(0, j))])
                else:
                    c.append("DETECTOR",
                             [rec(syn(cyc, j)), rec(syn(cyc - 1, j))])
            elif cyc >= 1:
                c.append("DETECTOR", [rec(syn(cyc, j)), rec(syn(cyc - 1, j))])
    for j in Z_POS:
        name = CHECK_AT[j]
        targets = [rec(fin(DIDX[qp])) for qp in CHECK_DEFS[name][1]]
        targets.append(rec(syn(num_cycles - 1, j)))
        c.append("DETECTOR", targets)
    c.append("OBSERVABLE_INCLUDE",
             [rec(fin(i)) for i in LOGICAL_Z_IDX], 0)


def num_detectors(num_cycles):
    """Z: 8*cycles, X: 8*(cycles-1), final: 8"""
    return 8 * num_cycles + 8 * (num_cycles - 1) + 8


# ==================================================================
# check-value / tensor utilities (representation shared by Stim and HW)
# ==================================================================
def split_stim_sample(raw, num_cycles):
    """compile_sampler() output (shots, 16C+17) -> (syn, dat).

    Stim uses MR on the ancillas, so the syn bits ARE the check values
    (hardware raw bits must first pass through the XOR chain of
    check_values())."""
    raw = np.asarray(raw, dtype=np.uint8)
    return raw[:, :N_CHECKS * num_cycles], raw[:, N_CHECKS * num_cycles:]


def sample_flips(circuit, shots, num_cycles, seed=None):
    """Sample MEASUREMENT FLIPS (error frame vs the noiseless reference)
    with stim.FlipSimulator -> (syn_flips, dat_flips).

    Why flips instead of raw measured values for the labels:
      The dual-use ancillas also measure X-checks, which project |0>_L into
      X-stabilizer eigenstates. As a result the INDIVIDUAL final data bits
      are random (only Z-stabilizer / logical-Z parities are deterministic),
      so raw final bits are NOT per-qubit X-error labels. The measurement
      flip of each final data qubit (net X frame at readout, including the
      final measurement flip) IS the well-defined per-qubit X-error label.

    Consistency guarantees (all quantities used downstream are XORs whose
    noiseless reference is deterministic, so flips == values there):
      * Z-check planes: reference is 0            -> flip == check value
      * X-check planes: cycle-to-cycle XOR        -> reference cancels
      * detectors:      all reference-deterministic XORs
      * logical label:  parity(dat_flips[LOGICAL_Z]) == parity of the
                        actually measured data bits (reference parity is 0)
    stabilizer randomization is disabled so that non-deterministic
    measurements are not artificially randomized (we track the pure error
    frame)."""
    fs = stim.FlipSimulator(batch_size=shots, seed=seed,
                            disable_stabilizer_randomization=True)
    fs.do(circuit)
    mf = fs.get_measurement_flips().T.astype(np.uint8)  # (shots, 16C+17)
    return split_stim_sample(mf, num_cycles)


def check_matrix_from_dict(vals, num_cycles):
    """heavyhex_37q.check_values() dict -> (shots, 16C) matrix.

    Bit index cyc*16 + j holds the cycle-cyc check value of CYCLE_ORDER[j]
    — the same layout as the raw Stim syn bits."""
    n = vals[CHECK_AT[0]].shape[0]
    mat = np.zeros((n, N_CHECKS * num_cycles), dtype=np.uint8)
    for j, name in enumerate(CHECK_AT):
        for cyc in range(num_cycles):
            mat[:, cyc * N_CHECKS + j] = vals[name][:, cyc]
    return mat


def syndrome_tensor(check_mat, num_cycles):
    """Check-value matrix (shots, 16C) -> 2D diamond-embedded tensor.

    Shape (shots, 2*num_cycles, 4, 5), channels [Z-plane, X-plane] x cycle:
      channel 2c   = Z-check detector plane (the value itself;
                     deterministically 0 for |0>_L)
      channel 2c+1 = X-check detector plane (cycle-to-cycle XOR; c=0 carries
                     no information because the first measurement is random,
                     so it is left as 0)
    The 8 cells of each plane sit at ANC_COORD (rung-derived diamond
    coordinates)."""
    shots = check_mat.shape[0]
    t = np.zeros((shots, 2 * num_cycles, *GRID_SHAPE), dtype=np.uint8)
    for j, name in enumerate(CHECK_AT):
        ctype, _, anc_phys, _ = CHECK_DEFS[name]
        r, col = ANC_COORD[anc_phys]
        for cyc in range(num_cycles):
            v = check_mat[:, cyc * N_CHECKS + j]
            if ctype == "Z":
                t[:, 2 * cyc, r, col] = v
            elif cyc >= 1:
                prev = check_mat[:, (cyc - 1) * N_CHECKS + j]
                t[:, 2 * cyc + 1, r, col] = v ^ prev
    return t


def detectors_from_dataset(check_mat, y_qubit, num_cycles):
    """(check-value matrix, final data 17 bits) -> Stim detector vector.

    Reconstructs detectors in exactly the order of _append_detectors().
    Used to feed hardware syndromes into MWPM (baseline/mwpm.py)."""
    shots = check_mat.shape[0]
    cols = []
    for cyc in range(num_cycles):
        for j, name in enumerate(CHECK_AT):
            v = check_mat[:, cyc * N_CHECKS + j]
            if name in Z_STABS:
                if cyc == 0:
                    cols.append(v)
                else:
                    cols.append(v ^ check_mat[:, (cyc - 1) * N_CHECKS + j])
            elif cyc >= 1:
                cols.append(v ^ check_mat[:, (cyc - 1) * N_CHECKS + j])
    for j in Z_POS:
        name = CHECK_AT[j]
        par = np.zeros(shots, dtype=np.uint8)
        for qp in CHECK_DEFS[name][1]:
            par ^= y_qubit[:, DIDX[qp]]
        cols.append(par ^ check_mat[:, (num_cycles - 1) * N_CHECKS + j])
    det = np.stack(cols, axis=1).astype(np.uint8)
    assert det.shape[1] == num_detectors(num_cycles)
    return det


def detectors_from_tensor(tensor, y_qubit):
    """Saved dataset (features tensor, y_qubit) -> Stim detector vector.

    Inverse view of syndrome_tensor(): the Z-plane values ARE the Z
    detectors and the X-plane values ARE the X detectors (cycle-to-cycle
    XOR), so they suffice to reconstruct all detectors (the raw X cycle-0
    bit is not used by any detector definition). Order matches
    _append_detectors()."""
    tensor = np.asarray(tensor, dtype=np.uint8)
    y_qubit = np.asarray(y_qubit, dtype=np.uint8)
    shots, ch = tensor.shape[0], tensor.shape[1]
    assert ch % 2 == 0
    num_cycles = ch // 2
    coord = {name: ANC_COORD[CHECK_DEFS[name][2]] for name in CHECK_AT}
    cols = []
    for cyc in range(num_cycles):
        for name in CHECK_AT:
            r, col = coord[name]
            if name in Z_STABS:
                if cyc == 0:
                    cols.append(tensor[:, 0, r, col])
                else:
                    cols.append(tensor[:, 2 * cyc, r, col]
                                ^ tensor[:, 2 * (cyc - 1), r, col])
            elif cyc >= 1:
                cols.append(tensor[:, 2 * cyc + 1, r, col])
    for j in Z_POS:
        name = CHECK_AT[j]
        r, col = coord[name]
        par = np.zeros(shots, dtype=np.uint8)
        for qp in CHECK_DEFS[name][1]:
            par ^= y_qubit[:, DIDX[qp]]
        cols.append(par ^ tensor[:, 2 * (num_cycles - 1), r, col])
    det = np.stack(cols, axis=1).astype(np.uint8)
    assert det.shape[1] == num_detectors(num_cycles)
    return det


def logical_label(y_qubit):
    """Final data 17 bits -> logical Z flip label (LOGICAL_Z parity)."""
    lab = np.zeros(y_qubit.shape[0], dtype=np.uint8)
    for i in LOGICAL_Z_IDX:
        lab ^= y_qubit[:, i]
    return lab


if __name__ == "__main__":
    c = build_stim_circuit(3, "X", 0.01,
                           "realistic/dp0.001_mf0.01_rf0.01_gd0.008")
    print(f"cycles=3  qubits={c.num_qubits}  measurements={c.num_measurements}")
    print(f"detectors={c.num_detectors} (expected {num_detectors(3)})")
    dem = c.detector_error_model(decompose_errors=True)
    print(f"DEM instructions: {len(dem)}")
    print(f"ANC_COORD = {ANC_COORD}")
