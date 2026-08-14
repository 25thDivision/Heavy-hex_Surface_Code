#!/usr/bin/env python3
"""
Rotated surface code d=3 — abstract Stim circuit generator
==========================================================
Same conventions as the heavy-hex generator (heavyhex33_stim.py):

  * per-cycle measurement order = circuits.rotatedSurface.rotatedSurface3.CYCLE_ORDER
    (8 checks; syn bit index = cycle*8 + j) — identical to the hardware
    circuit's clbit layout
  * Stim treats every ancilla with MR; the hardware never resets, and
    rotatedSurface3.check_values() (per-ancilla XOR chain) recovers the check
    values — the two streams agree at the CHECK-VALUE level (same
    contract as heavy-hex, README §4)
  * CX schedule inside a cycle = the SAME fixed hook-safe 4-layer order
    as the hardware circuit (rotatedSurface3.Z_CORNER_ORDER / X_CORNER_ORDER), so
    hook-error propagation is identical between Stim and Aer
    (verification/verify_rotatedSurface3.py checks this)
  * detectors: Z-checks from cycle 0 (anchored on the deterministic 0 of
    |0>_L), X-checks from cycle 1 (temporal XOR), then 4 final-Z
    detectors (final-data support parity ^ last Z-check value);
    observable = logical Z = parity of the data column x=1
  * labels: final-data MEASUREMENT FLIPS via stim.FlipSimulator (the
    X-checks project |0>_L, so individual final bits are random and only
    parities are deterministic — same subtlety as heavy-hex)
  * noise model: identical 4-parameter profiles (data_depol / meas_flip /
    reset_flip / gate_depol) applied at the same circuit positions

Tensor: (shots, 2*num_cycles, 4, 4) — the 8 ancillas on the 4x4
plaquette-vertex grid (rotatedSurface3.ANC_GRID), channels [Z-plane, X-plane] per
cycle exactly like the heavy-hex diamond tensor.
"""
import sys
from pathlib import Path

import numpy as np
import stim

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from circuits.rotatedSurface.rotatedSurface3 import (  # noqa: E402
    DATA_COORDS, DIDX, NUM_DATA, N_CHECKS, N_LAYERS, CYCLE_ORDER,
    CHECK_DEFS, Z_NAMES, X_NAMES, Z_POS, ANC_GRID, GRID_SHAPE,
    LOGICAL_Z_IDX, DISTANCE, check_values)
from dataset_generation.heavyhex33_stim import (  # noqa: E402
    NOISE_PROFILES, noise_tag, is_qpu_profile)

# Stim qubit indices: data 0..8 (DATA_COORDS order), ancillas 9..16
# (CYCLE_ORDER order) — same local layout as the hardware circuit
AIDX = {n: NUM_DATA + i for i, n in enumerate(CYCLE_ORDER)}


def num_detectors(num_cycles):
    """Z: 4*cycles, X: 4*(cycles-1), final: 4"""
    return 4 * num_cycles + 4 * (num_cycles - 1) + 4


def build_rotatedSurface3_stim_circuit(num_cycles=3, noise_type="X", p=0.0,
                            noise_profile="ideal/dp0_mf0_rf0_gd0",
                            inject=None, inject_ops=()):
    """Abstract rotatedSurface3 Stim circuit. Only the initial state |0>_L.

    Args mirror heavyhex33_stim.build_stim_circuit; additionally
    inject_ops = list of (pauli, ("data", coord)|("anc", name), cycle,
    after_layer) for hook-error verification (same meaning as
    rotatedSurface3.RotatedSurface3Hardware.build_circuit)."""
    if is_qpu_profile(noise_profile):
        raise ValueError("qpu_avg profiles are heavy-hex only for now")
    prof = (NOISE_PROFILES[noise_profile] if isinstance(noise_profile, str)
            else dict(noise_profile))
    dp, mf = prof["data_depol"], prof["meas_flip"]
    rf, gd = prof["reset_flip"], prof["gate_depol"]

    data = list(range(NUM_DATA))
    ancs = [AIDX[n] for n in CYCLE_ORDER]
    xancs = [AIDX[n] for n in X_NAMES]
    c = stim.Circuit()

    c.append("R", data + ancs)
    if rf > 0:
        c.append("X_ERROR", data + ancs, rf)
    if p > 0:
        c.append("X_ERROR" if noise_type == "X" else "Z_ERROR", data, p)

    for cyc in range(num_cycles):
        if dp > 0:
            c.append("DEPOLARIZE1", data, dp)
        c.append("H", xancs)
        if gd > 0:
            c.append("DEPOLARIZE1", xancs, gd)
        for layer in range(N_LAYERS):
            for name in CYCLE_ORDER:
                ctype = CHECK_DEFS[name][0]
                for ll, coord in CHECK_DEFS[name][3]:
                    if ll != layer:
                        continue
                    pair = ([DIDX[coord], AIDX[name]] if ctype == "Z"
                            else [AIDX[name], DIDX[coord]])
                    c.append("CX", pair)
                    if gd > 0:
                        c.append("DEPOLARIZE2", pair, gd)
            for pauli, tgt, icyc, ilayer in inject_ops:
                if icyc == cyc and ilayer == layer:
                    idx = (AIDX[tgt[1]] if tgt[0] == "anc"
                           else DIDX[tgt[1]])
                    c.append(pauli.upper(), [idx])
        c.append("H", xancs)
        if gd > 0:
            c.append("DEPOLARIZE1", xancs, gd)
        for name in CYCLE_ORDER:                       # MR in bit order
            a = AIDX[name]
            if mf > 0:
                c.append("X_ERROR", [a], mf)
            c.append("MR", [a])
            if rf > 0:
                c.append("X_ERROR", [a], rf)
        if inject is not None and inject[2] == cyc:
            pauli, dc, _ = inject
            c.append(pauli.upper(), [DIDX[dc]])

    if mf > 0:
        c.append("X_ERROR", data, mf)
    c.append("M", data)
    _append_detectors(c, num_cycles)
    return c


def _append_detectors(c, num_cycles):
    """Detector order (must match detectors_from_tensor/-dataset below):
      1) per cycle, iterate CYCLE_ORDER: Z-checks every cycle (cycle 0
         anchored on the deterministic 0), X-checks from cycle 1
      2) 4 final-Z detectors (Z_POS order)"""
    M = N_CHECKS * num_cycles + NUM_DATA
    rec = lambda k: stim.target_rec(k - M)  # noqa: E731
    syn = lambda cyc, j: cyc * N_CHECKS + j  # noqa: E731
    fin = lambda i: N_CHECKS * num_cycles + i  # noqa: E731

    for cyc in range(num_cycles):
        for j, name in enumerate(CYCLE_ORDER):
            if name in Z_NAMES:
                if cyc == 0:
                    c.append("DETECTOR", [rec(syn(0, j))])
                else:
                    c.append("DETECTOR",
                             [rec(syn(cyc, j)), rec(syn(cyc - 1, j))])
            elif cyc >= 1:
                c.append("DETECTOR", [rec(syn(cyc, j)), rec(syn(cyc - 1, j))])
    for j in Z_POS:
        name = CYCLE_ORDER[j]
        targets = [rec(fin(DIDX[qc])) for qc in CHECK_DEFS[name][1]]
        targets.append(rec(syn(num_cycles - 1, j)))
        c.append("DETECTOR", targets)
    c.append("OBSERVABLE_INCLUDE", [rec(fin(i)) for i in LOGICAL_Z_IDX], 0)


# ==================================================================
# sampling / tensor / detector utilities (mirror heavyhex33_stim)
# ==================================================================
def split_rotatedSurface3_sample(raw, num_cycles):
    raw = np.asarray(raw, dtype=np.uint8)
    return raw[:, :N_CHECKS * num_cycles], raw[:, N_CHECKS * num_cycles:]


def sample_flips_rotatedSurface3(circuit, shots, num_cycles, seed=None):
    """FlipSimulator measurement flips -> (syn check-value flips, data
    flips). Stim uses MR, so syn flips ARE check-value flips."""
    fs = stim.FlipSimulator(batch_size=shots, seed=seed,
                            disable_stabilizer_randomization=True)
    fs.do(circuit)
    mf = fs.get_measurement_flips().T.astype(np.uint8)
    return split_rotatedSurface3_sample(mf, num_cycles)


def syndrome_tensor_rotatedSurface3(check_mat, num_cycles):
    """(shots, 8C) check values -> (shots, 2C, 4, 4) tensor, channels
    [Z-plane (value), X-plane (cycle-to-cycle XOR; c=0 stays 0)]."""
    shots = check_mat.shape[0]
    t = np.zeros((shots, 2 * num_cycles, *GRID_SHAPE), dtype=np.uint8)
    for j, name in enumerate(CYCLE_ORDER):
        r, col = ANC_GRID[name]
        for cyc in range(num_cycles):
            v = check_mat[:, cyc * N_CHECKS + j]
            if name in Z_NAMES:
                t[:, 2 * cyc, r, col] = v
            elif cyc >= 1:
                prev = check_mat[:, (cyc - 1) * N_CHECKS + j]
                t[:, 2 * cyc + 1, r, col] = v ^ prev
    return t


def check_matrix_from_dict_rotatedSurface3(vals, num_cycles):
    n = vals[CYCLE_ORDER[0]].shape[0]
    mat = np.zeros((n, N_CHECKS * num_cycles), dtype=np.uint8)
    for j, name in enumerate(CYCLE_ORDER):
        for cyc in range(num_cycles):
            mat[:, cyc * N_CHECKS + j] = vals[name][:, cyc]
    return mat


def detectors_from_dataset_rotatedSurface3(check_mat, y_qubit, num_cycles):
    """(check values, final 9 bits) -> detector vectors in
    _append_detectors order (hardware/MWPM path)."""
    shots = check_mat.shape[0]
    cols = []
    for cyc in range(num_cycles):
        for j, name in enumerate(CYCLE_ORDER):
            v = check_mat[:, cyc * N_CHECKS + j]
            if name in Z_NAMES:
                if cyc == 0:
                    cols.append(v)
                else:
                    cols.append(v ^ check_mat[:, (cyc - 1) * N_CHECKS + j])
            elif cyc >= 1:
                cols.append(v ^ check_mat[:, (cyc - 1) * N_CHECKS + j])
    for j in Z_POS:
        name = CYCLE_ORDER[j]
        par = np.zeros(shots, dtype=np.uint8)
        for qc in CHECK_DEFS[name][1]:
            par ^= y_qubit[:, DIDX[qc]]
        cols.append(par ^ check_mat[:, (num_cycles - 1) * N_CHECKS + j])
    det = np.stack(cols, axis=1).astype(np.uint8)
    assert det.shape[1] == num_detectors(num_cycles)
    return det


def detectors_from_tensor_rotatedSurface3(tensor, y_qubit):
    """Saved dataset (features, labels) -> detector vectors (MWPM path)."""
    tensor = np.asarray(tensor, dtype=np.uint8)
    y_qubit = np.asarray(y_qubit, dtype=np.uint8)
    shots, ch = tensor.shape[0], tensor.shape[1]
    assert ch % 2 == 0
    num_cycles = ch // 2
    cols = []
    for cyc in range(num_cycles):
        for name in CYCLE_ORDER:
            r, col = ANC_GRID[name]
            if name in Z_NAMES:
                if cyc == 0:
                    cols.append(tensor[:, 0, r, col])
                else:
                    cols.append(tensor[:, 2 * cyc, r, col]
                                ^ tensor[:, 2 * (cyc - 1), r, col])
            elif cyc >= 1:
                cols.append(tensor[:, 2 * cyc + 1, r, col])
    for j in Z_POS:
        name = CYCLE_ORDER[j]
        r, col = ANC_GRID[name]
        par = np.zeros(shots, dtype=np.uint8)
        for qc in CHECK_DEFS[name][1]:
            par ^= y_qubit[:, DIDX[qc]]
        cols.append(par ^ tensor[:, 2 * (num_cycles - 1), r, col])
    det = np.stack(cols, axis=1).astype(np.uint8)
    assert det.shape[1] == num_detectors(num_cycles)
    return det


def logical_label_rotatedSurface3(y_qubit):
    """Final data 9 bits -> logical Z flip (column x=1 parity)."""
    lab = np.zeros(y_qubit.shape[0], dtype=np.uint8)
    for i in LOGICAL_Z_IDX:
        lab ^= y_qubit[:, i]
    return lab


if __name__ == "__main__":
    c = build_rotatedSurface3_stim_circuit(3, "X", 0.01,
                                "realistic/dp0.001_mf0.01_rf0.01_gd0.008")
    print(f"cycles=3 qubits={c.num_qubits} meas={c.num_measurements} "
          f"detectors={c.num_detectors} (expected {num_detectors(3)})")
    dem = c.detector_error_model(decompose_errors=True)
    print(f"DEM instructions: {len(dem)}")
    c0 = build_rotatedSurface3_stim_circuit(3)
    det = c0.compile_detector_sampler().sample(shots=256,
                                               append_observables=True)
    print(f"noiseless detectors all-zero: {np.asarray(det).sum() == 0}")
