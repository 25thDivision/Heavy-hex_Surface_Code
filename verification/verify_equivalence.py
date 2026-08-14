#!/usr/bin/env python3
"""
Contract gate (README §2): bit-level equivalence between the abstract Stim
circuit and the hardware depth-7 circuit
==========================================================================
Dataset generation (T2) must NOT proceed until this script prints ALL PASS.

  [A] With zero noise, every Stim detector is deterministic
      (detector samples are all zero)
  [B] Stim raw-stream invariants: Z-check values are 0, X-check
      cycle-to-cycle XORs are 0, final data parity == last Z-check value,
      logical Z parity is 0
  [C] The hardware circuit (noiseless Aer) satisfies the same invariants
      through check_values(), with the same bit layout (cyc*16+j,
      CYCLE_ORDER)
  [D] For injected single errors, the set of checks firing at cycle 1 is
      identical between Stim and Aer (semantic confirmation of
      order/meaning agreement)
  [E] Hardware-shaped 37q Stim circuit (heavyhex37_qpu_stim, the QPU
      calibration-profile generator; bridges + no-reset ancillas):
      (a) with zero noise, every detector is deterministic
      (b) injected single errors fire the same checks as Aer (raw
          measured values through the per-ancilla XOR chain, like [D])
      (c) the deterministic detector stream (incl. the final-Z detectors
          and the observable) is bit-identical to the abstract 25q Stim
          circuit's — check-value-level equivalence of the XOR chain vs
          MR, the same contract as hardware vs Stim (a deterministic
          detector stream determines the Z-check value stream by
          telescoping, and the X-check stream up to its random cycle-0
          reference)
      Dataset generation from qpu/* profiles must NOT proceed until this
      section passes together with the rest (ALL PASS).
"""
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from qiskit_aer import AerSimulator

from heavyhex_circuits.heavyhex_37q import Z_STABS, X_STABS
from heavyhex_circuits.heavyhex_depth7_opt_for_37q import (
    HeavyHex37QDepthOpt, check_values, N_CHECKS)
from dataset_generation.heavyhex33_stim import (  # noqa: E402
    build_stim_circuit, split_stim_sample, check_matrix_from_dict,
    num_detectors, DIDX, LOGICAL_Z_IDX, CHECK_AT)
from dataset_generation.heavyhex37_qpu_stim import (  # noqa: E402
    build_qpu_stim_circuit)

STIM_SHOTS = 512
AER_SHOTS = 400
CYCLES = 3
DET_EPS = 0.002


def stim_run(inject=None):
    c = build_stim_circuit(CYCLES, "X", 0.0, "ideal/dp0_mf0_rf0_gd0", inject=inject)
    raw = c.compile_sampler().sample(shots=STIM_SHOTS)
    syn, dat = split_stim_sample(raw, CYCLES)
    return c, syn, dat, np.ones(syn.shape[0])


def aer_run(inject=None):
    qc = HeavyHex37QDepthOpt(CYCLES).build_circuit(0, inject)
    counts = AerSimulator().run(qc, shots=AER_SHOTS).result().get_counts()
    syn, dat, w = [], [], []
    for bs, cnt in counts.items():
        d, s = bs.split()
        syn.append([int(b) for b in s[::-1]])
        dat.append([int(b) for b in d[::-1]])
        w.append(cnt)
    syn, dat, w = np.array(syn, dtype=np.uint8), np.array(dat, dtype=np.uint8), np.array(w)
    vals = check_values(syn, CYCLES)
    return check_matrix_from_dict(vals, CYCLES), dat, w


def p1(bits, w):
    return float((bits * w).sum() / w.sum())


def invariants(tag, mat, dat, w):
    """Noiseless invariant checks on a check-value matrix (shots, 16C)."""
    fails = []
    pos = {name: j for j, name in enumerate(CHECK_AT)}
    for name in Z_STABS:
        for c in range(CYCLES):
            p = p1(mat[:, c * N_CHECKS + pos[name]], w)
            if p > DET_EPS:
                fails.append(f"Z:{name}@c{c}={p:.3f}")
    for name in X_STABS:
        for c in range(CYCLES - 1):
            x0 = mat[:, c * N_CHECKS + pos[name]]
            x1 = mat[:, (c + 1) * N_CHECKS + pos[name]]
            if (p := p1(x0 ^ x1, w)) > DET_EPS:
                fails.append(f"Xxor:{name}@c{c}={p:.3f}")
    for name, supp in Z_STABS.items():
        par = np.bitwise_xor.reduce(dat[:, [DIDX[q] for q in supp]], axis=1)
        if (p := p1(par ^ mat[:, (CYCLES - 1) * N_CHECKS + pos[name]], w)) > DET_EPS:
            fails.append(f"final:{name}={p:.3f}")
    zl = np.bitwise_xor.reduce(dat[:, LOGICAL_Z_IDX], axis=1)
    if (p := p1(zl, w)) > DET_EPS:
        fails.append(f"Z_L={p:.3f}")
    ok = not fails
    print(f"[{tag}] invariants: {'PASS' if ok else 'FAIL ' + str(fails[:8])}")
    return ok


def fired_at_cycle1(mat, w):
    """Set of checks whose detector fires at cycle 1 in an injection run."""
    fired = []
    pos = {name: j for j, name in enumerate(CHECK_AT)}
    for name in CHECK_AT:
        j = pos[name]
        if name in Z_STABS:
            p = p1(mat[:, 1 * N_CHECKS + j], w)
        else:
            p = p1(mat[:, 1 * N_CHECKS + j] ^ mat[:, 0 * N_CHECKS + j], w)
        if p > 1 - DET_EPS:
            fired.append(name)
        elif p > DET_EPS:
            fired.append(f"{name}?{p:.2f}")
    return sorted(fired)


def main():
    ok = True

    # [A] Stim detector determinism
    c = build_stim_circuit(CYCLES, "X", 0.0, "ideal/dp0_mf0_rf0_gd0")
    det = np.asarray(c.compile_detector_sampler().sample(
        shots=STIM_SHOTS, append_observables=True), dtype=np.uint8)
    a_ok = det.sum() == 0 and det.shape[1] == num_detectors(CYCLES) + 1
    print(f"[A] Stim noiseless detectors+observable all-zero: "
          f"{'PASS' if a_ok else 'FAIL'} (shape={det.shape}, sum={det.sum()})")
    ok &= a_ok

    # [B] Stim raw-stream invariants
    _, syn, dat, w = stim_run()
    ok &= invariants("B:Stim", syn, dat, w)

    # [C] Hardware-circuit invariants on Aer (same bit layout)
    aer_mat, aer_dat, aer_w = aer_run()
    assert aer_mat.shape[1] == syn.shape[1] == N_CHECKS * CYCLES
    ok &= invariants("C:Aer", aer_mat, aer_dat, aer_w)

    # [D] single-error signature comparison (inject after cycle 0, read cycle 1)
    cases = [("X", 65), ("X", 25), ("X", 61), ("X", 89), ("X", 105),
             ("Z", 65), ("Z", 43), ("Z", 69), ("Z", 107), ("Z", 81)]
    print("[D] single-error signatures (Stim vs Aer):")
    for pauli, dq in cases:
        _, s_syn, _, s_w = stim_run(inject=(pauli, dq, 0))
        a_mat, _, a_w = aer_run(inject=(pauli, dq, 0))
        f_stim = fired_at_cycle1(s_syn, s_w)
        f_aer = fired_at_cycle1(a_mat, a_w)
        stabs = Z_STABS if pauli == "X" else X_STABS
        exp = sorted(n for n, s in stabs.items() if dq in s)
        good = f_stim == f_aer == exp
        ok &= good
        print(f"  {pauli} q{dq}: stim={f_stim} aer={f_aer} exp={exp} "
              f"{'PASS' if good else 'FAIL'}")

    # [E] hardware-shaped 37q Stim circuit (QPU-profile generator)
    # (a) noiseless detector determinism
    c37 = build_qpu_stim_circuit(CYCLES)
    det37 = np.asarray(c37.compile_detector_sampler().sample(
        shots=STIM_SHOTS, append_observables=True), dtype=np.uint8)
    ea_ok = (det37.sum() == 0
             and det37.shape[1] == num_detectors(CYCLES) + 1)
    print(f"[E-a] 37q HW-shaped Stim noiseless detectors+observable "
          f"all-zero: {'PASS' if ea_ok else 'FAIL'} "
          f"(shape={det37.shape}, sum={det37.sum()})")
    ok &= ea_ok

    # (b) single-error signatures vs Aer (raw values -> XOR chain, like
    #     the Aer path), (c) deterministic detector-stream equality vs
    #     the abstract 25q Stim circuit (incl. final-Z detectors).
    #     Detectors are reconstructed from RAW measured values with
    #     detectors_from_dataset (anchored to the |0>_L expectations) —
    #     stim's own detector sampler cannot be used for injected-Pauli
    #     checks, because it takes its reference from a noiseless run of
    #     the same circuit and absorbs the explicit Pauli into it.
    from dataset_generation.heavyhex33_stim import detectors_from_dataset

    def qpu_value_run(inject=None, shots=64):
        c = build_qpu_stim_circuit(CYCLES, inject=inject)
        raw = np.asarray(c.compile_sampler().sample(shots), dtype=np.uint8)
        syn, dat = split_stim_sample(raw, CYCLES)
        return check_matrix_from_dict(check_values(syn, CYCLES), CYCLES), dat

    def det_stream(mat, dat):
        """(deterministic?, first reconstructed detector row)"""
        det = detectors_from_dataset(mat, dat, CYCLES)
        return bool((det == det[0]).all()), det[0]

    print("[E-b/c] 37q Stim vs Aer signatures / vs 25q Stim detector "
          "streams:")
    for pauli, dq in cases:
        inject = (pauli, dq, 0)
        q_mat, q_dat = qpu_value_run(inject)
        f_qpu = fired_at_cycle1(q_mat, np.ones(q_mat.shape[0]))
        a_mat, _, a_w = aer_run(inject=inject)
        f_aer = fired_at_cycle1(a_mat, a_w)
        stabs = Z_STABS if pauli == "X" else X_STABS
        exp = sorted(n for n, s in stabs.items() if dq in s)
        _, s_syn, s_dat, _ = stim_run(inject=inject)
        det25_ok, det25 = det_stream(s_syn, s_dat)   # MR values directly
        det37_ok, det37 = det_stream(q_mat, q_dat)
        stream_ok = (det25_ok and det37_ok and np.array_equal(det25, det37)
                     and det25.any())                # must actually fire
        good = (f_qpu == f_aer == exp) and stream_ok
        ok &= good
        print(f"  {pauli} q{dq}: 37q={f_qpu} aer={f_aer} exp={exp} "
              f"detstream={'==' if stream_ok else '!='}25q "
              f"{'PASS' if good else 'FAIL'}")

    print(f"\nOVERALL: {'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
