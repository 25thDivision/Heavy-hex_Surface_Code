#!/usr/bin/env python3
"""
dd_utils.py -- Dynamical Decoupling, production style (April pipeline).
=======================================================================
Replicates the original build_circuit_with_dd approach on the new builders:
  transpiled circuit -> ALAPScheduleAnalysis -> PadDynamicalDecoupling
  (backend-target timing, skip_reset_qubits=True), default sequence XY4,
  with XY8 / XX4 / XX2 options. NOTE (Qiskit >=2.x): the pass now requires
every sequence gate to exist in the target basis; Heron (cz,rz,sx,x) has no
native Y, so XY-family raises TranspilerError there. Default is therefore
XX4 -- the same adaptation this project already adopted for ibm_miami
("XX sequence due to native gate constraint"). XX suppresses the dominant
dephasing axis, which is DD's main role in this setting.

Usage in the pipeline (after transpile, before submission):

    from dd_utils import apply_dd
    qc_dd = apply_dd(qc_transpiled, backend.target, sequence="XY4")

For local testing without a real backend, pass a GenericBackendV2 target.
Note: Heron basis (cz, rz, sx, x) lacks a native Y; translate_to_basis=True
(default) runs a BasisTranslator afterwards so the output stays ISA-ready.
"""
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import ALAPScheduleAnalysis, PadDynamicalDecoupling
from qiskit.circuit.library import XGate, YGate

SEQUENCES = {
    'XY4': [XGate(), YGate(), XGate(), YGate()],
    'XY8': [XGate(), YGate(), XGate(), YGate(),
            YGate(), XGate(), YGate(), XGate()],
    'XX4': [XGate(), XGate(), XGate(), XGate()],
    'XX2': [XGate(), XGate()],
}


def apply_dd(qc_transpiled, target, sequence='XX4', translate_to_basis=True):
    """Insert DD pulses into idle gaps of a transpiled circuit.

    qc_transpiled: circuit already transpiled to the device (layout fixed,
                   basis gates, no routing needed).
    target:        backend.target (real) or GenericBackendV2(...).target.
    """
    seq = SEQUENCES[sequence]
    pm = PassManager([
        ALAPScheduleAnalysis(target=target),
        PadDynamicalDecoupling(target=target, dd_sequence=seq,
                               skip_reset_qubits=True),
    ])
    qc_dd = pm.run(qc_transpiled)
    if translate_to_basis and sequence.startswith('XY'):
        from qiskit.transpiler.passes import BasisTranslator
        from qiskit.circuit.equivalence_library import (
            SessionEquivalenceLibrary as sel)
        basis = [g for g in ('cz', 'rz', 'sx', 'x', 'id', 'delay',
                             'measure', 'reset', 'barrier')]
        qc_dd = PassManager([BasisTranslator(sel, basis)]).run(qc_dd)
    return qc_dd


def dd_pulse_stats(qc_dd):
    ops = qc_dd.count_ops()
    return {k: ops.get(k, 0) for k in ('x', 'y', 'sx', 'rz', 'delay')}
