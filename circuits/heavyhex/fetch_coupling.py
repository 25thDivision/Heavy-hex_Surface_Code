#!/usr/bin/env python3
"""
fetch_coupling.py -- dump backend coupling maps for the heavy-hex pipeline.
===============================================================================
Run this FIRST in any pipeline; every circuit builder consumes its output.

Usage:
  python fetch_coupling.py --token <API_KEY> --instance <CRN> ibm_boston ibm_yonsei
  # or with environment variables QISKIT_IBM_TOKEN / QISKIT_IBM_INSTANCE:
  python fetch_coupling.py ibm_boston

Writes coupling_<backend>.json next to this script (or --outdir), containing
{name, num_qubits, coupling_map, basis_gates} -- the exact format expected by
heavyhex_general.DepthOptDiamond / GeneralDiamondCircuit and by
heavyhex_37q.validate_backend.

Typical pipeline preamble:

    from fetch_coupling import fetch
    from heavyhex_37q import validate_backend          # (3,3) V6 path
    from heavyhex_depthopt import DepthOptDiamond         # (3,5)/(5,3) path

    path = fetch("ibm_yonsei", token=TOKEN, instance=CRN)
    validate_backend(path)                # raises if the 37q patch won't fit
    gc = DepthOptDiamond(3, 5, path)      # asserts its own edges internally
"""
import argparse
import json
import os
import sys


def fetch(backend_name, token=None, instance=None, outdir='.'):
    """Fetch one backend's coupling map; returns the written file path."""
    from qiskit_ibm_runtime import QiskitRuntimeService
    kwargs = {}
    tok = token or os.environ.get('QISKIT_IBM_TOKEN')
    ins = instance or os.environ.get('QISKIT_IBM_INSTANCE')
    if tok:
        kwargs['token'] = tok
    if ins:
        kwargs['instance'] = ins
    service = QiskitRuntimeService(**kwargs)      # saved account if no token
    b = service.backend(backend_name)
    info = {
        'name': backend_name,
        'num_qubits': b.num_qubits,
        'coupling_map': [list(e) for e in b.coupling_map],
        'basis_gates': list(b.operation_names),
    }
    path = os.path.join(outdir, f'coupling_{backend_name}.json')
    json.dump(info, open(path, 'w'))
    print(f"{backend_name}: {b.num_qubits}q, "
          f"{len(info['coupling_map'])} directed edges -> {path}")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('backends', nargs='+', help='backend names, e.g. ibm_boston')
    ap.add_argument('--token', default=None, help='IBM Quantum Platform API key')
    ap.add_argument('--instance', default=None, help='instance CRN string')
    ap.add_argument('--outdir', default='.', help='output directory')
    a = ap.parse_args()
    paths = []
    for name in a.backends:
        try:
            paths.append(fetch(name, a.token, a.instance, a.outdir))
        except Exception as e:
            print(f"ERROR fetching {name}: {e}", file=sys.stderr)
            sys.exit(1)
    return paths


if __name__ == '__main__':
    main()
