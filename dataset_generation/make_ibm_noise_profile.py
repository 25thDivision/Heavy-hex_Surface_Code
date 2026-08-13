#!/usr/bin/env python3
"""
Create a location-dependent Stim noise profile from a saved IBM QPU run.

This script DOES NOT hard-code calibration numbers. It reads the exact
backend Target snapshot saved by hardware/run_hw.py at submission time:

    hardware/runs/<RUN_ID>/target.pkl
    hardware/runs/<RUN_ID>/job.json

and writes a profile into repo-root noise_profiles.json.

The generated profile is consumed by the patched
`dataset_generation/heavyhex33_stim.py` and mirrors the actual 37-qubit
HeavyHex37QDepthOpt gate schedule (bridge qubits + no-reset ancillas).

What is calibrated:
  * measurement/readout error per measured physical patch qubit
  * H-proxy error per ancilla (prefers sx error; falls back to mean 1Q error)
  * 2Q depolarizing error per physical hardware edge used by the 37q schedule

What is NOT claimed to be exact:
  * coherent/correlated/crosstalk errors
  * drift after the saved calibration
  * idle/T1/T2 noise during delays/DD
  * exact conversion of IBM-reported gate infidelity into a Pauli channel

So this is a calibration-aware, location-dependent approximation, not a
full reproduction of the hardware noise process.

Examples
--------
Preview only:
  python dataset_generation/make_ibm_noise_profile.py \
      --run-id d9u255c98n5s73929m5g --print-only

Write to noise_profiles.json:
  python dataset_generation/make_ibm_noise_profile.py \
      --run-id d9u255c98n5s73929m5g

Optionally choose a profile key:
  python dataset_generation/make_ibm_noise_profile.py \
      --run-id d9u255c98n5s73929m5g \
      --name ibm/yonsei_d9u255
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from pathlib import Path
from statistics import mean

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from heavyhex_circuits.heavyhex_37q import (  # noqa: E402
    ALL_PHYS, DATA_PHYS, ANC_PHYS, embedding_for, br,
)
from heavyhex_circuits.heavyhex_depth7_opt_for_37q import (  # noqa: E402
    FOLDS, RUNG,
)


def _finite_prob(x):
    """Return x as a valid probability, or None."""
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    return min(max(x, 0.0), 1.0)


def _prop_error(prop):
    return _finite_prob(getattr(prop, "error", None))


def _target_ops(target):
    """Yield (operation_name, qargs, instruction_properties)."""
    for op_name in target.operation_names:
        try:
            qmap = target[op_name]
        except Exception:
            continue
        if not hasattr(qmap, "items"):
            continue
        for qargs, prop in qmap.items():
            if qargs is None:
                continue
            try:
                qargs = tuple(int(q) for q in qargs)
            except TypeError:
                continue
            yield op_name, qargs, prop


def _collect_target_errors(target):
    """Collect calibration errors indexed by device qubit / edge."""
    measure = {}
    reset = {}
    oneq = {}
    sx = {}
    twoq = {}
    twoq_names = {}

    for name, qargs, prop in _target_ops(target):
        err = _prop_error(prop)
        if err is None:
            continue
        lname = name.lower()
        if len(qargs) == 1:
            q = qargs[0]
            if lname == "measure":
                measure[q] = err
            elif lname == "reset":
                reset[q] = err
            elif lname not in {"delay", "barrier"}:
                oneq.setdefault(q, []).append(err)
                if lname == "sx":
                    sx.setdefault(q, []).append(err)
        elif len(qargs) == 2:
            e = tuple(sorted(qargs))
            twoq.setdefault(e, []).append(err)
            twoq_names.setdefault(e, set()).add(lname)

    oneq_mean = {q: mean(v) for q, v in oneq.items() if v}
    sx_mean = {q: mean(v) for q, v in sx.items() if v}
    twoq_mean = {e: mean(v) for e, v in twoq.items() if v}
    return measure, reset, oneq_mean, sx_mean, twoq_mean, twoq_names


def _schedule_edges_patch():
    """Undirected PATCH-LABEL edges physically used by HeavyHex37QDepthOpt."""
    edges = set()
    for rnd in (1, 2):
        for outer, rep in FOLDS[rnd]:
            b = br(outer, rep)
            edges.add(tuple(sorted((outer, b))))
            edges.add(tuple(sorted((b, rep))))
    for anc, (u, v) in RUNG.items():
        edges.add(tuple(sorted((anc, u))))
        edges.add(tuple(sorted((anc, v))))
    return sorted(edges)


def _patch_key(q):
    return str(int(q))


def _edge_key(u, v):
    a, b = sorted((int(u), int(v)))
    return f"{a}-{b}"


def build_profile(run_dir: Path, name: str | None = None):
    job_path = run_dir / "job.json"
    target_path = run_dir / "target.pkl"
    if not job_path.exists():
        raise FileNotFoundError(f"missing {job_path}")
    if not target_path.exists():
        raise FileNotFoundError(f"missing {target_path}")

    job = json.load(open(job_path, encoding="utf-8"))
    backend = job.get("backend")
    if not backend:
        raise RuntimeError(f"{job_path} has no 'backend' field")

    with open(target_path, "rb") as f:
        target = pickle.load(f)

    emb = embedding_for(backend)  # patch label -> device qubit
    inv_emb = {int(dev): int(patch) for patch, dev in emb.items()}

    measure, reset, oneq, sx, twoq, twoq_names = _collect_target_errors(target)

    # Per-patch-qubit readout. Only data + ancilla are measured in the code;
    # bridge values are kept for provenance if available.
    readout_patch = {}
    h_patch = {}
    reset_patch = {}
    oneq_patch = {}
    missing_readout = []

    for patch in ALL_PHYS:
        dev = int(emb[patch])
        if dev in measure:
            readout_patch[_patch_key(patch)] = measure[dev]
        elif patch in DATA_PHYS or patch in ANC_PHYS:
            missing_readout.append((patch, dev))
        if dev in oneq:
            oneq_patch[_patch_key(patch)] = oneq[dev]
        if dev in reset:
            reset_patch[_patch_key(patch)] = reset[dev]
        # H on the depth-opt circuit occurs on ancillas. IBM typically
        # implements H using virtual RZ + SX-like physical pulses, so sx is
        # the preferred calibration proxy; mean 1Q error is the fallback.
        if patch in ANC_PHYS:
            if dev in sx:
                h_patch[_patch_key(patch)] = sx[dev]
            elif dev in oneq:
                h_patch[_patch_key(patch)] = oneq[dev]

    # Physical 2Q calibration for every hardware edge actually used by the
    # depth-optimized schedule, converted back to patch labels.
    twoq_patch = {}
    twoq_gate_names = {}
    missing_2q = []
    for pu, pv in _schedule_edges_patch():
        du, dv = int(emb[pu]), int(emb[pv])
        de = tuple(sorted((du, dv)))
        key = _edge_key(pu, pv)
        if de in twoq:
            twoq_patch[key] = twoq[de]
            twoq_gate_names[key] = sorted(twoq_names.get(de, []))
        else:
            missing_2q.append(((pu, pv), (du, dv)))

    if missing_readout:
        print("WARNING: missing readout calibration for patch/device qubits:")
        print("   ", missing_readout)
    if missing_2q:
        print("WARNING: missing 2Q calibration for required patch/device edges:")
        for x in missing_2q:
            print("   ", x)

    run_id = run_dir.name
    short = run_id[:8]
    if name is None:
        clean_backend = backend.removeprefix("ibm_")
        name = f"ibm/{clean_backend}_{short}"

    # IMPORTANT: reset_error is recorded for provenance, but the patched
    # hardware-shaped Stim circuit does NOT apply cycle reset noise because
    # HeavyHex37QDepthOpt uses no-reset ancillas. The initial Stim R only
    # establishes the known |0> starting state.
    profile = {
        "mode": "ibm_calibration_v1",
        "backend": backend,
        "run_id": run_id,
        "source": str(run_dir.as_posix()),
        "embedding_patch_to_device": {
            _patch_key(p): int(emb[p]) for p in ALL_PHYS
        },
        "readout_error": readout_patch,
        "h_error": h_patch,
        "one_qubit_error": oneq_patch,
        "two_qubit_error": twoq_patch,
        "two_qubit_gate_names": twoq_gate_names,
        "reset_error": reset_patch,
        "model_notes": {
            "schedule": "HeavyHex37QDepthOpt physical 37q schedule",
            "ancilla_reset": "no-reset, matching hardware",
            "idle_noise": "not modeled",
            "correlated_noise": "not modeled",
            "gate_channel": "IBM gate infidelity used as local depolarizing probability proxy",
        },
    }
    return name, profile


def _summary(name, profile):
    def stats(d):
        vals = [float(x) for x in d.values()]
        if not vals:
            return "n=0"
        return (f"n={len(vals)}, mean={mean(vals):.6g}, "
                f"min={min(vals):.6g}, max={max(vals):.6g}")

    print(f"profile: {name}")
    print(f"backend: {profile['backend']}")
    print(f"run_id : {profile['run_id']}")
    print(f"readout: {stats(profile['readout_error'])}")
    print(f"H proxy: {stats(profile['h_error'])}")
    print(f"1Q     : {stats(profile['one_qubit_error'])}")
    print(f"2Q     : {stats(profile['two_qubit_error'])}")
    print(f"reset  : {stats(profile['reset_error'])} (recorded; not used per cycle)")


def main():
    ap = argparse.ArgumentParser(
        description="Build location-dependent IBM calibration noise profile")
    ap.add_argument("--run-id", required=True,
                    help="hardware/runs/<RUN_ID> directory name")
    ap.add_argument("--name", default=None,
                    help="profile key, e.g. ibm/yonsei_d9u255")
    ap.add_argument("--runs-dir", default=str(_ROOT / "hardware" / "runs"))
    ap.add_argument("--profiles", default=str(_ROOT / "noise_profiles.json"),
                    help="noise_profiles.json to update")
    ap.add_argument("--print-only", action="store_true")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace an existing profile with the same key")
    args = ap.parse_args()

    run_dir = Path(args.runs_dir) / args.run_id
    name, profile = build_profile(run_dir, args.name)
    _summary(name, profile)

    if args.print_only:
        print(json.dumps({name: profile}, indent=2))
        return

    profiles_path = Path(args.profiles)
    profiles = json.load(open(profiles_path, encoding="utf-8")) if profiles_path.exists() else {}
    if name in profiles and not args.overwrite:
        raise SystemExit(
            f"profile '{name}' already exists in {profiles_path}; "
            f"use --overwrite or choose --name")
    profiles[name] = profile
    with open(profiles_path, "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=2)
        f.write("\n")
    print(f"saved -> {profiles_path}")
    print(f"next noise key: {name}")


if __name__ == "__main__":
    main()