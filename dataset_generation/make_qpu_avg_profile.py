#!/usr/bin/env python3
"""
QPU calibration-average noise profile generator (mode: qpu_avg_v1)
==================================================================
Scans hardware/runs/*/ (the per-submission snapshot folders written by
hardware/run_hw.py), selects the latest N non-dry-run submissions of ONE
backend, averages their calibration snapshots, and registers the result
as a noise profile in noise_profiles.json:

  name: qpu/<backend>_avg<N>_<hash8>
        (hash8 = first 8 hex chars of sha256 over the sorted run-id
         list — a different run combination gets a different name, so
         datasets/checkpoints generated from it keep their lineage)

Per-run extraction (target.pkl preferred, properties.json fallback):
  * per-qubit readout error            (measure)
  * per-qubit 1Q gate error            (sx preferred, x fallback)
  * per-physical-edge 2Q gate error    (any 2-qubit op: ecr/cz/cx;
                                        directions averaged)
Values are arithmetically averaged across runs (per quantity, over the
runs where it is present), then mapped from device qubits to the 37q
patch labels through heavyhex_37q.embedding_for(backend).

The profile stores provenance (run ids, submitted_at, per-run source,
generation time) but NO local absolute paths. Runs of different backends
are never mixed — pass --backend if the runs folder contains several.

The profile is consumed by dataset_generation/heavyhex37_qpu_stim.py
(hardware-shaped 37q Stim circuit); the plain 4-parameter profiles keep
using heavyhex33_stim.build_stim_circuit. Profiles with a "mode" key are
excluded from the default training grid (ALL_NOISE) — select them
explicitly with -n qpu/<name>.

Usage:
  python dataset_generation/make_qpu_avg_profile.py                 # latest 5
  python dataset_generation/make_qpu_avg_profile.py --n-runs 3
  python dataset_generation/make_qpu_avg_profile.py --backend ibm_yonsei
  python dataset_generation/make_qpu_avg_profile.py --dry-run       # print only
"""
import argparse
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from heavyhex_circuits.heavyhex_37q import (  # noqa: E402
    ALL_PHYS, embedding_for, required_edges)


# ------------------------------------------------------------------
# per-run calibration extraction (device-qubit keyed)
# ------------------------------------------------------------------
def _extract_from_target(target):
    """qiskit Target -> (readout, err1q, err2q) dicts, device-indexed."""
    readout, err1q, err2q = {}, {}, {}

    def props_of(name):
        try:
            return target[name] or {}
        except Exception:
            return {}

    for qargs, p in props_of("measure").items():
        if qargs and p is not None and p.error is not None:
            readout[qargs[0]] = float(p.error)
    for gate in ("sx", "x"):                       # sx preferred
        props = props_of(gate)
        found = False
        for qargs, p in props.items():
            if qargs and p is not None and p.error is not None:
                err1q[qargs[0]] = float(p.error)
                found = True
        if found:
            break
    twoq = {}                                      # (u,v) sorted -> [errs]
    for name in target.operation_names:
        for qargs, p in props_of(name).items():
            if (qargs and len(qargs) == 2 and p is not None
                    and p.error is not None):
                twoq.setdefault(tuple(sorted(qargs)), []).append(
                    float(p.error))
    err2q = {e: sum(v) / len(v) for e, v in twoq.items()}
    return readout, err1q, err2q


def _extract_from_properties(props_dict):
    """BackendProperties.to_dict() JSON -> same three dicts."""
    readout, err1q, err2q = {}, {}, {}
    for q, items in enumerate(props_dict.get("qubits", [])):
        for it in items:
            if it.get("name") == "readout_error" and it.get("value") is not None:
                readout[q] = float(it["value"])
    onq = {}                                        # gate -> {q: err}
    twoq = {}
    for g in props_dict.get("gates", []):
        err = None
        for par in g.get("parameters", []):
            if par.get("name") == "gate_error" and par.get("value") is not None:
                err = float(par["value"])
        if err is None:
            continue
        qs = g.get("qubits", [])
        if len(qs) == 1:
            onq.setdefault(g.get("gate"), {})[qs[0]] = err
        elif len(qs) == 2:
            twoq.setdefault(tuple(sorted(qs)), []).append(err)
    for gate in ("sx", "x"):                        # sx preferred
        if onq.get(gate):
            err1q = onq[gate]
            break
    err2q = {e: sum(v) / len(v) for e, v in twoq.items()}
    return readout, err1q, err2q


def extract_run(run_dir):
    """One run folder -> (calib dicts, source name)."""
    tpkl = run_dir / "target.pkl"
    if tpkl.exists():
        try:
            with open(tpkl, "rb") as f:
                target = pickle.load(f)
            return _extract_from_target(target), "target.pkl"
        except Exception as e:
            print(f"   WARNING: {run_dir.name}: target.pkl unreadable "
                  f"({e}), falling back to properties.json")
    pjson = run_dir / "properties.json"
    if pjson.exists():
        return _extract_from_properties(json.load(open(pjson))), \
            "properties.json"
    raise FileNotFoundError(
        f"{run_dir.name}: neither target.pkl nor properties.json")


# ------------------------------------------------------------------
# run selection / averaging / patch mapping
# ------------------------------------------------------------------
def select_runs(runs_dir, backend, n_runs):
    """Latest n_runs non-dry-run folders of ONE backend (never mixed)."""
    cands = []
    for d in sorted(Path(runs_dir).iterdir()):
        meta_path = d / "job.json"
        if not d.is_dir() or not meta_path.exists():
            continue
        meta = json.load(open(meta_path))
        if meta.get("dry_run"):
            continue
        cands.append((meta.get("backend"), meta.get("submitted_at", ""), d))
    backends = sorted({b for b, _, _ in cands})
    if not cands:
        sys.exit(f"no non-dry-run submissions found under {runs_dir}")
    if backend is None:
        if len(backends) > 1:
            sys.exit(f"runs of several backends present ({backends}) — "
                     f"pick one with --backend (never mixed).")
        backend = backends[0]
    picked = sorted([c for c in cands if c[0] == backend],
                    key=lambda c: c[1], reverse=True)[:n_runs]
    if not picked:
        sys.exit(f"no non-dry-run submissions of backend '{backend}' "
                 f"(available: {backends})")
    if len(picked) < n_runs:
        print(f"WARNING: only {len(picked)} run(s) of '{backend}' available "
              f"(requested {n_runs}) — averaging over what exists.")
    return backend, picked


def average_profiles(per_run):
    """List of (readout, err1q, err2q) -> arithmetic mean per key
    (over the runs where the key is present)."""
    out = []
    for slot in range(3):
        acc = {}
        for dicts in per_run:
            for k, v in dicts[slot].items():
                acc.setdefault(k, []).append(v)
        out.append({k: sum(v) / len(v) for k, v in acc.items()})
    return out


def map_to_patch(backend, readout, err1q, err2q):
    """Device-qubit keyed calib -> 37q patch-label keyed profile fields."""
    emb = embedding_for(backend)
    miss = []
    p_read, p_1q, p_2q = {}, {}, {}
    for p in ALL_PHYS:
        dq = emb[p]
        if dq in readout:
            p_read[str(p)] = readout[dq]
        else:
            miss.append(f"readout q{dq}")
        if dq in err1q:
            p_1q[str(p)] = err1q[dq]
        else:
            miss.append(f"1q q{dq}")
    for u, v in required_edges():
        dev = tuple(sorted((emb[u], emb[v])))
        key = f"{min(u, v)}-{max(u, v)}"
        if dev in err2q:
            p_2q[key] = err2q[dev]
        else:
            miss.append(f"2q {dev}")
    if miss:
        sys.exit(f"calibration values missing for the 37q patch after "
                 f"averaging: {miss[:10]}{'...' if len(miss) > 10 else ''}")
    return p_read, p_1q, p_2q


def main():
    ap = argparse.ArgumentParser(
        description="Average QPU calibration snapshots into a noise profile")
    ap.add_argument("--runs-dir", default=str(_ROOT / "hardware" / "runs"))
    ap.add_argument("--n-runs", type=int, default=5,
                    help="latest N non-dry-run submissions (default 5)")
    ap.add_argument("--backend", default=None,
                    help="required only if runs of several backends exist")
    ap.add_argument("--profiles", default=str(_ROOT / "noise_profiles.json"))
    ap.add_argument("--dry-run", action="store_true",
                    help="print the profile instead of writing it")
    args = ap.parse_args()

    backend, picked = select_runs(args.runs_dir, args.backend, args.n_runs)
    print(f"backend {backend}: averaging {len(picked)} run(s)")
    per_run, sources = [], []
    for _, sub, d in picked:
        calib, src = extract_run(d)
        per_run.append(calib)
        sources.append(src)
        print(f"   {d.name} ({sub}, {src}): "
              f"{len(calib[0])} readout / {len(calib[1])} 1q / "
              f"{len(calib[2])} 2q values")
    readout, err1q, err2q = average_profiles(per_run)
    p_read, p_1q, p_2q = map_to_patch(backend, readout, err1q, err2q)

    run_ids = [d.name for _, _, d in picked]
    h8 = hashlib.sha256(",".join(sorted(run_ids)).encode()).hexdigest()[:8]
    name = f"qpu/{backend}_avg{len(picked)}_{h8}"
    profile = {
        "mode": "qpu_avg_v1",
        "backend": backend,
        "n_runs": len(picked),
        "provenance": {
            "run_ids": run_ids,
            "submitted_at": [sub for _, sub, _ in picked],
            "source": sources,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "readout": p_read,
        "error_1q": p_1q,
        "error_2q": p_2q,
    }
    mean = lambda d: sum(d.values()) / len(d)  # noqa: E731
    print(f"profile {name}: mean readout={mean(p_read):.4f} "
          f"1q={mean(p_1q):.5f} 2q={mean(p_2q):.4f}")
    if args.dry_run:
        print(json.dumps({name: profile}, indent=2))
        return
    profiles = json.load(open(args.profiles))
    if name in profiles:
        print(f"NOTE: {name} already registered — overwriting (same run "
              f"combination, refreshed provenance).")
    profiles[name] = profile
    with open(args.profiles, "w") as f:
        json.dump(profiles, f, indent=2)
        f.write("\n")
    print(f"registered -> {args.profiles}")
    print(f"next: python dataset_generation/make_dataset.py -n {name} "
          f"--smoke   (gate: verification/verify_equivalence.py ALL PASS "
          f"first)")


if __name__ == "__main__":
    main()
