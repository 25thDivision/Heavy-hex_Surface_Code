#!/usr/bin/env python3
"""
QPU calibration-average noise profile generator (mode: qpu_avg_v1)
==================================================================
Scans hardware/runs/*/ (the per-submission snapshot folders written by
hardware/run_hw.py), selects the latest N non-dry-run submissions of ONE
backend AND one code family, averages their calibration snapshots, and
registers the result as a noise profile in noise_profiles.json:

  name: qpu/<backend>_<code>_avg<N>_<YYYYMMDD>[_<suffix>]
        <code>     heavyhex | surface
        <YYYYMMDD> date of the NEWEST submitted_at among the averaged
                   runs (human-readable lineage marker)
        <suffix>   optional --suffix for explicit disambiguation
  Lineage guard: if the key already exists in noise_profiles.json with a
  DIFFERENT provenance run-id list, the script aborts and prints the
  difference (datasets/checkpoints named after the key must never
  silently change meaning). Identical run lists just refresh provenance.
  Keys in the old hash format (qpu/<backend>_avg<N>_<hash8>) keep
  working (is_qpu_profile is mode-based) but are heavyhex-only;
  regeneration under the new naming is recommended.

Per-run extraction (target.pkl preferred, properties.json fallback):
  * per-qubit readout error            (measure)
  * per-qubit 1Q gate error            (sx preferred, x fallback)
  * per-physical-edge 2Q gate error    (any 2-qubit op: ecr/cz/cx;
                                        directions averaged)
Values are arithmetically averaged across runs (per quantity, over the
runs where it is present), then mapped from device qubits to patch
labels:
  * heavyhex: 37q patch physical labels (heavyhex_37q.embedding_for)
  * surface : rsc3 patch LOCAL indices 0..16 in rsc_circuits.rsc3
              ALL_COORDS order (data 0-8, ancillas 9-16 in CYCLE_ORDER),
              device qubits via rsc3.embedding_for_surface; edges from
              rsc3.required_edges_surface()

Run selection: job.json's backend / submitted_at / dry_run / code decide
membership — dry-runs are excluded, runs of another code are excluded
(a missing "code" field means heavyhex, the pre-code-axis format), and
runs of different backends are never mixed (pass --backend if the runs
folder contains several). The profile stores provenance (run ids,
submitted_at, per-run source, code, generation time) but NO local
absolute paths.

Consumers: heavyhex profiles -> heavyhex37_qpu_stim (37q circuit),
surface profiles -> rsc3_qpu_stim (17q circuit). Plain 4-parameter
profiles keep using the abstract generators. Profiles with a "mode" key
are excluded from the default training grid (ALL_NOISE) — select them
explicitly with -n qpu/<name>.

Usage:
  python dataset_generation/make_qpu_avg_profile.py                 # latest 5
  python dataset_generation/make_qpu_avg_profile.py --n-runs 3
  python dataset_generation/make_qpu_avg_profile.py --code surface \\
         --backend ibm_miami
  python dataset_generation/make_qpu_avg_profile.py --dry-run       # print only
"""
import argparse
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
def select_runs(runs_dir, backend, n_runs, code):
    """Latest n_runs non-dry-run folders of ONE backend and one code.

    job.json's "code" field filters the code family; runs without the
    field (pre-code-axis submissions) count as heavyhex. Backends are
    never mixed."""
    cands = []
    for d in sorted(Path(runs_dir).iterdir()):
        meta_path = d / "job.json"
        if not d.is_dir() or not meta_path.exists():
            continue
        meta = json.load(open(meta_path))
        if meta.get("dry_run"):
            continue
        if meta.get("code", "heavyhex") != code:
            continue
        cands.append((meta.get("backend"), meta.get("submitted_at", ""), d))
    backends = sorted({b for b, _, _ in cands})
    if not cands:
        sys.exit(f"no non-dry-run '{code}' submissions found under "
                 f"{runs_dir}")
    if backend is None:
        if len(backends) > 1:
            sys.exit(f"runs of several backends present ({backends}) — "
                     f"pick one with --backend (never mixed).")
        backend = backends[0]
    picked = sorted([c for c in cands if c[0] == backend],
                    key=lambda c: c[1], reverse=True)[:n_runs]
    if not picked:
        sys.exit(f"no non-dry-run '{code}' submissions of backend "
                 f"'{backend}' (available: {backends})")
    if len(picked) < n_runs:
        print(f"WARNING: only {len(picked)} run(s) of '{backend}'/{code} "
              f"available (requested {n_runs}) — averaging over what "
              f"exists.")
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


def _patch_layout(code, backend):
    """(patch labels, patch edges, label -> device qubit) of a code.

    heavyhex labels = 37q physical labels; surface labels = rsc3 LOCAL
    indices (ALL_COORDS order — the module's existing constant, no new
    labeling scheme)."""
    if code == "surface":
        from rsc_circuits.rsc3 import (
            ALL_COORDS, L, embedding_for_surface, required_edges_surface)
        emb_c = embedding_for_surface(backend)
        labels = [L[c] for c in ALL_COORDS]           # 0..16
        dev_of = {L[c]: emb_c[c] for c in ALL_COORDS}
        edges = [(L[u], L[v]) for u, v in required_edges_surface()]
        return labels, edges, dev_of
    emb = embedding_for(backend)
    return list(ALL_PHYS), required_edges(), dict(emb)


def map_to_patch(backend, code, readout, err1q, err2q):
    """Device-qubit keyed calib -> patch-label keyed profile fields."""
    labels, edges, dev_of = _patch_layout(code, backend)
    miss = []
    p_read, p_1q, p_2q = {}, {}, {}
    for p in labels:
        dq = dev_of[p]
        if dq in readout:
            p_read[str(p)] = readout[dq]
        else:
            miss.append(f"readout q{dq}")
        if dq in err1q:
            p_1q[str(p)] = err1q[dq]
        else:
            miss.append(f"1q q{dq}")
    for u, v in edges:
        dev = tuple(sorted((dev_of[u], dev_of[v])))
        key = f"{min(u, v)}-{max(u, v)}"
        if dev in err2q:
            p_2q[key] = err2q[dev]
        else:
            miss.append(f"2q {dev}")
    if miss:
        sys.exit(f"calibration values missing for the {code} patch after "
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
    ap.add_argument("--code", choices=["heavyhex", "surface"],
                    default="heavyhex",
                    help="code family — filters runs by job.json's "
                         "'code' (missing = heavyhex) and selects the "
                         "patch mapping")
    ap.add_argument("--suffix", default=None,
                    help="optional explicit name suffix "
                         "(qpu/<backend>_<code>_avg<N>_<date>_<suffix>)")
    ap.add_argument("--profiles", default=str(_ROOT / "noise_profiles.json"))
    ap.add_argument("--dry-run", action="store_true",
                    help="print the profile instead of writing it")
    args = ap.parse_args()

    backend, picked = select_runs(args.runs_dir, args.backend, args.n_runs,
                                  args.code)
    print(f"backend {backend} / {args.code}: averaging {len(picked)} run(s)")
    per_run, sources = [], []
    for _, sub, d in picked:
        calib, src = extract_run(d)
        per_run.append(calib)
        sources.append(src)
        print(f"   {d.name} ({sub}, {src}): "
              f"{len(calib[0])} readout / {len(calib[1])} 1q / "
              f"{len(calib[2])} 2q values")
    readout, err1q, err2q = average_profiles(per_run)
    p_read, p_1q, p_2q = map_to_patch(backend, args.code,
                                      readout, err1q, err2q)

    run_ids = [d.name for _, _, d in picked]
    # date suffix = newest submitted_at among the averaged runs (YYYYMMDD)
    newest = max(sub for _, sub, _ in picked)
    date8 = newest[:10].replace("-", "")
    name = f"qpu/{backend}_{args.code}_avg{len(picked)}_{date8}"
    if args.suffix:
        name += f"_{args.suffix}"
    profile = {
        "mode": "qpu_avg_v1",
        "backend": backend,
        "code": args.code,
        "n_runs": len(picked),
        "provenance": {
            "run_ids": run_ids,
            "submitted_at": [sub for _, sub, _ in picked],
            "source": sources,
            "code": args.code,
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
        # lineage guard: the same key must always mean the same runs —
        # datasets/checkpoints carry the key in their names
        old_ids = sorted(profiles[name].get("provenance", {})
                         .get("run_ids", []))
        if old_ids != sorted(run_ids):
            sys.exit(
                f"REFUSING to overwrite {name}: it is already registered "
                f"with a different run combination.\n"
                f"  registered: {old_ids}\n"
                f"  selected  : {sorted(run_ids)}\n"
                f"Use --suffix to register the new combination under a "
                f"distinct name (lineage of existing datasets/checkpoints "
                f"stays intact).")
        print(f"NOTE: {name} already registered with the same runs — "
              f"refreshing provenance.")
    profiles[name] = profile
    with open(args.profiles, "w") as f:
        json.dump(profiles, f, indent=2)
        f.write("\n")
    print(f"registered -> {args.profiles}")
    gate = ("verification/verify_equivalence.py" if args.code == "heavyhex"
            else "verification/verify_rsc3.py")
    print(f"next: python dataset_generation/make_dataset.py --code "
          f"{args.code} -n {name} --smoke   (gate: {gate} ALL PASS first)")


if __name__ == "__main__":
    main()
