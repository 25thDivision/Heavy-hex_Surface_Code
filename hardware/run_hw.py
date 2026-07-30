#!/usr/bin/env python3
"""
Hardware validation on IBM Quantum backends
===========================================
Pipeline (submit):
  keys.json -> fetch_coupling.fetch() -> validate_backend()
    -> HeavyHex37QDepthOpt circuit -> transpile(initial_layout=ALL_PHYS,
    optimization_level=1) -> dd_utils.apply_dd(tqc, backend.target)
    -> QPU environment snapshot -> SamplerV2 submit
Pipeline (analyze):
  raw results -> check_values() XOR-chain syndrome recovery
    -> syndrome tensor -> CNN + MWPM decoding -> LER report

Every submission gets its own folder so the run can be re-analyzed (and
the QPU environment of that moment stays on record):

  hardware/runs/<job_id>/          (dry-run: dryrun_<YYYYMMDD-HHMMSS>/)
    job.json             backend, cycles, shots, dd, timestamps,
                         local package versions, circuit stats
    coupling.json        coupling map / basis gates (fetch_coupling output)
    properties.json      backend.properties() calibration snapshot
                         (T1/T2, gate & readout errors, calibration times)
    configuration.json   backend.configuration() (dt, processor type, ...)
    target.pkl           backend.target pickle (what transpile/DD consumed)
    circuit.qpy          the exact transpiled+DD ISA circuit submitted
    raw.npz              measured raw bits (written by analyze)
    job_metrics.json     job timestamps/usage (written by analyze)

Notes:
  * Default backend is ibm_yonsei; switch with --backend ibm_boston.
  * Credentials come ONLY from keys.json at the repo root (copy
    keys.example.json; keys.json is gitignored). Never hardcode tokens/CRNs.
  * Hardware provides no per-qubit ground truth, so only LER is reported
    (ECR is simulation-only; see evaluation/metrics.py).
  * properties.json records the last calibration before the run — it is
    the best available record, not a live snapshot of drift. You can feed
    it to qiskit-aer's NoiseModel.from_backend() to re-simulate the run
    under that day's calibration.

Usage:
  python hardware/run_hw.py submit  [--backend ibm_yonsei] [--shots 50000]
                                    [--cycles 3] [--dd XX4] [--dry-run]
  python hardware/run_hw.py analyze --job-id <ID> --ckpt checkpoint/CNN_....pt
  python hardware/run_hw.py analyze --npz hardware/runs/<ID>/raw.npz --ckpt ...
"""
import argparse
import json
import pickle
import shutil
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from heavyhex_circuits.heavyhex_37q import ALL_PHYS, validate_backend  # noqa: E402
from heavyhex_circuits.heavyhex_depth7_opt_for_37q import (  # noqa: E402
    HeavyHex37QDepthOpt, check_values, N_CHECKS)
from heavyhex_circuits.fetch_coupling import fetch  # noqa: E402
from heavyhex_circuits.dd_utils import apply_dd, dd_pulse_stats  # noqa: E402

from dataset_generation.heavyhex33_stim import (  # noqa: E402
    check_matrix_from_dict, syndrome_tensor, logical_label, ACTIVE_NOISE)

RUNS_DIR = _ROOT / "hardware" / "runs"


def load_keys():
    path = _ROOT / "keys.json"
    if not path.exists():
        sys.exit(f"keys.json not found at {path}. "
                 f"Copy keys.example.json to keys.json and fill it in.")
    keys = json.load(open(path))
    for k in ("ibm_token", "ibm_instance"):
        if not keys.get(k) or keys[k].startswith("YOUR_"):
            sys.exit(f"keys.json: '{k}' is not set.")
    return keys


def get_service(keys):
    from qiskit_ibm_runtime import QiskitRuntimeService
    return QiskitRuntimeService(token=keys["ibm_token"],
                                instance=keys["ibm_instance"])


def _json_dump(obj, path):
    # default=str: calibration snapshots contain datetime objects
    json.dump(obj, open(path, "w"), indent=2, default=str)


def _local_versions():
    import qiskit
    import qiskit_ibm_runtime
    return {"python": sys.version.split()[0],
            "qiskit": qiskit.__version__,
            "qiskit_ibm_runtime": qiskit_ibm_runtime.__version__}


def snapshot_backend(backend, run_dir):
    """Save everything the backend exposes about the QPU at this moment.

    Best-effort: each item is saved independently so one missing API
    doesn't lose the others."""
    try:
        props = backend.properties()
        if props is not None:
            _json_dump(props.to_dict(), run_dir / "properties.json")
            print(f"   saved properties.json (calibration snapshot)")
    except Exception as e:
        print(f"   WARNING: could not save properties: {e}")
    try:
        _json_dump(backend.configuration().to_dict(),
                   run_dir / "configuration.json")
        print(f"   saved configuration.json")
    except Exception as e:
        print(f"   WARNING: could not save configuration: {e}")
    try:
        with open(run_dir / "target.pkl", "wb") as f:
            pickle.dump(backend.target, f)
        print(f"   saved target.pkl")
    except Exception as e:
        print(f"   WARNING: could not pickle target: {e}")


def cmd_submit(args):
    from qiskit import transpile, qpy
    from qiskit_ibm_runtime import SamplerV2

    keys = load_keys()
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    # 1) coupling map + patch validation (must run before circuit building)
    coupling_path = fetch(args.backend, token=keys["ibm_token"],
                          instance=keys["ibm_instance"], outdir=str(RUNS_DIR))
    validate_backend(coupling_path)
    print(f"backend '{args.backend}': 37q patch validated")

    # 2) hardware circuit
    qc = HeavyHex37QDepthOpt(args.cycles).build_circuit()

    # 3) transpile with the fixed physical layout, then insert DD
    service = get_service(keys)
    backend = service.backend(args.backend)
    tqc = transpile(qc, backend=backend, initial_layout=ALL_PHYS,
                    optimization_level=1)
    tqc = apply_dd(tqc, backend.target, sequence=args.dd)
    print(f"transpiled+DD({args.dd}): depth={tqc.depth()}, "
          f"pulses={dd_pulse_stats(tqc)}")

    # 4) submit (unless rehearsing), then snapshot everything into the
    #    run folder named after the job id
    if args.dry_run:
        job = None
        run_dir = RUNS_DIR / f"dryrun_{time.strftime('%Y%m%d-%H%M%S')}"
        print("--dry-run: not submitting (snapshot is still saved).")
    else:
        sampler = SamplerV2(mode=backend)
        job = sampler.run([tqc], shots=args.shots)
        run_dir = RUNS_DIR / job.job_id()
    run_dir.mkdir(parents=True, exist_ok=True)

    shutil.move(coupling_path, run_dir / "coupling.json")
    snapshot_backend(backend, run_dir)
    with open(run_dir / "circuit.qpy", "wb") as f:
        qpy.dump(tqc, f)
    print("   saved circuit.qpy (exact ISA circuit incl. DD delays)")

    meta = {"job_id": job.job_id() if job else None,
            "backend": args.backend, "cycles": args.cycles,
            "shots": args.shots, "dd": args.dd,
            "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "dry_run": args.dry_run,
            "transpiled_depth": tqc.depth(),
            "transpiled_ops": {k: int(v) for k, v in tqc.count_ops().items()},
            "versions": _local_versions()}
    _json_dump(meta, run_dir / "job.json")
    print(f"run folder: {run_dir}")
    if job:
        print(f"submitted: job_id={job.job_id()}")
        print(f"next: python hardware/run_hw.py analyze "
              f"--job-id {job.job_id()} --ckpt <checkpoint.pt>")


def fetch_raw(args):
    """Return (syn, dat, cycles): raw bits in clbit-index order."""
    if args.npz:
        d = np.load(args.npz)
        return (d["syn"].astype(np.uint8), d["dat"].astype(np.uint8),
                int(d["cycles"]))

    run_dir = RUNS_DIR / args.job_id
    cached = run_dir / "raw.npz"
    if cached.exists():
        d = np.load(cached)
        return (d["syn"].astype(np.uint8), d["dat"].astype(np.uint8),
                int(d["cycles"]))

    keys = load_keys()
    service = get_service(keys)
    job = service.job(args.job_id)
    cycles = args.cycles
    meta_path = run_dir / "job.json"
    if meta_path.exists():
        cycles = json.load(open(meta_path)).get("cycles", cycles)
    res = job.result()[0]
    # order='little' -> column i is clbit i, i.e. syn bit cyc*16+j and
    # dat bit i in DATA_PHYS order — exactly the check_values() layout
    syn = res.data["syn"].to_bool_array(order="little").astype(np.uint8)
    dat = res.data["dat"].to_bool_array(order="little").astype(np.uint8)
    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(run_dir / "raw.npz", syn=syn, dat=dat, cycles=cycles)
    print(f"saved raw results -> {run_dir / 'raw.npz'}")
    try:
        _json_dump({"metrics": job.metrics()}, run_dir / "job_metrics.json")
        print(f"saved job_metrics.json")
    except Exception as e:
        print(f"WARNING: could not save job metrics: {e}")
    return syn, dat, cycles


def cmd_analyze(args):
    syn, dat, cycles = fetch_raw(args)
    shots = syn.shape[0]
    assert syn.shape[1] == N_CHECKS * cycles and dat.shape[1] == 17
    print(f"shots={shots}, cycles={cycles}")

    # no-reset raw -> check values via per-ancilla XOR chains
    vals = check_values(syn, cycles)
    check_mat = check_matrix_from_dict(vals, cycles)
    y_logical = logical_label(dat)
    raw_ler = float(y_logical.mean())

    rows = [("raw (no decoding)", raw_ler)]

    # MWPM baseline (DEM weights from the reference noise profile)
    from baseline.mwpm import build_matching, mwpm_ler_from_hardware
    matching = build_matching(cycles, "X", args.mwpm_p, args.mwpm_profile)
    rows.append(("MWPM", mwpm_ler_from_hardware(check_mat, dat, cycles,
                                                matching)))

    # CNN
    if args.ckpt:
        import torch
        if args.solution:
            from solutions.cnn_solution import HeavyHexCNN
        else:
            from model.cnn_skeleton import HeavyHexCNN
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        model = HeavyHexCNN(in_channels=2 * cycles)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        tensor = syndrome_tensor(check_mat, cycles)
        preds = []
        with torch.no_grad():
            for i in range(0, shots, 8192):
                xb = torch.from_numpy(tensor[i:i + 8192])
                _, ll = model(xb)
                preds.append((ll.numpy().ravel() > 0).astype(np.uint8))
        pred = np.concatenate(preds)
        from evaluation.metrics import ler
        rows.append((f"CNN ({Path(args.ckpt).name})", ler(pred, y_logical)))
    else:
        print("(no --ckpt given: skipping CNN)")

    print(f"\n{'decoder':<40} {'LER':>8}")
    for name, v in rows:
        print(f"{name:<40} {v:>8.4f}")


def main():
    ap = argparse.ArgumentParser(description="Hardware validation pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("submit", help="build, transpile, apply DD and submit")
    s.add_argument("--backend", default="ibm_yonsei",
                   help="ibm_yonsei (default) or ibm_boston")
    s.add_argument("--cycles", type=int, default=3)
    s.add_argument("--shots", type=int, default=50_000)
    s.add_argument("--dd", default="XX4",
                   choices=["XX2", "XX4", "XY4", "XY8"],
                   help="DD sequence (XX4 default; Heron has no native Y)")
    s.add_argument("--dry-run", action="store_true",
                   help="do everything except the actual submission")
    s.set_defaults(func=cmd_submit)

    a = sub.add_parser("analyze", help="fetch results, decode, report LER")
    a.add_argument("--job-id", default=None)
    a.add_argument("--npz", default=None,
                   help="previously saved raw npz (offline re-analysis)")
    a.add_argument("--cycles", type=int, default=3,
                   help="fallback if job metadata is missing")
    a.add_argument("--ckpt", default=None, help="trained CNN checkpoint (.pt)")
    a.add_argument("--solution", action="store_true",
                   help="load the model class from solutions/ instead of "
                        "model/cnn_skeleton.py")
    a.add_argument("--mwpm-profile", default=ACTIVE_NOISE[0],
                   help="noise profile used for the MWPM DEM weights")
    a.add_argument("--mwpm-p", type=float, default=0.005)
    a.set_defaults(func=cmd_analyze)

    args = ap.parse_args()
    if args.cmd == "analyze" and not (args.job_id or args.npz):
        ap.error("analyze requires --job-id or --npz")
    args.func(args)


if __name__ == "__main__":
    main()
