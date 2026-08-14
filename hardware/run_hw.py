#!/usr/bin/env python3
"""
Hardware validation on IBM Quantum backends
===========================================
Pipeline (submit):
  keys.json -> fetch_coupling.fetch() -> validate_backend()
    -> HeavyHex37QDepthOpt circuit -> transpile(initial_layout =
    ALL_PHYS mapped through embedding_for(backend), optimization_level=1)
    -> dd_utils.apply_dd(tqc, backend.target)
    -> QPU environment snapshot -> SamplerV2 submit
Pipeline (analyze):
  raw results -> check_values() XOR-chain syndrome recovery
    -> syndrome tensor -> CNN + MWPM decoding -> LER report
    -> results/hardware/hw_<run_id>.csv
Pipeline (all): submit -> wait_for_job -> analyze in one go.

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
  * The official metric is the head-LER (logical head). Hardware provides
    no per-qubit ground truth, so `ECR (diagnostic, sim-only)` cannot be
    computed here; `parity_LER (diagnostic)` can (it only needs the
    logical ground truth) and is reported next to the LER, as is the
    `LER/MWPM ratio` (see evaluation/metrics.py).
  * properties.json records the last calibration before the run — it is
    the best available record, not a live snapshot of drift. You can feed
    it to qiskit-aer's NoiseModel.from_backend() to re-simulate the run
    under that day's calibration.

Usage:
  python hardware/run_hw.py         [submit options] [analyze options]
                                    [--poll 30]   # default = all
  python hardware/run_hw.py submit  [--backend ibm_yonsei] [--shots 50000]
                                    [--cycles 3] [--dd XX4] [--dry-run]
  python hardware/run_hw.py analyze --job-id <ID> [--ckpt checkpoint/CNN_....pt]
  python hardware/run_hw.py analyze --npz hardware/runs/<ID>/raw.npz [--ckpt ...]
  (no subcommand runs all: submit -> wait -> analyze in one go;
   analyze without --ckpt evaluates every checkpoint/*.pt)
"""
import argparse
import csv
import json
import pickle
import shutil
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from heavyhex_circuits.heavyhex_37q import (  # noqa: E402
    ALL_PHYS, validate_backend, embedding_for)
from heavyhex_circuits.heavyhex_depth7_opt_for_37q import (  # noqa: E402
    HeavyHex37QDepthOpt, check_values, N_CHECKS)
from heavyhex_circuits.fetch_coupling import fetch  # noqa: E402
from heavyhex_circuits.dd_utils import apply_dd, dd_pulse_stats  # noqa: E402

from dataset_generation.heavyhex33_stim import (  # noqa: E402
    check_matrix_from_dict, syndrome_tensor, logical_label, ALL_NOISE)

RUNS_DIR = _ROOT / "hardware" / "runs"
RESULTS_HW_DIR = _ROOT / "results" / "hardware"


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

    # 1) coupling map + patch validation (must run before circuit
    #    building), then 2) the code's hardware circuit
    coupling_path = fetch(args.backend, token=keys["ibm_token"],
                          instance=keys["ibm_instance"], outdir=str(RUNS_DIR))
    if args.code == "surface":
        from rsc_circuits.rsc3 import (
            RSC3Hardware, ALL_COORDS, embedding_for_surface,
            validate_backend_surface)
        validate_backend_surface(coupling_path)
        print(f"backend '{args.backend}': rsc3 17q patch validated "
              f"(45-degree embedding, no SWAPs)")
        qc = RSC3Hardware(args.cycles).build_circuit()
        layout = [embedding_for_surface(args.backend)[c]
                  for c in ALL_COORDS]
    else:
        validate_backend(coupling_path)
        print(f"backend '{args.backend}': 37q patch validated")
        qc = HeavyHex37QDepthOpt(args.cycles).build_circuit()
        layout = [embedding_for(args.backend)[p] for p in ALL_PHYS]

    # 3) transpile with the fixed physical layout (patch labels mapped to
    #    this backend's device qubits), then insert DD
    service = get_service(keys)
    backend = service.backend(args.backend)
    tqc = transpile(qc, backend=backend, initial_layout=layout,
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
            "backend": args.backend, "code": args.code, "cycles": args.cycles,
            "shots": args.shots, "dd": args.dd,
            "initial_layout": layout,
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
    return job


def wait_for_job(job, poll=30):
    """Poll until the job reaches a final state, printing state changes.
    Returns on DONE; exits on ERROR/CANCELLED."""
    last = None
    while True:
        status = job.status()
        name = getattr(status, "name", status)  # enum (old) or str (new)
        if name != last:
            print(f"[{time.strftime('%H:%M:%S')}] "
                  f"job {job.job_id()}: {name}")
            last = name
        if name == "DONE":
            return
        if name in ("ERROR", "CANCELLED"):
            sys.exit(f"job {job.job_id()} ended as {name} — "
                     f"nothing to analyze.")
        time.sleep(poll)


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
    wait_for_job(job, poll=getattr(args, "poll", 30))
    cycles = args.cycles
    meta_path = run_dir / "job.json"
    if meta_path.exists():
        cycles = json.load(open(meta_path)).get("cycles", cycles)
    res = job.result()[0]
    # order='little' -> column i is clbit i, i.e. syn bit cyc*16+j and
    # dat bit i in DATA_PHYS order — exactly the check_values() layout
    syn = res.data["syn"].to_bool_array(order="little").astype(np.uint8)
    dat = res.data["data"].to_bool_array(order="little").astype(np.uint8)
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
    # the run's code family: job.json wins (submit recorded it), else the
    # --code / default heavyhex (offline --npz re-analysis)
    code = getattr(args, "code", None) or "heavyhex"
    if args.job_id:
        meta_path = RUNS_DIR / args.job_id / "job.json"
        if meta_path.exists():
            code = json.load(open(meta_path)).get("code", code)

    if code == "surface":
        from rsc_circuits.rsc3 import (
            check_values as cv_fn, N_CHECKS as n_checks, NUM_DATA as n_data)
        from dataset_generation.rsc3_stim import (
            check_matrix_from_dict_rsc3 as mat_fn,
            syndrome_tensor_rsc3 as tensor_fn,
            logical_label_rsc3 as logical_fn)
    else:
        cv_fn, n_checks, n_data = check_values, N_CHECKS, 17
        mat_fn, tensor_fn, logical_fn = (check_matrix_from_dict,
                                         syndrome_tensor, logical_label)

    syn, dat, cycles = fetch_raw(args)
    shots = syn.shape[0]
    assert syn.shape[1] == n_checks * cycles and dat.shape[1] == n_data, \
        f"raw shapes {syn.shape}/{dat.shape} do not match code '{code}'"
    print(f"shots={shots}, cycles={cycles}, code={code}")

    # no-reset raw -> check values via per-ancilla XOR chains
    check_mat = mat_fn(cv_fn(syn, cycles), cycles)
    y_logical = logical_fn(dat)
    raw_ler = float(y_logical.mean())

    # row = (decoder, LER, parity_LER (diagnostic) or None,
    #        LER/MWPM ratio or None)
    rows = [("raw (no decoding)", raw_ler, None, None)]

    # MWPM baseline (DEM weights from the reference noise profile)
    from baseline.mwpm import build_matching, mwpm_ler_from_hardware
    matching = build_matching(cycles, "X", args.mwpm_p, args.mwpm_profile,
                              code)
    mwpm_ler = mwpm_ler_from_hardware(check_mat, dat, cycles, matching,
                                      code)
    rows.append(("MWPM", mwpm_ler, None, None))

    # model head: a single --ckpt, or every matching checkpoint if none
    # was given ({MODEL} prefix keeps cnn/gnn apart; surface checkpoints
    # carry the "_surface_" tag, legacy heavyhex names carry no code tag)
    model_ul = args.model.upper()
    if args.ckpt:
        ckpts = [Path(args.ckpt)]
    else:
        ckpts = sorted(p for p in (_ROOT / "checkpoint").glob(
            f"{model_ul}_*.pt")
            if ("_surface_" in p.name) == (code == "surface"))
        if not ckpts:
            print(f"(no --ckpt and no checkpoint/{model_ul}_*.pt for "
                  f"code '{code}': skipping {model_ul})")
    if ckpts:
        import torch
        from model import get_model_module, get_model_class, CODE_SPECS
        mod = get_model_module(args.model, args.solution)
        model_cls = get_model_class(args.model, args.solution)
        from evaluation.metrics import ler, parity_ler_from_qubit_logits
        tensor = tensor_fn(check_mat, cycles)
        # model-specific input prep (e.g. the GNN appends the final-Z
        # detector channel, computed from the measured final data bits —
        # the hardware counterpart of the simulation labels)
        if hasattr(mod, "prepare_features"):
            tensor = np.asarray(mod.prepare_features(tensor, dat, code))
        for ckpt_path in ckpts:
            ckpt = torch.load(ckpt_path, map_location="cpu",
                              weights_only=False)
            model = model_cls(in_channels=2 * cycles,
                              num_qubits=CODE_SPECS[code]["num_qubits"],
                              code=code)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            preds, q_logits = [], []
            with torch.no_grad():
                for i in range(0, shots, 8192):
                    xb = torch.from_numpy(tensor[i:i + 8192])
                    ql, ll = model(xb)
                    preds.append((ll.numpy().ravel() > 0).astype(np.uint8))
                    q_logits.append(ql.numpy())
            pred = np.concatenate(preds)
            model_ler = ler(pred, y_logical)
            parity_ler = parity_ler_from_qubit_logits(
                np.concatenate(q_logits), y_logical, code)
            ratio = model_ler / mwpm_ler if mwpm_ler else None
            rows.append((f"{model_ul} ({ckpt_path.name})", model_ler,
                         parity_ler, ratio))

    def _fmt(v, spec=".4f"):
        return format(v, spec) if v is not None else "N/A"

    print(f"\n{'decoder':<55} {'LER':>8} "
          f"{'parity_LER (diagnostic)':>24} {'LER/MWPM ratio':>15}")
    for name, v, pl, ratio in rows:
        print(f"{name:<55} {v:>8.4f} {_fmt(pl):>24} {_fmt(ratio):>15}")

    # persist the report next to the training results
    run_id = args.job_id or Path(args.npz).resolve().parent.name
    RESULTS_HW_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_HW_DIR / f"hw_{run_id}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["decoder", "ler", "parity_LER (diagnostic)",
                    "LER/MWPM ratio", "shots", "cycles", "run_id"])
        for name, v, pl, ratio in rows:
            w.writerow([name, f"{v:.6f}", _fmt(pl, ".6f"),
                        _fmt(ratio, ".6f"), shots, cycles, run_id])
    print(f"saved -> {csv_path}")


def cmd_all(args):
    job = cmd_submit(args)
    if args.dry_run:
        print("--dry-run: skipping wait/analyze.")
        return
    args.job_id = job.job_id()
    args.npz = None
    wait_for_job(job, poll=args.poll)
    cmd_analyze(args)


def _submit_opts(p):
    p.add_argument("--backend", default="ibm_yonsei",
                   help="ibm_yonsei (default) or ibm_boston")
    p.add_argument("--code", choices=["heavyhex", "surface"],
                   default="heavyhex",
                   help="code family (surface support lands with the rsc3 "
                        "milestone)")
    p.add_argument("--cycles", type=int, default=3)
    p.add_argument("--shots", type=int, default=50_000)
    p.add_argument("--dd", default="XX4",
                   choices=["XX2", "XX4", "XY4", "XY8"],
                   help="DD sequence (XX4 default; Heron has no native Y)")
    p.add_argument("--dry-run", action="store_true",
                   help="do everything except the actual submission")


def _analyze_opts(p, cycles=True):
    if cycles:   # standalone analyze parser ('all' inherits submit's)
        p.add_argument("--cycles", type=int, default=3,
                       help="fallback if job metadata is missing")
        p.add_argument("--code", choices=["heavyhex", "surface"],
                       default=None,
                       help="fallback if job metadata is missing "
                            "(e.g. offline --npz re-analysis)")
    p.add_argument("--model", choices=["cnn", "gnn"], default="cnn",
                   help="decoder architecture used for --ckpt loading; "
                        "without --ckpt, evaluates checkpoint/{MODEL}_*.pt")
    p.add_argument("--ckpt", default=None,
                   help="trained model checkpoint (.pt); omit to evaluate "
                        "every checkpoint/{MODEL}_*.pt")
    p.add_argument("--solution", action="store_true",
                   help="load the model class from solutions/ instead of "
                        "model/<model>_skeleton.py")
    p.add_argument("--mwpm-profile", default=ALL_NOISE[0],
                   help="noise profile used for the MWPM DEM weights")
    p.add_argument("--mwpm-p", type=float, default=0.005)


def main():
    ap = argparse.ArgumentParser(description="Hardware validation pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("submit", help="build, transpile, apply DD and submit")
    _submit_opts(s)
    s.set_defaults(func=cmd_submit)

    a = sub.add_parser("analyze", help="fetch results, decode, report LER")
    a.add_argument("--job-id", default=None)
    a.add_argument("--npz", default=None,
                   help="previously saved raw npz (offline re-analysis)")
    _analyze_opts(a)
    a.set_defaults(func=cmd_analyze)

    al = sub.add_parser("all", help="submit, wait for the job, then analyze")
    _submit_opts(al)
    _analyze_opts(al, cycles=False)   # submit already owns --cycles
    al.add_argument("--poll", type=int, default=30,
                    help="job status poll interval in seconds")
    al.set_defaults(func=cmd_all)

    # 서브커맨드 생략 시 all(제출->대기->분석 원샷)로 동작.
    argv = sys.argv[1:]
    if argv[:1] not in (["submit"], ["analyze"], ["all"], ["-h"], ["--help"]):
        argv.insert(0, "all")
    args = ap.parse_args(argv)
    if args.cmd == "analyze" and not (args.job_id or args.npz):
        ap.error("analyze requires --job-id or --npz")
    args.func(args)


if __name__ == "__main__":
    main()
