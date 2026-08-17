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
Submission model: ONE job per (backend, code) carrying --pubs PUBs of
the same ISA circuit (PUB = Primitive Unified Bloc, the V2 primitives'
(circuit, parameter values, shots) execution unit) — a single queue
entry buys shots x pubs. Backend limits (max_shots / max_experiments)
are queried and the plan auto-splits into extra PUBs/jobs (job.json then
carries job_ids). Every submission is registered in
hardware/pending_jobs.json; `collect` (run at the start of each
pipeline loop) analyzes whatever finished and carries the rest over, so
the pipeline never blocks on the queue.

Pipeline (analyze):
  raw results (all PUBs concatenated, pub_shots recorded)
    -> check_values() XOR-chain syndrome recovery
    -> syndrome tensor -> CNN/GNN + MWPM decoding -> LER report with a
    ler_std_over_pubs column (per-PUB LER spread)
    -> results/hardware/<backend>_<code>_<timestamp>.csv
  First collection also stores run_started_at (job metrics) in job.json
  and the calibration AS OF execution time in properties_run.json —
  make_qpu_avg_profile prefers this over the submission-time snapshot.
Pipeline (all): submit -> wait_for_job(s) -> analyze in one go
  (manual use; the sbatch pipeline uses submit + collect).

Every submission gets its own folder so the run can be re-analyzed (and
the QPU environment of that moment stays on record):

  hardware/runs/<backend>_<YYYYMMDD-HHMMSS>/   (dry-run: ..._dryrun/;
    the job id lives in job.json — analyze --job-id finds the folder
    by scanning job.json, so legacy job-id-named folders still work)
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

from circuits.heavyhex.heavyhex_37q import (  # noqa: E402
    ALL_PHYS, validate_backend, embedding_for)
from circuits.heavyhex.heavyhex_depth7_opt_for_37q import (  # noqa: E402
    HeavyHex37QDepthOpt, check_values, N_CHECKS)
from circuits.heavyhex.fetch_coupling import fetch  # noqa: E402
from circuits.heavyhex.dd_utils import apply_dd, dd_pulse_stats  # noqa: E402

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

    # 1) coupling map + backend handle, then 2) the code's hardware
    #    circuit and its device placement
    coupling_path = fetch(args.backend, token=keys["ibm_token"],
                          instance=keys["ibm_instance"], outdir=str(RUNS_DIR))
    service = get_service(keys)
    backend = service.backend(args.backend)
    placement_info = None
    if args.code == "surface":
        # surface (miami)는 자동 배치 비활성 — observable frame 문제 확정
        # 전까지 정적 45도 임베딩 유지 (placement 모듈 자체는 코드 공용)
        from circuits.rotatedSurface.rotatedSurface3 import (
            RotatedSurface3Hardware, ALL_COORDS, embedding_for_surface,
            validate_backend_surface)
        validate_backend_surface(coupling_path)
        print(f"backend '{args.backend}': rotatedSurface3 17q patch validated "
              f"(45-degree embedding, no SWAPs)")
        qc = RotatedSurface3Hardware(args.cycles).build_circuit()
        layout = [embedding_for_surface(args.backend)[c]
                  for c in ALL_COORDS]
    else:
        # heavyhex: 캘리브레이션 인지 자동 배치 (루프 체인 동안 고정 —
        # hardware/placement_<backend>_heavyhex.json 재사용, 임계값 위반
        # 시에만 재탐색; --reselect-layout으로 강제 재선택)
        from circuits.placement import resolve_placement
        from circuits.heavyhex.heavyhex_37q import required_edges
        qc = HeavyHex37QDepthOpt(args.cycles).build_circuit()
        cm = json.load(open(coupling_path))
        try:
            static_map = embedding_for(args.backend)
        except RuntimeError:
            static_map = None
        mapping, placement_info = resolve_placement(
            args.backend, "heavyhex", _ROOT / "hardware",
            ALL_PHYS, required_edges(), qc,
            [tuple(e) for e in cm["coupling_map"]], backend.target,
            static_mapping=static_map,
            reselect=getattr(args, "reselect_layout", False))
        if mapping is None:
            print("WARNING: 유효한 자동 배치 없음 — 정적 임베딩으로 폴백")
            validate_backend(coupling_path)
            mapping = embedding_for(args.backend)
            placement_info = {"fallback": "static",
                              **(placement_info or {})}
        else:
            # 선택된 배치가 required edges를 실제로 덮는지 이중 확인
            eset = {tuple(sorted(e)) for e in cm["coupling_map"]}
            for u, v in required_edges():
                assert tuple(sorted((mapping[u], mapping[v]))) in eset
            print(f"backend '{args.backend}': 37q patch placed "
                  f"(auto, qubits {min(mapping.values())}"
                  f"–{max(mapping.values())})")
        layout = [mapping[p] for p in ALL_PHYS]

    # 3) transpile with the fixed physical layout (patch labels mapped to
    #    this backend's device qubits), then insert DD
    tqc = transpile(qc, backend=backend, initial_layout=layout,
                    optimization_level=1)
    tqc = apply_dd(tqc, backend.target, sequence=args.dd)
    print(f"transpiled+DD({args.dd}): depth={tqc.depth()}, "
          f"pulses={dd_pulse_stats(tqc)}")

    # 4) submit (unless rehearsing) as ONE job carrying --pubs PUBs of
    #    the same ISA circuit. PUB = Primitive Unified Bloc, the
    #    (circuit, parameter values, shots) execution unit of the V2
    #    primitives — packing N PUBs into one job claims a SINGLE queue
    #    entry for N x shots total. Backend limits (max_shots per PUB,
    #    max_experiments per job) are queried and the plan auto-splits;
    #    if the lookup fails we submit as requested with a warning.
    #    The run folder is named <backend>_<timestamp> (job ids live in
    #    job.json — analyze/collect find the folder by scanning job.json,
    #    so legacy job-id-named folders keep working too).
    ts = time.strftime("%Y%m%d-%H%M%S")
    jobs = []
    if args.dry_run:
        run_dir = RUNS_DIR / f"{args.backend}_{ts}_dryrun"
        shots_per_pub, per_job = args.shots, [args.pubs]
        print(f"--dry-run: not submitting (PUB plan: {args.pubs} PUB x "
              f"{args.shots} shots in 1 job; snapshot is still saved).")
    else:
        max_shots = max_exp = None
        try:
            cfg_b = backend.configuration()
            max_shots = getattr(cfg_b, "max_shots", None)
            max_exp = getattr(cfg_b, "max_experiments", None)
        except Exception as e:
            print(f"WARNING: backend.configuration() lookup failed ({e}) "
                  f"— submitting without limit checks")
        shots_per_pub, per_job = _plan_pubs(args.shots, args.pubs,
                                            max_shots, max_exp)
        if shots_per_pub != args.shots or len(per_job) > 1:
            print(f"backend limits (max_shots={max_shots}, "
                  f"max_experiments={max_exp}) -> {shots_per_pub} "
                  f"shots/PUB, PUBs per job: {per_job}")
        sampler = SamplerV2(mode=backend)
        for n in per_job:
            jobs.append(sampler.run([tqc] * n, shots=shots_per_pub))
        run_dir = RUNS_DIR / f"{args.backend}_{ts}"
    while run_dir.exists():                  # same-second collision guard
        run_dir = run_dir.with_name(run_dir.name + "b")
    run_dir.mkdir(parents=True, exist_ok=True)

    shutil.move(coupling_path, run_dir / "coupling.json")
    snapshot_backend(backend, run_dir)
    with open(run_dir / "circuit.qpy", "wb") as f:
        qpy.dump(tqc, f)
    print("   saved circuit.qpy (exact ISA circuit incl. DD delays)")

    job_ids = [j.job_id() for j in jobs]
    meta = {"job_id": job_ids[0] if job_ids else None,
            "job_ids": job_ids,
            "n_pubs": sum(per_job), "shots_per_pub": shots_per_pub,
            "backend": args.backend, "code": args.code, "cycles": args.cycles,
            "shots": args.shots, "pubs": args.pubs, "dd": args.dd,
            "initial_layout": layout,
            "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "dry_run": args.dry_run,
            "transpiled_depth": tqc.depth(),
            "transpiled_ops": {k: int(v) for k, v in tqc.count_ops().items()},
            "placement": placement_info,
            "versions": _local_versions()}
    _json_dump(meta, run_dir / "job.json")
    print(f"run folder: {run_dir}")
    if jobs:
        pend = _load_pending()
        pend.append({"job_id": job_ids[0], "job_ids": job_ids,
                     "backend": args.backend, "code": args.code,
                     "run_dir": run_dir.name,
                     "submitted_at": meta["submitted_at"]})
        _save_pending(pend)
        print(f"submitted: {len(jobs)} job(s) {job_ids} — "
              f"{sum(per_job)} PUB x {shots_per_pub} shots, registered in "
              f"pending_jobs.json (다음 루프의 수거 단계가 자동 분석)")
        print(f"manual: python hardware/run_hw.py analyze "
              f"--job-id {job_ids[0]}")
    return jobs


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


def find_run_dir(job_id):
    """Locate the run folder of a job id.

    Run folders are named <backend>_<timestamp> (job.json holds the job
    id), so look there first by scanning job.json; a folder literally
    named after the job id (legacy layout) also matches. Returns None
    when no local folder exists (e.g. analyzing on another machine)."""
    direct = RUNS_DIR / job_id
    if direct.exists():
        return direct
    if RUNS_DIR.exists():
        for p in sorted(RUNS_DIR.iterdir()):
            meta_path = p / "job.json"
            if not (p.is_dir() and meta_path.exists()):
                continue
            try:
                if json.load(open(meta_path)).get("job_id") == job_id:
                    return p
            except Exception:
                continue
    return None


PENDING_PATH = _ROOT / "hardware" / "pending_jobs.json"


def _load_pending():
    if not PENDING_PATH.exists():
        return []
    try:
        return json.load(open(PENDING_PATH))
    except Exception as e:
        print(f"WARNING: pending_jobs.json unreadable ({e}) — "
              f"treating as empty")
        return []


def _save_pending(entries):
    PENDING_PATH.parent.mkdir(parents=True, exist_ok=True)
    _json_dump(entries, PENDING_PATH)


def _remove_pending(job_id):
    entries = _load_pending()
    kept = [e for e in entries
            if e.get("job_id") != job_id
            and job_id not in (e.get("job_ids") or [])]
    if len(kept) != len(entries):
        _save_pending(kept)


def _plan_pubs(shots, pubs, max_shots, max_experiments):
    """Backend-limit-aware PUB plan -> (shots per PUB, PUBs per job).

    Keeps the total ~ shots*pubs: a PUB whose shots exceed max_shots is
    subdivided; when the PUB count exceeds max_experiments the job is
    split (list entry = one job's PUB count)."""
    import math
    shots_per_pub, n_pubs = shots, pubs
    if max_shots and shots_per_pub > max_shots:
        k = math.ceil(shots_per_pub / max_shots)
        shots_per_pub = math.ceil(shots / k)
        n_pubs = pubs * k
    per_job, remaining = [], n_pubs
    cap = max_experiments if max_experiments else n_pubs
    while remaining > 0:
        n = min(remaining, cap)
        per_job.append(n)
        remaining -= n
    return shots_per_pub, per_job


def _latest_qpu_profile(backend, code):
    """Newest registered qpu/<backend>_<code>_* profile name (or None) —
    the collect path uses it for the MWPM DEM weights."""
    try:
        from dataset_generation import load_config
        profs = load_config().get("noise_profiles", {})
    except Exception:
        return None
    cands = [(v.get("provenance", {}).get("generated_at", ""), k)
             for k, v in profs.items()
             if k.startswith(f"qpu/{backend}_{code}_")]
    return max(cands)[1] if cands else None


def fetch_raw(args):
    """Return (syn, dat, cycles, pub_shots): raw bits in clbit order.

    Multi-PUB / multi-job aware: PUB results are concatenated in order
    and pub_shots records each PUB's shot count. Legacy caches / loose
    npz without pub_shots count as a single PUB (pub_shots=None).
    On first collection this also records run_started_at (job metrics
    "running" timestamp) into job.json and saves the calibration AS OF
    execution time to properties_run.json."""
    if args.npz:
        d = np.load(args.npz)
        ps = d["pub_shots"] if "pub_shots" in d.files else None
        return (d["syn"].astype(np.uint8), d["dat"].astype(np.uint8),
                int(d["cycles"]), ps)

    # legacy fallback name if the job has no local run folder yet
    run_dir = find_run_dir(args.job_id) or (RUNS_DIR / args.job_id)
    cached = run_dir / "raw.npz"
    if cached.exists():
        d = np.load(cached)
        ps = d["pub_shots"] if "pub_shots" in d.files else None
        return (d["syn"].astype(np.uint8), d["dat"].astype(np.uint8),
                int(d["cycles"]), ps)

    keys = load_keys()
    service = get_service(keys)
    cycles = args.cycles
    meta, job_ids = {}, [args.job_id]
    meta_path = run_dir / "job.json"
    if meta_path.exists():
        meta = json.load(open(meta_path))
        cycles = meta.get("cycles", cycles)
        job_ids = meta.get("job_ids") or [meta.get("job_id") or args.job_id]

    syns, dats, pub_shots, metrics_all = [], [], [], []
    first_running = None
    for jid in job_ids:
        job = service.job(jid)
        wait_for_job(job, poll=getattr(args, "poll", 30))
        # order='little' -> column i is clbit i, i.e. syn bit cyc*16+j and
        # dat bit i in DATA_PHYS order — exactly the check_values() layout
        for res in job.result():
            s = res.data["syn"].to_bool_array(order="little").astype(np.uint8)
            t = res.data["data"].to_bool_array(order="little").astype(np.uint8)
            syns.append(s)
            dats.append(t)
            pub_shots.append(s.shape[0])
        try:
            mt = job.metrics()
            metrics_all.append(mt)
            r = (mt.get("timestamps") or {}).get("running")
            if r and (first_running is None or r < first_running):
                first_running = r
        except Exception as e:
            print(f"WARNING: could not read metrics of {jid}: {e}")
    syn = np.concatenate(syns)
    dat = np.concatenate(dats)
    pub_arr = np.asarray(pub_shots, dtype=np.int64)
    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(run_dir / "raw.npz", syn=syn, dat=dat,
                        cycles=cycles, pub_shots=pub_arr)
    print(f"saved raw results -> {run_dir / 'raw.npz'} "
          f"({len(pub_shots)} PUB x ~{pub_shots[0] if pub_shots else 0} shots)")
    try:
        _json_dump({"metrics": metrics_all}, run_dir / "job_metrics.json")
        print(f"saved job_metrics.json")
    except Exception as e:
        print(f"WARNING: could not save job metrics: {e}")

    # execution-time calibration snapshot (submission-time snapshots can
    # be hours older than the actual run when the queue is long)
    try:
        import datetime as _dt
        if first_running:
            run_ts = first_running
            run_dt = _dt.datetime.fromisoformat(run_ts.replace("Z", "+00:00"))
        else:
            run_ts = time.strftime("%Y-%m-%dT%H:%M:%S")
            run_dt = None
            print("WARNING: no 'running' timestamp in job metrics — "
                  "using collection time; properties_run falls back to "
                  "current calibration")
        if meta:
            meta["run_started_at"] = run_ts
            _json_dump(meta, meta_path)
        if meta.get("backend"):
            bk = service.backend(meta["backend"])
            props = (bk.properties(datetime=run_dt) if run_dt
                     else bk.properties())
            if props is not None:
                _json_dump(props.to_dict(), run_dir / "properties_run.json")
                print("saved properties_run.json (calibration at run time)")
    except Exception as e:
        print(f"WARNING: run-time calibration snapshot failed: {e}")
    return syn, dat, cycles, pub_arr


def cmd_analyze(args):
    # run metadata: job.json wins (submit recorded it), else the --code /
    # default heavyhex (offline --npz re-analysis). backend/submitted_at
    # feed the report name and columns.
    code = getattr(args, "code", None) or "heavyhex"
    hw_backend, submitted_at, job_id = None, None, args.job_id
    run_dir = None
    if args.job_id:
        run_dir = find_run_dir(args.job_id)
    elif args.npz:
        run_dir = Path(args.npz).resolve().parent
    meta_path = (run_dir / "job.json") if run_dir else None
    if meta_path and meta_path.exists():
        meta = json.load(open(meta_path))
        code = meta.get("code", code)
        hw_backend = meta.get("backend")
        submitted_at = meta.get("submitted_at")
        job_id = meta.get("job_id", job_id)

    if code == "surface":
        from circuits.rotatedSurface.rotatedSurface3 import (
            check_values as cv_fn, N_CHECKS as n_checks, NUM_DATA as n_data)
        from dataset_generation.rotatedSurface3_stim import (
            check_matrix_from_dict_rotatedSurface3 as mat_fn,
            syndrome_tensor_rotatedSurface3 as tensor_fn,
            logical_label_rotatedSurface3 as logical_fn)
    else:
        cv_fn, n_checks, n_data = check_values, N_CHECKS, 17
        mat_fn, tensor_fn, logical_fn = (check_matrix_from_dict,
                                         syndrome_tensor, logical_label)

    syn, dat, cycles, pub_shots = fetch_raw(args)
    shots = syn.shape[0]
    assert syn.shape[1] == n_checks * cycles and dat.shape[1] == n_data, \
        f"raw shapes {syn.shape}/{dat.shape} do not match code '{code}'"
    n_pubs = len(pub_shots) if pub_shots is not None else 1
    print(f"shots={shots} ({n_pubs} PUB), cycles={cycles}, code={code}")

    # no-reset raw -> check values via per-ancilla XOR chains
    check_mat = mat_fn(cv_fn(syn, cycles), cycles)
    y_logical = logical_fn(dat)
    raw_ler = float(y_logical.mean())

    # PUB boundaries for the per-PUB LER spread (legacy runs without
    # pub_shots count as one PUB -> std N/A)
    bounds = None
    if pub_shots is not None and len(pub_shots) > 1:
        edges = np.concatenate([[0], np.cumsum(pub_shots)])
        bounds = [(int(edges[i]), int(edges[i + 1]))
                  for i in range(len(pub_shots))]

    def std_over_pubs(pred=None, values=None):
        """Sample std of the per-PUB LERs (pred vs y_logical, or raw
        values)."""
        if bounds is None:
            return None
        lers = []
        for a, b in bounds:
            if values is not None:
                lers.append(float(values[a:b].mean()))
            else:
                lers.append(float((pred[a:b] != y_logical[a:b]).mean()))
        return float(np.std(lers, ddof=1))

    # row = (decoder, LER, ler_std_over_pubs or None,
    #        parity_LER (diagnostic) or None, LER/MWPM ratio or None,
    #        best_epoch or None, total_epochs or None)
    rows = [("raw (no decoding)", raw_ler, std_over_pubs(values=y_logical),
             None, None, None, None)]

    # MWPM baseline (DEM weights from the reference noise profile)
    from baseline.mwpm import build_matching, mwpm_ler_from_hardware
    matching = build_matching(cycles, "X", args.mwpm_p, args.mwpm_profile,
                              code)
    mwpm_ler, mwpm_pred = mwpm_ler_from_hardware(check_mat, dat, cycles,
                                                 matching, code,
                                                 return_pred=True)
    rows.append(("MWPM", mwpm_ler, std_over_pubs(pred=mwpm_pred),
                 None, None, None, None))

    # model head: a single --ckpt, or every matching checkpoint if none
    # was given. The architecture is INFERRED from the {MODEL}_ filename
    # prefix (CNN_/GNN_), so one analyze evaluates cnn and gnn rows in
    # the same report; --model narrows it to one architecture. Surface
    # checkpoints carry the "_surface_" tag, legacy heavyhex names carry
    # no code tag.
    from model import MODEL_REGISTRY

    def model_of(path):
        prefix = path.name.split("_", 1)[0].lower()
        return prefix if prefix in MODEL_REGISTRY else None

    if args.ckpt:
        p = Path(args.ckpt)
        ckpts = [(model_of(p) or args.model or "cnn", p)]
    else:
        ckpts = []
        for p in sorted((_ROOT / "checkpoint").glob("*.pt")):
            if p.name.endswith(".resume.pt"):
                continue                    # training state, not a model
            mname = model_of(p)
            if mname is None:
                print(f"(skipping {p.name}: unknown model prefix)")
                continue
            if ("_surface_" in p.name) != (code == "surface"):
                continue
            if args.model and mname != args.model:
                continue
            ckpts.append((mname, p))
        if not ckpts:
            want = args.model or "cnn/gnn"
            print(f"(no --ckpt and no matching checkpoint/*.pt for "
                  f"code '{code}' / model {want}: skipping)")
    if ckpts:
        import torch
        from model import get_model_module, get_model_class, CODE_SPECS
        from evaluation.metrics import ler, parity_ler_from_qubit_logits
        base_tensor = tensor_fn(check_mat, cycles)
        prepared = {}       # model name -> model-ready input tensor
        for mname, ckpt_path in ckpts:
            mod = get_model_module(mname, args.solution)
            if mname not in prepared:
                # model-specific input prep (e.g. the GNN appends the
                # final-Z detector channel, computed from the measured
                # final data bits — the hardware counterpart of the
                # simulation labels)
                t = base_tensor
                if hasattr(mod, "prepare_features"):
                    t = np.asarray(mod.prepare_features(base_tensor, dat,
                                                        code))
                prepared[mname] = t
            tensor = prepared[mname]
            model_cls = get_model_class(mname, args.solution)
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
            # "weight from epoch <best> out of <total> trained"
            best_ep = ckpt.get("best_epoch", ckpt.get("epoch"))
            total_ep = ckpt.get("total_epochs")
            rows.append((f"{mname.upper()} ({ckpt_path.name})", model_ler,
                         std_over_pubs(pred=pred), parity_ler, ratio,
                         best_ep, total_ep))

    def _fmt(v, spec=".4f"):
        return format(v, spec) if v is not None else "N/A"

    print(f"\n{'decoder':<55} {'LER':>8} {'ler_std':>8} "
          f"{'parity_LER (diagnostic)':>24} {'LER/MWPM ratio':>15} "
          f"{'best_ep':>8} {'total_ep':>9}")
    for name, v, sd, pl, ratio, be, te in rows:
        print(f"{name:<55} {v:>8.4f} {_fmt(sd):>8} {_fmt(pl):>24} "
              f"{_fmt(ratio):>15} "
              f"{_fmt(be, 'd') if be is not None else 'N/A':>8} "
              f"{_fmt(te, 'd') if te is not None else 'N/A':>9}")

    # persist the report next to the training results
    # report file: <backend>_<code>_<timestamp>.csv, timestamp = the
    # run's submitted_at (falls back to the analysis time when the run
    # metadata is unavailable, e.g. a loose --npz file)
    job_id = job_id or (Path(args.npz).resolve().parent.name if args.npz
                        else None)
    code_label = "rotatedSurface" if code == "surface" else "heavyhex"
    ts = (submitted_at.replace("-", "").replace(":", "").replace("T", "-")
          if submitted_at else time.strftime("%Y%m%d-%H%M%S"))
    RESULTS_HW_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_HW_DIR / f"{hw_backend or 'unknown'}_{code_label}_{ts}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["decoder", "ler", "ler_std_over_pubs",
                    "parity_LER (diagnostic)", "LER/MWPM ratio", "shots",
                    "cycles", "best_epoch", "total_epochs", "backend",
                    "timestamp", "job_id"])
        for name, v, sd, pl, ratio, be, te in rows:
            w.writerow([name, f"{v:.6f}", _fmt(sd, ".6f"),
                        _fmt(pl, ".6f"), _fmt(ratio, ".6f"), shots, cycles,
                        be if be is not None else "",
                        te if te is not None else "",
                        hw_backend or "unknown", submitted_at or ts,
                        job_id or ""])
    print(f"saved -> {csv_path}")
    if args.job_id:
        _remove_pending(args.job_id)


def cmd_collect(args):
    """Check every entry of pending_jobs.json; analyze what finished.

    Called at the start of each pipeline loop: DONE -> analyze and drop,
    QUEUED/RUNNING -> carry over, ERROR/CANCELLED -> log the reason
    (hardware/collect_failed.log) and drop. This path never raises /
    sys.exits, so the sbatch loop chain (set -e) survives any single
    job's failure."""
    pending = _load_pending()
    if not pending:
        print("pending 없음")
        return
    try:
        service = get_service(load_keys())
    except (Exception, SystemExit) as e:
        print(f"WARNING: 서비스 연결 불가({e}) — 수거 보류")
        return
    keep = []
    for ent in pending:
        jids = ent.get("job_ids") or [ent.get("job_id")]
        label = f"{ent.get('backend')}/{ent.get('code')} {jids}"
        try:
            names = []
            for jid in jids:
                st = service.job(jid).status()
                names.append(getattr(st, "name", st))
        except (Exception, SystemExit) as e:
            print(f"  {label}: 상태 조회 실패({e}) — 이월")
            keep.append(ent)
            continue
        if any(n in ("ERROR", "CANCELLED") for n in names):
            print(f"  {label}: {names} — 실패로 기록하고 제거")
            try:
                with open(_ROOT / "hardware" / "collect_failed.log",
                          "a") as f:
                    f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} "
                            f"{label} {names}\n")
            except Exception:
                pass
            continue
        if all(n == "DONE" for n in names):
            print(f"  {label}: DONE — 분석 시작")
            mwpm_prof = (_latest_qpu_profile(ent.get("backend"),
                                             ent.get("code"))
                         or args.mwpm_profile)
            ns = argparse.Namespace(
                job_id=jids[0], npz=None, cycles=3,
                code=ent.get("code"), model=args.model, ckpt=args.ckpt,
                solution=args.solution, mwpm_profile=mwpm_prof,
                mwpm_p=args.mwpm_p, poll=30)
            try:
                cmd_analyze(ns)
            except (Exception, SystemExit) as e:
                print(f"  {label}: 분석 실패({e}) — 이월")
                keep.append(ent)
            continue
        print(f"  {label}: {names} — 이월")
        keep.append(ent)
    _save_pending(keep)
    print(f"수거 완료 — pending 잔여 {len(keep)}건")


def cmd_all(args):
    jobs = cmd_submit(args)
    if args.dry_run:
        print("--dry-run: skipping wait/analyze.")
        return
    args.job_id = jobs[0].job_id()
    args.npz = None
    for j in jobs:
        wait_for_job(j, poll=args.poll)
    cmd_analyze(args)


def _submit_opts(p):
    p.add_argument("--backend", default="ibm_yonsei",
                   help="ibm_yonsei (default) or ibm_boston")
    p.add_argument("--code", choices=["heavyhex", "surface"],
                   default="heavyhex",
                   help="code family (surface support lands with the rotatedSurface3 "
                        "milestone)")
    p.add_argument("--cycles", type=int, default=3)
    p.add_argument("--shots", type=int, default=50_000,
                   help="shots per PUB")
    p.add_argument("--pubs", type=int, default=1,
                   help="number of PUBs (same ISA circuit repeated) in "
                        "the single submitted job — total shots = "
                        "shots x pubs with ONE queue entry")
    p.add_argument("--dd", default="XX4",
                   choices=["XX2", "XX4", "XY4", "XY8"],
                   help="DD sequence (XX4 default; Heron has no native Y)")
    p.add_argument("--dry-run", action="store_true",
                   help="do everything except the actual submission")
    p.add_argument("--reselect-layout", action="store_true",
                   help="ignore the pinned placement file and search a "
                        "fresh calibration-optimal placement (heavyhex "
                        "auto-placement only; default: reuse the pinned "
                        "placement unless it violates a threshold)")


def _analyze_opts(p, cycles=True):
    if cycles:   # standalone analyze parser ('all' inherits submit's)
        p.add_argument("--cycles", type=int, default=3,
                       help="fallback if job metadata is missing")
        p.add_argument("--code", choices=["heavyhex", "surface"],
                       default=None,
                       help="fallback if job metadata is missing "
                            "(e.g. offline --npz re-analysis)")
    p.add_argument("--model", choices=["cnn", "gnn"], default=None,
                   help="restrict the evaluation to one architecture; "
                        "default: evaluate every matching checkpoint, "
                        "inferring cnn/gnn from the {MODEL}_ filename "
                        "prefix")
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

    al = sub.add_parser("all", help="submit, wait for the job, then analyze"
                                    " (manual one-shot; the pipeline uses "
                                    "submit + collect instead)")
    _submit_opts(al)
    _analyze_opts(al, cycles=False)   # submit already owns --cycles
    al.add_argument("--poll", type=int, default=30,
                    help="job status poll interval in seconds")
    al.set_defaults(func=cmd_all)

    co = sub.add_parser("collect",
                        help="analyze finished pending jobs "
                             "(pending_jobs.json), carry the rest over")
    _analyze_opts(co)
    co.set_defaults(func=cmd_collect)

    # 서브커맨드 생략 시 all(제출->대기->분석 원샷)로 동작.
    argv = sys.argv[1:]
    if argv[:1] not in (["submit"], ["analyze"], ["all"], ["collect"],
                        ["-h"], ["--help"]):
        argv.insert(0, "all")
    args = ap.parse_args(argv)
    if args.cmd == "analyze" and not (args.job_id or args.npz):
        ap.error("analyze requires --job-id or --npz")
    args.func(args)


if __name__ == "__main__":
    main()
