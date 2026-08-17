#!/usr/bin/env python3
"""
Calibration-aware automatic patch placement (code-agnostic)
===========================================================
Finds every way the patch's required-edge graph embeds into the device
coupling map (rustworkx VF2, subgraph=True, induced=False — the patch
graph need not be an induced subgraph of the device), scores each valid
placement with the current calibration, and picks the minimum-expected-
error one. Dead qubits (error reported as None or >= 1.0, IBM's failure
marker) and qubits/edges beyond the exclusion thresholds knock a
placement out entirely.

Score (Vezvaee et al., arXiv:2510.18847 Appendix D):
  gate terms:  sum over pattern edges  (#2q gates on edge)  x err2q
             + sum over pattern qubits (#measurements x readout
                                        + #1q gates x err1q)
  idle term (Eq. D1/D2): the circuit is sliced into layers; per layer a
    qubit idles for t_idle = max_j(t_op^j) - t_op^i, and contributes
      p_x = p_y = t_idle / (4 T1)
      p_z = (t_idle / 2) (1/T2 - 1/(2 T1))    (clipped at 0)
    DD is ignored and raw T1/T2 are used — the score only needs to RANK
    placements, not predict absolute fidelity.

Pinned-placement policy (used by hardware/run_hw.py):
  the first submission of a loop chain selects a placement and records
  it in hardware/placement_<backend>_<code>.json; later submissions
  reuse it. Only when the recorded placement violates a threshold under
  the CURRENT calibration is a re-search triggered (warning + reason
  recorded). --reselect-layout forces a fresh search.
"""
import json
import time
from pathlib import Path

import rustworkx as rx

# exclusion thresholds (module constants, not knobs)
MAX_READOUT = 0.2
MAX_ERR1Q = 0.05
MAX_ERR2Q = 0.15
MIN_T1 = 20e-6          # seconds — below this the qubit is unusable
MIN_T2 = 20e-6
BAD_VALUE = 1.0         # IBM reports 1.0 (or None) for failed calibration

# nominal durations when the target reports none (ranking fallback)
NOMINAL_DUR = {"2q": 68e-9, "1q": 32e-9, "meas": 1.2e-6}

ENUM_CAP = 10_000       # VF2 enumeration cap (symmetry blow-up guard)


def _valid(v, limit):
    return v is not None and v < BAD_VALUE and v <= limit


def calib_from_target(target):
    """Target -> per-qubit/edge quality dict for scoring/exclusion.

    {readout, dur_meas, err1q, dur_1q, t1, t2} per qubit and
    {err2q, dur_2q} per sorted edge; missing entries stay absent and
    count as invalid when a placement needs them."""
    q = {"readout": {}, "dur_meas": {}, "err1q": {}, "dur_1q": {},
         "t1": {}, "t2": {}, "err2q": {}, "dur_2q": {}}

    def props_of(name):
        try:
            return target[name] or {}
        except Exception:
            return {}

    for qargs, p in props_of("measure").items():
        if qargs and p is not None:
            q["readout"][qargs[0]] = p.error
            if p.duration:
                q["dur_meas"][qargs[0]] = p.duration
    for gate in ("sx", "x"):
        props = props_of(gate)
        found = False
        for qargs, p in props.items():
            if qargs and p is not None:
                q["err1q"][qargs[0]] = p.error
                if p.duration:
                    q["dur_1q"][qargs[0]] = p.duration
                found = True
        if found:
            break
    for name in target.operation_names:
        for qargs, p in props_of(name).items():
            if qargs and len(qargs) == 2 and p is not None:
                e = tuple(sorted(qargs))
                prev = q["err2q"].get(e)
                if p.error is not None and (prev is None or p.error < prev):
                    q["err2q"][e] = p.error
                if p.duration and e not in q["dur_2q"]:
                    q["dur_2q"][e] = p.duration
    qp = getattr(target, "qubit_properties", None) or []
    for i, prop in enumerate(qp):
        if prop is not None:
            q["t1"][i] = getattr(prop, "t1", None)
            q["t2"][i] = getattr(prop, "t2", None)
    return q


def circuit_layers(qc, labels):
    """Slice the (local-index) patch circuit into layers of per-label
    ops: [{label: ("2q", partner) | ("1q", n) | ("meas", None)}].
    H counts as two 1q pulses (sx proxy); barriers only split layers."""
    from qiskit.converters import circuit_to_dag
    dag = circuit_to_dag(qc)
    layers = []
    for layer in dag.layers():
        ops = {}
        for node in layer["graph"].op_nodes():
            name = node.op.name
            idxs = [qc.find_bit(qb).index for qb in node.qargs]
            if name == "barrier":
                continue
            if len(idxs) == 2:
                a, b = labels[idxs[0]], labels[idxs[1]]
                ops[a] = ("2q", b)
                ops[b] = ("2q", a)
            elif name == "measure":
                ops[labels[idxs[0]]] = ("meas", None)
            elif name == "h":
                ops[labels[idxs[0]]] = ("1q", 2)
            else:
                ops[labels[idxs[0]]] = ("1q", 1)
        if ops:
            layers.append(ops)
    return layers


def score_placement(mapping, labels, pattern_edges, layers, calib):
    """(patch label -> device qubit) -> (score, None) or (None, reason).

    Exclusion first (dead values / thresholds / missing calib), then the
    gate + idle score."""
    for lb in labels:
        dq = mapping[lb]
        ro = calib["readout"].get(dq)
        e1 = calib["err1q"].get(dq)
        t1 = calib["t1"].get(dq)
        t2 = calib["t2"].get(dq)
        if not _valid(ro, MAX_READOUT):
            return None, f"q{dq}(label {lb}) readout={ro}"
        if not _valid(e1, MAX_ERR1Q):
            return None, f"q{dq}(label {lb}) 1q={e1}"
        if t1 is None or t1 < MIN_T1:
            return None, f"q{dq}(label {lb}) T1={t1}"
        if t2 is None or t2 < MIN_T2:
            return None, f"q{dq}(label {lb}) T2={t2}"
    for u, v in pattern_edges:
        de = tuple(sorted((mapping[u], mapping[v])))
        e2 = calib["err2q"].get(de)
        if not _valid(e2, MAX_ERR2Q):
            return None, f"edge {de}(labels {u}-{v}) 2q={e2}"

    score = 0.0
    for layer in layers:
        durs = {}
        for lb, (kind, extra) in layer.items():
            dq = mapping[lb]
            if kind == "2q":
                de = tuple(sorted((dq, mapping[extra])))
                d = calib["dur_2q"].get(de) or NOMINAL_DUR["2q"]
                score += 0.5 * calib["err2q"][de]     # half per endpoint
            elif kind == "meas":
                d = calib["dur_meas"].get(dq) or NOMINAL_DUR["meas"]
                score += calib["readout"][dq]
            else:
                d = extra * (calib["dur_1q"].get(dq) or NOMINAL_DUR["1q"])
                score += extra * calib["err1q"][dq]
            durs[lb] = d
        t_max = max(durs.values())
        for lb in labels:
            t_idle = t_max - durs.get(lb, 0.0)
            if t_idle <= 0:
                continue
            dq = mapping[lb]
            t1, t2 = calib["t1"][dq], calib["t2"][dq]
            p_xy = 2 * (t_idle / (4 * t1))            # p_x + p_y
            p_z = max(0.0, (t_idle / 2) * (1 / t2 - 1 / (2 * t1)))
            score += p_xy + p_z
    return score, None


def enumerate_placements(pattern_edges, coupling_edges, cap=ENUM_CAP):
    """Yield mappings {pattern label -> device qubit} (VF2 subgraph
    search, non-induced). Capped; symmetry dedup happens in the caller
    (best score per used-qubit set)."""
    labels = sorted({x for e in pattern_edges for x in e})
    lidx = {lb: i for i, lb in enumerate(labels)}
    pat = rx.PyGraph()
    pat.add_nodes_from(range(len(labels)))
    pat.add_edges_from([(lidx[u], lidx[v], None) for u, v in pattern_edges])
    dev_nodes = sorted({x for e in coupling_edges for x in e})
    didx = {q: i for i, q in enumerate(dev_nodes)}
    dev = rx.PyGraph()
    dev.add_nodes_from(range(len(dev_nodes)))
    dev.add_edges_from([(didx[a], didx[b], None)
                        for a, b in {tuple(sorted(e))
                                     for e in coupling_edges}])
    n = 0
    for vf2 in rx.vf2_mapping(dev, pat, subgraph=True, induced=False):
        # vf2 maps device node index -> pattern node index
        inv = {pat_i: dev_i for dev_i, pat_i in vf2.items()}
        yield {labels[i]: dev_nodes[inv[i]] for i in range(len(labels))}
        n += 1
        if n >= cap:
            print(f"WARNING: placement enumeration capped at {cap}")
            break


def select_placement(labels, pattern_edges, qc, coupling_edges, target,
                     static_mapping=None, cap=ENUM_CAP):
    """Search + score every placement; return a result dict or None.

    {mapping, score, runners_up: [(score, sorted qubits)...],
     static_score (or exclusion reason), n_candidates, n_excluded}"""
    calib = calib_from_target(target)
    layers = circuit_layers(qc, labels)
    best = {}                    # frozenset(qubits) -> (score, mapping)
    excluded = 0
    for mapping in enumerate_placements(pattern_edges, coupling_edges,
                                        cap):
        s, why = score_placement(mapping, labels, pattern_edges, layers,
                                 calib)
        if s is None:
            excluded += 1
            continue
        key = frozenset(mapping.values())
        if key not in best or s < best[key][0]:
            best[key] = (s, mapping)
    if not best:
        return None
    ranked = sorted(best.values(), key=lambda t: t[0])
    static_score = None
    if static_mapping is not None:
        ss, why = score_placement(static_mapping, labels, pattern_edges,
                                  layers, calib)
        static_score = ss if ss is not None else f"excluded ({why})"
    s0, m0 = ranked[0]
    return {"mapping": m0, "score": s0,
            "runners_up": [(s, sorted(m.values()))
                           for s, m in ranked[1:4]],
            "static_score": static_score,
            "n_candidates": len(best), "n_excluded": excluded}


def check_placement(mapping, labels, pattern_edges, target):
    """Re-validate a pinned placement under CURRENT calibration.
    Returns (ok, reason)."""
    calib = calib_from_target(target)
    # thresholds only (no scoring): reuse the exclusion part
    s, why = score_placement(mapping, labels, pattern_edges, [], calib)
    return (why is None), why


def placement_path(state_dir, backend, code):
    return Path(state_dir) / f"placement_{backend}_{code}.json"


def resolve_placement(backend_name, code, state_dir, labels,
                      pattern_edges, qc, coupling_edges, target,
                      static_mapping=None, reselect=False):
    """Pinned-placement policy — returns (mapping, info dict) or
    (None, info) when no valid placement exists (caller falls back to
    the static embedding).

    info carries everything job.json should record: mapping, score,
    runners_up, static_score, reselected(bool), reselect_reason,
    chosen_at."""
    path = placement_path(state_dir, backend_name, code)
    reason = None
    if path.exists() and not reselect:
        rec = json.load(open(path))
        # JSON keys are strings — coerce back to int patch labels
        mapping = {int(k): v for k, v in rec["mapping"].items()}
        ok, why = check_placement(mapping, labels, pattern_edges, target)
        if ok:
            print(f"placement: {path.name} 재사용 (score at selection: "
                  f"{rec.get('score'):.6f})")
            info = dict(rec)
            info.update({"reselected": False, "reselect_reason": None})
            return mapping, info
        reason = why
        print(f"WARNING: 기록된 배치가 현재 캘리브레이션 임계값 위반 "
              f"({why}) — 재탐색")
    res = select_placement(labels, pattern_edges, qc, coupling_edges,
                           target, static_mapping=static_mapping)
    if res is None:
        return None, {"reselected": True, "reselect_reason": reason,
                      "error": "no valid placement"}
    info = {"mapping": {str(k): v for k, v in res["mapping"].items()},
            "score": res["score"], "runners_up": res["runners_up"],
            "static_score": res["static_score"],
            "n_candidates": res["n_candidates"],
            "n_excluded": res["n_excluded"],
            "chosen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "reselected": path.exists() or reselect,
            "reselect_reason": reason}
    Path(state_dir).mkdir(parents=True, exist_ok=True)
    json.dump(info, open(path, "w"), indent=2)
    print(f"placement: 신규 선택 score={res['score']:.6f} "
          f"(후보 {res['n_candidates']}, 배제 {res['n_excluded']}, "
          f"정적 임베딩 score={res['static_score']}) -> {path.name}")
    return res["mapping"], info
