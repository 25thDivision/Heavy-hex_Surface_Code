#!/usr/bin/env python3
"""
General heavy-hex diamond code circuit builder (correctness-first).
==================================================================
Works for any solved (dx,dz) diamond code from diamond_generator:
  - per-check isolated gadgets (proven v2/v5 method):
      Z-check: parity folds -> CX(rep->anc) -> M -> unfold
      X-check: H on support -> same in H-frame -> H undo
  - checks whose support cannot reach a free ancilla directly are measured
    via cancel-fold relays (v2 X58 technique): parity chains through
    intervening qubits with pre-cancel CXs; provably pure, all-classical.
  - dedicated rung ancillas, never reset; values via per-ancilla XOR chains.

Physical mapping to Heron (boston/aachen/pittsburgh):
  code (r,c) -> chip row r+1, chip col c+1:
      data/bridge: qubit = (r+1)*20 + (c+1)
      rung anc between code rows r,r+1 at col c: the chip rung qubit adjacent
      to both endpoints (looked up from the coupling map).
"""
import json
import numpy as np
from itertools import combinations
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from heavyhex_curcuits.diamond_generator import build_code, gf2_rank, rung_of_check


# --------------------------------------------------------------- solving
def solve(dx, dz, cache=True):
    fn = f"/home/claude/sol_{dx}{dz}.json"
    if cache:
        try:
            d = json.load(open(fn))
            return d
        except FileNotFoundError:
            pass
    code = build_code(dx, dz)
    plq = code.plaquettes()
    fulls = {k: v[0] for k, v in plq.items() if v[1]}
    parts = [v[0] for k, v in plq.items() if not v[1]]
    dset = set(code.data)
    diag = [((r, c), (r + 1, c + dc)) for (r, c) in code.data
            for dc in (-2, 2) if (r + 1, c + dc) in dset]
    P = parts + diag
    Zf = [s for k, s in fulls.items() if code.checkerboard_type(*k) == 'Z']
    Xf = [s for k, s in fulls.items() if code.checkerboard_type(*k) == 'X']
    vecs = np.array([code.vec(s) for s in P])
    PP = (vecs @ vecs.T) % 2
    PZf = (vecs @ np.array([code.vec(s) for s in Zf]).T) % 2
    PXf = (vecs @ np.array([code.vec(s) for s in Xf]).T) % 2
    okZ = [i for i in range(len(P)) if not PXf[i].any()]
    okX = [i for i in range(len(P)) if not PZf[i].any()]
    nb = code.n - 1 - len(Zf) - len(Xf)
    sol = None
    for nz in range(2, nb - 1):
        for zsel in combinations(okZ, nz):
            zs = set(zsel)
            for xsel in combinations([i for i in okX if i not in zs], nb - nz):
                xs = set(xsel)
                if any(PP[i, j] for i in xs for j in zs):
                    continue
                Z = Zf + [P[i] for i in zs]
                X = Xf + [P[i] for i in xs]
                if gf2_rank(np.array([code.vec(s) for s in Z])) + \
                   gf2_rank(np.array([code.vec(s) for s in X])) != code.n - 1:
                    continue
                dzv, zl = code.min_logical(Z, X, maxw=max(dx, dz))
                dxv, xl = code.min_logical(X, Z, maxw=max(dx, dz))
                if dzv >= dz and dxv >= dx:
                    sol = dict(Z=[list(map(list, s)) for s in Z],
                               X=[list(map(list, s)) for s in X],
                               zl=list(map(list, zl)), xl=list(map(list, xl)),
                               dz=dzv, dx=dxv)
                    break
            if sol:
                break
        if sol:
            break
    assert sol, f"no solution for ({dx},{dz})"
    json.dump(sol, open(fn, 'w'))
    return sol


# --------------------------------------------------------------- mapping
def chip_maps(dx, dz, coupling_json):
    """returns (code, phys_of_node, adj) where nodes are code data (r,c),
    bridges ('b',r,c) between (r,c)-(r,c+2), rung ancs ('a',r,c)."""
    cm = json.load(open(coupling_json))
    edges = {tuple(sorted(e)) for e in cm['coupling_map']}
    import collections
    adjq = collections.defaultdict(set)
    for a, b in edges:
        adjq[a].add(b)
        adjq[b].add(a)
    code = build_code(dx, dz)
    phys = {}
    for (r, c) in code.data:
        phys[(r, c)] = (r + 1) * 20 + (c + 1)
    for (r, c) in code.data:
        if (r, c + 2) in code.dset:
            phys[('b', r, c)] = (r + 1) * 20 + (c + 2)
    for (u, v) in code.rungs:
        pu, pv = phys[u], phys[v]
        rq = (adjq[pu] & adjq[pv])
        assert len(rq) == 1, (u, v, rq)
        phys[('a',) + u] = rq.pop()
    # verify all needed couplings exist
    for (r, c) in code.data:
        if (r, c + 2) in code.dset:
            assert tuple(sorted((phys[(r, c)], phys[('b', r, c)]))) in edges
            assert tuple(sorted((phys[('b', r, c)], phys[(r, c + 2)]))) in edges
    return code, phys


# --------------------------------------------------------------- assignment
def assign(dx, dz):
    """returns per-check: anc node, arms [(outer_chain...)], relay paths."""
    sol = solve(dx, dz)
    code = build_code(dx, dz)
    fullset = None
    checks = []          # (name, type, supp tuple-of-(r,c))
    for i, s in enumerate(sol['Z']):
        checks.append((f"Z{i}", 'Z', tuple(map(tuple, s))))
    for i, s in enumerate(sol['X']):
        checks.append((f"X{i}", 'X', tuple(map(tuple, s))))
    used = {r: 0 for r in code.rungs}
    plan = {}
    deferred = []
    for name, t, supp in checks:
        opts = [r for r in rung_of_check(code, supp) if used[r] < 2]
        if opts:
            # prefer rung fully inside support
            opts.sort(key=lambda r: -len(set(r) & set(supp)))
            r = opts[0]
            used[r] += 1
            plan[name] = dict(t=t, supp=supp, anc=r, relay=False)
        else:
            deferred.append((name, t, supp))
    free = [r for r, u in used.items() if u < 2]
    for name, t, supp in deferred:
        # nearest free rung by center distance
        cy = np.mean([q[0] for q in supp])
        cx = np.mean([q[1] for q in supp])
        free.sort(key=lambda r: abs(r[0][0] + .5 - cy) + abs(r[0][1] - cx) / 2)
        r = free.pop(0)
        used[r] += 1
        plan[name] = dict(t=t, supp=supp, anc=r, relay=True)
    return code, sol, plan


# --------------------------------------------------------------- circuit
class GeneralDiamondCircuit:
    def __init__(self, dx, dz, coupling_json, num_cycles=3):
        self.dx, self.dz = dx, dz
        self.code, self.phys = chip_maps(dx, dz, coupling_json)
        _, self.sol, self.plan = assign(dx, dz)
        self.num_cycles = num_cycles
        # local indices
        self.nodes = sorted(self.phys, key=lambda k: self.phys[k])
        self.L = {n: i for i, n in enumerate(self.nodes)}
        self.nq = len(self.nodes)
        self.anc_of = {name: ('a',) + p['anc'][0] for name, p in self.plan.items()}
        self.order = sorted(self.plan)        # deterministic cycle order
        # patch graph for relay paths
        import collections
        self.adj = collections.defaultdict(list)
        for (r, c) in self.code.data:
            if ('b', r, c) in self.phys:
                self.adj[(r, c)].append(('b', r, c))
                self.adj[('b', r, c)].append((r, c))
                self.adj[('b', r, c)].append((r, c + 2))
                self.adj[(r, c + 2)].append(('b', r, c))
        for (u, v) in self.code.rungs:
            a = ('a',) + u
            self.adj[u].append(a)
            self.adj[a].append(u)
            self.adj[a].append(v)
            self.adj[v].append(a)
        # relay residuals: first interior node of each chain if it is an anc
        self.residuals = {}
        for name, p in self.plan.items():
            if not p['relay']:
                continue
            anc = ('a',) + tuple(p['anc'][0])
            res = []
            for qsupp in p['supp']:
                path = self._path_lazy(qsupp, anc)
                if len(path) > 2 and isinstance(path[1], tuple) and path[1][0] == 'a':
                    res.append(path[1])
            self.residuals[name] = res

    def _path_lazy(self, src, dst):
        if not hasattr(self, '_adj_ready'):
            self._adj_ready = True
        return self._path(src, dst)

    def _path(self, src, dst):
        import collections
        prev = {src: None}
        dq = collections.deque([src])
        while dq:
            x = dq.popleft()
            if x == dst:
                out = []
                while x is not None:
                    out.append(x)
                    x = prev[x]
                return out[::-1]
            for y in self.adj[x]:
                if y not in prev:
                    prev[y] = x
                    dq.append(y)
        raise RuntimeError((src, dst))

    # ---- gadgets (all-classical XOR networks; provably pure) ----
    def _direct_gadget(self, qc, p, creg, bit):
        t, supp, (u, v) = p['t'], p['supp'], p['anc']
        anc = ('a',) + (u,)[0:0] or None
        anc = ('a',) + u
        reps = [q for q in (u, v) if q in supp]
        # fold arms: BFS tree from reps within supp (horizontal steps)
        sset = set(supp)
        parent = {q: None for q in reps}
        frontier = list(reps)
        while frontier:
            nxt = []
            for q in frontier:
                for dc in (-2, 2):
                    w = (q[0], q[1] + dc)
                    if w in sset and w not in parent:
                        parent[w] = q
                        nxt.append(w)
            frontier = nxt
        # emit: deepest-first folds (outer -> parent) as 2-CX parity folds
        depth = {q: 0 for q in reps}
        stack = [q for q in parent if parent[q]]
        for q in sorted(parent, key=lambda q: 0 if parent[q] is None else 1):
            if parent[q] is not None:
                depth[q] = depth[parent[q]] + 1
        arms = sorted([q for q in parent if parent[q]], key=lambda q: -depth[q])
        if t == 'X':
            for q in supp:
                qc.h(self.L[q])
        for q in arms:                       # fold
            b = self._bridge(q, parent[q])
            qc.cx(self.L[q], self.L[b])
            qc.cx(self.L[b], self.L[parent[q]])
        for q in reps:
            qc.cx(self.L[q], self.L[anc])
        qc.measure(self.L[anc], creg[bit])
        for q in arms[::-1]:                 # unfold
            b = self._bridge(q, parent[q])
            qc.cx(self.L[b], self.L[parent[q]])
            qc.cx(self.L[q], self.L[b])
        if t == 'X':
            for q in supp:
                qc.h(self.L[q])

    def _bridge(self, a, b):
        (r, c1), (_, c2) = a, b
        return ('b', r, min(c1, c2))

    def _relay_gadget(self, qc, p, creg, bit):
        """cancel-fold: for each support qubit, parity-chain to the ancilla;
        pre-cancel every intermediate that is data or (no-reset) ancilla."""
        t, supp, (u, v) = p['t'], p['supp'], p['anc']
        anc = ('a',) + u
        if t == 'X':
            for q in supp:
                qc.h(self.L[q])
        chains = [self._path(q, anc) for q in supp]
        # telescoping pre-cancel, ASCENDING, interiors i>=2 only (never touch
        # the H-rotated support qubit p0). Residual contribution = p1 of each
        # chain: a bridge (|0>, vanishes) or a rung ancilla whose Z-value is
        # the last recorded raw -> cancelled in SOFTWARE at decode time.
        for path in chains:
            for i in range(2, len(path) - 1):
                qc.cx(self.L[path[i]], self.L[path[i - 1]])
        for path in chains:                 # forward chains
            for a, b in zip(path, path[1:]):
                qc.cx(self.L[a], self.L[b])
        qc.measure(self.L[anc], creg[bit])
        for path in chains:                 # unfold chain (minus final hop)
            for a, b in list(zip(path, path[1:]))[-2::-1]:
                qc.cx(self.L[a], self.L[b])
        for path in chains:                 # unfold pre-cancels
            for i in range(len(path) - 2, 1, -1):
                qc.cx(self.L[path[i]], self.L[path[i - 1]])
        if t == 'X':
            for q in supp:
                qc.h(self.L[q])

    def build_circuit(self, initial_state=0, inject=None):
        q = QuantumRegister(self.nq, 'q')
        ncyc = self.num_cycles
        syn = ClassicalRegister(len(self.order) * ncyc, 'syn')
        dat = ClassicalRegister(self.code.n, 'data')
        qc = QuantumCircuit(q, syn, dat)
        if initial_state == 1:
            for s in self.sol['xl']:
                qc.x(self.L[tuple(s)])
        bit = 0
        for cyc in range(ncyc):
            for name in self.order:
                p = self.plan[name]
                if p['relay']:
                    self._relay_gadget(qc, p, syn, bit)
                else:
                    self._direct_gadget(qc, p, syn, bit)
                bit += 1
            qc.barrier()
            if inject is not None and inject[2] == cyc:
                pauli, dq_, _ = inject
                getattr(qc, pauli.lower())(self.L[tuple(dq_)])
                qc.barrier()
        for i, d in enumerate(sorted(self.code.data)):
            qc.measure(self.L[d], dat[i])
        return qc

    def check_values(self, syn_rows, num_cycles):
        n = syn_rows.shape[0]
        vals = {nm: np.zeros((n, num_cycles), dtype=int) for nm in self.order}
        prev = {}
        for cyc in range(num_cycles):
            for j, nm in enumerate(self.order):
                raw = syn_rows[:, cyc * len(self.order) + j]
                a = self.anc_of[nm]
                v = raw ^ prev.get(a, 0)
                for ra in getattr(self, 'residuals', {}).get(nm, []):
                    v = v ^ prev.get(ra, 0)      # software residual cancel
                vals[nm][:, cyc] = v
                prev[a] = raw.copy()
        return vals
