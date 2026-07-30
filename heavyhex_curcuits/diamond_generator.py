#!/usr/bin/env python3
"""
Generalized (dx,dz) heavy-hex diamond surface-code generator.
=============================================================
Reconstructs the Vezvaee-family code from geometry:
  1. data on brick-lattice vertices inside a diamond (row-width profile)
  2. checks = all plaquettes (full + truncated) with checkerboard types
  3. each rung (vertical edge) = dedicated ancilla for its <=2 flanking checks
  4. boundary completion solved by exhaustive search for target (dz, dx)

Coordinates: (row r, col c); data at given (r, c); horizontal neighbours at
c +/- 2 (bridge between); vertical rungs exist at (r, r+1, c) iff both data
present AND c on the brick pattern for that row pair (alternating offset).
Physical qubit numbering is assigned later from a device coupling map.
"""
import numpy as np
from itertools import combinations, product


def gf2_rank(A):
    A = A.copy() % 2
    r = 0
    for c in range(A.shape[1]):
        p = next((i for i in range(r, A.shape[0]) if A[i, c]), None)
        if p is None:
            continue
        A[[r, p]] = A[[p, r]]
        for i in range(A.shape[0]):
            if i != r and A[i, c]:
                A[i] ^= A[r]
        r += 1
    return r


class DiamondCode:
    def __init__(self, row_profile, row_offsets, rung_phase=0):
        """
        row_profile: data count per row (odd widths), e.g. (3,3): [1,3,5,5,3]
        row_offsets: starting col of each row (cols step by 2 within a row)
        rung_phase: brick alternation phase (rungs at c%4 == (2*(rp+phase))%4)
        """
        self.rows = len(row_profile)
        self.data = []                       # list of (r, c)
        for r, (w, off) in enumerate(zip(row_profile, row_offsets)):
            for k in range(w):
                self.data.append((r, off + 2 * k))
        self.dset = set(self.data)
        self.idx = {d: i for i, d in enumerate(self.data)}
        self.n = len(self.data)
        # rungs: vertical data pairs on the brick pattern
        self.rungs = []
        for (r, c) in self.data:
            if (r + 1, c) in self.dset and c % 4 == (2 * (r + rung_phase)) % 4:
                self.rungs.append(((r, c), (r + 1, c)))

    # ---------------- plaquettes ----------------
    def plaquettes(self):
        """unit squares (col pair c..c+2, row pair r..r+1) with >=2 corners.
        Returns dict key=(r, c) -> (corners_present, full?)"""
        out = {}
        rs = [r for r, _ in self.data]
        cs = [c for _, c in self.data]
        for r in range(min(rs) - 1, max(rs) + 1):
            for c in range(min(cs) - 2, max(cs) + 2, 2):
                corners = [(r, c), (r, c + 2), (r + 1, c), (r + 1, c + 2)]
                pres = tuple(q for q in corners if q in self.dset)
                if len(pres) >= 2:
                    out[(r, c)] = (pres, len(pres) == 4)
        return out

    def checkerboard_type(self, r, c, phase=0):
        return 'Z' if ((c // 2) + r + phase) % 2 == 0 else 'X'

    # ---------------- stabilizer solving ----------------
    def vec(self, supp):
        v = np.zeros(self.n, dtype=int)
        v[[self.idx[q] for q in supp]] = 1
        return v

    def min_logical(self, S_list, O_list, maxw=7):
        S = np.array([self.vec(s) for s in S_list])
        O = np.array([self.vec(s) for s in O_list])
        rS = gf2_rank(S)
        for w in range(1, maxw + 1):
            for supp in combinations(range(self.n), w):
                v = np.zeros(self.n, dtype=int)
                v[list(supp)] = 1
                if np.any((O @ v) % 2):
                    continue
                if gf2_rank(np.vstack([S, v])) == rS:
                    continue
                return w, tuple(self.data[i] for i in supp)
        return maxw + 1, None

    def verify(self, Z, X):
        if any(len(set(a) & set(b)) % 2 for a in X for b in Z):
            return None
        rz = gf2_rank(np.array([self.vec(s) for s in Z]))
        rx = gf2_rank(np.array([self.vec(s) for s in X]))
        if rz + rx != self.n - 1:
            return None
        dz, zl = self.min_logical(Z, X)
        dx, xl = self.min_logical(X, Z)
        return dz, dx, zl, xl


def build_33_reference():
    """(3,3): profile [1,3,5,5,3]; regression target = verified V6 set."""
    prof = [1, 3, 5, 5, 3]
    offs = [4, 2, 0, 0, 2]
    code = DiamondCode(prof, offs, rung_phase=0)
    return code


if __name__ == '__main__':
    code = build_33_reference()
    print(f"(3,3): data={code.n} rungs={len(code.rungs)}")
    plq = code.plaquettes()
    full = {k: v for k, v in plq.items() if v[1]}
    part = {k: v for k, v in plq.items() if not v[1]}
    print(f"full plaquettes={len(full)} partial(>=2 corners)={len(part)}")
    for k in sorted(full):
        print("  full", k, full[k][0], code.checkerboard_type(*k))


# ====================== generalized construction & solver ======================
def build_code(dx, dz, drop='max'):
    """Data = even sublattice of (u,v) in [0,2dx-1]x[0,2dz-1] minus one corner.
    Convert to (r,c): r=(u+v)//2, c = u-v+(2*dz-1). Returns DiamondCode."""
    pts = [(u, v) for u in range(2 * dx) for v in range(2 * dz)
           if (u + v) % 2 == 0]
    # drop the (u,v) corner with largest u+v (bottom apex) among even sites
    apex = max(pts, key=lambda p: (p[0] + p[1], p[0]))
    if drop == 'max':
        pts.remove(apex)
    rc = sorted(((u + v) // 2, u - v + (2 * dz - 2)) for u, v in pts)
    rows = sorted(set(r for r, _ in rc))
    prof, offs = [], []
    for r in rows:
        cs = sorted(c for rr, c in rc if rr == r)
        prof.append(len(cs)); offs.append(cs[0])
    # rung phase: match (3,3) convention (phase 0 reproduces verified rungs)
    return DiamondCode(prof, offs, rung_phase=0)


def rung_of_check(code, supp):
    """rungs whose pair intersects supp AND every support qubit reachable
    from a rep by a horizontal chain (c +/- 2 steps) staying inside supp."""
    out = []
    sset = set(supp)
    for (a, b) in code.rungs:
        reps = [q for q in (a, b) if q in sset]
        if not reps:
            continue
        seen = set(reps)
        frontier = list(reps)
        while frontier:
            nxt = []
            for (r, c) in frontier:
                for dc in (-2, 2):
                    q = (r, c + dc)
                    if q in sset and q not in seen:
                        seen.add(q)
                        nxt.append(q)
            frontier = nxt
        if seen == sset:
            out.append((a, b))
    return out


def solve_code(dx, dz, verbose=True, maxw=None):
    """full plaquettes fixed; boundary slots searched; returns solution dict."""
    from itertools import product as iproduct
    code = build_code(dx, dz)
    plq = code.plaquettes()
    fulls = {k: v[0] for k, v in plq.items() if v[1]}
    parts = {k: v[0] for k, v in plq.items() if not v[1]}
    rungs = list(code.rungs)
    n_stab = code.n - 1
    if maxw is None:
        maxw = max(dx, dz)

    # mandatory full checks with checkerboard types, assigned to contained rung
    fixed = []          # (type, supp, rung)
    used = {r: [] for r in rungs}
    for k, supp in fulls.items():
        t = code.checkerboard_type(*k)
        cands = [r for r in rung_of_check(code, supp)
                 if set(r) <= set(supp)]
        assert len(cands) == 1, (k, cands)
        fixed.append((t, supp, cands[0]))
        used[cands[0]].append(t)

    # boundary candidates: each ancilla has capacity 2 (one check per round);
    # slot TYPE is free (solid-color ancillas allowed, per Fig 1a legend)
    slots = {r: ['*'] * (2 - len(used[r])) for r in rungs}
    cand_by_rung = {r: [] for r in rungs}
    for k, supp in parts.items():
        for r in rung_of_check(code, supp):
            if r in cand_by_rung:
                for t in 'ZX':
                    cand_by_rung[r].append((t, supp))
    open_slots = []
    for r in rungs:
        for _ in slots[r]:
            open_slots.append(r)
        if slots[r] and not cand_by_rung[r]:
            if verbose:
                print(f"  rung {r}: NO candidates")
            return None
    if verbose:
        print(f"({dx},{dz}): data={code.n} rungs={len(rungs)} fulls={len(fixed)} "
              f"open slots={len(open_slots)} "
              f"space={np.prod([float(len(cand_by_rung[r])) for r in open_slots]):.0f}")

    Zf = [s for t, s, _ in fixed if t == 'Z']
    Xf = [s for t, s, _ in fixed if t == 'X']

    best = (0, 0)
    for combo in iproduct(*[cand_by_rung[r] for r in open_slots]):
        # a rung's two boundary checks must be distinct supports
        seen = {}
        clash = False
        for r, (t, s) in zip(open_slots, combo):
            if s in seen.get(r, ()):
                clash = True
                break
            seen.setdefault(r, []).append(s)
        if clash:
            continue
        Z = list(Zf); X = list(Xf)
        for r, (t, s) in zip(open_slots, combo):
            (Z if t == 'Z' else X).append(s)
        if len(Z) + len(X) != n_stab:
            continue
        if any(len(set(a) & set(b)) % 2 for a in X for b in Z):
            continue
        rz = gf2_rank(np.array([code.vec(s) for s in Z]))
        rx = gf2_rank(np.array([code.vec(s) for s in X]))
        if rz + rx != n_stab:
            continue
        dzv, zl = code.min_logical(Z, X, maxw=maxw)
        dxv, xl = code.min_logical(X, Z, maxw=maxw)
        if dzv >= dz and dxv >= dx:
            sol = dict(code=code, Z=Z, X=X, dz=dzv, dxv=dxv, zl=zl, xl=xl,
                       slots=list(zip(open_slots, combo)), fixed=fixed)
            if verbose:
                print(f"  SOLUTION dz={dzv} dx={dxv} Z_L={zl} X_L={xl}")
            return sol
        if (min(dzv, dxv), dzv + dxv) > (min(best), sum(best)):
            best = (dzv, dxv)
    if verbose:
        print(f"  no solution meeting (dz={dz},dx={dx}); best={best}")
    return None
