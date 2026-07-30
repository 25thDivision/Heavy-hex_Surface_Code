#!/usr/bin/env python3
"""
Depth-optimized general diamond circuit (V6 scheme generalized).
================================================================
Per cycle: two rounds. Each ancilla measures one of its two checks per round.
Within a round:
  fold layer: shared TRUE-CX arms via 3-CX bridge relays.
     arm (ctrl -> tgt): transmits Z-info to tgt (Z-check reads tgt as rep)
     and X-info stays extendable at ctrl (X-check reads ctrl via anc-control).
  anc couplings: Z-check: CX(rep -> anc) on its rung reps;
                 X-check: H(anc), CX(anc -> read), H(anc).
  measure all round ancillas (no reset; XOR-chain decode).
  unfold = same relay layer (palindrome, self-inverse).
  relay checks (diagonal boundary pairs): isolated cancel-fold gadgets
  appended after the unfold of their round (patch in code frame).

Round assignment: exhaustive over per-ancilla binary choice with arm-conflict
constraints (no qubit is both an arm target and an arm control of different
arms within one round; no duplicated targets).
"""
import numpy as np
from itertools import product
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from heavyhex_curcuits.heavyhex_general import GeneralDiamondCircuit


class DepthOptDiamond(GeneralDiamondCircuit):
    def __init__(self, dx, dz, coupling_json, num_cycles=3):
        super().__init__(dx, dz, coupling_json, num_cycles)
        self._compute_arms()
        self._color_rounds()

    # ---------------- arms per check ----------------
    def _compute_arms(self):
        self.arms = {}          # name -> list of (ctrl, tgt)
        self.reads = {}         # name -> qubits coupled to anc
        for name, p in self.plan.items():
            if p['relay']:
                self.arms[name] = []
                self.reads[name] = []
                continue
            t, supp, (u, v) = p['t'], p['supp'], p['anc']
            reps = [q for q in (u, v) if q in supp]
            others = [q for q in supp if q not in reps]
            arms = []
            for q in others:
                near = [r for r in reps if r[0] == q[0] and abs(r[1]-q[1]) == 2]
                assert near, (name, q, reps)   # all solved checks are 1-step
                r = near[0]
                arms.append((q, r) if t == 'Z' else (r, q))
            self.arms[name] = arms
            self.reads[name] = reps

    # ---------------- round coloring ----------------
    def _color_rounds(self):
        ancs = sorted({p['anc'] for p in self.plan.values()})
        by_anc = {a: [nm for nm, p in self.plan.items() if p['anc'] == a]
                  for a in ancs}
        assert all(len(v) == 2 for v in by_anc.values())

        def ok(round_set):
            tgts, ctrls, armset = {}, {}, set()
            for nm in round_set:
                for (c, t) in self.arms[nm]:
                    if (c, t) in armset:
                        continue
                    if t in tgts or t in ctrls or c in tgts:
                        # allow exact shared arm only (handled above)
                        return False
                    armset.add((c, t))
                    tgts[t] = nm
                    ctrls[c] = nm
            return True

        def greedy_fit(names):
            """returns (fitted, demoted) keeping arm-compatible subset."""
            fitted, demoted = [], []
            tgts, ctrls, armset = {}, {}, set()
            for nm in sorted(names, key=lambda n: -len(self.arms[n])):
                bad = False
                new = []
                for (c, t) in self.arms[nm]:
                    if (c, t) in armset:
                        continue
                    if t in tgts or t in ctrls or c in tgts:
                        bad = True
                        break
                    new.append((c, t))
                if bad:
                    demoted.append(nm)
                else:
                    for (c, t) in new:
                        armset.add((c, t))
                        tgts[t] = nm
                        ctrls[c] = nm
                    fitted.append(nm)
            return fitted, demoted

        best = None
        for bits in product((0, 1), repeat=len(ancs)):
            r1 = [by_anc[a][b] for a, b in zip(ancs, bits)]
            r2 = [by_anc[a][1 - b] for a, b in zip(ancs, bits)]
            f1, d1 = greedy_fit([n for n in r1 if not self.plan[n]['relay']])
            f2, d2 = greedy_fit([n for n in r2 if not self.plan[n]['relay']])
            score = len(d1) + len(d2)
            if best is None or score < best[0]:
                best = (score, r1, r2, set(d1) | set(d2))
            if score == 0:
                break
        _, r1, r2, demoted = best
        self.round1, self.round2 = r1, r2
        self.demoted = demoted

        def emit_order(names):
            direct = [n for n in sorted(names)
                      if not self.plan[n]['relay'] and n not in demoted]
            iso = [n for n in sorted(names) if n in demoted]
            rel = [n for n in sorted(names) if self.plan[n]['relay']]
            return direct + iso + rel
        # decode order must equal physical emission order (residual bookkeeping)
        self.order = emit_order(r1) + emit_order(r2)

    # ---------------- circuit ----------------
    def _relay_layer(self, qc, arms):
        for c, t in arms:
            qc.cx(self.L[c], self.L[self._bridge(c, t)])
        for c, t in arms:
            qc.cx(self.L[self._bridge(c, t)], self.L[t])
        for c, t in arms:
            qc.cx(self.L[c], self.L[self._bridge(c, t)])

    def _round(self, qc, names, creg, bits):
        direct = [n for n in sorted(names)
                  if not self.plan[n]['relay'] and n not in self.demoted]
        relays = [n for n in sorted(names) if self.plan[n]['relay']]
        isolated = [n for n in sorted(names) if n in self.demoted]
        armset = []
        for n in direct:
            for a in self.arms[n]:
                if a not in armset:
                    armset.append(a)
        self._relay_layer(qc, armset)                    # fold
        for n in direct:
            p = self.plan[n]
            anc = ('a',) + tuple(p['anc'][0])
            if p['t'] == 'Z':
                for q in self.reads[n]:
                    qc.cx(self.L[q], self.L[anc])
            else:
                qc.h(self.L[anc])
                for q in self.reads[n]:
                    qc.cx(self.L[anc], self.L[q])
                qc.h(self.L[anc])
        for n in direct:
            anc = ('a',) + tuple(self.plan[n]['anc'][0])
            qc.measure(self.L[anc], creg[bits[n]])
        self._relay_layer(qc, armset)                    # unfold
        for n in isolated:                               # demoted direct checks
            self._direct_gadget(qc, self.plan[n], creg, bits[n])
        for n in relays:                                 # diagonal relay checks
            self._relay_gadget(qc, self.plan[n], creg, bits[n])

    def build_circuit(self, initial_state=0, inject=None):
        q = QuantumRegister(self.nq, 'q')
        ncyc = self.num_cycles
        syn = ClassicalRegister(len(self.order) * ncyc, 'syn')
        dat = ClassicalRegister(self.code.n, 'data')
        qc = QuantumCircuit(q, syn, dat)
        if initial_state == 1:
            for s in self.sol['xl']:
                qc.x(self.L[tuple(s)])
        for cyc in range(ncyc):
            base = cyc * len(self.order)
            bits = {n: base + i for i, n in enumerate(self.order)}
            self._round(qc, self.round1, syn, bits)
            self._round(qc, self.round2, syn, bits)
            qc.barrier()
            if inject is not None and inject[2] == cyc:
                pauli, dq_, _ = inject
                getattr(qc, pauli.lower())(self.L[tuple(dq_)])
                qc.barrier()
        for i, d in enumerate(sorted(self.code.data)):
            qc.measure(self.L[d], dat[i])
        return qc


if __name__ == '__main__':
    CJ = '/mnt/user-data/uploads/coupling_ibm_boston.json'
    for dx, dz in [(3, 3), (3, 5), (5, 3)]:
        gc = DepthOptDiamond(dx, dz, CJ, num_cycles=2)
        qc = gc.build_circuit()
        ops = qc.count_ops()
        print(f"({dx},{dz}): depth={qc.depth()/2:.0f}/cyc "
              f"cx={ops.get('cx')/2:.0f}/cyc h={ops.get('h',0)/2:.0f}/cyc "
              f"demoted={len(gc.demoted)}")
