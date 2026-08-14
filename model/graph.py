#!/usr/bin/env python3
"""
Detector graph construction for the GNN decoder — shared infrastructure
=======================================================================
(you don't need to modify this file; the model you complete lives in
model/gnn_skeleton.py)

Node = one Stim DETECTOR, in exactly the order of
heavyhex33_stim._append_detectors (the order MWPM consumes):
  1) per cycle, iterate CYCLE_ORDER: Z-check detectors at every cycle,
     X-check detectors from cycle 1 on (temporal XOR)
  2) 8 final-Z detectors (Z_POS order)
For cycles=3 that is 8*3 + 8*2 + 8 = 48 nodes.

The final-Z detector nodes make the GNN input identical to MWPM's. This
is NOT label leakage for the official metric: the final-Z detectors are
Z-stabilizer parities of the final readout, and the logical Z operator
(data [69,87,105]) is not a product of Z-stabilizers, so the logical
label is not derivable from them. It IS an input asymmetry vs the CNN,
whose (2C,4,5) tensor carries no final-data-derived syndrome (see
README.md).

Input tensor: the dataset npz format is unchanged. `augment_features`
appends ONE extra channel to the (B, 2C, 4, 5) syndrome tensor carrying
the 8 final-Z detector bits (final-data Z-support parity XOR last-cycle
Z-check value, placed at the Z-ancilla's ANC_COORD cell):
    simulation: final_bits = per-qubit flip labels (npz "labels")
    hardware:   final_bits = measured final data bits (DATA_PHYS order)
The two give the same detector bits as detectors_from_tensor /
detectors_from_dataset (the noiseless reference parity is 0).

Static edges, per (code, cycles):
  * spatial : two detectors of the same cycle whose checks share a data
              qubit
  * temporal: same check, adjacent cycles
  * final-Z : support sharing among final-Z nodes + an edge to the same
              check's last-cycle Z detector
"""
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

NODE_FEATS = 6          # static per-node features (see below)
IN_FEATS = NODE_FEATS + 1   # + the detector value itself


def code_adapter(code):
    """Per-code detector-graph ingredients, one shape for every code:
      check_at     ordered check names (bit order within a cycle)
      z_names      set of Z-check names
      z_pos        positions of Z-checks in check_at (final-Z order)
      support_idx  name -> data indices (into the label vector)
      coord        name -> (row, col) on the tensor grid
      grid         (H, W) of the tensor planes
      num_detectors(num_cycles)"""
    if code == "heavyhex":
        from dataset_generation.heavyhex33_stim import (
            CHECK_AT, Z_POS, ANC_COORD, DIDX, GRID_SHAPE, num_detectors)
        from circuits.heavyhex.heavyhex_37q import CHECK_DEFS, Z_STABS
        return dict(
            check_at=list(CHECK_AT), z_names=set(Z_STABS),
            z_pos=list(Z_POS),
            support_idx={n: [DIDX[q] for q in CHECK_DEFS[n][1]]
                         for n in CHECK_AT},
            coord={n: ANC_COORD[CHECK_DEFS[n][2]] for n in CHECK_AT},
            grid=GRID_SHAPE, num_detectors=num_detectors)
    if code == "surface":
        from dataset_generation.rotatedSurface3_stim import num_detectors
        from circuits.rotatedSurface.rotatedSurface3 import (
            CYCLE_ORDER, Z_NAMES, Z_POS, CHECK_DEFS, ANC_GRID, DIDX,
            GRID_SHAPE)
        return dict(
            check_at=list(CYCLE_ORDER), z_names=set(Z_NAMES),
            z_pos=list(Z_POS),
            support_idx={n: [DIDX[c] for c in CHECK_DEFS[n][1]]
                         for n in CYCLE_ORDER},
            coord=dict(ANC_GRID), grid=GRID_SHAPE,
            num_detectors=num_detectors)
    raise ValueError(f"unknown code '{code}'")


class GraphBuilder:
    """Static detector graph + tensor -> node-value extraction for one
    (code, num_cycles).

    Exposes (all numpy, consumed as torch buffers by the model):
      nodes           list of (kind, name, cyc) in detector order
                      (kind: "Z" | "X" | "final")
      static_feats    (N, 6) float32 —
                      [is_Z, is_X, is_final, cyc/C (final=1.0),
                       row/(H-1), col/(W-1)]
      adjacency       (N, N) float32 0/1, symmetric, no self-loops
      norm_adjacency  (N, N) float32 — D^-1/2 (A+I) D^-1/2
      src1, src2      (N,) int64 flat indices into the AUGMENTED tensor
                      (2C+1 channels, flattened C*H*W); the node's
                      detector bit is x[src1] ^ x[src2] (src2 == -1
                      means no second term)
    """

    def __init__(self, code="heavyhex", num_cycles=3):
        self.code = code
        self.num_cycles = int(num_cycles)
        C = self.num_cycles
        ad = code_adapter(code)
        self.adapter = ad
        check_at, z_names, z_pos = ad["check_at"], ad["z_names"], ad["z_pos"]
        coord, support_idx = ad["coord"], ad["support_idx"]
        grid_h, grid_w = ad["grid"]
        self.grid = (grid_h, grid_w)

        # ---- nodes, in _append_detectors order --------------------------
        nodes = []
        for cyc in range(C):
            for name in check_at:
                if name in z_names:
                    nodes.append(("Z", name, cyc))
                elif cyc >= 1:
                    nodes.append(("X", name, cyc))
        for j in z_pos:
            nodes.append(("final", check_at[j], C))
        assert len(nodes) == ad["num_detectors"](C)
        self.nodes = nodes
        self.num_nodes = len(nodes)
        idx = {(k, n, c): i for i, (k, n, c) in enumerate(nodes)}

        # ---- static node features ---------------------------------------
        feats = np.zeros((self.num_nodes, NODE_FEATS), dtype=np.float32)
        for i, (kind, name, cyc) in enumerate(nodes):
            r, col = coord[name]
            feats[i] = [kind == "Z", kind == "X", kind == "final",
                        cyc / C, r / (grid_h - 1), col / (grid_w - 1)]
        self.static_feats = feats

        # ---- detector-bit sources in the augmented (2C+1)-channel tensor
        flat = lambda ch, r, c: (ch * grid_h + r) * grid_w + c  # noqa: E731
        src1 = np.zeros(self.num_nodes, dtype=np.int64)
        src2 = np.full(self.num_nodes, -1, dtype=np.int64)
        for i, (kind, name, cyc) in enumerate(nodes):
            r, col = coord[name]
            if kind == "Z":
                src1[i] = flat(2 * cyc, r, col)
                if cyc >= 1:
                    src2[i] = flat(2 * (cyc - 1), r, col)
            elif kind == "X":
                src1[i] = flat(2 * cyc + 1, r, col)
            else:                                   # final-Z channel
                src1[i] = flat(2 * C, r, col)
        self.src1, self.src2 = src1, src2

        # ---- static edges ------------------------------------------------
        A = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)

        def link(a, b):
            if a != b:
                A[a, b] = A[b, a] = 1.0

        names = list(check_at)
        share = {(a, b): bool(set(support_idx[a]) & set(support_idx[b]))
                 for a in names for b in names}
        # spatial: same cycle, supports share a data qubit
        for cyc in range(C):
            live = [(k, n, c) for (k, n, c) in nodes
                    if c == cyc and k != "final"]
            for ia, (ka, na, ca) in enumerate(live):
                for kb, nb, cb in live[ia + 1:]:
                    if share[(na, nb)]:
                        link(idx[(ka, na, ca)], idx[(kb, nb, cb)])
        # temporal: same check, adjacent cycles
        for kind, name, cyc in nodes:
            if kind in ("Z", "X") and (kind, name, cyc + 1) in idx:
                link(idx[(kind, name, cyc)], idx[(kind, name, cyc + 1)])
        # final-Z: support sharing among final nodes + own last-cycle Z
        finals = [(k, n, c) for (k, n, c) in nodes if k == "final"]
        for ia, (ka, na, ca) in enumerate(finals):
            for kb, nb, cb in finals[ia + 1:]:
                if share[(na, nb)]:
                    link(idx[(ka, na, ca)], idx[(kb, nb, cb)])
        for ka, na, ca in finals:
            link(idx[(ka, na, ca)], idx[("Z", na, C - 1)])
        self.adjacency = A

        # symmetric normalization with self-loops: D^-1/2 (A+I) D^-1/2
        A_hat = A + np.eye(self.num_nodes, dtype=np.float32)
        d = A_hat.sum(1)
        d_inv_sqrt = 1.0 / np.sqrt(d)
        self.norm_adjacency = (A_hat * d_inv_sqrt[None, :]
                               * d_inv_sqrt[:, None]).astype(np.float32)

    def node_values(self, x):
        """Augmented tensor (B, 2C+1, H, W) uint8 -> detector bits (B, N).

        Torch, stays on x's device. Raises if x lacks the final-Z channel
        (build it with augment_features first)."""
        C = self.num_cycles
        if x.shape[1] != 2 * C + 1:
            raise ValueError(
                f"expected an augmented ({2 * C + 1}-channel) tensor, got "
                f"{tuple(x.shape)} — run augment_features(features, "
                f"final_bits) (train.py/run_hw.py do this automatically "
                f"for --model gnn)")
        xf = x.reshape(x.shape[0], -1)
        src1 = torch.as_tensor(self.src1, device=x.device)
        src2 = torch.as_tensor(self.src2, device=x.device)
        v = xf[:, src1]
        has2 = src2 >= 0
        v2 = xf[:, src2.clamp(min=0)] * has2.to(xf.dtype)
        return torch.bitwise_xor(v, v2)


def augment_features(features, final_bits, code="heavyhex"):
    """(B, 2C, H, W) syndrome tensor + final data bits -> (B, 2C+1, H, W).

    The extra channel holds the final-Z detector bits (support parity of
    final_bits XOR last-cycle Z-check plane) at the Z-ancilla cells.
    final_bits: simulation -> npz "labels" (measurement flips); hardware
    -> measured final data bits. Accepts torch tensors (any device) or
    numpy arrays; returns the same kind."""
    ad = code_adapter(code)
    grid_h, grid_w = ad["grid"]
    was_numpy = isinstance(features, np.ndarray)
    feats = torch.as_tensor(features)
    bits = torch.as_tensor(final_bits, device=feats.device)
    assert feats.shape[1] % 2 == 0, "already augmented?"
    assert feats.shape[2:] == (grid_h, grid_w), \
        f"tensor grid {tuple(feats.shape[2:])} != {code} grid"
    C = feats.shape[1] // 2
    B = feats.shape[0]
    extra = torch.zeros((B, 1, grid_h, grid_w), dtype=feats.dtype,
                        device=feats.device)
    last_z = feats[:, 2 * (C - 1)]
    for j in ad["z_pos"]:
        name = ad["check_at"][j]
        r, col = ad["coord"][name]
        par = torch.zeros(B, dtype=feats.dtype, device=feats.device)
        for qi in ad["support_idx"][name]:
            par = torch.bitwise_xor(par, bits[:, qi])
        extra[:, 0, r, col] = torch.bitwise_xor(par, last_z[:, r, col])
    out = torch.cat([feats, extra], dim=1)
    return out.numpy() if was_numpy else out


if __name__ == "__main__":
    for code in ("heavyhex", "surface"):
        gb = GraphBuilder(code, 3)
        deg = gb.adjacency.sum(1)
        kinds = [k for k, _, _ in gb.nodes]
        print(f"{code}: nodes={gb.num_nodes} "
              f"edges={int(gb.adjacency.sum() // 2)} "
              f"deg min/mean/max = {deg.min():.0f}/{deg.mean():.1f}/"
              f"{deg.max():.0f} | Z={kinds.count('Z')} "
              f"X={kinds.count('X')} final={kinds.count('final')}")
