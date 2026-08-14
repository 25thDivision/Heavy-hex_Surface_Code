#!/usr/bin/env python3
"""
Dual-head GNN decoder for the (3,3) heavy-hex surface code.

This is the file you complete for --model gnn. Fill in every `TODO`
block below — the rest of the pipeline (dataset generation, graph
construction, training loop, evaluation, MWPM baseline, hardware run) is
already done and will run as soon as you finish this file.

Graph : nodes = Stim detectors (order of heavyhex33_stim._append_detectors;
        48 nodes for cycles=3 incl. the 8 final-Z detectors — same input
        as MWPM), static edges from model/graph.GraphBuilder (spatial
        support-sharing / temporal / final-Z links). The graph plumbing
        (adjacency, static node features, detector-bit extraction) is
        provided — you only design the network on top of it.
Input : (B, 2*num_cycles + 1, 4, 5) uint8 AUGMENTED syndrome tensor —
        the dataset tensor plus one channel with the 8 final-Z detector
        bits (model/graph.augment_features; train.py / hardware/run_hw.py
        apply it automatically for --model gnn via prepare_features).
Output: qubit_logits   (B, 17) — per-qubit X-error head
                                 (diagnostics: ECR, parity_LER)
        logical_logits (B, 1)  — logical Z flip head
                                 (official metric: head-LER)

Keep the interfaces exactly as they are — the training/eval/hardware
scripts call them as-is:
  * HeavyHexGNN(in_channels, num_qubits, ...) with
    forward(x) -> (qubit_logits, logical_logits)
    (in_channels = 2*num_cycles, same convention as the CNN; the model
     derives num_cycles = in_channels // 2)
  * compute_loss(...) -> (total_loss, loss_logical, loss_qubit)
    The loss is LER-first: BCE on the logical head is the MAIN loss and
    BCE on the per-qubit head is an AUXILIARY loss scaled by aux_weight
    (default 0.5).
  * prepare_features(...) — provided, do not change.

Constraint: pure torch only — no torch_geometric or other graph
libraries. The graph is small (48 nodes), so message passing works well
as a dense matmul with the normalized adjacency buffer `self.adj`:
neighbor aggregation of node states h (B, N, H) is simply `self.adj @ h`.

Hints:
  * Encode each node's 7 features (detector bit + 6 static) with a
    Linear -> hidden_dim, then run 3-4 message-passing rounds, e.g.
        h' = h + ReLU(LayerNorm(W_self h + adj @ (W_neigh h)))
    (residual + LayerNorm keep deeper stacks stable).
  * Pool over nodes (mean and/or max), then a shared FC -> two linear
    heads, exactly like the CNN.
  * Use torch.nn.functional.binary_cross_entropy_with_logits (the heads
    output logits, not probabilities).
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model.graph import GraphBuilder, augment_features, IN_FEATS  # noqa: E402

NUM_QUBITS = 17


def prepare_features(features, final_bits, code="heavyhex"):
    """Dataset tensor (B,2C,4,5) + final bits (B,17) -> augmented input.

    train.py passes the per-qubit flip labels; hardware/run_hw.py passes
    the measured final data bits. Same detector bits either way (see
    model/graph.py). Provided — do not change."""
    return augment_features(features, final_bits, code)


class HeavyHexGNN(nn.Module):
    """in_channels = 2*num_cycles (same convention as the CNN)."""

    def __init__(self, in_channels=6, num_qubits=NUM_QUBITS,
                 hidden_dim=128, num_layers=4, fc_dim=256, dropout=0.1,
                 code="heavyhex"):
        super().__init__()
        # provided: static graph buffers (do not change) ------------------
        num_cycles = in_channels // 2
        gb = GraphBuilder(code, num_cycles)
        self.graph = gb
        self.register_buffer("adj",
                             torch.from_numpy(gb.norm_adjacency))
        self.register_buffer("static_feats",
                             torch.from_numpy(gb.static_feats))
        # ------------------------------------------------------------------
        # TODO 1/3 — network construction
        # * self.encoder: Linear(IN_FEATS, hidden_dim) for the per-node
        #   input [detector bit, 6 static features]
        # * message-passing layers (num_layers rounds, pure torch — use
        #   `self.adj @ h` for neighbor aggregation; see the module
        #   docstring for a layer recipe)
        # * self.shared: pooled node states -> Linear -> ReLU ->
        #   Dropout(dropout) (mirror the CNN's shared FC block)
        # ------------------------------------------------------------------
        raise NotImplementedError("TODO: define the GNN layers")

        # ------------------------------------------------------------------
        # TODO 2/3 — the two heads
        # self.head_qubit  : Linear(fc_dim, num_qubits)  (17 outputs, ECR)
        # self.head_logical: Linear(fc_dim, 1)           (1 output, LER)
        # ------------------------------------------------------------------

    def forward(self, x):
        # provided: detector bits + static features per node --------------
        det = self.graph.node_values(x).float()          # (B, N)
        stat = self.static_feats.unsqueeze(0).expand(x.shape[0], -1, -1)
        h = torch.cat([det.unsqueeze(-1), stat], dim=-1)  # (B, N, 7)
        # ------------------------------------------------------------------
        # TODO 1/3 (cont.) — run h through the encoder and the
        # message-passing rounds, pool over the node dimension, apply
        # self.shared, then return the two head outputs as a tuple:
        # (qubit_logits, logical_logits)
        # ------------------------------------------------------------------
        raise NotImplementedError("TODO: implement forward")


def compute_loss(qubit_logits, logical_logits, y_qubit, y_logical,
                 aux_weight=0.5, qubit_pos_weight=None):
    """Return (total_loss, loss_logical, loss_qubit).

    total = BCE(logical_logits, y_logical) + aux_weight * BCE(qubit_logits,
    y_qubit). qubit_pos_weight (optional, shape (17,)) goes into the
    per-qubit BCE's pos_weight to counter class imbalance
    (e.g. pos_weight=(1-p)/p).
    """
    # ----------------------------------------------------------------------
    # TODO 3/3 — loss computation
    # * logical head: F.binary_cross_entropy_with_logits on the squeezed
    #   (B,) logits vs y_logical.float()  -> MAIN loss
    # * per-qubit head: F.binary_cross_entropy_with_logits on (B,17) vs
    #   y_qubit.float(), pos_weight=qubit_pos_weight  -> AUXILIARY loss
    # * total = main + aux_weight * auxiliary
    # ----------------------------------------------------------------------
    raise NotImplementedError("TODO: implement the two-head loss")
