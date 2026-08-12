#!/usr/bin/env python3
"""
Evaluation metrics: LER (official) / ECR, parity-LER (diagnostics)
==================================================================
* LER (Logical Error Rate) — **the official metric**, shared by
  simulation and hardware. This is the single number the project is
  judged on (best val LER selects the checkpoint; test head-LER is the
  final score). The fraction of logical Z flips remaining after applying
  the decoder's logical prediction:
      LER = mean( pred_logical_flip != true_logical_flip )
  true_logical_flip is the LOGICAL_Z parity of the final data measurement,
  so it is computable from hardware raw data as well.
  "head-LER" = this LER computed from the logical head's logits.

* ECR (Error Correction Rate) — **diagnostic, simulation only**.
  Requires per-qubit ground truth (the 17-bit final-data label), so it can
  only be computed on Stim datasets (hardware has no per-qubit ground
  truth):
      preds = (outputs > 0)            # logit sign = 0.5 threshold
      error_mask = (labels == 1)       # bits that actually hold an error
      ECR = (preds[error_mask] == 1).sum() / error_mask.sum()
  i.e. the conditional accuracy (detection rate) restricted to
  error-holding qubits.

* parity-LER — **diagnostic**, derived from the per-qubit head. The
  logical flip implied by the per-qubit predictions (parity of the
  predicted mask over the LOGICAL_Z data qubits) compared against the
  true logical flip. See parity_ler_from_qubit_logits below.
"""
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from heavyhex_circuits.heavyhex_37q import DATA_PHYS, LOGICAL_Z  # noqa: E402

# LOGICAL_Z physical labels -> indices into the 17-bit DATA_PHYS-ordered
# per-qubit vector (derived, never hardcoded)
LOGICAL_Z_DATA_IDX = [DATA_PHYS.index(p) for p in LOGICAL_Z]


def ecr(logits, labels):
    """ECR (diagnostic, sim-only). logits: (N,17) real logits,
    labels: (N,17) 0/1.

    Simulation only (needs per-qubit ground truth). Reported for
    diagnosis only — the official metric is the head-LER."""
    logits = np.asarray(logits)
    labels = np.asarray(labels)
    preds = (logits > 0).astype(np.uint8)
    error_mask = labels == 1
    total = error_mask.sum()
    if total == 0:
        return 0.0
    return float((preds[error_mask] == 1).sum() / total)


def bit_accuracy(logits, labels):
    """Auxiliary metric: overall bit accuracy over the per-qubit head."""
    preds = (np.asarray(logits) > 0)
    labels = np.asarray(labels).astype(bool)
    return float((preds == labels).mean())


def ler(pred_logical, true_logical):
    """LER after applying the logical prediction. Simulation & hardware.

    pred_logical: (N,) 0/1 — decoder-predicted logical flip
                  (binarize logits with >0 before passing)
    true_logical: (N,) 0/1 — actual logical flip (LOGICAL_Z parity)"""
    p = np.asarray(pred_logical).astype(np.uint8).ravel()
    t = np.asarray(true_logical).astype(np.uint8).ravel()
    assert p.shape == t.shape
    return float((p != t).mean())


def ler_from_logits(logical_logits, true_logical):
    """Convenience: logical-head logits -> head-LER (the official metric)."""
    return ler(np.asarray(logical_logits).ravel() > 0, true_logical)


def parity_ler_from_qubit_logits(qubit_logits, true_logical):
    """parity_LER (diagnostic): LER implied by the per-qubit head.

    pred_qubit_mask = (qubit_logits > 0); the predicted logical flip is
    parity(pred_qubit_mask[LOGICAL_Z data indices]), compared against the
    true logical flip:
        parity_LER = mean( parity(pred_qubit_mask[LOGICAL_Z]) != y_logical )

    The head-LER (logical head) is the official metric; this one is a
    diagnostic of *how* the model achieves it — whether it actually
    learned per-qubit correction or shortcut-learned the logical
    classification. If head-LER looks good but parity-LER is much worse,
    the per-qubit head is not really decoding: consider adjusting the aux
    loss weight (--aux-weight).

    qubit_logits: (N,17) real logits from the per-qubit head
    true_logical: (N,) 0/1 — actual logical flip (LOGICAL_Z parity)"""
    preds = (np.asarray(qubit_logits) > 0).astype(np.uint8)
    parity = np.bitwise_xor.reduce(preds[:, LOGICAL_Z_DATA_IDX], axis=1)
    return ler(parity, true_logical)
