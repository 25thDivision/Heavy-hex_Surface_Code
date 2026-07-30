#!/usr/bin/env python3
"""
Evaluation metrics: ECR / LER
=============================
* ECR (Error Correction Rate) — **simulation only**.
  Requires per-qubit ground truth (the 17-bit final-data label), so it can
  only be computed on Stim datasets (hardware has no per-qubit ground
  truth). The definition is identical to KCS run_stim_simulation.py
  (train_and_evaluate, L263-271):
      preds = (outputs > 0)            # logit sign = 0.5 threshold
      error_mask = (labels == 1)       # bits that actually hold an error
      ECR = (preds[error_mask] == 1).sum() / error_mask.sum()
  i.e. the conditional accuracy (detection rate) restricted to
  error-holding qubits.

* LER (Logical Error Rate) — **shared by simulation and hardware**.
  The fraction of logical Z flips remaining after applying the decoder's
  logical prediction:
      LER = mean( pred_logical_flip != true_logical_flip )
  true_logical_flip is the LOGICAL_Z parity of the final data measurement,
  so it is computable from hardware raw data as well.
"""
import numpy as np


def ecr(logits, labels):
    """KCS-defined ECR. logits: (N,17) real logits, labels: (N,17) 0/1.

    Simulation only (needs per-qubit ground truth)."""
    logits = np.asarray(logits)
    labels = np.asarray(labels)
    preds = (logits > 0).astype(np.uint8)
    error_mask = labels == 1
    total = error_mask.sum()
    if total == 0:
        return 0.0
    return float((preds[error_mask] == 1).sum() / total)


def bit_accuracy(logits, labels):
    """Auxiliary metric: overall bit accuracy (same definition as the
    Accuracy(%) column in KCS)."""
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
    """Convenience: logical-head logits -> LER."""
    return ler(np.asarray(logical_logits).ravel() > 0, true_logical)
