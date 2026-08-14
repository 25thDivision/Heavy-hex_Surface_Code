#!/usr/bin/env python3
"""
Data loading utilities — you don't need to modify this file
===========================================================
Loads the npz files written by make_dataset.py and builds GPU-resident
tensor loaders. FastTensorDataLoader is much faster than DataLoader
workers for small fixed-size tensors.
"""
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from dataset_generation.heavyhex33_stim import noise_tag, DISTANCE  # noqa: E402


def npz_path(data_dir, code, noise, split, cycles, p, error_type):
    """dataset/<code>/<noise_tag>/<split>_d..._c..._p..._<type>.npz

    The <code> level ({heavyhex, surface}) was introduced with the --code
    axis; pre-existing files saved directly under dataset/<noise_tag>/
    belong to heavyhex and can be moved in with a one-line mv (see
    README.md)."""
    return (Path(data_dir) / code / noise_tag(noise)
            / f"{split}_d{DISTANCE}_c{cycles}_p{p}_{error_type}.npz")


def load_split(data_dir, code, noise, split, cycles, p, error_type,
               device="cpu"):
    """npz -> (features, y_qubit, y_logical) torch tensors (uint8, on device)."""
    path = npz_path(data_dir, code, noise, split, cycles, p, error_type)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run dataset_generation/make_dataset.py first.")
    d = np.load(path)
    feats = torch.from_numpy(np.ascontiguousarray(d["features"])).to(device)
    y_q = torch.from_numpy(np.ascontiguousarray(d["labels"])).to(device)
    y_l = torch.from_numpy(np.ascontiguousarray(d["logical_labels"])).to(device)
    return feats, y_q, y_l


class FastTensorDataLoader:
    """Minimal loader for GPU-resident tensors."""

    def __init__(self, *tensors, batch_size=2048, shuffle=False):
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = list(tensors)
        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            idx = torch.randperm(self.dataset_len, device=self.tensors[0].device)
            self.tensors = [t[idx] for t in self.tensors]
        self._i = 0
        return self

    def __next__(self):
        if self._i >= self.dataset_len:
            raise StopIteration
        j = min(self._i + self.batch_size, self.dataset_len)
        batch = [t[self._i:j] for t in self.tensors]
        self._i = j
        return batch

    def __len__(self):
        return (self.dataset_len + self.batch_size - 1) // self.batch_size
