#!/usr/bin/env python3
"""
MWPM baseline decoder (PyMatching, Stim DEM based)
==================================================
Builds a matching decoder from the abstract Stim circuit's detector error
model:
    stim.Circuit.detector_error_model(decompose_errors=True)
      -> pymatching.Matching.from_detector_error_model(...)

Works on both data sources:
  * Stim datasets: detectors are reconstructed from (features, labels) via
    heavyhex33_stim.detectors_from_tensor
  * hardware runs: detectors come from check_values() output + final data
    bits via heavyhex33_stim.detectors_from_dataset

Both reconstructions follow the exact detector order of
heavyhex33_stim._append_detectors, so the same Matching object decodes
either stream. The predicted observable flip is compared against the true
logical flip to produce the LER.

Usage (evaluate one dataset file and print a table):
  python baseline/mwpm.py -n realistic/dp0.001_mf0.01_rf0.01_gd0.008 -p 0.005
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pymatching

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from dataset_generation.heavyhex33_stim import (  # noqa: E402
    build_stim_circuit, detectors_from_tensor, detectors_from_dataset,
    noise_tag, is_qpu_profile, NOISE_PROFILES,
    DISTANCE, ERROR_RATES, ERROR_TYPES, ALL_NOISE)
from evaluation.metrics import ler  # noqa: E402


def build_matching(num_cycles, error_type, p, noise_profile,
                   code="heavyhex"):
    """Matching decoder from the DEM of the noisy reference circuit.

    Plain 4-parameter profiles use the code's abstract circuit (25q
    heavy-hex / 17q rsc3); QPU calibration profiles (mode: qpu_avg_v1)
    use the code's hardware-shaped circuit — detectors follow the same
    order either way, so the same detector reconstructions feed any of
    the Matching objects."""
    if code == "surface":
        if is_qpu_profile(noise_profile):
            from dataset_generation.rsc3_qpu_stim import (
                build_rsc3_qpu_stim_circuit)
            prof = (NOISE_PROFILES[noise_profile]
                    if isinstance(noise_profile, str) else noise_profile)
            circuit = build_rsc3_qpu_stim_circuit(num_cycles, error_type,
                                                  p, prof)
        else:
            from dataset_generation.rsc3_stim import (
                build_rsc3_stim_circuit)
            circuit = build_rsc3_stim_circuit(num_cycles, error_type, p,
                                              noise_profile)
    elif is_qpu_profile(noise_profile):
        from dataset_generation.heavyhex37_qpu_stim import (
            build_qpu_stim_circuit)
        prof = (NOISE_PROFILES[noise_profile]
                if isinstance(noise_profile, str) else noise_profile)
        circuit = build_qpu_stim_circuit(num_cycles, error_type, p, prof)
    else:
        circuit = build_stim_circuit(num_cycles, error_type, p,
                                     noise_profile)
    dem = circuit.detector_error_model(decompose_errors=True)
    return pymatching.Matching.from_detector_error_model(dem)


def decode_ler(matching, detectors, true_logical):
    """Batch-decode detector vectors -> (LER, predicted flips)."""
    pred = np.asarray(matching.decode_batch(detectors), dtype=np.uint8)
    pred = pred.reshape(pred.shape[0], -1)[:, 0]
    return ler(pred, true_logical), pred


def mwpm_ler_from_dataset(npz_file, matching=None):
    """Evaluate MWPM LER on a saved dataset npz (any code; the npz's
    "code" field selects the detector reconstruction, heavyhex when
    absent — pre-code-axis files)."""
    d = np.load(npz_file)
    code = str(d["code"]) if "code" in d.files else "heavyhex"
    if matching is None:
        matching = build_matching(int(d["num_cycles"]), str(d["error_type"]),
                                  float(d["error_rate"]),
                                  str(d["noise_profile"]), code)
    if code == "surface":
        from dataset_generation.rsc3_stim import detectors_from_tensor_rsc3
        det = detectors_from_tensor_rsc3(d["features"], d["labels"])
    else:
        det = detectors_from_tensor(d["features"], d["labels"])
    return decode_ler(matching, det, d["logical_labels"])[0]


def mwpm_ler_from_hardware(check_mat, dat, num_cycles, matching,
                           code="heavyhex"):
    """Evaluate MWPM LER on hardware data.

    check_mat: (shots, N_CHECKS*C) check-value matrix (check_values()
               through the code's check_matrix layout)
    dat:       (shots, num_data) final data bits
    """
    if code == "surface":
        from dataset_generation.rsc3_stim import (
            detectors_from_dataset_rsc3, logical_label_rsc3)
        det = detectors_from_dataset_rsc3(check_mat, dat, num_cycles)
        return decode_ler(matching, det, logical_label_rsc3(dat))[0]
    from dataset_generation.heavyhex33_stim import logical_label
    det = detectors_from_dataset(check_mat, dat, num_cycles)
    return decode_ler(matching, det, logical_label(dat))[0]


def main():
    ap = argparse.ArgumentParser(description="MWPM baseline evaluation")
    ap.add_argument("-n", "--noise", nargs="+", default=ALL_NOISE)
    ap.add_argument("-p", "--rates", nargs="+", type=float, default=ERROR_RATES)
    ap.add_argument("-e", "--error-types", nargs="+", default=ERROR_TYPES)
    ap.add_argument("--code", choices=["heavyhex", "surface"],
                    default="heavyhex",
                    help="code family (dataset/<code>/ subtree)")
    ap.add_argument("--cycles", type=int, default=3)
    ap.add_argument("--split", default="test")
    ap.add_argument("--data-dir", default=str(_ROOT / "dataset"))
    args = ap.parse_args()

    print(f"{'noise':<42} {'p':>6} {'type':>4} {'raw_LER':>8} {'MWPM_LER':>9}")
    for noise in args.noise:
        for p in args.rates:
            for et in args.error_types:
                f = (Path(args.data_dir) / args.code / noise_tag(noise) /
                     f"{args.split}_d{DISTANCE}_c{args.cycles}_p{p}_{et}.npz")
                if not f.exists():
                    print(f"{noise:<42} {p:>6} {et:>4}   (missing: {f.name})")
                    continue
                d = np.load(f)
                raw = float(np.asarray(d["logical_labels"]).mean())
                mler = mwpm_ler_from_dataset(f)
                print(f"{noise:<42} {p:>6} {et:>4} {raw:>8.4f} {mler:>9.4f}")


if __name__ == "__main__":
    main()
