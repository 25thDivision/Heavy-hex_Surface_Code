#!/usr/bin/env python3
"""
(3,3) Heavy-Hex CNN dataset generation
======================================
Inherits the (error_type x error_rate) grid and per-config shot counts of
KCS stim_simulation:
  * error_types / error_rates / noise_profiles / active_noise:
      KCS/stim_simulation/config.json  (quoted in heavyhex33_stim.py)
  * shots (d=3 entries): TRAIN 10,000,000 / TEST 100,000
      KCS/stim_simulation/simulation/generate_dataset_image.py
      (TRAIN_SAMPLES={3: 10000000}, TEST_SAMPLES={3: 100000})
  * train/val split: same as KCS — not a ratio split; train and test files
    are generated separately with independent seeds, and training uses the
    test file as the validation set (same usage as KCS
    run_stim_simulation.py).

Sample format:
  features:        (N, 2*num_cycles, 4, 5) uint8 — 2D diamond-embedded tensor
                   (channels [Z-plane, X-plane] x cycle,
                    heavyhex33_stim.syndrome_tensor)
  labels:          (N, 17) uint8 — per-qubit X-error label at final readout:
                   the final-data MEASUREMENT FLIPS (error frame vs the
                   noiseless reference, via stim.FlipSimulator).
                   NOTE: the raw final bits themselves are individually
                   random because the X-checks project |0>_L into X-eigen-
                   states; only parities are deterministic. See
                   heavyhex33_stim.sample_flips for why flips are the
                   correct per-qubit label (used for ECR).
  logical_labels:  (N,)  uint8 — logical Z flip (LER label). Equals the
                   parity of `labels` over LOGICAL_Z, which is identical to
                   the parity of the actually measured data [69,87,105]
                   (the reference parity is deterministically 0).

Usage:
  python dataset_generation/make_dataset.py --smoke     # quick smoke (10k/2k)
  python dataset_generation/make_dataset.py             # full KCS grid (default shots)
  python dataset_generation/make_dataset.py -n realistic/dp0.001_mf0.01_rf0.01_gd0.008 -p 0.01
  python dataset_generation/make_dataset.py --config train_sweep.json
      # every (noise, p, type, cycles) combo the sweep config needs
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from dataset_generation import load_options, load_sweep  # noqa: E402
from dataset_generation.heavyhex33_stim import (  # noqa: E402
    build_stim_circuit, sample_flips, syndrome_tensor, logical_label,
    noise_tag, DISTANCE, ERROR_TYPES, ERROR_RATES, ACTIVE_NOISE,
    NOISE_PROFILES)

# Inherited from KCS generate_dataset_image.py (d=3 entries)
TRAIN_SAMPLES = 10_000_000
TEST_SAMPLES = 100_000
CHUNK = 1_000_000


def parse_args():
    ap = argparse.ArgumentParser(description="(3,3) heavy-hex dataset generator")
    ap.add_argument("-n", "--noise", nargs="+", default=None,
                    help=f"noise profile names (default: {ACTIVE_NOISE})")
    ap.add_argument("-p", "--rates", nargs="+", type=float, default=None,
                    help=f"Error_Rate list (default: {ERROR_RATES})")
    ap.add_argument("-e", "--error-types", nargs="+", default=None,
                    help=f"Error_Type list (default: {ERROR_TYPES})")
    ap.add_argument("--cycles", type=int, default=3,
                    help="number of QEC cycles (default 3, same as HW run)")
    ap.add_argument("--train-samples", type=int, default=TRAIN_SAMPLES)
    ap.add_argument("--test-samples", type=int, default=TEST_SAMPLES)
    ap.add_argument("--outdir", default=str(_ROOT / "dataset"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true",
                    help="smoke test: train 10k / test 2k")
    ap.add_argument("--config", default=None,
                    help="sweep-config JSON; generates every (noise, p, "
                         "type, cycles) combo the sweep needs. Defaults "
                         "to train_sweep.json at the repo root when no "
                         "selection args (-n/-p/-e/--smoke) are given; "
                         "'none' disables")
    # train_options.json (repo root, if present) replaces the hardcoded
    # defaults (e.g. train/test sample counts); explicit CLI args still win
    opts = load_options("dataset")
    if opts:
        ap.set_defaults(**opts)
        print(f"train_options.json: {opts}")
    args = ap.parse_args()
    # auto-sweep: no explicit config and no explicit selection -> pick up
    # the default sweep file if it exists
    if args.config == "none":
        args.config = None
    elif args.config is None and not (args.noise or args.rates
                                      or args.error_types or args.smoke):
        default_sweep = _ROOT / "train_sweep.json"
        if default_sweep.exists():
            args.config = str(default_sweep)
    return args


def sweep_combos(args):
    """Resolve the (noise, p, error_type, cycles) combos to generate."""
    if args.config:
        combos = []
        for run in load_sweep(args.config):
            for n in run.get("noise") or ACTIVE_NOISE:
                for p in run.get("rates") or ERROR_RATES:
                    for et in run.get("error_types") or ERROR_TYPES:
                        c = (n, p, et, run.get("cycles", args.cycles))
                        if c not in combos:
                            combos.append(c)
        return combos
    noises = args.noise if args.noise else ACTIVE_NOISE
    rates = args.rates if args.rates else ERROR_RATES
    etypes = args.error_types if args.error_types else ERROR_TYPES
    return [(n, p, et, args.cycles)
            for n in noises for p in rates for et in etypes]


def generate_split(circuit, num_cycles, total, seed, desc):
    """Sample `total` shots in CHUNK batches; return (features, labels, logical).

    Uses FlipSimulator flips: identical to measured values for every
    downstream quantity (Z-planes, X-plane XORs, detectors, logical
    parity), while additionally providing well-defined per-qubit labels."""
    feats = np.zeros((total, 2 * num_cycles, 4, 5), dtype=np.uint8)
    labels = np.zeros((total, 17), dtype=np.uint8)
    done, chunk_i = 0, 0
    t0 = time.time()
    while done < total:
        n = min(CHUNK, total - done)
        syn, dat = sample_flips(circuit, n, num_cycles,
                                seed=seed * 100003 + chunk_i)
        feats[done:done + n] = syndrome_tensor(syn, num_cycles)
        labels[done:done + n] = dat
        done += n
        chunk_i += 1
        print(f"      {desc}: {done:,}/{total:,} ({time.time() - t0:.1f}s)",
              flush=True)
    return feats, labels, logical_label(labels)


def main():
    args = parse_args()
    combos = sweep_combos(args)
    n_train = 10_000 if args.smoke else args.train_samples
    n_test = 2_000 if args.smoke else args.test_samples
    outdir = Path(args.outdir)

    print("=== (3,3) dataset generation ===")
    if args.config:
        print(f"sweep: {args.config}")
    print(f"{len(combos)} combo(s) (noise, p, type, cycles):")
    for c in combos:
        print(f"   {c}")
    print(f"train={n_train:,} test={n_test:,} -> {outdir}")

    for noise, p, et, cycles in combos:
        if noise not in NOISE_PROFILES:
            print(f"WARNING: unknown noise profile '{noise}', skipping")
            continue
        # folder name = parameter tag only (no 'realistic/' level),
        # e.g. dataset/dp0.001_mf0.01_rf0.01_gd0.008/
        ndir = outdir / noise_tag(noise)
        ndir.mkdir(parents=True, exist_ok=True)
        circuit = build_stim_circuit(cycles, et, p, noise)
        for split, n, seed_off in (("train", n_train, 0),
                                   ("test", n_test, 1)):
            fname = ndir / (f"{split}_d{DISTANCE}_c{cycles}"
                            f"_p{p}_{et}.npz")
            if fname.exists():
                print(f"   skip (exists): {fname}")
                continue
            print(f"   >>> {noise} p={p} {et} c={cycles} [{split}]")
            # independent train/test seeds (KCS generates the two
            # files separately, i.e. independent samples)
            seed = args.seed * 1000 + hash((noise, p, et)) % 10007 + seed_off
            f, l, y = generate_split(circuit, cycles, n,
                                     seed & 0x7FFFFFFF, split)
            # atomic write: dump to a temp file, then rename. A
            # crashed/concurrent run can never leave a half-written
            # npz under the final name (the exists-skip above would
            # otherwise trust it and training would hit BadZipFile).
            tmp = fname.with_name(f".{fname.name}.{os.getpid()}.tmp")
            try:
                with open(tmp, "wb") as fh:
                    np.savez_compressed(
                        fh, features=f, labels=l, logical_labels=y,
                        num_cycles=cycles, noise_profile=noise,
                        error_rate=p, error_type=et)
                tmp.replace(fname)
            finally:
                tmp.unlink(missing_ok=True)
            ler0 = float(y.mean())
            print(f"      saved {fname.name}: features{f.shape}, "
                  f"raw logical-flip rate={ler0:.4f}")
    print("=== done ===")


if __name__ == "__main__":
    main()
