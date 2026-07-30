#!/usr/bin/env python3
"""
Training entry point — you don't need to modify this file
==========================================================
Trains the dual-head CNN (model/cnn_skeleton.py — the file you complete)
on the Stim datasets and logs ECR / LER per epoch. Loop structure, early
stopping, checkpointing and CSV logging follow KCS
run_stim_simulation.py; model selection is by best validation LER
(the loss is LER-first), with patience-based early stopping.

The "validation set" is the independently generated test file, exactly as
in KCS (train/test files, not a ratio split).

Usage:
  python train.py --smoke                    # quick end-to-end check
  python train.py -n realistic/dp0.001_mf0.01_rf0.01_gd0.008 -p 0.005
  python train.py --all                      # full KCS grid
  python train.py --solution ...             # reference model, if you have solutions/
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from dataset_generation.heavyhex33_stim import (  # noqa: E402
    noise_tag, DISTANCE, ERROR_RATES, ERROR_TYPES, ACTIVE_NOISE)
from model.data import load_split, FastTensorDataLoader  # noqa: E402
from evaluation.metrics import ecr, bit_accuracy, ler_from_logits  # noqa: E402

# KCS run_stim_simulation.py / config.json defaults
MAX_EPOCHS = 30
PATIENCE = 5
BATCH_SIZE = 2048
LR = 1e-3


def parse_args():
    ap = argparse.ArgumentParser(description="Train the heavy-hex CNN decoder")
    ap.add_argument("-n", "--noise", nargs="+", default=None)
    ap.add_argument("-p", "--rates", nargs="+", type=float, default=None)
    ap.add_argument("-e", "--error-types", nargs="+", default=None)
    ap.add_argument("--all", action="store_true",
                    help="run the full KCS grid (all active noise x rates)")
    ap.add_argument("--cycles", type=int, default=3)
    ap.add_argument("--data-dir", default=str(_ROOT / "dataset"))
    ap.add_argument("--outdir", default=str(_ROOT / "results"))
    ap.add_argument("--ckpt-dir", default=str(_ROOT / "checkpoint"))
    ap.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    ap.add_argument("--patience", type=int, default=PATIENCE)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--aux-weight", type=float, default=0.5,
                    help="weight of the per-qubit auxiliary BCE loss")
    ap.add_argument("--pos-weight", choices=["none", "linear", "sqrt"],
                    default="none",
                    help="per-qubit BCE pos_weight from p (KCS used linear "
                         "on injected-mask labels; our labels differ, so "
                         "default is none)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--mwpm", action="store_true",
                    help="also evaluate the MWPM baseline on the val set")
    ap.add_argument("--solution", action="store_true",
                    help="use the reference model from solutions/ "
                         "(not part of the distributed repo)")
    ap.add_argument("--smoke", action="store_true",
                    help="single config, 3 epochs, tiny batch count")
    return ap.parse_args()


def ensure_coupling_json(backend="ibm_yonsei"):
    """Make sure a coupling_<backend>.json exists (fetch it if missing).

    Training itself never touches the coupling map — it only matters for
    the circuit tests and the hardware steps — so this is best-effort:
    without keys.json or network access it just prints a note and moves on
    (e.g. on offline slurm GPU nodes)."""
    if list(_ROOT.glob("coupling_*.json")):
        return
    keys_path = _ROOT / "keys.json"
    if not keys_path.exists():
        print("NOTE: no coupling_*.json and no keys.json — skipping coupling "
              "fetch (training doesn't need it; hardware/tests do).")
        return
    try:
        import json
        from heavyhex_circuits.fetch_coupling import fetch
        keys = json.load(open(keys_path))
        fetch(backend, token=keys.get("ibm_token"),
              instance=keys.get("ibm_instance"), outdir=str(_ROOT))
    except Exception as e:
        print(f"WARNING: coupling fetch failed ({e}) — continuing, "
              f"training doesn't need it.")


def get_model_module(use_solution):
    if use_solution:
        from solutions import cnn_solution as mod
    else:
        from model import cnn_skeleton as mod
    return mod


def evaluate(model, loader, mod, aux_weight, pos_weight, device):
    model.eval()
    q_logits, l_logits, y_qs, y_ls = [], [], [], []
    val_loss, nb, t_inf, n_inf = 0.0, 0, 0.0, 0
    with torch.no_grad():
        for xb, yq, yl in loader:
            t0 = time.time()
            ql, ll = model(xb)
            t_inf += time.time() - t0
            n_inf += xb.shape[0]
            loss, _, _ = mod.compute_loss(ql, ll, yq, yl, aux_weight, pos_weight)
            val_loss += loss.item()
            nb += 1
            q_logits.append(ql.float().cpu().numpy())
            l_logits.append(ll.float().cpu().numpy())
            y_qs.append(yq.cpu().numpy())
            y_ls.append(yl.cpu().numpy())
    q_logits = np.concatenate(q_logits)
    l_logits = np.concatenate(l_logits)
    y_q = np.concatenate(y_qs)
    y_l = np.concatenate(y_ls)
    return {
        "val_loss": val_loss / max(nb, 1),
        "ecr": ecr(q_logits, y_q),
        "acc": bit_accuracy(q_logits, y_q),
        "ler": ler_from_logits(l_logits, y_l),
        "raw_ler": float(y_l.mean()),
        "inf_ms": (t_inf / max(n_inf, 1)) * 1000,
    }


def train_one(args, mod, noise, p, et, device):
    print(f"\n{'=' * 70}\n>>> {noise} | p={p} | {et} | cycles={args.cycles}")
    Xtr, yqtr, yltr = load_split(args.data_dir, noise, "train", args.cycles,
                                 p, et, device)
    Xva, yqva, ylva = load_split(args.data_dir, noise, "test", args.cycles,
                                 p, et, device)
    print(f"    train {tuple(Xtr.shape)}  val {tuple(Xva.shape)}  -> {device}")

    train_loader = FastTensorDataLoader(Xtr, yqtr, yltr,
                                        batch_size=args.batch_size, shuffle=True)
    val_loader = FastTensorDataLoader(Xva, yqva, ylva,
                                      batch_size=args.batch_size)

    model = mod.HeavyHexCNN(in_channels=2 * args.cycles).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    pos_weight = None
    if args.pos_weight != "none":
        w = (1.0 - p) / p
        if args.pos_weight == "sqrt":
            w = float(np.sqrt(w))
        pos_weight = torch.full((17,), w, device=device)

    # flat layout: results/CNN_d3_c3_p0.005_dp0.001_mf0.01_rf0.01_gd0.008.csv
    # (d = code distance, c = number of QEC cycles; the error type is
    #  omitted since the KCS grid is X-only)
    tag = f"d{DISTANCE}_c{args.cycles}_p{p}_{noise_tag(noise)}"
    result_dir = Path(args.outdir)
    ckpt_dir = Path(args.ckpt_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    csv_path = result_dir / f"CNN_{tag}.csv"
    ckpt_path = ckpt_dir / f"CNN_{tag}.pt"

    best = {"ler": float("inf"), "ecr": 0.0, "acc": 0.0, "epoch": 0}
    patience = 0
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train_Loss", "Val_Loss", "Val_ECR",
                         "Val_Acc", "Val_LER", "Raw_LER", "Inference_ms"])
        for epoch in range(1, args.epochs + 1):
            model.train()
            train_loss, nb = 0.0, 0
            for xb, yq, yl in train_loader:
                optimizer.zero_grad()
                ql, ll = model(xb)
                loss, _, _ = mod.compute_loss(ql, ll, yq, yl,
                                              args.aux_weight, pos_weight)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                nb += 1
            train_loss /= max(nb, 1)

            m = evaluate(model, val_loader, mod, args.aux_weight,
                         pos_weight, device)
            writer.writerow([epoch, f"{train_loss:.6f}",
                             f"{m['val_loss']:.6f}", f"{m['ecr']:.4f}",
                             f"{m['acc']:.4f}", f"{m['ler']:.4f}",
                             f"{m['raw_ler']:.4f}", f"{m['inf_ms']:.4f}"])
            f.flush()
            print(f"    [Ep {epoch:02d}] loss {train_loss:.4f}/"
                  f"{m['val_loss']:.4f} | ECR {m['ecr']:.2%} | "
                  f"Acc {m['acc']:.2%} | LER {m['ler']:.4f} "
                  f"(raw {m['raw_ler']:.4f}) | pat {patience}")

            if m["ler"] < best["ler"]:
                best = {"ler": m["ler"], "ecr": m["ecr"], "acc": m["acc"],
                        "epoch": epoch}
                patience = 0
                torch.save({"model_state_dict": model.state_dict(),
                            "epoch": epoch, "val_ler": m["ler"],
                            "val_ecr": m["ecr"], "config": vars(args),
                            "noise": noise, "p": p, "error_type": et},
                           ckpt_path)
            else:
                patience += 1
                if patience >= args.patience:
                    print("    -> early stopping")
                    break

    row = {"noise": noise, "p": p, "type": et, **best,
           "raw_ler": m["raw_ler"]}
    if args.mwpm:
        from baseline.mwpm import mwpm_ler_from_dataset
        from model.data import npz_path
        row["mwpm_ler"] = mwpm_ler_from_dataset(
            npz_path(args.data_dir, noise, "test", args.cycles, p, et))
    return row


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    noises = args.noise or (ACTIVE_NOISE if args.all else [ACTIVE_NOISE[0]])
    rates = args.rates or (ERROR_RATES if (args.all or not args.smoke)
                           else [ERROR_RATES[0]])
    etypes = args.error_types or ERROR_TYPES
    if args.smoke:
        args.epochs = min(args.epochs, 3)
        noises, rates = noises[:1], rates[:1]

    ensure_coupling_json()
    mod = get_model_module(args.solution)
    print(f"model module: {mod.__name__} | device: {device}")

    rows = []
    for noise in noises:
        for p in rates:
            for et in etypes:
                try:
                    rows.append(train_one(args, mod, noise, p, et, device))
                except FileNotFoundError as e:
                    print(f"    skipped: {e}")
                except NotImplementedError as e:
                    print(f"\nERROR: the model is not implemented yet — "
                          f"you need to fill in model/cnn_skeleton.py ({e})")
                    sys.exit(1)

    if rows:
        print(f"\n{'=' * 70}\nSummary (best epoch per config)")
        cols = ["noise", "p", "type", "epoch", "ecr", "acc", "raw_ler", "ler"]
        if any("mwpm_ler" in r for r in rows):
            cols.append("mwpm_ler")
        print(" | ".join(f"{c:>8}" for c in cols))
        for r in rows:
            print(" | ".join(
                (f"{r.get(c, ''):>8.4f}" if isinstance(r.get(c), float)
                 else f"{str(r.get(c, '')):>8}") for c in cols))


if __name__ == "__main__":
    main()
