#!/usr/bin/env python3
"""
Training entry point — you don't need to modify this file
==========================================================
Trains a dual-head decoder (--model {cnn,gnn}; the skeleton file you
complete lives at model/<name>_skeleton.py) on the Stim datasets and logs
the metrics per epoch. Checkpoint selection is by best validation
head-LER (the loss is LER-first), with patience-based early stopping.
--code {heavyhex,surface} selects the code family (dataset subtree and
result/checkpoint tag).

Metric status: the **head-LER** (logical head) is the official metric.
`ECR (diagnostic, sim-only)` and `parity_LER (diagnostic)` are reported
alongside for diagnosis only, and `LER/MWPM ratio` (present when --mwpm
ran on the same data) is the single comparison number across runs.

The "validation set" is the independently generated test file
(train/test files, not a ratio split).

Usage:
  python train.py --model cnn --smoke        # quick end-to-end check
  python train.py -n realistic/dp0.001_mf0.01_rf0.01_gd0.008 -p 0.005
  python train.py --all                      # full noise x rate grid
  python train.py --config train_sweep.json  # JSON-driven sweep (see below)
  python train.py --solution ...             # reference model, if you have solutions/

Options file (train_options.json at the repo root, auto-loaded when it
exists): replaces the hardcoded defaults — "train" section for the
hyperparameters here, "dataset" section for make_dataset.py's sample
counts, top-level "cycles" shared by both. Explicit CLI arguments still
override it, and sweep run entries override both.

Sweep config: a JSON of {"defaults": {...}, "runs": [{...}]}. If
train_sweep.json exists at the repo root it is used automatically —
unless you pass --config, select explicitly with -n/-p/-e/--all/--smoke,
or disable with --config none.
Each run entry may set noise / rates / error_types plus hyperparameter
overrides (cycles, epochs, patience, batch_size, lr, aux_weight,
pos_weight, mwpm) and an optional "name" appended to the result/checkpoint
tag (use it when two runs share the same noise/p/cycles, or their files
would overwrite each other). Keys a run omits fall back to the CLI
arguments. make_dataset.py --config takes the same file to generate the
matching datasets.
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

from dataset_generation import load_options  # noqa: E402
from dataset_generation.heavyhex33_stim import (  # noqa: E402
    noise_tag, DISTANCE, ERROR_RATES, ERROR_TYPES, ALL_NOISE)
from model import get_model_module, get_model_class, MODEL_NAMES  # noqa: E402
from model.data import load_split, FastTensorDataLoader  # noqa: E402
from evaluation.metrics import (  # noqa: E402
    ecr, bit_accuracy, ler_from_logits, parity_ler_from_qubit_logits)

# training-loop defaults
MAX_EPOCHS = 30
PATIENCE = 5
BATCH_SIZE = 2048
LR = 1e-3


def parse_args():
    ap = argparse.ArgumentParser(description="Train a QEC decoder model")
    ap.add_argument("--model", choices=MODEL_NAMES, default="cnn",
                    help="decoder architecture (model/<name>_skeleton.py); "
                         "reflected in result/checkpoint names "
                         "({MODEL}_{tag})")
    ap.add_argument("--code", choices=["heavyhex", "surface"],
                    default="heavyhex",
                    help="code family; selects dataset/<code>/ and is part "
                         "of the result/checkpoint tag (surface lands with "
                         "the rsc3 milestone)")
    ap.add_argument("-n", "--noise", nargs="+", default=None)
    ap.add_argument("-p", "--rates", nargs="+", type=float, default=None)
    ap.add_argument("-e", "--error-types", nargs="+", default=None)
    ap.add_argument("--all", action="store_true",
                    help="run the full grid (all active noise x rates)")
    ap.add_argument("--cycles", type=int, default=3)
    ap.add_argument("--data-dir", default=str(_ROOT / "dataset"))
    ap.add_argument("--outdir", default=str(_ROOT / "results" / "train"))
    ap.add_argument("--ckpt-dir", default=str(_ROOT / "checkpoint"))
    ap.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    ap.add_argument("--patience", type=int, default=PATIENCE)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--aux-weight", type=float, default=0.5,
                    help="weight of the per-qubit auxiliary BCE loss")
    ap.add_argument("--pos-weight", choices=["none", "linear", "sqrt"],
                    default="none",
                    help="per-qubit BCE pos_weight from p (linear was used "
                         "on injected-mask labels historically; our labels "
                         "differ, so default is none)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--mwpm", action="store_true",
                    help="also evaluate the MWPM baseline on the val set")
    ap.add_argument("--solution", action="store_true",
                    help="use the reference model from solutions/ "
                         "(not part of the distributed repo)")
    ap.add_argument("--smoke", action="store_true",
                    help="single config, 3 epochs, tiny batch count")
    ap.add_argument("--config", default=None,
                    help="sweep-config JSON: loop training over its run "
                         "entries. Defaults to train_sweep.json at the "
                         "repo root when no selection args (-n/-p/-e/"
                         "--all/--smoke) are given; 'none' disables")
    # train_options.json (repo root, if present) replaces the hardcoded
    # defaults; explicit CLI arguments and sweep entries still win
    opts = load_options("train")
    if opts:
        ap.set_defaults(**opts)
        print(f"train_options.json: {opts}")
    args = ap.parse_args()
    # auto-sweep: no explicit config and no explicit selection -> pick up
    # the default sweep file if it exists
    if args.config == "none":
        args.config = None
    elif args.config is None and not (args.noise or args.rates
                                      or args.error_types or args.all
                                      or args.smoke):
        default_sweep = _ROOT / "train_sweep.json"
        if default_sweep.exists():
            args.config = str(default_sweep)
    return args


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
        "parity_ler": parity_ler_from_qubit_logits(q_logits, y_l),
        "raw_ler": float(y_l.mean()),
        "inf_ms": (t_inf / max(n_inf, 1)) * 1000,
    }


def mwpm_ratio(cnn_ler, mwpm_ler):
    """CNN head-LER / MWPM LER on the same data; 'N/A' without MWPM."""
    if mwpm_ler is None or mwpm_ler == 0:
        return None
    return cnn_ler / mwpm_ler


def fmt_ratio(ratio):
    return f"{ratio:.4f}" if ratio is not None else "N/A"


def train_one(args, mod, noise, p, et, device):
    print(f"\n{'=' * 70}\n>>> {args.model}/{args.code} | {noise} | p={p} | "
          f"{et} | cycles={args.cycles}")
    Xtr, yqtr, yltr = load_split(args.data_dir, args.code, noise, "train",
                                 args.cycles, p, et, device)
    Xva, yqva, ylva = load_split(args.data_dir, args.code, noise, "test",
                                 args.cycles, p, et, device)
    # model-specific input prep (e.g. the GNN appends the final-Z
    # detector channel from the final bits — see model/graph.py)
    if hasattr(mod, "prepare_features"):
        Xtr = mod.prepare_features(Xtr, yqtr, args.code)
        Xva = mod.prepare_features(Xva, yqva, args.code)
    print(f"    train {tuple(Xtr.shape)}  val {tuple(Xva.shape)}  -> {device}")

    train_loader = FastTensorDataLoader(Xtr, yqtr, yltr,
                                        batch_size=args.batch_size, shuffle=True)
    val_loader = FastTensorDataLoader(Xva, yqva, ylva,
                                      batch_size=args.batch_size)

    model_cls = get_model_class(args.model, args.solution)
    model = model_cls(in_channels=2 * args.cycles).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    pos_weight = None
    if args.pos_weight != "none":
        w = (1.0 - p) / p
        if args.pos_weight == "sqrt":
            w = float(np.sqrt(w))
        pos_weight = torch.full((17,), w, device=device)

    # flat layout: results/train/CNN_heavyhex_d3_c3_p0.005_dp0.001_....csv
    # ({MODEL}_{code}_d..., d = code distance, c = number of QEC cycles;
    #  the error type is omitted since the grid is X-only)
    tag = f"{args.code}_d{DISTANCE}_c{args.cycles}_p{p}_{noise_tag(noise)}"
    if getattr(args, "run_name", ""):
        tag += f"_{args.run_name}"
    prefix = args.model.upper()
    result_dir = Path(args.outdir)
    ckpt_dir = Path(args.ckpt_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    csv_path = result_dir / f"{prefix}_{tag}.csv"
    ckpt_path = ckpt_dir / f"{prefix}_{tag}.pt"

    # MWPM baseline on the same val data (once, before the loop) so every
    # evaluation report can carry the LER/MWPM ratio column
    mwpm_ler = None
    if args.mwpm:
        from baseline.mwpm import mwpm_ler_from_dataset
        from model.data import npz_path
        mwpm_ler = mwpm_ler_from_dataset(
            npz_path(args.data_dir, args.code, noise, "test",
                     args.cycles, p, et))
        print(f"    MWPM baseline LER (val data): {mwpm_ler:.4f}")

    best = {"ler": float("inf"), "ecr": 0.0, "acc": 0.0,
            "parity_ler": float("nan"), "epoch": 0}
    patience = 0
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train_Loss", "Val_Loss",
                         "Val_ECR (diagnostic, sim-only)", "Val_Acc",
                         "Val_LER", "Val_parity_LER (diagnostic)",
                         "Raw_LER", "LER/MWPM ratio", "Inference_ms"])
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
            ratio = mwpm_ratio(m["ler"], mwpm_ler)
            writer.writerow([epoch, f"{train_loss:.6f}",
                             f"{m['val_loss']:.6f}", f"{m['ecr']:.4f}",
                             f"{m['acc']:.4f}", f"{m['ler']:.4f}",
                             f"{m['parity_ler']:.4f}",
                             f"{m['raw_ler']:.4f}", fmt_ratio(ratio),
                             f"{m['inf_ms']:.4f}"])
            f.flush()
            print(f"    [Ep {epoch:02d}] loss {train_loss:.4f}/"
                  f"{m['val_loss']:.4f} | "
                  f"ECR (diagnostic, sim-only) {m['ecr']:.2%} | "
                  f"Acc {m['acc']:.2%} | LER {m['ler']:.4f} "
                  f"(raw {m['raw_ler']:.4f}) | "
                  f"parity_LER (diagnostic) {m['parity_ler']:.4f} | "
                  f"LER/MWPM ratio {fmt_ratio(ratio)} | pat {patience}")

            if m["ler"] < best["ler"]:
                best = {"ler": m["ler"], "ecr": m["ecr"], "acc": m["acc"],
                        "parity_ler": m["parity_ler"], "epoch": epoch}
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

    row = {"model": args.model, "noise": noise, "p": p, "type": et, **best,
           "raw_ler": m["raw_ler"]}
    if mwpm_ler is not None:
        row["mwpm_ler"] = mwpm_ler
    row["ler_mwpm_ratio"] = mwpm_ratio(best["ler"], mwpm_ler)
    return row


def expand_config(args):
    """--config JSON -> one args namespace per run entry."""
    from dataset_generation import load_sweep
    run_list = []
    for run in load_sweep(args.config):
        ra = argparse.Namespace(**vars(args))
        ra.run_name = run.pop("name", "")
        for k, v in run.items():
            setattr(ra, k, v)
        run_list.append(ra)
    return run_list


def main():
    args = parse_args()
    if args.code != "heavyhex":
        sys.exit(f"--code {args.code}: not implemented yet — the rotated "
                 f"surface code path arrives with the rsc3 milestone.")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    run_list = expand_config(args) if args.config else [args]

    ensure_coupling_json()
    mod = get_model_module(args.model, args.solution)
    print(f"model module: {mod.__name__} | device: {device}"
          + (f" | sweep: {args.config} ({len(run_list)} runs)"
             if args.config else ""))

    rows = []
    for ra in run_list:
        # sweep entries may override the model per run ("model" key), so
        # resolve the module per entry (importlib caches repeats)
        mod = get_model_module(ra.model, ra.solution)
        noises = ra.noise or (ALL_NOISE if ra.all else [ALL_NOISE[0]])
        rates = ra.rates or (ERROR_RATES if (ra.all or not ra.smoke)
                             else [ERROR_RATES[0]])
        etypes = ra.error_types or ERROR_TYPES
        if ra.smoke:
            ra.epochs = min(ra.epochs, 3)
            noises, rates = noises[:1], rates[:1]

        for noise in noises:
            for p in rates:
                for et in etypes:
                    try:
                        rows.append(train_one(ra, mod, noise, p, et, device))
                    except FileNotFoundError as e:
                        print(f"    skipped: {e}")
                    except NotImplementedError as e:
                        print(f"\nERROR: the model is not implemented yet — "
                              f"you need to fill in "
                              f"model/{ra.model}_skeleton.py ({e})")
                        sys.exit(1)

    if rows:
        print(f"\n{'=' * 70}\nSummary (best epoch per config; official "
              f"metric = head-LER, 'ler' column)")
        cols = [("model", "model"),
                ("noise", "noise"), ("p", "p"), ("type", "type"),
                ("epoch", "epoch"),
                ("ecr", "ECR (diagnostic, sim-only)"),
                ("acc", "acc"), ("raw_ler", "raw_ler"), ("ler", "ler"),
                ("parity_ler", "parity_LER (diagnostic)")]
        if any(r.get("mwpm_ler") is not None for r in rows):
            cols.append(("mwpm_ler", "mwpm_ler"))
        cols.append(("ler_mwpm_ratio", "LER/MWPM ratio"))

        def cell(r, key):
            v = r.get(key)
            if key == "ler_mwpm_ratio":
                return fmt_ratio(v)
            if isinstance(v, float):
                return f"{v:.4f}"
            return str(v) if v is not None else ""

        widths = [max(len(label), 8) for _, label in cols]
        print(" | ".join(f"{label:>{w}}"
                         for (_, label), w in zip(cols, widths)))
        for r in rows:
            print(" | ".join(f"{cell(r, key):>{w}}"
                             for (key, _), w in zip(cols, widths)))


if __name__ == "__main__":
    main()
