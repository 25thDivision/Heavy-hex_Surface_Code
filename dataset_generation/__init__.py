"""Shared helpers for the pipeline's JSON config files.

* train_sweep.json (load_sweep, explicit --config): WHAT to run — the
  (noise, p, ...) combos make_dataset.py generates and train.py loops over.
* train_options.json (load_options, auto-loaded from the repo root when
  present): HOW to run — default hyperparameters and dataset sample
  counts. CLI arguments still override it, and sweep run entries override
  both.
"""
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

# keys allowed in a run entry (train.py consumes the full set;
# make_dataset.py only reads noise / rates / error_types / cycles)
SWEEP_KEYS = {"noise", "rates", "error_types", "cycles", "epochs",
              "patience", "batch_size", "lr", "aux_weight", "pos_weight",
              "mwpm", "name"}
_LIST_KEYS = ("noise", "rates", "error_types")


def load_sweep(path):
    """Parse a sweep-config JSON into a list of per-run override dicts.

    Accepted layouts:
      {"defaults": {...}, "runs": [{...}, ...]}    or just    [{...}, ...]
    Each run inherits `defaults` and then applies its own keys.
    noise/rates/error_types accept a scalar or a list (normalized to
    lists); unknown keys are rejected so typos fail fast."""
    cfg = json.load(open(path))
    if isinstance(cfg, list):
        defaults, runs = {}, cfg
    else:
        defaults, runs = cfg.get("defaults", {}), cfg.get("runs", [{}])
    out = []
    for i, run in enumerate(runs):
        merged = {**defaults, **run}
        unknown = set(merged) - SWEEP_KEYS
        if unknown:
            raise ValueError(
                f"sweep config {path}: unknown keys {sorted(unknown)} in "
                f"run {i} (allowed: {sorted(SWEEP_KEYS)})")
        for k in _LIST_KEYS:
            if k in merged and not isinstance(merged[k], list):
                merged[k] = [merged[k]]
        out.append(merged)
    return out


# keys allowed per train_options.json section (argparse dest names)
OPTION_KEYS = {
    "train": {"cycles", "epochs", "patience", "batch_size", "lr",
              "aux_weight", "pos_weight", "mwpm",
              "data_dir", "outdir", "ckpt_dir"},
    "dataset": {"cycles", "train_samples", "test_samples", "seed",
                "outdir"},
}


def load_options(section, path=None):
    """Read train_options.json and return `section` ("train"/"dataset")
    as a dict of argparse defaults.

    Missing file (or section) -> {} so the hardcoded defaults apply.
    A top-level "cycles" is shared by both sections (dataset files and
    training must agree on it). Unknown sections/keys are rejected so
    typos fail fast."""
    p = Path(path) if path else _ROOT / "train_options.json"
    if not p.exists():
        return {}
    cfg = json.load(open(p))
    bad = set(cfg) - set(OPTION_KEYS) - {"cycles"}
    if bad:
        raise ValueError(
            f"{p.name}: unknown sections {sorted(bad)} "
            f"(allowed: {sorted(OPTION_KEYS)} and top-level 'cycles')")
    opts = {"cycles": cfg["cycles"]} if "cycles" in cfg else {}
    opts.update(cfg.get(section, {}))
    unknown = set(opts) - OPTION_KEYS[section]
    if unknown:
        raise ValueError(
            f"{p.name}: unknown keys {sorted(unknown)} in '{section}' "
            f"(allowed: {sorted(OPTION_KEYS[section])})")
    return opts
