"""Shared helpers for the pipeline's single JSON config (config.json).

config.json at the repo root holds every user-facing setting in one
file, one section per axis:

* "noise_profiles" — the noise-profile registry (WHICH physics): the
  hand-edited 4-parameter profiles plus the qpu_avg_v1 entries that
  make_qpu_avg_profile.py registers programmatically (the tool updates
  ONLY this section and leaves the rest of the file untouched).
* "train" / "dataset" / top-level "cycles" (load_options) — HOW to run:
  default hyperparameters and dataset sample counts. CLI arguments
  still override them, and sweep run entries override both.
* "sweep" (load_sweep) — WHAT to run: the (noise, p, model, ...) run
  entries make_dataset.py generates for and train.py loops over. When
  the section exists it is applied automatically; disable with
  --config none, or point --config at a standalone sweep JSON (either
  the bare {"defaults", "runs"} layout or another full config file) to
  run a different experiment set.

Precedence: config.json defaults < CLI arguments < sweep run entries.
"""
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = _ROOT / "config.json"
CONFIG_SECTIONS = {"noise_profiles", "train", "dataset", "sweep", "cycles",
                   "pipeline"}


def load_config(path=None):
    """Read config.json (or an explicit path) with top-level-section
    validation. Missing file -> {} so hardcoded defaults apply."""
    p = Path(path) if path else CONFIG_PATH
    if not p.exists():
        return {}
    cfg = json.load(open(p))
    bad = set(cfg) - CONFIG_SECTIONS
    if bad:
        raise ValueError(
            f"{p.name}: unknown top-level sections {sorted(bad)} "
            f"(allowed: {sorted(CONFIG_SECTIONS)})")
    return cfg


# keys allowed in a sweep run entry (train.py consumes the full set;
# make_dataset.py only reads noise / rates / error_types / cycles).
# "model" lets one sweep train cnn and gnn runs on the same data, so the
# summary prints a combined CNN/GNN/MWPM table.
SWEEP_KEYS = {"noise", "rates", "error_types", "cycles", "epochs",
              "patience", "min_delta", "batch_size", "lr", "aux_weight",
              "pos_weight", "mwpm", "name", "model"}
_LIST_KEYS = ("noise", "rates", "error_types")


def load_sweep(path=None):
    """Parse the sweep definition into a list of per-run override dicts.

    Default (path=None): the "sweep" section of config.json.
    Explicit path: a standalone sweep JSON in one of the layouts
      {"defaults": {...}, "runs": [{...}, ...]}    or    [{...}, ...]
    or another full config file (its "sweep" section is used).
    Each run inherits `defaults` and then applies its own keys.
    noise/rates/error_types accept a scalar or a list (normalized to
    lists); unknown keys are rejected so typos fail fast."""
    where = path or f"{CONFIG_PATH.name}[sweep]"
    if path:
        cfg = json.load(open(path))
        if isinstance(cfg, dict) and "runs" not in cfg and "sweep" in cfg:
            cfg = cfg["sweep"]              # full config file passed
    else:
        cfg = load_config().get("sweep")
        if cfg is None:
            raise ValueError(f"{CONFIG_PATH.name} has no 'sweep' section")
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
                f"sweep config {where}: unknown keys {sorted(unknown)} in "
                f"run {i} (allowed: {sorted(SWEEP_KEYS)})")
        for k in _LIST_KEYS:
            if k in merged and not isinstance(merged[k], list):
                merged[k] = [merged[k]]
        out.append(merged)
    return out


def has_sweep(path=None):
    """True when config.json defines a sweep section (auto-applied)."""
    return bool(load_config(path).get("sweep"))


# keys allowed per config section consumed by load_options
OPTION_KEYS = {
    "train": {"cycles", "epochs", "patience", "min_delta", "batch_size",
              "lr", "aux_weight", "pos_weight", "mwpm", "amp",
              "data_dir", "outdir", "ckpt_dir"},
    "dataset": {"cycles", "train_samples", "test_samples", "seed",
                "outdir"},
}


def load_options(section, path=None):
    """Return config.json's `section` ("train"/"dataset") as a dict of
    argparse defaults.

    Missing file (or section) -> {} so the hardcoded defaults apply.
    The top-level "cycles" is shared by both sections (dataset files and
    training must agree on it). Unknown keys are rejected so typos fail
    fast."""
    cfg = load_config(path)
    opts = {"cycles": cfg["cycles"]} if "cycles" in cfg else {}
    opts.update(cfg.get(section, {}))
    unknown = set(opts) - OPTION_KEYS[section]
    if unknown:
        raise ValueError(
            f"config.json: unknown keys {sorted(unknown)} in '{section}' "
            f"(allowed: {sorted(OPTION_KEYS[section])})")
    return opts
