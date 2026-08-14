"""Model registry — single place that resolves --model names.

Each model ships as a skeleton/solution pair with the same interface
(README §1): a decoder class `<Class>(in_channels=2*cycles, ...)` with
`forward(x) -> (qubit_logits, logical_logits)` plus a module-level
`compute_loss(...) -> (total, loss_logical, loss_qubit)`.
`get_model_module` / `get_model_class` unify skeleton vs solution loading
for train.py and hardware/run_hw.py.
"""
import importlib

# name -> (skeleton module, solution module, decoder class name)
MODEL_REGISTRY = {
    "cnn": ("model.cnn_skeleton", "solutions.cnn_solution", "HeavyHexCNN"),
    "gnn": ("model.gnn_skeleton", "solutions.gnn_solution", "HeavyHexGNN"),
}
MODEL_NAMES = sorted(MODEL_REGISTRY)


def get_model_module(model_name, use_solution=False):
    """Import and return the skeleton (or solution) module of a model."""
    try:
        skeleton, solution, _ = MODEL_REGISTRY[model_name]
    except KeyError:
        raise ValueError(f"unknown model '{model_name}' "
                         f"(available: {MODEL_NAMES})")
    target = solution if use_solution else skeleton
    try:
        return importlib.import_module(target)
    except ModuleNotFoundError as e:
        if use_solution:
            raise ModuleNotFoundError(
                f"{target} not found — solutions/ is not part of the "
                f"distributed repo (drop --solution to use the skeleton)"
            ) from e
        raise


def get_model_class(model_name, use_solution=False):
    """Return the decoder class (e.g. HeavyHexCNN) of a model."""
    mod = get_model_module(model_name, use_solution)
    return getattr(mod, MODEL_REGISTRY[model_name][2])
