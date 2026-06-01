from __future__ import annotations

import sys
from pathlib import Path
import importlib.util


def _load_runtime_module():
    repo_root = Path(__file__).resolve().parents[3]
    runtime_path = repo_root / "Countdown-Code" / "verl" / "verl" / "verl" / "utils" / "mech_interp.py"
    if not runtime_path.exists():
        raise FileNotFoundError(f"Could not find VERL mech-interp runtime at {runtime_path}")
    spec = importlib.util.spec_from_file_location("_em_verl_mech_interp_runtime", runtime_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import VERL mech-interp runtime from {runtime_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_runtime = _load_runtime_module()

EM_VECTOR_FORMAT = _runtime.EM_VECTOR_FORMAT
EMVectorRuntime = _runtime.EMVectorRuntime
ResidualVectorHookContext = _runtime.ResidualVectorHookContext
add_vector = _runtime.add_vector
normalize_vector = _runtime.normalize_vector
project_onto_vector = _runtime.project_onto_vector
remove_projection = _runtime.remove_projection
resolve_transformer_layers = _runtime.resolve_transformer_layers
