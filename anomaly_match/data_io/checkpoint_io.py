#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.

"""Checkpoint I/O using safetensors for secure model serialization.

Replaces pickle-based ``torch.save`` / ``torch.load`` with safetensors to
prevent arbitrary code execution when loading untrusted model files.

Checkpoint layout inside a single ``.safetensors`` file:

* **Binary section** — all ``torch.Tensor`` values (model weights, optimizer
  momentum buffers, …) stored under namespaced keys
  (``train_model.<name>``, ``optimizer.state.<idx>.<buf>``, …).
* **Metadata header** — every non-tensor value is JSON-encoded into the
  ``Dict[str, str]`` metadata that safetensors carries in its header.
"""

from __future__ import annotations

import json
import pickle
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger

# ---------------------------------------------------------------------------
# JSON helpers for types that appear in checkpoint metadata
# ---------------------------------------------------------------------------


def _nullify_empty_dicts(obj: Any) -> Any:
    """Recursively replace empty dicts with ``None``.

    DotMap auto-creates empty child maps when accessing missing keys.  After
    ``toDict()`` these become ``{}``, which breaks fitsbolt's
    ``validate_config`` on reload (e.g. ``channel_combination`` is expected to
    be ``None`` or ``np.ndarray``, not ``{}``).
    """
    if isinstance(obj, dict):
        if len(obj) == 0:
            return None
        return {k: _nullify_empty_dicts(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_nullify_empty_dicts(v) for v in obj]
    return obj


def _prepare_for_json(obj: Any) -> Any:
    """Recursively convert non-JSON-native types to tagged representations.

    This is needed because ``IntEnum`` (which ``NormalisationMethod`` inherits
    from) is a subclass of ``int`` — the standard JSON encoder serializes it
    as a plain integer and never calls ``default()``.  By walking the
    structure up-front we ensure *all* special types are tagged.

    """
    # Enum check MUST come before int/float because IntEnum is also an int
    if isinstance(obj, Enum):
        return {"__enum__": type(obj).__name__, "name": obj.name}
    if isinstance(obj, np.dtype):
        return {"__numpy_dtype__": str(obj)}
    if isinstance(obj, type) and issubclass(obj, np.generic):
        return {"__numpy_dtype_type__": np.dtype(obj).str}
    if isinstance(obj, np.ndarray):
        return {"__numpy_array__": obj.tolist(), "dtype": str(obj.dtype)}
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: _prepare_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_prepare_for_json(v) for v in obj]
    return obj


class _CheckpointEncoder(json.JSONEncoder):
    """JSON encoder that handles checkpoint-specific types.

    Note: ``IntEnum`` values bypass ``default()`` because they *are* ints.
    Use :func:`_prepare_for_json` on the data **before** calling
    ``json.dumps`` to ensure those types are correctly tagged.
    """

    def default(self, obj: Any) -> Any:
        if isinstance(obj, Enum):
            return {"__enum__": type(obj).__name__, "name": obj.name}
        if isinstance(obj, np.dtype):
            return {"__numpy_dtype__": str(obj)}
        if isinstance(obj, type) and issubclass(obj, np.generic):
            return {"__numpy_dtype_type__": np.dtype(obj).str}
        if isinstance(obj, np.ndarray):
            return {"__numpy_array__": obj.tolist(), "dtype": str(obj.dtype)}
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def _checkpoint_object_hook(obj: dict) -> Any:
    """JSON object-hook that restores checkpoint-specific types."""
    if "__enum__" in obj:
        enum_name = obj["__enum__"]
        if enum_name == "NormalisationMethod":
            from fitsbolt.normalisation.NormalisationMethod import NormalisationMethod

            return NormalisationMethod[obj["name"]]
        return f"{enum_name}.{obj['name']}"
    if "__numpy_dtype__" in obj:
        return np.dtype(obj["__numpy_dtype__"])
    if "__numpy_dtype_type__" in obj:
        return np.dtype(obj["__numpy_dtype_type__"]).type
    if "__numpy_array__" in obj:
        return np.array(obj["__numpy_array__"], dtype=obj["dtype"])
    return obj


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def save_checkpoint(save_state: dict[str, Any], path: str | Path) -> Path:
    """Save a model checkpoint in safetensors format.

    Tensors are stored in the safetensors binary section; everything else is
    JSON-encoded into the safetensors metadata header.

    Args:
        save_state: Checkpoint dictionary (same keys as previously passed to
            ``torch.save``).
        path: Destination file path. The extension is forced to
            ``.safetensors``.

    Returns:
        The actual path written (with ``.safetensors`` extension).
    """
    from safetensors.torch import save_file

    path = Path(path).with_suffix(".safetensors")

    tensors: dict[str, torch.Tensor] = {}
    metadata: dict[str, str] = {}

    # ---- model state-dicts ------------------------------------------------
    for model_key in ("train_model", "eval_model"):
        state_dict = save_state.get(model_key)
        if state_dict is None:
            continue
        for param_name, tensor in state_dict.items():
            tensors[f"{model_key}.{param_name}"] = tensor.detach().clone().contiguous()

    # ---- optimizer state --------------------------------------------------
    opt_state = save_state.get("optimizer")
    if opt_state is not None:
        opt_skeleton: dict[str, Any] = {
            "state": {},
            "param_groups": opt_state.get("param_groups", []),
        }
        for param_idx, state in opt_state.get("state", {}).items():
            idx_str = str(param_idx)
            opt_skeleton["state"][idx_str] = {}
            for key, val in state.items():
                if isinstance(val, torch.Tensor):
                    tensors[f"optimizer.state.{param_idx}.{key}"] = (
                        val.detach().clone().contiguous()
                    )
                    opt_skeleton["state"][idx_str][key] = "__tensor__"
                else:
                    opt_skeleton["state"][idx_str][key] = val
        metadata["optimizer"] = json.dumps(_prepare_for_json(opt_skeleton), cls=_CheckpointEncoder)
    else:
        metadata["optimizer"] = "null"

    # ---- scheduler state --------------------------------------------------
    sched_state = save_state.get("scheduler")
    metadata["scheduler"] = (
        json.dumps(_prepare_for_json(sched_state), cls=_CheckpointEncoder)
        if sched_state is not None
        else "null"
    )

    # ---- scalar / enum metadata -------------------------------------------
    for key in (
        "it",
        "total_it",
        "best_eval_acc",
        "best_it",
        "num_channels",
        "net",
        "normalisation_method",
        "last_normalisation_method",
    ):
        metadata[key] = json.dumps(_prepare_for_json(save_state.get(key)), cls=_CheckpointEncoder)

    # ---- fitsbolt config (DotMap → dict → JSON) ---------------------------
    fb_cfg = save_state.get("fitsbolt_cfg")
    if fb_cfg is not None:
        cfg_dict = fb_cfg.toDict() if hasattr(fb_cfg, "toDict") else fb_cfg
        # DotMap auto-creates empty child maps on missing-key access (e.g.
        # channel_combination).  After toDict() these become empty dicts {},
        # which break fitsbolt's validate_config on reload.  Normalize
        # leaf-level empty dicts to None.
        cfg_dict = _nullify_empty_dicts(cfg_dict)
        metadata["fitsbolt_cfg"] = json.dumps(_prepare_for_json(cfg_dict), cls=_CheckpointEncoder)
    else:
        metadata["fitsbolt_cfg"] = "null"

    # ---- labeled-data CSV -------------------------------------------------
    csv_str = save_state.get("labeled_data_csv")
    if csv_str is not None:
        metadata["labeled_data_csv"] = csv_str

    # safetensors requires at least one tensor
    if not tensors:
        tensors["__placeholder__"] = torch.zeros(1)

    save_file(tensors, str(path), metadata=metadata)
    logger.debug(f"Saved checkpoint in safetensors format: {path}")
    return path


def _load_trusted_legacy_checkpoint(path: Path, map_location: str) -> dict[str, Any]:
    if path.suffix.lower() == ".pkl":
        try:
            with path.open("rb") as file:
                return pickle.load(file)
        except (pickle.UnpicklingError, EOFError):
            pass

    return torch.load(path, map_location=map_location, weights_only=False)


def convert_legacy_checkpoint_to_safetensors(
    legacy_path: str | Path,
    output_path: str | Path | None = None,
    *,
    trusted: bool = False,
    map_location: str = "cpu",
) -> Path:
    """Convert one trusted pickle-based checkpoint to ``.safetensors``.

    This deliberately unsafe deserialization is isolated from normal checkpoint
    loading and must be explicitly enabled for each conversion.
    """
    if not trusted:
        raise ValueError(
            "Conversion requires trusted=True. Legacy PyTorch/pickle checkpoints "
            "can execute arbitrary code when loaded; only convert files from trusted sources."
        )

    legacy_path = Path(legacy_path)
    if legacy_path.suffix.lower() not in {".pth", ".pkl"}:
        raise ValueError(
            f"Unsupported legacy checkpoint extension {legacy_path.suffix!r}; "
            "expected .pth or .pkl."
        )
    if not legacy_path.is_file():
        raise FileNotFoundError(f"Legacy checkpoint not found: {legacy_path}")

    checkpoint = _load_trusted_legacy_checkpoint(legacy_path, map_location)
    if not isinstance(checkpoint, dict):
        raise ValueError("Legacy checkpoint must contain a checkpoint dictionary.")
    for model_key in ("train_model", "eval_model"):
        state_dict = checkpoint.get(model_key)
        if not isinstance(state_dict, dict) or not all(
            isinstance(value, torch.Tensor) for value in state_dict.values()
        ):
            raise ValueError(
                f"Legacy checkpoint key {model_key!r} must contain a tensor state dictionary."
            )

    return save_checkpoint(checkpoint, output_path or legacy_path.with_suffix(".safetensors"))


def load_checkpoint(path: str | Path, device: str = "cpu") -> dict[str, Any]:
    """Load a model checkpoint from a ``.safetensors`` file.

    Args:
        path: Path to the ``.safetensors`` checkpoint file.
        device: Device to map tensors to (default ``"cpu"``).

    Returns:
        Checkpoint dictionary with the same structure as originally saved.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    from safetensors import safe_open
    from safetensors.torch import load_file

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    all_tensors = load_file(str(path), device=device)

    with safe_open(str(path), framework="pt", device=device) as f:
        raw_metadata = f.metadata() or {}

    checkpoint: dict[str, Any] = {}

    # ---- model state-dicts ------------------------------------------------
    for model_key in ("train_model", "eval_model"):
        prefix = f"{model_key}."
        state_dict = {k[len(prefix) :]: v for k, v in all_tensors.items() if k.startswith(prefix)}
        if state_dict:
            checkpoint[model_key] = state_dict

    # ---- optimizer state --------------------------------------------------
    opt_skeleton = json.loads(
        raw_metadata.get("optimizer", "null"), object_hook=_checkpoint_object_hook
    )
    if opt_skeleton is not None:
        new_state: dict[int, dict] = {}
        for idx_str, state in opt_skeleton.get("state", {}).items():
            restored: dict[str, Any] = {}
            for key, val in state.items():
                if val == "__tensor__":
                    restored[key] = all_tensors[f"optimizer.state.{idx_str}.{key}"]
                else:
                    restored[key] = val
            new_state[int(idx_str)] = restored
        opt_skeleton["state"] = new_state
        checkpoint["optimizer"] = opt_skeleton
    else:
        checkpoint["optimizer"] = None

    # ---- scheduler state --------------------------------------------------
    checkpoint["scheduler"] = json.loads(
        raw_metadata.get("scheduler", "null"), object_hook=_checkpoint_object_hook
    )

    # ---- scalar / enum metadata -------------------------------------------
    for key in (
        "it",
        "total_it",
        "best_eval_acc",
        "best_it",
        "num_channels",
        "net",
        "normalisation_method",
        "last_normalisation_method",
    ):
        checkpoint[key] = json.loads(
            raw_metadata.get(key, "null"), object_hook=_checkpoint_object_hook
        )

    # ---- fitsbolt config --------------------------------------------------
    fb_data = json.loads(
        raw_metadata.get("fitsbolt_cfg", "null"), object_hook=_checkpoint_object_hook
    )
    if fb_data is not None:
        from dotmap import DotMap

        # _dynamic=False prevents DotMap from auto-creating empty child maps
        # on missing-key access, which would break fitsbolt's validate_config
        # (e.g. channel_combination should stay absent, not become DotMap()).
        checkpoint["fitsbolt_cfg"] = DotMap(fb_data, _dynamic=False)
    else:
        checkpoint["fitsbolt_cfg"] = None

    # ---- labeled-data CSV -------------------------------------------------
    if "labeled_data_csv" in raw_metadata:
        checkpoint["labeled_data_csv"] = raw_metadata["labeled_data_csv"]

    return checkpoint
