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
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import torch
from dotmap import DotMap
from fitsbolt.cfg.create_config import create_config as fb_create_cfg
from fitsbolt.normalisation.NormalisationMethod import NormalisationMethod
from loguru import logger
from safetensors import safe_open
from safetensors.torch import load_file as safetensors_load_file
from safetensors.torch import save_file as safetensors_save_file

# ---------------------------------------------------------------------------
# JSON helpers for types that appear in checkpoint metadata
# ---------------------------------------------------------------------------


def _nullify_empty_dicts(obj: Any) -> Any:
    """Recursively replace empty dicts with ``None``.

    DotMap auto-creates empty child maps when accessing missing keys.  After
    ``toDict()`` these become ``{}``, which breaks fitsbolt's
    ``validate_config`` on reload (e.g. ``channel_combination`` is expected to
    be ``None`` or ``np.ndarray``, not ``{}``).

    Returns:
        The input structure with empty dicts replaced by ``None``.
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

    Returns:
        JSON-serializable version of *obj*.
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
    """JSON object-hook that restores checkpoint-specific types.

    Returns:
        The original dict, or a restored Python object if a tag was found.
    """
    if "__enum__" in obj:
        enum_name = obj["__enum__"]
        if enum_name == "NormalisationMethod":
            return NormalisationMethod[obj["name"]]
        return f"{enum_name}.{obj['name']}"
    if "__numpy_dtype__" in obj:
        return np.dtype(obj["__numpy_dtype__"])
    if "__numpy_dtype_type__" in obj:
        return np.dtype(obj["__numpy_dtype_type__"]).type
    if "__numpy_array__" in obj:
        return np.array(obj["__numpy_array__"], dtype=obj["dtype"])
    return obj


def _deep_fill(defaults: dict, override: dict) -> dict:
    """Merge *override* onto *defaults*, keeping every override value.

    Recurses into nested dicts so a partially-populated ``override`` submap gains
    the keys it lacks from ``defaults`` without losing the keys it has.  One
    format upgrade is applied: where the default is a list but the override is a
    bare number, the override is wrapped in a single-element list.  fitsbolt's
    per-channel normalisation params (e.g. ``midtones.percentile``) were scalars
    in older releases and are lists in 0.3.x, and its validator now iterates
    them — a stored scalar would otherwise raise ``TypeError``.

    Args:
        defaults: Fallback values supplying keys ``override`` is missing.
        override: Authoritative values; win on every key they define.

    Returns:
        A new dict with every ``override`` key preserved and every
        ``defaults``-only key filled in.
    """
    result = dict(defaults)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_fill(result[key], value)
        elif (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and isinstance(result.get(key), list)
        ):
            result[key] = [value]
        else:
            result[key] = value
    return result


def _backfill_fitsbolt_cfg(fb_data: dict) -> dict:
    """Fill in fitsbolt config keys added after the checkpoint was written.

    A checkpoint embeds whatever fitsbolt config schema was current at training
    time.  Later fitsbolt releases add keys the normalisation code then reads
    directly — e.g. ``normalisation.minmax_n_samples`` in 0.3.x — so an older
    embedded config raises ``AttributeError`` mid-normalisation.  Rebuild a fresh
    default config for the checkpoint's own core parameters and overlay the
    stored values on top: every key current fitsbolt expects becomes present,
    while every training-time value is preserved unchanged (overlay wins, so the
    rebuild only contributes keys the stored config lacked).

    Args:
        fb_data: The decoded (not yet DotMap-wrapped) fitsbolt config dict.

    Returns:
        The config with any missing current-schema keys backfilled from
        fitsbolt's defaults.  Values already present are untouched.
    """
    # ``.get`` fallbacks (not direct access) only because this reconstructs a
    # *reference* default to source new keys from — a checkpoint predating any of
    # these top-level keys should still migrate rather than crash here.
    reference = fb_create_cfg(
        output_dtype=fb_data.get("output_dtype", np.uint8),
        size=fb_data.get("size", [224, 224]),
        fits_extension=fb_data.get("fits_extension"),
        interpolation_order=fb_data.get("interpolation_order", 1),
        n_output_channels=fb_data.get("n_output_channels", 3),
        normalisation_method=NormalisationMethod(fb_data.get("normalisation_method", 0)),
        num_workers=fb_data.get("num_workers", 1),
    ).toDict()
    return _deep_fill(reference, fb_data)


def _decode_fitsbolt_cfg(raw_metadata: dict) -> DotMap | None:
    """Decode the ``fitsbolt_cfg`` entry from a safetensors metadata header.

    Returns:
        The fitsbolt config as a ``_dynamic=False`` DotMap (so missing-key
        access raises instead of auto-creating empty child maps, which would
        break fitsbolt's ``validate_config``), or ``None`` when no fitsbolt
        config was stored.  Keys that post-date the checkpoint's fitsbolt version
        are backfilled from current defaults (see :func:`_backfill_fitsbolt_cfg`)
        so an older embedded config still validates and normalises.
    """
    fb_data = json.loads(
        raw_metadata.get("fitsbolt_cfg", "null"), object_hook=_checkpoint_object_hook
    )
    if fb_data is None:
        return None
    return DotMap(_backfill_fitsbolt_cfg(fb_data), _dynamic=False)


def _decode_channel_combination(raw_metadata: dict):
    """Decode the ``channel_combination`` entry from a metadata header.

    Returns:
        The band-mixing matrix as an ``np.ndarray`` (or list), or ``None`` when
        none was stored.  A non-array value is treated as ``None`` so the combine
        sites never receive an empty DotMap.
    """
    cc = json.loads(
        raw_metadata.get("channel_combination", "null"), object_hook=_checkpoint_object_hook
    )
    return cc if isinstance(cc, (list, tuple, np.ndarray)) else None


def _fitsbolt_normalisation_fields(fitsbolt_cfg) -> dict[str, Any]:
    """Extract the cfg.normalisation-shaped fields from a fitsbolt config.

    Shared by :func:`read_model_normalisation` (which only reports them) and
    :func:`sync_normalisation_from_checkpoint` (which writes them onto a cfg) so
    the two readers can't disagree about how a checkpoint maps to
    ``cfg.normalisation``.  ``channel_combination`` is *not* here — it lives
    outside the fitsbolt config (see :func:`_decode_channel_combination`).

    Args:
        fitsbolt_cfg: A fitsbolt config (dict or DotMap).  ``size``,
            ``normalisation_method`` and ``n_output_channels`` are required —
            accessed directly so a malformed config raises rather than silently
            yielding ``None``.

    Returns:
        Dict with ``image_size``, ``normalisation_method`` (a
        :class:`NormalisationMethod`) and ``n_output_channels``.
    """
    # ``list()`` accepts the stored JSON list and an in-memory tuple alike;
    # ``NormalisationMethod()`` is idempotent on an enum and decodes the stored
    # int — so no isinstance branching is needed, and a genuinely malformed
    # value raises here rather than slipping through.
    return {
        "image_size": list(fitsbolt_cfg["size"]),
        "normalisation_method": NormalisationMethod(fitsbolt_cfg["normalisation_method"]),
        "n_output_channels": fitsbolt_cfg["n_output_channels"],
    }


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
    # ``channel_combination`` is an AnomalyMatch-level band-mixing matrix
    # (n_out x n_in) applied *outside* fitsbolt — on the GPU in
    # ``cutana_batch_to_model_tensor_gpu`` for prediction.  It is therefore not a
    # fitsbolt_cfg field, and must be persisted here in its own right, or a model
    # trained on more bands than its input channels (e.g. 4-band Euclid -> 3
    # channels) cannot be scored: prediction would have no matrix to reduce the
    # bands and would feed the model the wrong channel count.  Stored as an
    # ``np.ndarray`` (or ``None``).
    for key in (
        "it",
        "total_it",
        "best_eval_acc",
        "best_it",
        "num_channels",
        "net",
        "normalisation_method",
        "last_normalisation_method",
        "channel_combination",
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

    safetensors_save_file(tensors, str(path), metadata=metadata)
    logger.debug(f"Saved checkpoint in safetensors format: {path}")
    return path


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
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    all_tensors = safetensors_load_file(str(path), device=device)

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

    # ---- fitsbolt config + channel combination ----------------------------
    checkpoint["fitsbolt_cfg"] = _decode_fitsbolt_cfg(raw_metadata)
    checkpoint["channel_combination"] = _decode_channel_combination(raw_metadata)

    # ---- labeled-data CSV -------------------------------------------------
    if "labeled_data_csv" in raw_metadata:
        checkpoint["labeled_data_csv"] = raw_metadata["labeled_data_csv"]

    return checkpoint


def _read_metadata_header(path: str | Path) -> dict:
    """Return a checkpoint's safetensors metadata header without loading tensors.

    ``safe_open`` exposes the JSON metadata without materialising any tensor, so
    this is cheap enough to run on every model-chooser change.  Shared by
    :func:`read_model_normalisation` and :func:`read_checkpoint_normalisation` so
    the tensor-free read happens in exactly one place.

    Args:
        path: Path to the ``.safetensors`` checkpoint.

    Returns:
        The metadata header dict (empty when the file carries no metadata).

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    with safe_open(str(path), framework="pt", device="cpu") as f:
        return f.metadata() or {}


def read_model_normalisation(path: str | Path) -> dict[str, Any]:
    """Read a checkpoint's embedded normalisation settings without loading tensors.

    The prediction subprocess overrides ``cfg.fitsbolt_cfg`` (and ``net`` /
    ``num_channels``) from the checkpoint in :func:`load_model`, so the model's
    embedded settings — not anything chosen in the UI — are what inference
    actually runs.  The prediction setup screen reads them to show a read-only
    overview and to keep ``cfg.normalisation`` consistent with the model.

    Args:
        path: Path to the ``.safetensors`` checkpoint.

    Returns:
        Dict with ``net``, ``num_channels``, ``image_size`` (``[h, w]``),
        ``normalisation_method`` (:class:`NormalisationMethod`),
        ``n_output_channels`` and ``channel_combination``.  The normalisation
        fields are ``None`` when the checkpoint predates embedded normalisation
        metadata (no ``fitsbolt_cfg``).  Propagates ``FileNotFoundError`` from
        :func:`_read_metadata_header` when *path* does not exist.
    """
    raw_metadata = _read_metadata_header(path)

    net = json.loads(raw_metadata.get("net", "null"), object_hook=_checkpoint_object_hook)
    num_channels = json.loads(
        raw_metadata.get("num_channels", "null"), object_hook=_checkpoint_object_hook
    )
    fb = _decode_fitsbolt_cfg(raw_metadata)

    summary: dict[str, Any] = {
        "net": net,
        "num_channels": num_channels,
        "image_size": None,
        "normalisation_method": None,
        "n_output_channels": None,
        # Read only the standalone field — no fallback to fitsbolt's embedded copy.
        "channel_combination": _decode_channel_combination(raw_metadata),
    }
    if fb is not None:
        summary.update(_fitsbolt_normalisation_fields(fb))
    return summary


def read_checkpoint_normalisation(path: str | Path) -> tuple[DotMap | None, Any]:
    """Read a checkpoint's fitsbolt config + channel combination, tensor-free.

    Returns the two surfaces prediction needs to reproduce training's input
    pipeline: the fitsbolt config (resolution + per-channel normalisation) and
    the AnomalyMatch band-mixing matrix (applied outside fitsbolt, on the GPU).
    Cheap enough to call from the parent process before launching subprocesses.

    Args:
        path: Path to the ``.safetensors`` checkpoint.

    Returns:
        ``(fitsbolt_cfg, channel_combination)``: the fitsbolt config as a
        ``_dynamic=False`` DotMap (or ``None`` for a legacy checkpoint), and the
        channel-combination matrix as an ``np.ndarray``/list (or ``None``).  This
        returns the *raw* fitsbolt config that :func:`sync_normalisation_from_checkpoint`
        writes onto ``cfg.fitsbolt_cfg`` — distinct from
        :func:`read_model_normalisation`, which extracts the display-shaped
        ``image_size``/``normalisation_method`` fields and adds ``net`` /
        ``num_channels``.  Both share :func:`_read_metadata_header` and the
        ``_decode_*`` helpers, so they can't disagree on the read.  Propagates
        ``FileNotFoundError`` from :func:`_read_metadata_header` when *path* does
        not exist.
    """
    raw_metadata = _read_metadata_header(path)
    return _decode_fitsbolt_cfg(raw_metadata), _decode_channel_combination(raw_metadata)


def sync_normalisation_from_checkpoint(
    cfg, fitsbolt_cfg: DotMap | None, channel_combination
) -> bool:
    """Make a model checkpoint the single source of truth for normalisation on *cfg*.

    Prediction must feed the model images produced exactly as in training. Three
    config surfaces drive that, and all must agree with the model:

    * ``cfg.fitsbolt_cfg`` — the inference decode pipeline (image/Zarr paths) and
      the Cutana orchestrator's ``external_fitsbolt_cfg`` normalisation.
    * ``cfg.normalisation`` ``image_size`` / ``n_output_channels`` — the Cutana
      orchestrator ``target_resolution`` and output channel count.
    * ``cfg.normalisation.channel_combination`` — the band-mixing matrix applied
      on the GPU in ``cutana_batch_to_model_tensor_gpu`` (outside fitsbolt), which
      reduces e.g. 4-band Euclid input to the model's 3 channels.

    A stale ``cfg`` (left over from training, or the pickled UI default) would
    otherwise produce cutouts at the wrong resolution or with the wrong channel
    mixing — a silent, severe accuracy loss, or a hard channel-count crash. This
    overwrites all three from the checkpoint.

    Args:
        cfg: Configuration to update in place.
        fitsbolt_cfg: The model's embedded fitsbolt config (from
            :func:`read_checkpoint_normalisation` or a loaded checkpoint's
            ``"fitsbolt_cfg"``), or ``None`` for a legacy checkpoint.
        channel_combination: The model's band-mixing matrix (from the checkpoint's
            ``"channel_combination"``), or ``None``.

    Returns:
        ``True`` when *fitsbolt_cfg* was applied; ``False`` when it was ``None``
        (the caller must then decide how to handle a metadata-less checkpoint).

    Raises:
        ValueError: If *channel_combination* is a matrix whose row count does not
            equal the model's ``n_output_channels`` (a corrupt/inconsistent
            band-mixing matrix that would feed the model the wrong channel count).
    """
    if fitsbolt_cfg is None:
        return False
    fields = _fitsbolt_normalisation_fields(fitsbolt_cfg)
    cfg.fitsbolt_cfg = fitsbolt_cfg
    cfg.normalisation.image_size = fields["image_size"]
    cfg.normalisation.normalisation_method = fields["normalisation_method"]
    cfg.normalisation.n_output_channels = fields["n_output_channels"]
    # The standalone ``channel_combination`` field is the single source — no
    # fallback to fitsbolt's embedded copy.  A checkpoint that predates this
    # field and needs band mixing must be retrained (the GPU combine site fails
    # hard if the bands don't match the model's channels).  Guard the DotMap-None
    # pitfall: only keep a real array, else None.
    matrix = (
        channel_combination if isinstance(channel_combination, (list, tuple, np.ndarray)) else None
    )
    # A band-mixing matrix maps input bands to the model's output channels, so it
    # must have exactly ``n_output_channels`` rows.  A row count that disagrees
    # (e.g. a 3×4 matrix truncated to 2×4 by stale UI state) would feed the model
    # the wrong channel count — reject it here rather than let prediction run on a
    # mis-shaped matrix.
    if matrix is not None:
        rows = np.asarray(matrix).shape[0]
        if rows != fields["n_output_channels"]:
            raise ValueError(
                f"Model checkpoint's channel_combination has {rows} rows but the model "
                f"has {fields['n_output_channels']} output channels; the band-mixing "
                "matrix is inconsistent with the model. Retrain so the checkpoint stores "
                "a channel_combination whose row count matches n_output_channels."
            )
    cfg.normalisation.channel_combination = matrix
    return True
