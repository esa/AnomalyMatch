#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Centralised normalisation parameter bounds and defaults.

Defines the valid ranges and default values for normalisation
parameters. Used by both the UI widget and ``validate_config`` so
that limits are enforced consistently whether the user configures
via the GUI or programmatically.
"""

from __future__ import annotations

# ── Image resolution ────────────────────────────────────────────────
RESOLUTION_MIN = 64
RESOLUTION_MAX = 384
RESOLUTION_DEFAULT = 150

# ── Interpolation order ────────────────────────────────────────────
INTERPOLATION_MIN = 0
INTERPOLATION_MAX = 5

# ── ASINH parameters ───────────────────────────────────────────────
ASINH_SCALE_MIN = 0.01
ASINH_SCALE_MAX = 100.0
ASINH_SCALE_DEFAULT = 0.7

ASINH_CLIP_MIN = 0.1
ASINH_CLIP_MAX = 100.0
ASINH_CLIP_DEFAULT = 99.8

# ── Channel combination matrix ─────────────────────────────────────
CHANNEL_COMBINATION_MIN = -10.0
CHANNEL_COMBINATION_MAX = 10.0

# ── Output channels ────────────────────────────────────────────────
N_OUTPUT_CHANNELS_MIN = 1
N_OUTPUT_CHANNELS_MAX = 32

# ── Cutana cutout padding factor ──────────────────────────────────
CUTOUT_PADDING_FACTOR_MIN = 0.25
CUTOUT_PADDING_FACTOR_MAX = 10.0
CUTOUT_PADDING_FACTOR_DEFAULT = 1.0

# ── Default extension labels ───────────────────────────────────────
DEFAULT_IMAGE_EXTENSIONS = ["R", "G", "B"]

# ── Extraction-affecting normalisation fields ──────────────────────
# These change the *raw* pixels Cutana extracts from the mosaic — the field of
# view, which bands are read, how they are interpolated — rather than how those
# pixels are normalised afterwards.  Raw cutouts cached under one set of values
# are not reusable under another, which is why they (and only they) form the
# labeled cache's extraction hash.  Anything not listed here can be re-applied
# to already-extracted raws in memory.
EXTRACTION_AFFECTING_NORM_FIELDS = frozenset(
    {
        "fits_extension",
        "cutout_padding_factor",
        "apply_flux_conversion",
        "interpolation_order",
    }
)


def validate_normalisation(norm_cfg: dict) -> list[str]:
    """Validate normalisation parameters against bounds.

    Works with both ``DotMap`` and plain dict, so it can be called
    from ``validate_config`` (full cfg) and from the UI widget
    (extracted dict).

    Args:
        norm_cfg: Dict-like with normalisation fields (``image_size``,
            ``interpolation_order``, ``n_output_channels``,
            ``norm_asinh_scale``, ``norm_asinh_clip``).

    Returns:
        List of error strings, empty if all valid.
    """
    errors: list[str] = []

    size = norm_cfg.get("image_size")
    if size is not None:
        h = size[0] if isinstance(size, (list, tuple)) else size
        if not (RESOLUTION_MIN <= h <= RESOLUTION_MAX):
            errors.append(f"image_size {h} outside [{RESOLUTION_MIN}, {RESOLUTION_MAX}]")

    interp = norm_cfg.get("interpolation_order")
    if interp is not None and not (INTERPOLATION_MIN <= interp <= INTERPOLATION_MAX):
        errors.append(
            f"interpolation_order {interp} outside [{INTERPOLATION_MIN}, {INTERPOLATION_MAX}]"
        )

    n_ch = norm_cfg.get("n_output_channels")
    if n_ch is not None and not (N_OUTPUT_CHANNELS_MIN <= n_ch <= N_OUTPUT_CHANNELS_MAX):
        errors.append(
            f"n_output_channels {n_ch} outside [{N_OUTPUT_CHANNELS_MIN}, {N_OUTPUT_CHANNELS_MAX}]"
        )

    pad = norm_cfg.get("cutout_padding_factor")
    if pad is not None and not (CUTOUT_PADDING_FACTOR_MIN <= pad <= CUTOUT_PADDING_FACTOR_MAX):
        errors.append(
            f"cutout_padding_factor {pad} outside "
            f"[{CUTOUT_PADDING_FACTOR_MIN}, {CUTOUT_PADDING_FACTOR_MAX}]"
        )

    for values, name, lo, hi in [
        (norm_cfg.get("norm_asinh_scale"), "norm_asinh_scale", ASINH_SCALE_MIN, ASINH_SCALE_MAX),
        (norm_cfg.get("norm_asinh_clip"), "norm_asinh_clip", ASINH_CLIP_MIN, ASINH_CLIP_MAX),
    ]:
        if values is not None:
            for i, v in enumerate(values):
                if not (lo <= v <= hi):
                    errors.append(f"{name}[{i}]={v} outside [{lo}, {hi}]")

    return errors
