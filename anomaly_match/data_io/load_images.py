#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Functions for loading and processing images as a wrapper around fitsbolt."""

from __future__ import annotations

import os

import numpy as np
from astropy.io import fits as astropy_fits
from cutana.flux_conversion import convert_mosaic_to_flux
from dotmap import DotMap
from fitsbolt import batch_channel_combination
from fitsbolt.cfg.create_config import create_config as fb_create_cfg
from fitsbolt.image_loader import _process_image, load_and_process_images
from fitsbolt.normalisation.NormalisationMethod import NormalisationMethod
from loguru import logger
from PIL import Image as PILImage

# Exact PIL equivalents for skimage interpolation orders.
# Only orders with a true PIL match are included — others would silently
# produce different results than skimage, causing train/predict divergence.
_PIL_INTERPOLATION = {
    0: PILImage.NEAREST,
    1: PILImage.BILINEAR,
    3: PILImage.BICUBIC,
}


def apply_fits_flux_conversion(filepath: str, zeropoint_keyword: str = "MAGZERO") -> np.ndarray:
    """Read a FITS file and convert pixel values to flux density in Jansky.

    Uses the AB zeropoint from the FITS header and cutana's
    ``convert_mosaic_to_flux`` to ensure the training path applies the
    exact same conversion as the cutana prediction path.

    Args:
        filepath: Path to the FITS file.
        zeropoint_keyword: FITS header keyword for the AB zeropoint.

    Returns:
        np.ndarray: Flux-converted 2-D image in Jansky (float32).

    Raises:
        ValueError: If the primary HDU contains no image data.
    """
    with astropy_fits.open(filepath) as hdul:
        if hdul[0].data is None:
            raise ValueError(
                f"Primary HDU of '{filepath}' contains no image data. "
                f"Some observatories store image data in a different extension "
                f"(e.g. hdul[1]). This is not yet supported."
            )
        data = hdul[0].data.astype(np.float32)
        zeropoint = float(hdul[0].header[zeropoint_keyword])
    return convert_mosaic_to_flux(data, zeropoint)


def _pil_resize(
    image: np.ndarray, target_size: list[int], interpolation_order: int = 1
) -> np.ndarray:
    """Resize an HWC or HW numpy array using PIL (much faster than skimage).

    Args:
        image: numpy array of shape (H, W) or (H, W, C)
        target_size: [height, width]
        interpolation_order: 0=nearest, 1=bilinear, 3=bicubic

    Returns:
        Resized numpy array with same dtype.

    Raises:
        ValueError: If interpolation_order has no exact PIL equivalent.
    """
    h, w = target_size
    if image.shape[0] == h and image.shape[1] == w:
        return image
    if interpolation_order not in _PIL_INTERPOLATION:
        raise ValueError(
            f"interpolation_order={interpolation_order} has no exact PIL equivalent. "
            f"Supported orders for PIL resize: {sorted(_PIL_INTERPOLATION.keys())}. "
            f"Use one of these or switch to a normalisation method that uses skimage resize."
        )
    original_dtype = image.dtype
    resample = _PIL_INTERPOLATION[interpolation_order]
    # PIL doesn't support (H, W, 1) — squeeze singleton channel, restore after
    squeezed = image.ndim == 3 and image.shape[2] == 1
    if squeezed:
        image = image[:, :, 0]
    pil_img = PILImage.fromarray(image)
    # PIL.resize takes (width, height)
    pil_img = pil_img.resize((w, h), resample)
    result = np.array(pil_img)
    if squeezed:
        result = result[:, :, np.newaxis]
    if result.dtype != original_dtype:
        result = result.astype(original_dtype)
    return result


def get_fitsbolt_config(cfg: DotMap, size_override: list[int] | str | None = "default") -> DotMap:
    """Get the fitsbolt configuration from the provided configuration.

    Args:
        cfg: The configuration object.
        size_override: If "default", use cfg.normalisation.image_size.
                       If None, disable resizing (full resolution).
                       Otherwise use the provided value.

    Returns:
        DotMap: The cfg with a cfg.fitsbolt_cfg subdotmap.

    Raises:
        ValueError: If configuration does not include a normalisation config.
    """
    if not hasattr(cfg, "normalisation"):
        raise ValueError("Configuration must include a normalisation config for fitsbolt.")
    size = cfg.normalisation.image_size if size_override == "default" else size_override
    # Both this function and fitsbolt's load_and_process_images call
    # fb_create_cfg with the same params, so passing the pre-built cfg
    # is equivalent to passing individual params.
    fb_channel_combination = fitsbolt_channel_combination(cfg)
    # Adjust per-channel ASINH params to match n_output_channels — the UI
    # widget only emits values for the active method, so stale defaults
    # (e.g. 3-element list) can survive even after n_output_channels changes
    # (single-channel Cutana reduces, multi-band Cutana increases).
    n_out = cfg.normalisation.n_output_channels
    asinh_scale = cfg.normalisation.norm_asinh_scale
    asinh_clip = cfg.normalisation.norm_asinh_clip
    if isinstance(asinh_scale, (list, tuple)) and len(asinh_scale) != n_out:
        asinh_scale = _resize_param_list(asinh_scale, n_out)
    if isinstance(asinh_clip, (list, tuple)) and len(asinh_clip) != n_out:
        asinh_clip = _resize_param_list(asinh_clip, n_out)
    cfg.fitsbolt_cfg = fb_create_cfg(
        output_dtype=cfg.normalisation.output_dtype,
        size=size,
        fits_extension=cfg.normalisation.fits_extension,
        interpolation_order=cfg.normalisation.interpolation_order,
        n_output_channels=cfg.normalisation.n_output_channels,
        normalisation_method=cfg.normalisation.normalisation_method,
        channel_combination=fb_channel_combination,
        num_workers=max(cfg.num_workers, 1),
        norm_maximum_value=cfg.normalisation.norm_maximum_value,
        norm_minimum_value=cfg.normalisation.norm_minimum_value,
        norm_log_calculate_minimum_value=cfg.normalisation.norm_log_calculate_minimum_value,
        norm_crop_for_maximum_value=cfg.normalisation.norm_crop_for_maximum_value,
        norm_asinh_scale=asinh_scale,
        norm_asinh_clip=asinh_clip,
        norm_asinh_n_samples=cfg.normalisation.norm_asinh_n_samples,
        log_level="WARNING",
        force_dtype=True,
    )

    return cfg


def _resize_param_list(vals: list | tuple, n: int) -> list:
    """Trim or extend a per-channel parameter list to length *n*.

    When extending, the last value is repeated.

    Args:
        vals: Current parameter list.
        n: Target length.

    Returns:
        List of length *n*.
    """
    vals = list(vals)
    if len(vals) > n:
        return vals[:n]
    return vals + [vals[-1]] * (n - len(vals))


def normalise_channel_combination(
    channel_combination: np.ndarray | None,
) -> tuple[np.ndarray | None, list[int], bool]:
    """Rescale output rows (with non-negative weights) whose weights sum to more than 1.

    fitsbolt combines channels and then clips the result to the output dtype
    range (``np.clip(...).astype`` — see Lasloruhberg/fitsbolt#40).  A row of
    non-negative weights summing to more than 1 therefore pushes bright pixels
    past the range and **silently drops** that signal on uint8 output.  Scaling
    such a row to sum 1 turns it into a weighted average that can never exceed
    the input range, so no value is lost.

    Rows that already sum to ``<= 1`` are left untouched (identity, single-band
    broadcast, and any convex combination are unaffected).  Rows containing a
    negative weight are *not* rescaled — a scale can't bring them back into
    range — but their presence is reported so the caller can warn.

    Args:
        channel_combination: ``(n_out, n_in)`` matrix, or ``None``.

    Returns:
        Tuple ``(matrix, rescaled_rows, has_negative)``: ``matrix`` is the
        (possibly rescaled) array — a new array only when a row changed,
        otherwise the input unchanged — or ``None``; ``rescaled_rows`` lists the
        row indices that were rescaled; ``has_negative`` is ``True`` if any
        weight is negative.
    """
    if channel_combination is None:
        return None, [], False
    cc = np.asarray(channel_combination)
    work = cc.astype(np.float64, copy=False)
    has_negative = bool((work < 0).any())
    row_sums = work.sum(axis=1)
    rescaled_rows = [
        i for i in range(work.shape[0]) if row_sums[i] > 1.0 + 1e-9 and bool((work[i] >= 0).all())
    ]
    if not rescaled_rows:
        return cc, [], has_negative
    out_dtype = cc.dtype if np.issubdtype(cc.dtype, np.floating) else np.float32
    out = work.copy()
    for i in rescaled_rows:
        out[i] = out[i] / row_sums[i]
    return out.astype(out_dtype, copy=False), rescaled_rows, has_negative


def fitsbolt_applies_channel_combination(cfg: DotMap) -> bool:
    """Whether fitsbolt combined the bands in the file-decode path.

    Answers one narrow question: for an image fitsbolt has just decoded through
    :func:`load_and_process_wrapper` or
    :func:`load_and_process_single_wrapper`, did fitsbolt already apply the
    matrix?  It did whenever ``fits_extension`` is set, because
    :func:`get_fitsbolt_config` hands it one then; otherwise the matrix is
    applied afterwards, in :func:`finish_channel_combination`.

    **Only those two wrappers may consult this.** ``fits_extension`` also holds
    band *names* for Cutana catalogues, so this returns ``True`` for a Cutana or
    Zarr config even though the container decoders null the matrix out of their
    own fitsbolt config and combine it themselves — they call
    :func:`_apply_channel_combination` directly (see
    ``container_loaders.decode_zarr_image``) and deliberately bypass
    :func:`finish_channel_combination`. Routing them through it would take the
    "fitsbolt already combined" branch and drop the user's matrix entirely.

    Read the answer from here rather than from the fitsbolt config that came
    back: fitsbolt fills in a ``channel_combination`` of its own during
    ``load_and_process_images`` (``recompute_config_channel_combination``
    mutates the config in place), so by the time an image is decoded that field
    no longer says who is responsible for combining.

    Args:
        cfg: AnomalyMatch configuration.

    Returns:
        ``True`` when fitsbolt combined the bands during the file decode.

    Raises:
        TypeError: If ``fits_extension`` is neither ``None`` nor a form fitsbolt
            accepts. Guessing either way is wrong: ``True`` skips the post-hoc
            combine, ``False`` withholds the matrix from fitsbolt.
    """
    fits_extension = cfg.normalisation.fits_extension
    if fits_extension is None:
        return False
    if isinstance(fits_extension, (int, str, list, tuple)):
        return True
    raise TypeError(
        "normalisation.fits_extension must be None, an int, a str or a list of those, "
        f"got {type(fits_extension).__name__}"
    )


def blank_output_rows(channel_combination: np.ndarray | None) -> list[int]:
    """Find the output channels a ``channel_combination`` matrix blanks.

    AnomalyMatch reads a row whose weights are *all* zero as "leave this output
    channel empty" — the way a user drops a band they don't want the model to
    see without changing ``n_output_channels`` or retraining at a different
    channel count.

    A row that merely *sums* to zero is not blank: ``[1, -1]`` is the difference
    of two bands and produces a real output channel. Blanking it would replace
    the user's data with zeros, and — because the substitution in
    :func:`fitsbolt_safe_channel_combination` would also drop the negative
    weight before fitsbolt sees it — would do so instead of raising the error
    fitsbolt owes a FITS source with negative weights.

    The tolerance is per weight rather than on the row sum, so a row of
    negligible weights still counts as blank and never reaches fitsbolt's own
    zero-sum check.

    Args:
        channel_combination: ``(n_out, n_in)`` matrix, or ``None``.

    Returns:
        Sorted indices of the blank rows; empty when nothing is blanked.
    """
    if channel_combination is None:
        return []
    largest_weight = np.abs(np.asarray(channel_combination, dtype=np.float64)).max(axis=1)
    return [i for i in range(len(largest_weight)) if largest_weight[i] <= 1e-9]


def fitsbolt_safe_channel_combination(
    channel_combination: np.ndarray | None,
) -> np.ndarray | None:
    """Rewrite blank rows so fitsbolt's config validator accepts the matrix.

    fitsbolt rejects a zero-sum row outright (``must not sum to zero``) and
    re-validates on *every* ``load_and_process_images`` call, so a blanked
    channel cannot reach the FITS path as-is (#588).  Its combine step is a
    plain matmul that would handle the zero row correctly — only the validator
    stands in the way.  So each blank row is replaced by a copy of a surviving
    row, and :func:`zero_blank_output_channels` zeroes those channels again
    once fitsbolt is done.

    Duplicating a row that is already in the output leaves the *kept* channels
    bit-identical: the substituted channel adds no pixel value the stack didn't
    already contain, so the cross-channel minimum and maximum that
    CONVERSION_ONLY and ZSCALE clip against are unchanged, and LOG and ASINH
    normalise each channel independently anyway.

    Args:
        channel_combination: ``(n_out, n_in)`` matrix, or ``None``. Anything that
            is not a real array is treated as ``None`` — ``DotMap.copy()`` can
            turn a ``None`` config value into an empty ``DotMap()``.

    Returns:
        A validator-safe matrix of the same shape, the input unchanged when no
        row is blank, or ``None``.

    Raises:
        ValueError: If *every* row is blank — there is no surviving row to copy,
            and the result would be an entirely empty image rather than a
            deliberately blanked channel.
    """
    if not isinstance(channel_combination, (np.ndarray, list, tuple)):
        return None
    cc = np.asarray(channel_combination)
    blank_rows = blank_output_rows(cc)
    if not blank_rows:
        return cc
    surviving = [i for i in range(cc.shape[0]) if i not in set(blank_rows)]
    if not surviving:
        raise ValueError(
            "channel_combination blanks every output channel (every row has all-zero "
            "weights); the result would be an empty image. Give at least one output "
            "channel a non-zero weight."
        )
    safe = np.array(cc, dtype=np.result_type(cc.dtype, np.float32), copy=True)
    safe[blank_rows] = cc[surviving[0]]
    return safe


def fitsbolt_channel_combination(cfg: DotMap) -> np.ndarray | None:
    """The ``channel_combination`` to hand fitsbolt's ``create_config`` for *cfg*.

    fitsbolt applies the matrix to *every* input in 0.3.x (fitsbolt#39 no longer
    skips non-FITS arrays), so for non-FITS sources it gets ``None`` and AnomalyMatch
    applies the matrix post-hoc (:func:`finish_channel_combination` and the
    Cutana/Zarr decoders), which avoids a double application.  FITS inputs still go
    through fitsbolt, which combines bands within normalisation — but its validator
    rejects the all-zero rows AnomalyMatch uses to blank an output channel, so it gets
    the substituted matrix and :func:`finish_channel_combination` re-blanks afterwards
    (#588).

    :func:`get_fitsbolt_config` and ``validate_config`` both call this, so a config
    validates exactly when the matrix the run hands fitsbolt is acceptable.

    Args:
        cfg: AnomalyMatch configuration.

    Returns:
        The validator-safe matrix when fitsbolt applies it, otherwise ``None``.
    """
    if not fitsbolt_applies_channel_combination(cfg):
        return None
    return fitsbolt_safe_channel_combination(_get_channel_combination_array(cfg))


def fitsbolt_decode_config(cfg: DotMap) -> DotMap:
    """Per-decode copy of ``cfg.fitsbolt_cfg``, sized for what fitsbolt must return.

    When AnomalyMatch applies ``channel_combination`` itself (non-FITS sources —
    see :func:`fitsbolt_channel_combination`), fitsbolt must hand back the
    matrix's *input* channels so the post-hoc combine has something to map from.
    Asking it for ``n_output_channels`` instead makes it try to reduce, say,
    3 RGB channels to 1 with no matrix, which it refuses — so a 1- or 2-output
    matrix on an RGB image folder or Zarr store never decoded at all.

    A copy rather than an edit: ``cfg.fitsbolt_cfg.n_output_channels`` is saved
    into checkpoints as the model's channel count, and the shared config is read
    concurrently by preview and prediction threads.

    Args:
        cfg: AnomalyMatch configuration with ``fitsbolt_cfg`` attached.

    Returns:
        A fitsbolt config safe to mutate for a single decode.
    """
    fitsbolt_cfg = DotMap(cfg.fitsbolt_cfg.toDict(), _dynamic=False)
    if fitsbolt_applies_channel_combination(cfg):
        return fitsbolt_cfg
    channel_combination = _get_channel_combination_array(cfg)
    if channel_combination is None:
        return fitsbolt_cfg
    _request_input_channels(fitsbolt_cfg, cfg, channel_combination.shape[1])
    return fitsbolt_cfg


def _request_input_channels(fitsbolt_cfg: DotMap, cfg: DotMap, n_in: int) -> None:
    """Ask fitsbolt for *n_in* channels so AnomalyMatch can combine afterwards.

    The per-channel ASINH values are re-derived from ``cfg.normalisation``
    rather than from *fitsbolt_cfg*: :func:`get_fitsbolt_config` has already cut
    those lists to ``n_output_channels``, so resizing its copy would replace the
    user's per-band values with the first band's.

    Args:
        fitsbolt_cfg: Per-decode copy to edit in place.
        cfg: AnomalyMatch configuration holding the user's ASINH lists.
        n_in: Channels fitsbolt must return — the matrix's input count.
    """
    fitsbolt_cfg.n_output_channels = n_in
    for fb_key, cfg_key in (("asinh_scale", "norm_asinh_scale"), ("asinh_clip", "norm_asinh_clip")):
        values = cfg.normalisation[cfg_key]
        if isinstance(values, (list, tuple)):
            fitsbolt_cfg.normalisation[fb_key] = _resize_param_list(values, n_in)


def zero_blank_output_channels(image: np.ndarray, blank_rows: list[int]) -> np.ndarray:
    """Blank the output channels that :func:`fitsbolt_safe_channel_combination` filled in.

    Args:
        image: HWC image fitsbolt produced from the substituted matrix.
        blank_rows: Output channel indices to zero.

    Returns:
        The image with *blank_rows* zeroed. Modifies a copy, never the input.

    Raises:
        ValueError: If *image* has fewer channels than the matrix has rows — the
            decode produced a shape the matrix cannot describe, so zeroing by
            index would silently blank the wrong channel.
    """
    if not blank_rows:
        return image
    if image.ndim != 3 or image.shape[2] <= max(blank_rows):
        raise ValueError(
            f"Cannot blank output channel(s) {blank_rows} of an image with shape "
            f"{image.shape}; expected a HWC image with at least {max(blank_rows) + 1} "
            "channels. Decoded shape and channel_combination disagree."
        )
    blanked = image.copy()
    blanked[..., blank_rows] = 0
    return blanked


def finish_channel_combination(image: np.ndarray, cfg: DotMap) -> np.ndarray:
    """Complete channel combination for an image fitsbolt has just processed.

    Two paths meet here, and which one applies depends on whether fitsbolt
    already combined the bands (:func:`fitsbolt_applies_channel_combination`):

    * fitsbolt combined the bands (FITS inputs) — it ran the substituted matrix
      from :func:`fitsbolt_safe_channel_combination`, so the only work left is
      re-blanking the channels the user zeroed.
    * fitsbolt did not (``channel_combination=None`` for non-FITS inputs) — the
      full matrix is applied here as post-processing, the single application for
      those sources.

    For file decodes only. The Zarr and Cutana decoders combine their own output
    and must keep bypassing this — see
    :func:`fitsbolt_applies_channel_combination` for why routing them through it
    would drop the matrix.

    Args:
        image: HWC image returned by the fitsbolt pipeline.
        cfg: AnomalyMatch configuration holding ``normalisation.channel_combination``.

    Returns:
        The image with the configured channel combination fully applied.
    """
    channel_combination = _get_channel_combination_array(cfg)
    if channel_combination is None:
        return image
    if fitsbolt_applies_channel_combination(cfg):
        return zero_blank_output_channels(image, blank_output_rows(channel_combination))
    return _apply_channel_combination(image, channel_combination)


# Tracks channel-combination matrices we've already warned about, keyed by
# raw bytes.  _get_channel_combination_array runs once per decoded image on the
# prediction and training load paths, but the matrix is fixed for a run —
# without this guard a rescaled/clipping config would warn on every image.
_warned_channel_combinations: set[bytes] = set()


def _warn_channel_combination_once(
    channel_combination: np.ndarray,
    rescaled_rows: list[int],
    has_negative: bool,
) -> None:
    """Log each channel-combination advisory at most once per distinct matrix.

    Builds the advisories from the ``(rescaled_rows, has_negative)`` the caller
    already got from :func:`normalise_channel_combination` (so the prediction hot
    path normalises the matrix only once), plus the one fact those don't carry:
    rows of *channel_combination* whose weights are all zero (a blank output
    channel).
    We rescale rather than reject non-stochastic rows so the canonical VIS+NIR
    config — whose rows sum to 1.5 by design — keeps working, but the user should
    know their matrix was adjusted.  ``channel_combination`` is also the dedup key:
    the warning fires at most once per distinct matrix because the caller runs
    once per batch on the hot path.

    Args:
        channel_combination: The user's original matrix (pre-normalisation).
        rescaled_rows: Row indices rescaled by :func:`normalise_channel_combination`.
        has_negative: Whether the matrix contains a negative weight.
    """
    blank_rows = [i + 1 for i in blank_output_rows(channel_combination)]
    messages: list[str] = []
    if rescaled_rows:
        rows = ", ".join(str(i + 1) for i in rescaled_rows)
        messages.append(f"Row(s) {rows} sum to >1; rescaled to avoid clipping.")
    if blank_rows:
        rows = ", ".join(str(i) for i in blank_rows)
        messages.append(f"Row(s) {rows} have all-zero weights; that channel will be blank.")
    if has_negative:
        messages.append("Negative weights may clip to 0.")
    if not messages:
        return
    signature = np.ascontiguousarray(channel_combination).tobytes()
    if signature in _warned_channel_combinations:
        return
    _warned_channel_combinations.add(signature)
    for message in messages:
        logger.warning("channel_combination: {}", message)


def _get_channel_combination_array(cfg: DotMap) -> np.ndarray | None:
    """Extract channel_combination from config as a numpy array.

    Returns ``None`` when no channel combination is configured.  Handles
    the DotMap edge-case where ``None`` values become empty ``DotMap()``
    objects after a ``copy()`` call.  Weights are normalised via
    :func:`normalise_channel_combination` so every combine site (training,
    prediction, Cutana/Zarr/image paths) sees the same loss-free matrix.

    Args:
        cfg: Configuration with ``normalisation.channel_combination``.

    Returns:
        Numpy array of shape ``(n_out, n_in)``, or ``None``.
    """
    channel_combination = cfg.normalisation.channel_combination
    if isinstance(channel_combination, np.ndarray):
        cc = channel_combination
    elif isinstance(channel_combination, (list, tuple)):
        cc = np.asarray(channel_combination)
    else:
        return None
    normalised, rescaled_rows, has_negative = normalise_channel_combination(cc)
    _warn_channel_combination_once(cc, rescaled_rows, has_negative)
    return normalised


def _drop_unused_combination_columns(channel_combination: np.ndarray, n_bands: int) -> np.ndarray:
    """Drop all-zero columns from a ``channel_combination`` matrix to match a batch.

    The streaming extraction path skips reading FITS extensions whose
    ``channel_combination`` column is entirely zero — they contribute nothing to
    any output channel, so reading them is wasted I/O (each band is a separate
    windowed read).  The combine matmul then receives a batch with fewer bands
    than the full matrix has columns.  Removing the matching all-zero columns
    realigns the matrix to the bands actually present; the result is identical to
    keeping them, because a zero column contributes nothing.

    Args:
        channel_combination: ``(n_out, n_in_full)`` matrix.
        n_bands: Band count of the batch the matrix will be applied to.

    Returns:
        ``(n_out, n_bands)`` matrix with all-zero columns removed.

    Raises:
        ValueError: If dropping the all-zero columns does not yield exactly
            *n_bands* columns — extraction and combination then disagree on which
            bands are present for a reason other than unused-band pruning, which
            would feed the model the wrong data. Fail hard rather than combine
            misaligned bands.
    """
    keep = np.any(channel_combination != 0, axis=0)
    aligned = channel_combination[:, keep]
    if aligned.shape[1] != n_bands:
        raise ValueError(
            f"channel_combination has {channel_combination.shape[1]} columns; after "
            f"dropping {channel_combination.shape[1] - int(keep.sum())} all-zero "
            f"column(s), {aligned.shape[1]} remain but the batch has {n_bands} band(s). "
            "Extraction and channel combination disagree on which bands are present."
        )
    return aligned


def _apply_channel_combination(image: np.ndarray, channel_combination: np.ndarray) -> np.ndarray:
    """Apply a channel-combination matrix to an already-processed image.

    Delegates to fitsbolt's ``batch_channel_combination`` which expects a
    4-D batch, so we add/remove the batch axis around the call.

    Args:
        image: HWC numpy array with ``channel_combination.shape[1]``
            channels.
        channel_combination: Matrix of shape ``(n_out, n_in)`` where
            ``n_in == image.shape[2]``.

    Returns:
        Transformed HWC image with ``n_out`` channels.
    """
    if image.ndim != 3:
        return image
    result = batch_channel_combination(
        image[np.newaxis], channel_combination, output_dtype=image.dtype
    )
    return result[0]


def load_and_process_wrapper(
    filepaths: list[str],
    cfg: DotMap,
    desc: str = "Loading images",
    show_progress: bool = True,
) -> list[tuple[str, np.ndarray]]:
    """Load and process multiple image files using fitsbolt.

    Args:
        filepaths: List of image file paths to load.
        cfg: Configuration object with normalisation settings.
        desc: Description for the loading progress bar.
        show_progress: Whether to show a progress bar.

    Returns:
        List of (filepath, processed image) tuples.

    Raises:
        ValueError: If the number of loaded images does not match filepaths.
    """
    target_size = cfg.normalisation.image_size
    norm_method = cfg.normalisation.normalisation_method
    fits_exts = (".fits", ".fit", ".fts")
    do_flux = cfg.normalisation.apply_flux_conversion

    # Flux conversion needs a FITS MAGZERO header, so it only applies to
    # FITS inputs.  Non-FITS files (PNG/JPEG/...) always take the standard
    # path regardless of the flag.  Split the batch by which path each
    # file belongs to, then reassemble in original order at the end.
    flux_idx: list[int] = []
    standard_idx: list[int] = []
    for i, fp in enumerate(filepaths):
        is_fits = os.path.splitext(fp)[1].lower() in fits_exts
        (flux_idx if (do_flux and is_fits) else standard_idx).append(i)

    images: list[np.ndarray | None] = [None] * len(filepaths)

    if flux_idx:
        keyword = cfg.normalisation.flux_conversion_zeropoint_keyword
        cfg_full = get_fitsbolt_config(cfg)
        for i in flux_idx:
            converted = apply_fits_flux_conversion(filepaths[i], zeropoint_keyword=keyword)
            processed = process_single_wrapper(converted, cfg_full, desc=desc)
            images[i] = finish_channel_combination(processed, cfg)

    if standard_idx:
        # PIL resize is only safe for CONVERSION_ONLY: other normalisation
        # methods depend on the float64 dtype that fitsbolt's skimage
        # resize produces.
        use_pil_resize = norm_method == NormalisationMethod.CONVERSION_ONLY
        fb_size = None if use_pil_resize else target_size
        cfg_with_fb = get_fitsbolt_config(cfg, size_override=fb_size)
        standard_paths = [filepaths[i] for i in standard_idx]
        standard_images = load_and_process_images(
            standard_paths,
            cfg=fitsbolt_decode_config(cfg_with_fb),
            desc=desc,
            show_progress=show_progress,
        )
        if use_pil_resize and target_size is not None:
            interp = cfg.normalisation.interpolation_order
            standard_images = [_pil_resize(img, target_size, interp) for img in standard_images]

        standard_images = [finish_channel_combination(img, cfg) for img in standard_images]
        for i, img in zip(standard_idx, standard_images):
            images[i] = img

    if len(images) != len(filepaths) or any(img is None for img in images):
        raise ValueError(
            f"Mismatch between filepaths ({len(filepaths)}) and images ({len(images)})"
        )
    return [(fp, img) for fp, img in zip(filepaths, images)]


def load_and_process_single_wrapper(
    filepath: str,
    cfg: DotMap,
    desc: str = "image load and process",
    show_progress: bool = False,
    prediction: bool = False,
    size_override: list[int] | str | None = "default",
) -> np.ndarray:
    """Load and process a single image file.

    Creates a fitsbolt config if not part of current cfg.

    Args:
        filepath: The path to the image file.
        cfg: The configuration object of AnomalyMatch.
        desc: Description for the loading process. Defaults to "image load and process".
        show_progress: Whether to show progress. Defaults to False.
        prediction: Whether this is a prediction step. Defaults to False.
        size_override: If "default", use cfg.normalisation.image_size.
                       If None, disable resizing (full resolution).

    Returns:
        np.ndarray: The processed image in H,W,C format.
    """
    if not prediction:
        # cfg might have changed, update fitsbolt config
        cfg = get_fitsbolt_config(cfg, size_override=size_override)

    # cfg.fitsbolt_cfg is shared across calls (e.g. ImageCache re-loads per
    # thumbnail), so mutate a copy — forcing num_workers here must not pin it
    # to 1 for every other caller still holding this cfg.
    fitsbolt_cfg = fitsbolt_decode_config(cfg)
    fitsbolt_cfg.num_workers = 1
    image = load_and_process_images(
        filepath,
        cfg=fitsbolt_cfg,
        desc=desc,
        show_progress=show_progress,
    )

    return finish_channel_combination(image, cfg)


# in the future it might make sense to have a process wrapper, right now using legacy functionality
def process_single_wrapper(
    image: np.ndarray, cfg: DotMap, desc: str = "source", *, combine_post_hoc: bool = False
) -> np.ndarray:
    """Process a single image using fitsbolt.

    Args:
        image: Input image as numpy array
        cfg: Configuration object with fitsbolt_cfg set
        desc: Description for logging
        combine_post_hoc: Whether the caller applies ``channel_combination``
            itself afterwards (the Zarr decoder).  fitsbolt then gets no matrix
            and returns the image's own channels for the caller to map.  The
            Cutana decoder clears the matrix on its config instead, which takes
            the same path.

    Returns:
        Processed image

    Raises:
        ValueError: If fitsbolt_cfg is not properly set in cfg
    """
    # Validate fitsbolt_cfg exists and is valid
    # DotMap auto-creates empty DotMaps when accessing missing keys, so check for 'size' key
    if cfg.fitsbolt_cfg is None or (
        isinstance(cfg.fitsbolt_cfg, DotMap) and "size" not in cfg.fitsbolt_cfg
    ):
        raise ValueError(
            "fitsbolt_cfg is not set in configuration. "
            "Models must be saved with fitsbolt config for prediction. "
            "Please retrain and save the model to include fitsbolt config."
        )
    # A per-decode copy: the shared config is read concurrently by preview,
    # gallery and prediction threads, and its n_output_channels is saved into
    # checkpoints as the model's channel count.
    fitsbolt_cfg = DotMap(cfg.fitsbolt_cfg.toDict(), _dynamic=False)

    # Sanitize CC: DotMap.copy() can turn None into empty DotMap().
    # Also guard against fitsbolt's _dynamic=True auto-creating DotMap()
    # on attribute access.  Any CC that isn't a real array must be None.
    cc_val = fitsbolt_cfg.get("channel_combination")
    if combine_post_hoc or not isinstance(cc_val, (np.ndarray, list, tuple)):
        fitsbolt_cfg.channel_combination = None

    # processing requires n_expected channels from the input image
    n_channels = 1 if image.ndim == 2 else image.shape[-1]
    fitsbolt_cfg.n_expected_channels = n_channels
    # Without a matrix fitsbolt cannot turn n_channels into a different output
    # count; when the caller combines afterwards it must hand the channels back
    # unchanged.  Decided from the image in hand, not from fits_extension, which
    # holds band names for Cutana/Zarr configs (see
    # fitsbolt_applies_channel_combination).
    if fitsbolt_cfg.channel_combination is None and _get_channel_combination_array(cfg) is not None:
        _request_input_channels(fitsbolt_cfg, cfg, n_channels)
    fitsbolt_cfg.num_workers = 1  # for single image processing, force to 1
    return _process_image(image, fitsbolt_cfg, image_source=desc)


def detect_num_channels(root_dir: str, filenames: list[str]) -> int | None:
    """Detect the number of image channels from a sample image file.

    For FITS files, returns None since channel count depends on
    fits_extension and channel_combination settings.

    Args:
        root_dir: Directory containing the images.
        filenames: List of image filenames.

    Returns:
        Detected number of channels, or None if detection is not applicable.
    """
    if not filenames:
        return None

    sample_file = filenames[0]
    sample_path = os.path.join(root_dir, sample_file)
    ext = os.path.splitext(sample_file)[1].lower()

    # FITS channel count depends on extension/combination config
    if ext in (".fits", ".fit", ".fts"):
        return None

    try:
        img = PILImage.open(sample_path)
        num_channels = len(img.getbands())
        logger.debug(f"Detected {num_channels} channels from {sample_file}")
        return num_channels
    except Exception as e:
        logger.warning(f"Could not detect channels from {sample_file}: {e}")
        return None
