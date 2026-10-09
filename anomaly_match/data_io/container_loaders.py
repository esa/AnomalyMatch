#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.

"""Decode functions for reading images from Zarr containers and Cutana caches.

These are shared between the training data sources and the prediction subprocess
scripts. Centralising them avoids duplication and ensures consistent behaviour.
"""

from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import numpy as np
from cutana.system_monitor import SystemMonitor
from dotmap import DotMap
from fitsbolt import batch_channel_combination
from loguru import logger
from threadpoolctl import threadpool_limits

from anomaly_match.data_io.load_images import (
    _apply_channel_combination,
    _drop_unused_combination_columns,
    _get_channel_combination_array,
    _resize_param_list,
    get_fitsbolt_config,
    process_single_wrapper,
)


@lru_cache(maxsize=1)
def _effective_cpu_count() -> int:
    """Pod's effective CPU count, queried once and memoised for the process.

    ``SystemMonitor`` reads the Kubernetes cgroup CPU quota, so this is 8 on an
    8-core pod even when the node reports 64 physical cores.  Constructing it
    primes psutil (a one-off ~0.1 s cost), and the pod limit can't change while
    we run, so the result is cached for the process lifetime.  Tests reset it
    with ``_effective_cpu_count.cache_clear()``.

    Returns:
        The Kubernetes-aware CPU count to size the decode pool against.
    """
    return SystemMonitor().get_effective_cpu_count()


def _decode_thread_count(n_images: int) -> int:
    """Return the decode thread-pool size for *n_images* cutouts.

    Decoding a cached cutout is dominated (~80% of wall time) by the resize
    anti-aliasing gaussian filter (scipy ``ndimage``), a C routine that releases
    the GIL, so a thread pool scales near-linearly up to the pod's cores (issue
    #500).  Capped at the pod's *effective* CPU count (the Kubernetes quota, not
    the node's physical cores) to avoid oversubscription — which *slows* the
    decode — and never larger than the batch itself.

    Args:
        n_images: Number of cutouts to decode in this batch.

    Returns:
        Thread count in ``[1, effective_cpu_count]``.
    """
    return max(1, min(_effective_cpu_count(), n_images))


def decode_zarr_image(image_data: np.ndarray, cfg: DotMap) -> np.ndarray:
    """Read and preprocess raw pixel data from an upstream Zarr store.

    Runs the full fitsbolt pipeline (normalisation + resize) on the raw
    array values, then applies ``channel_combination`` as post-processing.
    The fitsbolt config carries ``channel_combination=None`` for non-FITS
    sources (see :func:`get_fitsbolt_config`), so this post-hoc combine is
    the single application — never a duplicate of a fitsbolt-side one.

    Args:
        image_data: Raw pixel array from Zarr (CHW or HWC).
        cfg: Configuration with ``fitsbolt_cfg`` already attached.

    Returns:
        Preprocessed image as HWC numpy array with
        ``cfg.normalisation.n_output_channels`` channels.
    """
    try:
        if not isinstance(image_data, np.ndarray):
            image_data = np.array(image_data)

        # CHW → HWC when the leading axis is the short one.  Comparing against
        # n_output_channels missed channel-first stores whenever a matrix
        # changes the channel count (a 1x3 matrix on a (3, H, W) store).
        if image_data.ndim == 3 and image_data.shape[0] < image_data.shape[-1]:
            image = image_data.transpose(1, 2, 0)
        else:
            image = image_data

        # fitsbolt's single-image pipeline requires an explicit channel axis.
        if image.ndim == 2:
            image = image[..., np.newaxis]

        # fitsbolt 0.3.x would combine raw arrays too, so it decodes without the
        # matrix and we apply it below as the single application.
        image = process_single_wrapper(image, cfg, desc="zarr", combine_post_hoc=True)

        channel_combination = _get_channel_combination_array(cfg)
        if channel_combination is not None:
            image = _apply_channel_combination(image, channel_combination)

        return image

    except Exception as e:
        logger.error("Error processing image from Zarr: {}", e)
        raise


def decode_zarr_gallery_images(
    raw_images: dict[str, np.ndarray], cfg: DotMap
) -> dict[str, np.ndarray]:
    """Decode raw Zarr pixels for gallery thumbnails the way the model saw them.

    The gallery reads images straight from the store, so without this its
    thumbnails ignored normalisation and ``channel_combination`` — a 1-output
    matrix still showed colour images.

    Args:
        raw_images: ``{filename: raw store pixels}``.
        cfg: Session configuration; its ``fitsbolt_cfg`` is rebuilt from
            ``cfg.normalisation`` first, as the preview does.

    Returns:
        ``{filename: decoded HWC image}``; images that fail to decode are logged
        and left out, so one bad entry does not blank the whole page.
    """
    if not raw_images:
        return {}
    cfg = get_fitsbolt_config(cfg)
    decoded: dict[str, np.ndarray] = {}
    for filename, raw in raw_images.items():
        try:
            decoded[filename] = decode_zarr_image(raw, cfg)
        except Exception:
            # decode_zarr_image has already logged the error.
            continue
    return decoded


def _resolve_channel_combination(
    batch_data: np.ndarray,
    channel_combination: np.ndarray | None,
    n_output_channels: int,
) -> np.ndarray | None:
    """Resolve the channel-combination matrix to apply to an ``(N, H, W, C)`` batch.

    Returns *channel_combination* when one is configured. When it is ``None`` but
    the batch has a single band, falls back to a broadcast matrix that replicates
    that band to *n_output_channels* (the implicit "replicate VIS to RGB" intent
    for single-extension sources). Returns ``None`` when the batch should pass
    through unchanged.

    Pure helper: callers read ``channel_combination``/``n_output_channels`` from
    the config (or a snapshot of it) and pass them in, so the resolution doesn't
    re-read a config that a concurrent thread might be mutating.

    Args:
        batch_data: ``(N, H, W, C)`` stack the matrix will be applied to.
        channel_combination: The configured ``(n_output, C)`` matrix, or ``None``.
        n_output_channels: Target channel count for the single-band broadcast.

    Returns:
        An ``(n_output, C)`` matrix, or ``None`` for passthrough.
    """
    if channel_combination is not None:
        # Streaming extraction drops bands whose column is all-zero, so the batch
        # can have fewer bands than the full matrix has columns; realign by
        # dropping the matching all-zero columns (a no-op when they already match).
        # Both callers pass a 4-D (N, H, W, C) batch, so the band count is shape[-1].
        n_bands = batch_data.shape[-1]
        if channel_combination.shape[1] != n_bands:
            channel_combination = _drop_unused_combination_columns(channel_combination, n_bands)
        return channel_combination
    is_single_band = batch_data.ndim == 4 and batch_data.shape[-1] == 1
    if is_single_band and n_output_channels > 1:
        return np.ones((n_output_channels, 1), dtype=np.float32)
    return None


def apply_channel_combination_to_cutana_image(image_data: np.ndarray, cfg: DotMap) -> np.ndarray:
    """Combine Cutana per-band output to ``n_output_channels`` via fitsbolt.

    Used on Cutana orchestrator output in the unlabeled training stream
    and the prediction subprocess, where Cutana is configured with the
    user's normalisation settings and returns per-band *already
    normalised* data.  The labeled cache read path uses
    :func:`decode_cutana_raw_images` instead, because that path sees
    unnormalised float32 and needs the full fitsbolt pipeline.

    When no ``channel_combination`` is configured and the input has
    exactly one band, the single channel is broadcast to
    ``n_output_channels`` (the implicit "replicate VIS to RGB" intent for
    single-extension sources).  Otherwise the image passes through
    unchanged.

    Args:
        image_data: HWC array from Cutana with ``n_extensions`` channels.
        cfg: Configuration holding ``normalisation.channel_combination``
            and ``normalisation.n_output_channels``.

    Returns:
        HWC numpy array with ``n_output_channels`` channels, same dtype
        as the input (which is already ``cfg.normalisation.output_dtype``
        because the orchestrator is built with matching ``data_type``).
    """
    if not isinstance(image_data, np.ndarray):
        image_data = np.array(image_data)
    # A non-3-D image has no channel axis for the matmul and can't be reshaped
    # into the (N, H, W, C) batch the helper expects, so pass it straight through.
    # Cutana cutouts are always contiguous HWC arrays, so this guard is defensive.
    if image_data.ndim != 3:
        return image_data
    # The combination and single-band broadcast live only in the batched helper.
    return apply_channel_combination_to_cutana_batch(image_data[np.newaxis], cfg)[0]


def apply_channel_combination_to_cutana_batch(batch_data: np.ndarray, cfg: DotMap) -> np.ndarray:
    """Batched form of :func:`apply_channel_combination_to_cutana_image`.

    Applies ``channel_combination`` to a whole ``(N, H, W, n_extensions)``
    stack in a single :func:`batch_channel_combination` matmul instead of a
    per-image Python loop.  On the prediction hot path this collapses the
    dominant cost: a per-image loop over the channel combination was ~89% of
    end-to-end wall time for multi-band (VIS+NIR) scoring (issue #458).

    Output is identical to stacking the per-image results — the loop variant
    just paid Python-call and batch-axis-juggling overhead per cutout.

    Args:
        batch_data: ``(N, H, W, n_extensions)`` array from Cutana, already
            per-band normalised (Cutana runs with identity channel weights).
        cfg: Configuration holding ``normalisation.channel_combination`` and
            ``normalisation.n_output_channels``.

    Returns:
        ``(N, H, W, n_output_channels)`` array, same dtype as the input.
        Contiguity is established by the consumer at the point it is needed
        (``cutana_batch_to_model_tensor`` re-contiguifies after its NHWC→NCHW
        transpose, just before ``torch.from_numpy``), not here.
    """
    if not isinstance(batch_data, np.ndarray):
        batch_data = np.asarray(batch_data)

    channel_combination = _resolve_channel_combination(
        batch_data, _get_channel_combination_array(cfg), cfg.normalisation.n_output_channels
    )
    if channel_combination is None:
        return batch_data
    return batch_channel_combination(batch_data, channel_combination, output_dtype=batch_data.dtype)


def decode_cutana_raw_images(images: list[np.ndarray], cfg: DotMap) -> list[np.ndarray]:
    """Decode a batch of raw Cutana cache images in one fitsbolt pass.

    Each image is expected to be an HWC ndarray with ``n_extensions``
    bands on the last axis — the layout the labeled cache stores after
    Cutana extraction, so 3-band RGB, 4-band Euclid (VIS + NISP Y/J/H)
    and any other per-band count work identically.

    The labeled cache stores unnormalised float32 cutouts at a fixed
    resolution (see
    :data:`anomaly_match.data_io.labeled_data_cache.LABELED_CACHE_RESOLUTION`).
    This function applies the user's current
    ``normalisation_method``/``output_dtype``/``image_size``/
    ``channel_combination`` on the fly, so the cache stays valid across
    all of those changes.

    Versus decoding each item in its own call, three things are shared
    across the batch: the ``fitsbolt_cfg`` save/restore dance happens once
    (instead of N times), the per-cutout resize+normalise is fanned out
    across the pod's cores (the resize anti-aliasing releases the GIL —
    issue #500), and the final ``channel_combination`` matmul runs once on
    a stacked ``(N,H,W,C)`` array via :func:`batch_channel_combination` —
    a measurable win on 100+ preview thumbnails and a thousand-plus
    labeled cutouts.

    fitsbolt 0.3.x combines raw arrays too, so we pin
    ``n_output_channels``/``n_expected_channels`` to the per-band count and
    clear ``channel_combination`` on the fitsbolt config, then apply the
    user's combination matrix post-hoc — mirroring :func:`decode_zarr_image`.

    Args:
        images: Raw HWC float arrays from the cache with ``n_extensions``
            channels at the cached resolution.  May be empty.
        cfg: Configuration with ``fitsbolt_cfg`` already attached.

    Returns:
        List of HWC numpy arrays at ``cfg.normalisation.image_size`` with
        ``cfg.normalisation.n_output_channels`` channels and the
        configured output dtype, one per input.  Empty list if *images*
        is empty.

    Raises:
        ValueError: If *images* mix band counts (the batch must be
            homogeneous — see the single-pass channel-combination resolve).
    """
    if not images:
        return []

    first = images[0]
    n_extensions = first.shape[-1] if first.ndim == 3 else 1

    # The single-pass channel-combination resolve rests on every cutout having
    # the same band count (derived from ``images[0]`` alone).  Assert it once so
    # a mis-wired caller fails loudly here rather than silently combining with
    # the wrong matrix (fail-hard, per the repo's no-fallback stance).
    for img in images:
        bands = img.shape[-1] if img.ndim == 3 else 1
        if bands != n_extensions:
            raise ValueError(
                f"decode_cutana_raw_images requires a homogeneous batch; got mixed "
                f"band counts ({n_extensions} and {bands})"
            )

    # Snapshot a *private* config for the whole batch.  Preview decode runs in a
    # background thread; if the user edits the normalisation widget mid-decode,
    # the UI thread replaces ``cfg.fitsbolt_cfg``, and ``process_single_wrapper``
    # re-reads it for every cutout — so later cutouts would be processed with a
    # different ``n_output_channels``, yielding a mix of e.g. 4- and 3-channel
    # arrays that ``np.stack`` rejects.
    #
    # ``fitsbolt_cfg`` must therefore be a fully independent copy.  We
    # ``copy.deepcopy`` it explicitly rather than trust ``cfg.copy()``:
    # ``DotMap.copy()``'s isolation is *type-dependent* — it clones nested DotMaps
    # and list leaves but SHARES numpy-array leaves (and has the known
    # ``None -> DotMap()`` quirk, see CLAUDE.md).  ``channel_combination`` is often
    # an ndarray, so relying on ``cfg.copy()`` here would only be safe by accident
    # (because today's pins happen to be key reassignments, not in-place leaf
    # mutations).  ``copy.deepcopy`` is type-independent, so the per-band pins below
    # can never leak into the caller's cfg and a concurrent UI swap of
    # ``cfg.fitsbolt_cfg`` can't change what this batch decodes with.  A newer edit
    # simply spawns a fresh, superseding preview job.
    batch_cfg = cfg.copy()
    batch_cfg.fitsbolt_cfg = copy.deepcopy(cfg.fitsbolt_cfg)
    fb_cfg = batch_cfg.fitsbolt_cfg

    # Read the user's requested combine target once, up front, before pinning the
    # per-band decode layout below.  These are the post-hoc combine target (the
    # user's ``normalisation`` settings) — distinct from the per-band layout pinned
    # onto ``fb_cfg`` — and reading them once into locals snapshots the values so
    # the two reads can't be split by a concurrent widget edit.
    user_channel_combination = _get_channel_combination_array(batch_cfg)
    user_n_output = batch_cfg.normalisation.n_output_channels

    # Pin the per-band layout for the shared decode pass; channel combination is
    # applied post-hoc below (fitsbolt only wires it through for FITS inputs).
    # Resizing the per-channel asinh params to the cached band count keeps
    # fitsbolt's "length N, expected M. Will use first element" warning quiet.
    fb_cfg.channel_combination = None
    fb_cfg.n_output_channels = n_extensions
    for attr in ("asinh_scale", "asinh_clip"):
        saved = fb_cfg.normalisation.get(attr)
        if isinstance(saved, (list, tuple)) and len(saved) != n_extensions:
            fb_cfg.normalisation[attr] = _resize_param_list(saved, n_extensions)

    # Resolve the channel-combination matrix once.  It depends only on the band
    # count (uniform across this homogeneous batch) and the user's snapshotted
    # settings — never on pixel data — so every slice applies the *same* matrix.
    # That lets each worker stack+combine its own slice in parallel rather than
    # leaving one big serial ``np.stack`` + matmul tail on the main thread.  We
    # resolve from a band-count placeholder (not the real stack, which doesn't
    # exist yet) and from the values snapshotted before the per-band pins above,
    # so a concurrent widget edit can't desync the combine from the decode pass.
    channel_combination = _resolve_channel_combination(
        np.empty((1, 1, 1, n_extensions)), user_channel_combination, user_n_output
    )

    def _decode_slice(slice_images):
        """Decode one contiguous slice end-to-end (resize + normalise + combine).

        Runs entirely inside a worker thread so the GIL-releasing resize and the
        numpy channel-combine matmul of different slices overlap.

        Args:
            slice_images: Contiguous sub-list of the batch's raw HWC cutouts.

        Returns:
            List of decoded HWC arrays for this slice, in input order.
        """
        processed = [
            process_single_wrapper(img, batch_cfg, desc="cutana_cache") for img in slice_images
        ]
        stack = np.stack(processed, axis=0)
        if channel_combination is not None:
            # Pin output dtype to the input's: fitsbolt defaults to float32,
            # which silently promoted the uint8 labeled cache to float32 and
            # crashed training on ``Image.fromarray`` (PIL has no 3-channel
            # float32 mode).  Bug was introduced alongside the batched
            # decoder (#407 review) because the previous per-image call
            # already passed ``output_dtype=image.dtype``.
            stack = batch_channel_combination(stack, channel_combination, output_dtype=stack.dtype)
        return list(stack)

    # The per-cutout decode is independent across images and dominated by the
    # GIL-releasing resize, so fan it out across the pod's cores.
    # ``process_single_wrapper`` decodes with its own copy of ``fb_cfg``, so the
    # threads share it read-only.  Output is bit-identical to the serial path
    # (verified across 1/2/4/8 workers, issue #500).
    workers = _decode_thread_count(len(images))
    if workers <= 1:
        return _decode_slice(images)

    # Split into ``workers`` *contiguous* slices so each thread runs a tight
    # decode loop.  Coarse slicing beats a per-image ``pool.map`` here: it keeps
    # the GIL-releasing resize overlapping across threads instead of paying
    # Python dispatch + GIL-handoff overhead on every one of the (thousand-plus)
    # cutouts.  ``pool.map`` preserves slice order and each slice preserves item
    # order, so the flattened result matches the input order.
    size = (len(images) + workers - 1) // workers
    slices = [images[i : i + size] for i in range(0, len(images), size)]
    # Pin numpy's native pools to one thread for the fan-out so the ``workers``
    # decode threads don't each spawn a full MKL pool and oversubscribe the pod's
    # cores — which made an 8-worker decode *slower* than a 4-worker one under
    # node load (issue #500).  ``threadpool_limits`` is entered once here, on the
    # main thread, *not* per worker: it mutates global native-library state and
    # is not safe to enter concurrently from many threads (doing so deadlocks —
    # an 8-worker / 400-cutout decode hung indefinitely; #503 review follow-up).
    # A main-thread entry pins MKL globally, which covers what this path actually
    # uses — the channel-combine matmul (BLAS) and the GIL-releasing scipy
    # ``ndimage`` resize, which is single-threaded and spawns no OpenMP pool. It
    # does *not* set each worker's per-thread OpenMP ICV (that is the calling
    # thread's), but no OpenMP-parallel routine runs in the decode, so nothing
    # oversubscribes in practice.
    with threadpool_limits(limits=1), ThreadPoolExecutor(max_workers=workers) as pool:
        return [img for sub in pool.map(_decode_slice, slices) for img in sub]
