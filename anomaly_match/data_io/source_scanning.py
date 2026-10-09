#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.

"""Source scanning utilities for counting and previewing data sources.

Provides backend functions for detecting source types, counting items, reading
label maps, and loading preview samples.  These are pure data I/O operations
that the UI layer delegates to via ``BackendInterface``.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import zarr
from dotmap import DotMap
from fitsbolt import SUPPORTED_IMAGE_EXTENSIONS
from loguru import logger

from anomaly_match.data_io.container_loaders import decode_cutana_raw_images, decode_zarr_image
from anomaly_match.data_io.find_images_in_folder import get_image_names_from_folder
from anomaly_match.data_io.labeled_data_cache import LabeledDataCache
from anomaly_match.data_io.load_images import (
    detect_num_channels,
    get_fitsbolt_config,
    load_and_process_single_wrapper,
)
from anomaly_match.datasets.cutana_source import _is_cutana_catalogue
from anomaly_match.datasets.training_data_source import DataSourceType, _auto_detect_source_type
from anomaly_match.image_processing.image_utils import ensure_uint8_hwc
from anomaly_match.prediction.cutana_loader import load_batch_cutouts
from anomaly_match.prediction.zarr_filenames import derive_filenames
from anomaly_match.utils.cutana_stream_utils import cutana_validate_files_and_count_sources


@dataclass
class SourceScanResult:
    """Result of scanning a source directory.

    Attributes:
        source_type: Detected source type.
        count: Number of sources found.
        image_files: File paths (only populated for image_folder sources).
    """

    source_type: DataSourceType = DataSourceType.IMAGE_FOLDER
    count: int = 0
    image_files: list[str] = field(default_factory=list)


def scan_and_count_sources(folder: str) -> SourceScanResult:
    """Detect source type and count items in a folder.

    Handles image folders, Zarr stores, and Cutana catalogues.

    Args:
        folder: Path to the data directory.

    Returns:
        A :class:`SourceScanResult` with source type, count, and file list.
    """
    source_type = _auto_detect_source_type(folder)

    if source_type == DataSourceType.ZARR:
        count = _count_zarr(folder)
        return SourceScanResult(source_type=source_type, count=count)

    if source_type == DataSourceType.CUTANA:
        count = _count_cutana(folder)
        return SourceScanResult(source_type=source_type, count=count)

    # Image folder
    image_files = get_image_names_from_folder(folder, recursive=True)
    return SourceScanResult(
        source_type=source_type,
        count=len(image_files),
        image_files=image_files,
    )


def _count_zarr(folder: str) -> int:
    """Count images across Zarr stores in *folder*.

    Returns:
        Total number of images found.
    """
    zarr_paths: list[str] = []
    if folder.rstrip(os.sep + "/").lower().endswith(".zarr"):
        zarr_paths = [folder]
    else:
        for entry in os.listdir(folder):
            if entry.lower().endswith(".zarr"):
                zarr_paths.append(os.path.join(folder, entry))

    count = 0
    for zp in zarr_paths:
        root = zarr.open_group(str(zp), mode="r")
        if "images" in root:
            count += root["images"].shape[0]
    return count


def _count_cutana(folder: str) -> int:
    """Count sources in Cutana catalogue files within *folder*.

    Returns:
        Total number of sources found.
    """
    count = 0
    for fname in os.listdir(folder):
        fpath = os.path.join(folder, fname)
        if not fname.endswith((".parquet", ".csv")):
            continue
        if not _is_cutana_catalogue(Path(fpath)):
            continue
        if fname.endswith(".parquet"):
            count += pq.read_metadata(fpath).num_rows
        else:
            with open(fpath) as f:
                count += sum(1 for _ in f) - 1
    return count


def is_zarr_store(path: str) -> bool:
    """Return whether *path* is a Zarr store, by name or on-disk marker.

    A Zarr v3 store is not required to have a ``.zarr`` name, so a top-level
    ``zarr.json`` is also accepted.

    Args:
        path: Directory path to check.

    Returns:
        True if *path* looks like a Zarr store.
    """
    return path.lower().endswith(".zarr") or os.path.exists(os.path.join(path, "zarr.json"))


def find_prediction_zarr_stores(folder: str) -> list[tuple[str, str]]:
    """Find Zarr store paths for prediction within *folder*.

    Treats *folder* itself as a single store when its name/contents mark it
    as one; otherwise looks one level down for `*.zarr` stores or
    `<batch>/images.zarr` chunks (the layout chunked prediction output uses).

    Args:
        folder: Directory to scan, or a direct path to a single store.

    Returns:
        List of (display_name, store_path) pairs.
    """
    folder = folder.rstrip(os.sep + "/")
    if is_zarr_store(folder):
        return [(os.path.basename(folder), folder)]

    stores: list[tuple[str, str]] = []
    for name in os.listdir(folder):
        full = os.path.join(folder, name)
        if is_zarr_store(full):
            stores.append((name, full))
        elif os.path.isdir(full) and os.path.exists(os.path.join(full, "images.zarr")):
            if os.path.exists(os.path.join(full, LabeledDataCache.CACHE_INFO_JSON)):
                continue
            stores.append((name, os.path.join(full, "images.zarr")))
    return stores


def detect_source_channel_count(
    folder: str, source_type: DataSourceType, image_files: list[str] | None = None
) -> int | None:
    """Return how many channels each image in a Zarr store or image folder has.

    The setup screen sizes the channel-combination matrix's input columns from
    this, so a matrix left over from a previous source (e.g. four Cutana bands)
    can't be applied to a three-channel store.

    Args:
        folder: Source directory, or a direct path to a single Zarr store.
        source_type: :attr:`DataSourceType.ZARR` or
            :attr:`DataSourceType.IMAGE_FOLDER`; other types return ``None``.
        image_files: Image filenames for image-folder sources (the first is
            sampled).

    Returns:
        The per-image channel count, or ``None`` when it cannot be read
        (e.g. FITS files, whose channels depend on ``fits_extension``).
    """
    if source_type == DataSourceType.IMAGE_FOLDER:
        return detect_num_channels(folder, image_files or [])
    if source_type != DataSourceType.ZARR:
        return None
    for _name, store_path in find_prediction_zarr_stores(folder):
        root = zarr.open_group(store_path, mode="r")
        if "images" not in root:
            continue
        shape = root["images"].shape
        if len(shape) == 3:  # (N, H, W): no channel axis
            return 1
        if len(shape) == 4:
            # Channel-first stores keep the short axis first, as decode_zarr_image assumes.
            return shape[1] if shape[1] < shape[-1] else shape[-1]
    return None


def detect_prediction_source_type(search_dir: str) -> DataSourceType:
    """Detect the dominant prediction source type for *search_dir*.

    Checks whether *search_dir* itself is a single Zarr store first. Otherwise
    scans immediate directory entries by precedence: any Zarr store wins,
    then any file passing `_is_cutana_catalogue` wins over image files; a
    folder with only non-catalogue CSV/parquet files falls back to CUTANA.

    Args:
        search_dir: Directory to scan, or a direct path to a single store.

    Returns:
        The detected `DataSourceType`. Defaults to `IMAGE_FOLDER` when
        *search_dir* doesn't exist or nothing recognizable is found.
    """
    # Strip trailing separators so ".zarr/" still matches ".zarr"
    lower = search_dir.rstrip(os.sep + "/").lower()
    if lower.endswith(".zarr"):
        return DataSourceType.ZARR

    if not search_dir or not os.path.exists(search_dir):
        logger.warning("Prediction search directory does not exist: {}", search_dir)
        return DataSourceType.IMAGE_FOLDER

    if find_prediction_zarr_stores(search_dir):
        return DataSourceType.ZARR

    has_image_file = False
    has_catalogue_ext = False
    for filename in os.listdir(search_dir):
        file_path = os.path.join(search_dir, filename)
        if not os.path.isfile(file_path):
            continue
        if filename.lower().endswith((".csv", ".parquet")):
            has_catalogue_ext = True
            if _is_cutana_catalogue(Path(file_path)):
                return DataSourceType.CUTANA
        elif os.path.splitext(filename.lower())[1] in SUPPORTED_IMAGE_EXTENSIONS:
            has_image_file = True

    if has_image_file:
        return DataSourceType.IMAGE_FOLDER
    return DataSourceType.CUTANA if has_catalogue_ext else DataSourceType.IMAGE_FOLDER


def detect_and_count_prediction_sources(folder: str) -> tuple[DataSourceType, int]:
    """Detect the prediction source type in *folder* and count its items.

    Args:
        folder: Directory to scan, or a direct path to a single Zarr store.

    Returns:
        Tuple of (detected type, total item count).
    """
    file_type = detect_prediction_source_type(folder)

    if file_type == DataSourceType.IMAGE_FOLDER:
        return DataSourceType.IMAGE_FOLDER, len(get_image_names_from_folder(folder, recursive=True))

    if file_type == DataSourceType.CUTANA:
        candidates = (
            Path(folder) / f for f in os.listdir(folder) if f.lower().endswith((".csv", ".parquet"))
        )
        # Only count files that actually look like Cutana catalogues — a
        # label CSV or other sidecar sitting next to the real catalogue
        # must not inflate the source count.
        catalogue_files = [f for f in candidates if _is_cutana_catalogue(f)]
        _valid, total_sources, _chunks = cutana_validate_files_and_count_sources(catalogue_files)
        return DataSourceType.CUTANA, total_sources

    total = 0
    for _name, store_path in find_prediction_zarr_stores(folder):
        try:
            root = zarr.open_group(store_path, mode="r")
            if "images" in root:
                total += root["images"].shape[0]
        except Exception as exc:
            logger.warning("Failed to count zarr images in {}: {}", store_path, exc)
    return DataSourceType.ZARR, total


def read_label_map(label_path: str) -> dict[str, str]:
    """Read a label CSV and return an ``{id: label}`` mapping.

    Args:
        label_path: Path to an existing label CSV file.

    Returns:
        Mapping from sample id to label string.

    Raises:
        FileNotFoundError: If ``label_path`` is empty or does not refer to an
            existing file. Callers must guard against the "no label selected"
            UI state before invoking this function.
        ValueError: If the CSV is missing the required ``id``/``label`` columns
            or contains rows with null values in either column.
    """
    if not label_path or not os.path.isfile(label_path):
        raise FileNotFoundError(f"Label file not found: {label_path!r}")
    ldf = pd.read_csv(label_path)
    missing = {"id", "label"} - set(ldf.columns)
    if missing:
        raise ValueError(f"Label file {label_path!r} missing required column(s): {sorted(missing)}")
    if ldf["id"].isna().any() or ldf["label"].isna().any():
        raise ValueError(f"Label file {label_path!r} contains rows with null id or label")
    return {str(k): v for k, v in zip(ldf["id"], ldf["label"])}


def load_preview_samples(
    cfg: DotMap,
    folder: str,
    source_type: DataSourceType | str,
    *,
    image_files: list[str] | None = None,
    source_ids: list[str] | None = None,
    max_items: int = 9,
    offset: int = 0,
    stale_check=None,
) -> list[tuple[str, np.ndarray]]:
    """Load sample images for preview display.

    Returns raw numpy arrays (HWC uint8) so the UI can convert to display
    format independently.

    Args:
        cfg: AnomalyMatch configuration with normalisation settings.
        folder: Path to the data directory.
        source_type: Detected source type.
        image_files: Files to load for image_folder sources.
        source_ids: Source IDs to load for Cutana or Zarr sources.
        max_items: Maximum number of preview samples to return.
        offset: Skip this many items from the start before loading
            ``max_items``.  Used by paged preview galleries (#427) so
            repeated calls with rising offsets walk the full candidate
            list one page at a time.
        stale_check: Optional zero-arg callable returning ``True`` when
            this preview request has been superseded.  Forwarded to the
            Cutana fallback loop so stale requests bail out quickly
            instead of scanning every catalogue.

    Returns:
        List of ``(name, image_array)`` tuples.
    """
    source_type = DataSourceType(source_type) if isinstance(source_type, str) else source_type

    # Always rebuild fitsbolt_cfg from cfg.normalisation before preview
    # decode.  The Cutana path reads cfg.fitsbolt_cfg directly (via
    # decode_cutana_raw_images → process_single_wrapper) and would
    # otherwise serve a stale normalisation_method/image_size whenever
    # the user changed those widgets — only ``channel_combination`` is
    # re-read from cfg.normalisation post-hoc.  Cheap (small dict
    # construction) and harmless for the image-folder path that
    # already rebuilds inside its loop.
    cfg = get_fitsbolt_config(cfg)

    if source_type == DataSourceType.IMAGE_FOLDER:
        files = (image_files or [])[offset : offset + max_items]
        return _preview_image_folder(cfg, folder, files, max_items)
    if source_type == DataSourceType.CUTANA:
        ids = (source_ids or [])[offset : offset + max_items]
        return _preview_cutana(cfg, folder, ids, max_items, stale_check=stale_check)
    if source_type == DataSourceType.ZARR:
        if source_ids is None:
            return _preview_zarr(cfg, folder, max_items, offset=offset)
        ids = source_ids[offset : offset + max_items]
        return _preview_zarr(cfg, folder, max_items, source_ids=ids)
    return []


def _preview_image_folder(
    cfg: DotMap, folder: str, image_files: list[str], max_items: int
) -> list[tuple[str, np.ndarray]]:
    """Load preview images from an image folder.

    Returns:
        List of ``(filename, image_array)`` tuples.
    """
    items: list[tuple[str, np.ndarray]] = []
    sample = image_files[:max_items]
    for fname in sample:
        filepath = fname if os.path.isabs(fname) else os.path.join(folder, fname)
        try:
            img = load_and_process_single_wrapper(
                filepath, cfg, desc="preview", show_progress=False
            )
            if img.ndim == 3 and img.shape[0] <= 4:
                img = img.transpose(1, 2, 0)
            items.append((os.path.basename(fname), img))
        except Exception as exc:
            logger.warning("Preview load failed for {}: {}", fname, exc)
    return items


def _preview_cutana(
    cfg: DotMap,
    folder: str,
    source_ids: list[str],
    max_items: int,
    *,
    stale_check=None,
) -> list[tuple[str, np.ndarray]]:
    """Load preview cutouts from the labeled cache, falling back to Cutana.

    Thin wrapper over :func:`load_cutana_preview` for the generic
    :func:`load_preview_samples` dispatch, which only needs the decoded images.

    Args:
        cfg: AnomalyMatch configuration.
        folder: Source catalogue directory (fallback search root).
        source_ids: Preview ids, already stratified by label.
        max_items: Cap on preview grid size.
        stale_check: Optional zero-arg callable returning ``True`` when
            this preview request has been superseded.

    Returns:
        List of ``(source_id, image_array)`` tuples, preserving caller order.
    """
    decoded, _ = load_cutana_preview(cfg, folder, source_ids, max_items, stale_check=stale_check)
    return decoded


def load_cutana_preview(
    cfg: DotMap,
    folder: str,
    source_ids: list[str],
    max_items: int,
    *,
    stale_check=None,
) -> tuple[list[tuple[str, np.ndarray]], list[tuple[str, np.ndarray]] | None]:
    """Load Cutana preview cutouts and the raw arrays worth holding in memory.

    Reads the labeled cache's raw (normalisation-independent) float32 cutouts
    *once* and decodes them in-process; for ids the cache can't serve it falls
    back to Cutana extraction (which bakes the current normalisation in, so
    those cutouts are not re-decodable).  Returning the raw arrays alongside the
    decoded images lets the setup screen hold them and re-decode on a
    normalisation change without touching the cache again (issue #501) — in a
    single read, rather than one read to hold and another to display.

    Args:
        cfg: AnomalyMatch configuration.
        folder: Source catalogue directory (fallback search root).
        source_ids: Preview ids, already stratified by label.
        max_items: Cap on preview grid size.
        stale_check: Optional zero-arg callable returning ``True`` when this
            preview request has been superseded; forwarded to the Cutana
            fallback so it bails out between catalogues.

    Returns:
        ``(decoded, holdable_raws)`` where *decoded* is the list of
        ``(source_id, uint8_hwc)`` in caller order, and *holdable_raws* is the
        ``(source_id, raw_float32)`` list when the cache fully covered the
        request (safe to hold + re-decode) or ``None`` otherwise (a partial /
        fallback preview has cutouts with normalisation already baked in).
    """
    preview_ids = [str(s) for s in source_ids[:max_items]]
    if not preview_ids:
        return [], None

    # Rebuild fitsbolt_cfg from the current cfg.normalisation once, up front, so
    # every downstream path sees it regardless of which one runs: the cache
    # re-decode (decode_raw_cutouts) and the catalogue-extraction fallback
    # (load_batch_cutouts reads cfg.fitsbolt_cfg directly).  Doing it here rather
    # than only on the cache-miss branch removes the easy-to-miss footgun where a
    # future caller path skips the rebuild.
    cfg = get_fitsbolt_config(cfg)

    # One raw read of whatever the cache can serve.
    raw_items = load_preview_raw_cutouts(cfg, preview_ids, max_items)
    decoded: list[tuple[str, np.ndarray]] = []
    if raw_items:
        try:
            decoded = decode_raw_cutouts(cfg, raw_items)
        except Exception as exc:
            # Degrade gracefully (the preview falls back / shows fewer items)
            # but surface the full traceback — a bare message hides where the
            # decode actually failed.
            logger.opt(exception=True).warning(
                "Preview batch decode failed for {} cached id(s): {}",
                len(raw_items),
                exc,
            )

    cached_ids = {source_id for source_id, _ in decoded}
    missing = [source_id for source_id in preview_ids if source_id not in cached_ids]
    if not missing:
        # Fully cache-backed: the raws are safe to hold and re-decode later.
        return decoded, raw_items

    # Fall back to catalogue extraction for ids the cache can't serve (cache
    # absent, unpopulated, or a partial rebuild).  That path bakes the current
    # normalisation into the cutouts, so the result is *not* holdable.
    # `stale_check` is threaded through so the fallback bails out between
    # catalogues when a newer request supersedes.
    batch = load_batch_cutouts(cfg, missing, search_dir=folder, stale_check=stale_check)
    if batch.partial:
        logger.warning(
            "Cutana preview fallback returned partial batch ({} of {} ids)",
            len(batch.cutouts),
            len(missing),
        )
    # Preserve caller order (stratified by label) rather than catalogue
    # discovery order — otherwise anomalies from later catalogue files would
    # always end up at the end of the preview grid.
    decoded.extend(
        (source_id, ensure_uint8_hwc(batch.cutouts[source_id]))
        for source_id in missing
        if source_id in batch.cutouts
    )
    return decoded, None


def load_preview_raw_cutouts(
    cfg: DotMap,
    source_ids: list[str],
    max_items: int,
) -> list[tuple[str, np.ndarray]]:
    """Return normalisation-independent raw cutouts from the labeled cache.

    The labeled cache stores per-band float32 cutouts at the cache resolution,
    *before* normalisation/resize/channel-combine.  Returning the raw arrays
    lets the setup-screen preview re-apply a changed normalisation in-process
    (issue #501) without re-reading the catalogue or FITS tiles — the read of
    these arrays is identical regardless of the user's normalisation settings.

    Ids the populated cache doesn't hold are omitted; the caller decides whether
    to fall back to catalogue extraction for them (that path bakes normalisation
    in, so its cutouts are *not* raw and can't be re-decoded later).

    Args:
        cfg: AnomalyMatch configuration; ``cfg.labeled_cache_path`` locates the
            cache.
        source_ids: Candidate ids, already stratified by label.
        max_items: Cap on the number of cutouts to read.

    Returns:
        ``list[(source_id, raw_float32_hwc)]`` for the cache-backed ids in caller
        order, or ``[]`` when no cache is configured/populated or it needs a
        rebuild.
    """
    preview_ids = [str(s) for s in source_ids[:max_items]]
    cache_path = cfg.labeled_cache_path
    if not cache_path:
        return []
    cache = LabeledDataCache(Path(cache_path))
    if not cache.is_populated or cache.needs_rebuild(cfg):
        return []

    cached_ids = set(cache.get_label_df()["id"].astype(str))
    in_cache = [source_id for source_id in preview_ids if source_id in cached_ids]
    if not in_cache:
        return []
    return cache.get_raw_images_by_id(in_cache)


def decode_raw_cutouts(
    cfg: DotMap, raw_items: list[tuple[str, np.ndarray]]
) -> list[tuple[str, np.ndarray]]:
    """Apply the current normalisation to raw cache cutouts.

    Rebuilds ``cfg.fitsbolt_cfg`` from ``cfg.normalisation`` and runs the full
    decode (normalise + resize + channel-combine) on the raw float32 arrays,
    returning display-ready HWC uint8.  This is the in-process re-decode the
    setup-screen preview uses to reflect a normalisation change without touching
    the cache again (issue #501).

    Args:
        cfg: AnomalyMatch configuration with the desired normalisation settings.
        raw_items: ``(source_id, raw_float32_hwc)`` pairs from
            :func:`load_preview_raw_cutouts`.

    Returns:
        ``list[(source_id, uint8_hwc)]`` in input order.
    """
    if not raw_items:
        return []
    cfg = get_fitsbolt_config(cfg)
    processed = decode_cutana_raw_images([raw_image for _, raw_image in raw_items], cfg)
    return [
        (source_id, ensure_uint8_hwc(processed_image))
        for (source_id, _), processed_image in zip(raw_items, processed, strict=True)
    ]


def _discover_zarr_paths(folder: str) -> list[str]:
    """Find the Zarr stores under *folder* that the preview may show.

    Delegates to :func:`find_prediction_zarr_stores`, so the preview sees the
    same stores prediction scores — in particular never the training-label
    cache (``labeled_data_cache/images.zarr``).

    Returns:
        List of filesystem paths to Zarr stores.
    """
    return [path for _name, path in find_prediction_zarr_stores(folder)]


def _preview_zarr_by_ids(
    cfg: DotMap, folder: str, source_ids: list[str]
) -> list[tuple[str, np.ndarray]]:
    """Load specific Zarr cutouts by their resolved source id.

    Used when the caller has already picked (and stratified-by-label) the
    exact ids to preview -- e.g. the training-setup gallery, which must
    only preview labeled cutouts rather than a random sample of the whole
    store.

    Args:
        cfg: AnomalyMatch configuration; images are decoded through
            :func:`decode_zarr_image`, as training decodes them.
        folder: Path to the data directory or single ``.zarr`` store.
        source_ids: Ids to load, in the order they should appear. Ids not
            found in any store are silently skipped.

    Returns:
        List of ``(source_id, image_array)`` tuples, in *source_ids* order.
    """
    if not source_ids:
        return []

    location: dict[str, tuple[str, int]] = {}
    for zpath in _discover_zarr_paths(folder):
        try:
            for idx, fn in enumerate(derive_filenames(zpath)):
                location.setdefault(str(fn), (zpath, idx))
        except Exception as exc:
            logger.warning("Zarr preview id lookup failed for {}: {}", zpath, exc)

    items: list[tuple[str, np.ndarray]] = []
    opened_roots: dict[str, zarr.Group] = {}
    for source_id in source_ids:
        loc = location.get(str(source_id))
        if loc is None:
            continue
        zpath, idx = loc
        try:
            if zpath not in opened_roots:
                opened_roots[zpath] = zarr.open_group(zpath, mode="r")
            root = opened_roots[zpath]
            decoded = decode_zarr_image(np.asarray(root["images"][idx]), cfg)
            items.append((source_id, ensure_uint8_hwc(decoded)))
        except Exception as exc:
            logger.warning("Zarr preview failed for {} [{}]: {}", zpath, idx, exc)
    return items


def _preview_zarr(
    cfg: DotMap,
    folder: str,
    max_items: int,
    *,
    offset: int = 0,
    source_ids: list[str] | None = None,
) -> list[tuple[str, np.ndarray]]:
    """Load preview images from Zarr stores, decoded exactly as training decodes them.

    When *source_ids* is given, loads exactly those ids (see
    :func:`_preview_zarr_by_ids`) -- used whenever the caller must only
    preview a specific set of cutouts, e.g. ones already stratified by
    label. Otherwise walks the index space deterministically from
    *offset* across all discovered ``.zarr`` stores in *folder*; on the
    first page (and only the first page) the in-store indices are
    randomised so a single-page preview still shows a varied sample, but
    subsequent pages return the next contiguous slice from the global
    index so pagination is well-defined (#427).

    Args:
        cfg: AnomalyMatch configuration with ``fitsbolt_cfg`` attached; its
            normalisation and ``channel_combination`` are applied through
            :func:`decode_zarr_image`, the decoder ``ZarrSource`` trains on.
        folder: Path to the data directory or single ``.zarr`` store.
        max_items: Maximum number of preview samples to return.
        offset: Skip this many items across the global concatenated
            index space before loading. Ignored when *source_ids* is given.
        source_ids: Specific ids to load, already paginated by the caller,
            or ``None`` to sample from the whole store.

    Returns:
        List of ``(source_id, image_array)`` tuples. The id is the real
        ``source_id``/``original_filename`` from the store's metadata
        parquet when present, else the same generated
        ``{prefix}__image_{idx:06d}`` id training and validation fall back
        to -- never a cache path or a raw ``store.zarr[idx]`` index.
    """
    if source_ids is not None:
        return _preview_zarr_by_ids(cfg, folder, source_ids)

    zarr_paths = _discover_zarr_paths(folder)
    items: list[tuple[str, np.ndarray]] = []
    skipped = 0
    for zpath in zarr_paths:
        try:
            root = zarr.open_group(zpath, mode="r")
            if "images" not in root:
                continue
            arr = root["images"]
            store_n = arr.shape[0]
            # Walk this store's slice of the global index window
            # [offset, offset+max_items).
            start_in_store = max(0, offset - skipped)
            if start_in_store >= store_n:
                skipped += store_n
                continue
            remaining = max_items - len(items)
            stop_in_store = min(store_n, start_in_store + remaining)
            if offset == 0:
                # Random sample on the first page only — paginated runs
                # are fully deterministic.
                indices = random.sample(range(store_n), stop_in_store - start_in_store)
            else:
                indices = list(range(start_in_store, stop_in_store))
            source_ids = derive_filenames(zpath)
            for idx in indices:
                decoded = decode_zarr_image(np.asarray(arr[idx]), cfg)
                items.append((str(source_ids[idx]), ensure_uint8_hwc(decoded)))
            skipped += store_n
        except Exception as exc:
            logger.warning("Zarr preview failed for {}: {}", zpath, exc)
        if len(items) >= max_items:
            break
    return items
