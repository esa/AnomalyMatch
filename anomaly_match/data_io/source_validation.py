#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.

"""Validate that labeled filenames/IDs exist in data sources.

Provides efficient validation for container sources (Zarr, Cutana) without
building full O(N) indexes. For each labeled item, produces a location tuple
that downstream code can use for targeted extraction.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from loguru import logger

from anomaly_match.datasets.training_data_source import DataSourceType


@dataclass
class ValidationResult:
    """Result of validating labeled files against a data source.

    Attributes:
        found: Filenames/IDs that exist in the source.
        missing: Filenames/IDs that were not found.
        found_locations: Map from filename to source-specific location info.
            - Image folder: ``{filename: filepath}``
            - Zarr: ``{filename: (store_path_str, local_index)}``
            - Cutana: ``{source_id: (catalogue_path_str, row_index)}``
        partial: ``True`` when the validator bailed out before scanning every
            catalogue (e.g. because ``stale_check`` reported the request was
            superseded).  When ``True``, ``found``/``missing``/
            ``found_locations`` reflect only the catalogues processed so far
            and must not be cached or used to build a labeled cache.
    """

    found: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    found_locations: dict[str, tuple] = field(default_factory=dict)
    partial: bool = False


def validate_labeled_files(
    source_dir: str,
    label_df: pd.DataFrame,
    source_type: str | DataSourceType,
    *,
    stale_check=None,
) -> ValidationResult:
    """Check which labeled files/IDs exist in the data source.

    Dispatches to source-specific validation. Never builds a full index
    of all N items in the source -- only checks the labeled subset.

    Args:
        source_dir: Path to the data directory or file.
        label_df: DataFrame with ``id`` and ``label`` columns.
        source_type: A :class:`DataSourceType` value (or its string equivalent).
        stale_check: Optional zero-arg callable returning ``True`` when
            this validation request has been superseded.  Only the
            Cutana path inspects it (the others are fast).

    Returns:
        ValidationResult with found/missing lists and location info.

    Raises:
        ValueError: If *source_type* is not recognised.
    """
    if source_type == DataSourceType.IMAGE_FOLDER:
        return _validate_image_folder(source_dir, label_df)
    elif source_type == DataSourceType.ZARR:
        return _validate_zarr(source_dir, label_df)
    elif source_type == DataSourceType.CUTANA:
        return _validate_cutana(source_dir, label_df, stale_check=stale_check)
    else:
        raise ValueError(f"Unknown source type: {source_type!r}")


# ---------------------------------------------------------------------------
# Image folder validation
# ---------------------------------------------------------------------------


def _validate_image_folder(source_dir: str, label_df: pd.DataFrame) -> ValidationResult:
    """Validate by checking file existence on disk.

    Returns:
        Validation result with filesystem paths as locations.
    """
    result = ValidationResult()
    # Label CSVs may store integer-valued IDs (Cutana source IDs) that
    # happen to be used against an image folder — coerce to str so
    # os.path.join never sees a non-path type.
    filenames = [str(x) for x in label_df["id"].tolist()]

    for fn in filenames:
        filepath = os.path.join(source_dir, fn)
        if os.path.exists(filepath):
            result.found.append(fn)
            result.found_locations[fn] = (filepath,)
        else:
            result.missing.append(fn)

    logger.debug("Image folder validation: {}/{} found", len(result.found), len(filenames))
    return result


# ---------------------------------------------------------------------------
# Zarr validation
# ---------------------------------------------------------------------------

# Matches generated filenames like "storename__image_000042"
_GENERATED_FN_RE = re.compile(r"^(.+)__image_(\d+)$")


def _validate_zarr(source_dir: str, label_df: pd.DataFrame) -> ValidationResult:
    """Validate labeled filenames against Zarr store metadata.

    Handles both generated filenames (index-based check) and metadata
    filenames (parquet lookup). Opens each store only to read its size
    and optional metadata -- never builds a full filename index.

    Returns:
        Validation result with ``(store_path, local_index)`` locations.
    """
    result = ValidationResult()
    filenames = [str(v) for v in label_df["id"].tolist()]
    if not filenames:
        return result

    # Discover stores (same logic as ZarrSource._discover_stores)
    stores = _discover_zarr_stores(source_dir)
    if not stores:
        result.missing = list(filenames)
        return result

    labeled_set = set(filenames)
    remaining = set(filenames)

    for store_path, images_arr in stores:
        count = images_arr.shape[0]
        store_filenames = _load_zarr_store_filenames(store_path, count)
        has_metadata = store_filenames is not None

        if has_metadata:
            # Metadata filenames: check intersection with labeled set
            _match_metadata_filenames(store_filenames, store_path, labeled_set, remaining, result)
        else:
            # Generated filenames: parse index from label, check range
            _match_generated_filenames(store_path, count, remaining, result)

    # Anything still in remaining was not found
    result.missing = list(remaining)
    logger.debug("Zarr validation: {}/{} found", len(result.found), len(filenames))
    return result


def _discover_zarr_stores(source_dir: str) -> list[tuple[Path, object]]:
    """Discover Zarr stores and return (path, images_array) pairs.

    Opens each store in read-only mode. Only returns stores that contain
    an ``images`` array.

    Returns:
        List of (path, images_array) pairs.
    """
    import zarr  # noqa: PLC0415

    path = Path(source_dir)
    zarr_paths: list[Path] = []

    if str(path).rstrip(os.sep + "/").lower().endswith(".zarr"):
        zarr_paths = [path]
    elif path.is_dir():
        zarr_paths = sorted(p for p in path.iterdir() if p.name.lower().endswith(".zarr"))

    stores = []
    for zp in zarr_paths:
        try:
            root = zarr.open_group(str(zp), mode="r")
            if "images" in root:
                stores.append((zp, root["images"]))
        except Exception as exc:
            logger.warning("Failed to open Zarr store {}: {}", zp, exc)
    return stores


def _load_zarr_store_filenames(store_path: Path, total: int) -> list[str] | None:
    """Try to load filenames from a Zarr store's metadata parquet.

    Returns:
        List of filenames if metadata exists, ``None`` if only generated
        filenames are available.
    """
    import zarr  # noqa: PLC0415

    root = zarr.open_group(str(store_path), mode="r")

    # Same discovery logic as ZarrSource._load_filenames
    candidates: list[Path] = []

    if "metadata_file" in root.attrs:
        candidate = Path(root.attrs["metadata_file"])
        if not candidate.is_absolute():
            candidate = store_path.parent / candidate.name
        candidates.append(candidate)

    candidates.append(store_path.parent / f"{store_path.stem}_metadata.parquet")

    if store_path.name == "images.zarr":
        candidates.append(store_path.parent / "images_metadata.parquet")

    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            df = pd.read_parquet(candidate)
            for col in ("original_filename", "filename", "source_id"):
                if col in df.columns:
                    filenames = df[col].tolist()
                    if len(filenames) == total:
                        return filenames
        except Exception as exc:
            logger.warning("Failed to read metadata {}: {}", candidate, exc)

    return None


def _match_metadata_filenames(
    store_filenames: list[str],
    store_path: Path,
    labeled_set: set[str],
    remaining: set[str],
    result: ValidationResult,
) -> None:
    """Match labeled filenames against a store's metadata filenames."""
    store_path_str = str(store_path)
    for idx, fn in enumerate(store_filenames):
        if fn in remaining:
            result.found.append(fn)
            result.found_locations[fn] = (store_path_str, idx)
            remaining.discard(fn)


def _match_generated_filenames(
    store_path: Path,
    store_count: int,
    remaining: set[str],
    result: ValidationResult,
) -> None:
    """Match labeled filenames that look like generated Zarr names.

    Generated filenames follow the pattern ``{prefix}__image_{idx:06d}``.
    """
    store_path_str = str(store_path)
    prefix = store_path.parent.name if store_path.name == "images.zarr" else store_path.stem

    matched = []
    for fn in remaining:
        m = _GENERATED_FN_RE.match(fn)
        if m and m.group(1) == prefix:
            idx = int(m.group(2))
            if 0 <= idx < store_count:
                result.found.append(fn)
                result.found_locations[fn] = (store_path_str, idx)
                matched.append(fn)

    for fn in matched:
        remaining.discard(fn)


# ---------------------------------------------------------------------------
# Cutana validation
# ---------------------------------------------------------------------------


def _validate_cutana(
    source_dir: str,
    label_df: pd.DataFrame,
    stale_check=None,
) -> ValidationResult:
    """Validate labeled source IDs against Cutana catalogue files.

    Reads only the ``SourceID`` column from each catalogue for efficient
    validation of large catalogues (millions to billions of rows).

    Emits an INFO-level progress line per catalogue so the setup-screen
    spinner can mirror real work instead of sitting on a static
    ``Validating labels...`` message for minutes on billion-row sources
    (see :class:`anomaly_match_ui.utils.progress_tap.LogProgressTap`).

    Between catalogues a ``stale_check`` (if supplied) short-circuits
    the loop so older setup-screen jobs stop re-reading the same 54
    parquets once a newer job is queued — that's what caused five-way
    concurrent validations to deadlock pyarrow on contested memmap'd
    reads.

    Returns:
        Validation result with ``(catalogue_path, row_index)`` locations.
    """
    result = ValidationResult()

    source_ids = label_df["id"].tolist()
    if not source_ids:
        return result

    labeled_set = set(str(sid) for sid in source_ids)
    remaining = set(labeled_set)
    total_labels = len(labeled_set)

    catalogue_files = _discover_cutana_catalogues(source_dir)
    if not catalogue_files:
        result.missing = list(remaining)
        return result

    n_cat = len(catalogue_files)
    logger.info("Validating labels against {} Cutana catalogue(s)", n_cat)

    for i, cat_path in enumerate(catalogue_files, start=1):
        if stale_check is not None and stale_check():
            logger.info("Validation superseded at catalogue {}/{} — bailing", i, n_cat)
            result.missing = list(remaining)
            result.partial = True
            return result
        if not remaining:
            logger.info(
                "Validating labels: catalogue {}/{} — all {} IDs matched, stopping",
                i,
                n_cat,
                total_labels,
            )
            break
        logger.info(
            "Validating labels: catalogue {}/{} ({}) — {}/{} matched so far",
            i,
            n_cat,
            cat_path.name,
            len(result.found),
            total_labels,
        )
        _match_cutana_source_ids(cat_path, remaining, result)

    result.missing = list(remaining)
    logger.info(
        "Cutana validation complete: {}/{} IDs matched across {} catalogue(s)",
        len(result.found),
        total_labels,
        n_cat,
    )
    return result


def _discover_cutana_catalogues(source_dir: str) -> list[Path]:
    """Discover Cutana catalogue files in the source directory.

    Returns:
        List of paths to valid Cutana catalogue files.
    """
    from anomaly_match.datasets.cutana_source import _is_cutana_catalogue  # noqa: PLC0415

    path = Path(source_dir)

    # Single file
    if path.is_file():
        if _is_cutana_catalogue(path):
            return [path]
        return []

    # Directory
    catalogues = []
    if path.is_dir():
        for entry in sorted(path.iterdir()):
            if entry.suffix.lower() in (".csv", ".parquet"):
                if _is_cutana_catalogue(entry):
                    catalogues.append(entry)
    return catalogues


def _match_cutana_source_ids(
    cat_path: Path,
    remaining: set[str],
    result: ValidationResult,
) -> None:
    """Check which labeled SourceIDs appear in a single catalogue file.

    Reads only the SourceID column to minimize memory usage.
    """
    cat_path_str = str(cat_path)
    try:
        if cat_path.suffix == ".parquet":
            df = pd.read_parquet(cat_path, columns=["SourceID"])
        else:
            df = pd.read_csv(cat_path, usecols=["SourceID"])
    except Exception as exc:
        logger.warning("Failed to read catalogue {}: {}", cat_path, exc)
        return

    # Convert to string for consistent matching
    source_ids_series = df["SourceID"].astype(str)
    matching_mask = source_ids_series.isin(remaining)

    if not matching_mask.any():
        return

    matched_indices = matching_mask[matching_mask].index
    for idx in matched_indices:
        sid = source_ids_series.iloc[idx]
        if sid in remaining:
            result.found.append(sid)
            result.found_locations[sid] = (cat_path_str, idx)
            remaining.discard(sid)
