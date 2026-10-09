#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Derive per-image filenames for a Zarr store.

Shared by :mod:`subprocess_scripts.prediction_process_zarr` (to tag its
``store_results`` writes) and :mod:`anomaly_match.pipeline.session` (to
detect when an entire Zarr file is already scored and skip the
subprocess spawn entirely).  Kept in one place so the two callers agree
byte-for-byte on what name a given index resolves to.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import zarr
from loguru import logger

# Naming conventions produced by the upstream ``images_to_zarr`` tool —
# AnomalyMatch consumes them as-is.  Centralised here so any rename can
# be made in one place; if the upstream schema is ever formalised the
# names should move with it (see issue thread on PR #426).
BATCH_FOLDER_ZARR_NAME = "images.zarr"
BATCH_FOLDER_METADATA_NAME = "images_metadata.parquet"

# Per-row filename columns the metadata sidecar may carry, in priority
# order — ``original_filename`` (set by image-folder ingest), ``filename``
# (legacy alias), ``source_id`` (set by Cutana exports).  The first one
# present wins; missing all three falls back to generated names.
_FILENAME_COLUMN_PRIORITY = ("original_filename", "filename", "source_id")


def _zarr_prefix(zarr_path: Path) -> str:
    """Return the prefix used when falling back to generated filenames."""
    # Batch-folder layout: <batch_dir>/images.zarr → use the parent name
    # so two batches never collide on "image_000000".  Standalone zarr
    # files fall back to the file stem.
    if zarr_path.name == BATCH_FOLDER_ZARR_NAME:
        return zarr_path.parent.name
    return zarr_path.stem


def _generated_names(prefix: str, num_images: int) -> list[str]:
    return [f"{prefix}__image_{i:06d}" for i in range(num_images)]


def _locate_metadata_file(zarr_path: Path, root_attrs: dict) -> Path | None:
    """Find the parquet sidecar that lists per-index filenames, if any.

    Args:
        zarr_path: Path to the zarr store.
        root_attrs: Dict copy of the zarr root's ``.attrs``.

    Returns:
        Path to the metadata parquet file, or ``None`` if no sidecar
        can be located.
    """
    metadata_file: Path | None = None

    if "metadata_file" in root_attrs:
        candidate = Path(root_attrs["metadata_file"])
        if not candidate.is_absolute():
            candidate = zarr_path.parent / candidate.name
        if candidate.exists():
            return candidate
        metadata_file = candidate  # remember for error logs; may not exist

    candidate = zarr_path.parent / f"{zarr_path.stem}_metadata.parquet"
    if candidate.exists():
        return candidate

    if zarr_path.name == BATCH_FOLDER_ZARR_NAME:
        candidate = zarr_path.parent / BATCH_FOLDER_METADATA_NAME
        if candidate.exists():
            return candidate

    return metadata_file if metadata_file and metadata_file.exists() else None


def derive_filenames(zarr_path: str | Path) -> list[str]:
    """Return the ``(filename ↔ zarr index)`` mapping used by prediction.

    Looks for a parquet sidecar — pointed at either by the zarr's
    ``metadata_file`` attribute or by ``<name>_metadata.parquet`` next
    to the store — and pulls per-row filenames from
    ``original_filename`` / ``filename`` / ``source_id`` (in that order
    of preference).  Falls back to ``{prefix}__image_{i:06d}`` when no
    metadata is found or the row count doesn't match.

    Args:
        zarr_path: Path to the zarr store (file or directory).

    Returns:
        List of filename strings, length equal to the zarr's ``images``
        array length.  Empty list if the ``images`` array is missing or
        unreadable.
    """
    zarr_path = Path(zarr_path)
    try:
        root = zarr.open_group(str(zarr_path), mode="r")
    except Exception as exc:
        logger.warning("Cannot open Zarr store {} for filename derivation: {}", zarr_path, exc)
        return []

    if "images" not in root:
        logger.warning("Zarr store {} has no 'images' array", zarr_path)
        return []

    num_images = root["images"].shape[0]
    prefix = _zarr_prefix(zarr_path)
    metadata_file = _locate_metadata_file(zarr_path, dict(root.attrs))

    if metadata_file is None:
        return _generated_names(prefix, num_images)

    try:
        metadata_df = pd.read_parquet(metadata_file)
    except Exception as exc:
        logger.warning("Failed to read zarr metadata {}: {}", metadata_file, exc)
        return _generated_names(prefix, num_images)

    for column in _FILENAME_COLUMN_PRIORITY:
        if column in metadata_df.columns:
            filenames = [str(v) for v in metadata_df[column].tolist()]
            break
    else:
        logger.warning("No filename column in {}", metadata_file)
        return _generated_names(prefix, num_images)

    if len(filenames) != num_images:
        logger.warning(
            "Filename count ({}) does not match image count ({}) in {}",
            len(filenames),
            num_images,
            zarr_path,
        )
        return _generated_names(prefix, num_images)

    return filenames
