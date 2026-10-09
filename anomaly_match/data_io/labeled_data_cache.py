#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.

"""Zarr-backed cache for labeled source data (container sources only).

Extracts labeled images once in the setup screen and carries them forward
across retrain cycles to avoid re-streaming from remote sources.

For **Zarr** sources the cache holds the store's raw array values —
normalisation is applied at training time, so normalisation changes never
need a rebuild.

For **Cutana** sources the cache holds per-band, *unnormalised*, float32
cutouts at :data:`LABELED_CACHE_RESOLUTION` (or larger if the user has
ever requested a higher ``image_size``).  The full fitsbolt pipeline —
normalisation method, output dtype, resize, and channel combination — is
applied at read time (see
:func:`anomaly_match.data_io.container_loaders.decode_cutana_raw_images`).
That decoupling means changing ``normalisation_method``,
``output_dtype``, ``channel_combination`` or ``n_output_channels`` does
**not** trigger a rebuild.  The only triggers are:

- Raw-extraction parameters (FITS extension list, padding factor, flux
  conversion, interpolation order) — these feed into Cutana's
  reprojection and so affect the stored pixel values.
- A requested ``image_size`` larger than the cached resolution — we can
  always resize down, never up.
- A change in the source's resolved *band count* — the cache stores one
  channel per band, so adding/removing a band (e.g. a new NIR-J entry in
  the catalogues' ``fits_file_paths``) makes the cached cutouts disagree
  with the fresh stream even when the config is untouched.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow.types as pat
import zarr
from cutana import StreamingOrchestrator
from dotmap import DotMap
from fitsbolt.normalisation.NormalisationMethod import NormalisationMethod
from loguru import logger
from tqdm import tqdm
from zarr.codecs import ZstdCodec

from anomaly_match.data_io.load_images import get_fitsbolt_config
from anomaly_match.data_io.source_validation import validate_labeled_files
from anomaly_match.datasets.cutana_source import (
    DIRECT_CUTOUT_MAX_SOURCES,
    _iter_batch_items,
    _resolve_extension_names,
    build_cutana_orchestrator_config,
    extract_cutouts_direct,
    first_cutana_catalogue,
)
from anomaly_match.datasets.training_data_source import DataSourceType, _auto_detect_source_type
from anomaly_match.utils.normalisation_parameters import EXTRACTION_AFFECTING_NORM_FIELDS
from anomaly_match.utils.tqdm_logging import tqdm_logging_file

LABELED_CACHE_RESOLUTION = 384
"""Default Cutana cutout resolution used for labeled-cache builds.

The cache stores unnormalised float32 data at this resolution (or larger
if the user has ever requested a higher ``image_size``), and the read
path resizes down to ``cfg.normalisation.image_size`` on the fly.  384 px
gives ~1.7 MB / cutout for 4 bands — affordable for thousands of
labeled sources, and leaves headroom for detail-view zoom without forcing
a rebuild.
"""

# Progress caption for the cache read.  Each cutout is a separate zarr chunk
# file (see ``_write_cache``'s ``chunks=(1, ...)``), so reading a few hundred
# over NFS is a long, otherwise-silent string of round-trips; surfacing this
# string lets the training status label and the setup-preview spinner show
# how far along the read is.  Kept in sync with the matcher in
# ``anomaly_match_ui/utils/progress_tap.py``.
_CACHE_READ_DESC = "Loading labeled cache"

# On-disk format version of the images zarr.  Bumped to 2 when the array moved
# from one-chunk-file-per-cutout (pathological on NFS: a separate file open per
# cutout) to zarr-v3 sharding + Zstd compression.  ``needs_rebuild`` rebuilds
# any cache whose ``cache_info.json`` predates this version, so existing caches
# are transparently upgraded once.
_CACHE_FORMAT_VERSION = 2

# Cutouts per shard file.  Sharding packs this many per-cutout chunks into one
# file, so reading the labeled set is a handful of file opens instead of one
# per cutout — the NFS round-trip cost that made the read slow.  At 1000, a
# typical labeled set (hundreds to ~1k cutouts) lands in a single shard file
# and larger sets shard into a few; random per-cutout reads stay cheap because
# zarr-v3 partial-reads the needed chunk from within the shard.
_CACHE_SHARD_SIZE = 1000

# Zstd level for the cached cutouts.  Low level: the win is fewer/smaller NFS
# reads, and astronomical cutouts (much near-zero sky) compress well even here;
# higher levels cost build CPU for little extra ratio.
_CACHE_COMPRESSION_LEVEL = 3

# Target number of cache-read heartbeats, regardless of how many cutouts are
# read.  ~20 is enough to look alive without flooding the log.
_CACHE_READ_TARGET_LINES = 20


def _cache_read_log_stride(total: int) -> int:
    """Heartbeat stride that yields at most ~``_CACHE_READ_TARGET_LINES`` lines.

    Uses ``ceil`` so the cap also holds in the ``20 < total < 40`` band, where
    floor division would collapse to a stride of 1 and log every cutout.

    Args:
        total: Number of cutouts that will be read.

    Returns:
        Log every Nth cutout (always ``>= 1``).
    """
    return max(1, math.ceil(total / _CACHE_READ_TARGET_LINES))


class LabeledDataCache:
    """Zarr-backed cache for raw labeled training images.

    Only used for container sources (Zarr, Cutana). Image folders load
    from disk directly and don't need caching.

    For Zarr the cache stores raw arrays — normalisation changes never
    trigger a rebuild.  For Cutana the cache stores per-band normalised
    output; ``channel_combination`` is applied at read time, so only the
    parameters in :data:`EXTRACTION_AFFECTING_NORM_FIELDS` trigger a rebuild
    (see :func:`_compute_extraction_hash`).  Resolution is tracked separately,
    via the stored ``image_shape``.

    Args:
        cache_dir: Directory to store the cache files.
    """

    IMAGES_ZARR = "images.zarr"
    METADATA_PARQUET = "metadata.parquet"
    CACHE_INFO_JSON = "cache_info.json"

    def __init__(self, cache_dir: Path) -> None:
        self._cache_dir = Path(cache_dir)

    @property
    def cache_dir(self) -> Path:
        """Path to the cache directory."""
        return self._cache_dir

    @property
    def source_type(self) -> str:
        """Return the source type stored in the cache info, or empty string."""
        info_path = self._cache_dir / self.CACHE_INFO_JSON
        if info_path.exists():
            with open(info_path) as f:
                return json.load(f).get("source_type", "")
        return ""

    @property
    def is_populated(self) -> bool:
        """Check if the cache has been built and contains data.

        Returns:
            True if the cache zarr and metadata files exist.
        """
        return is_labeled_cache_populated(self._cache_dir)

    # ------------------------------------------------------------------
    # Build / append
    # ------------------------------------------------------------------

    def build_from_zarr(
        self,
        found_locations: dict[str, tuple[str, int]],
        label_df: pd.DataFrame,
        cfg: DotMap,
    ) -> None:
        """Extract raw labeled images from Zarr stores and write to cache.

        Reads directly from known ``(store_path, local_index)`` positions
        without building a full source index.

        Args:
            found_locations: Map of ``id -> (store_path, local_index)``
                from validation.
            label_df: Label DataFrame with ``id``, ``label`` columns.
            cfg: Configuration (used for extraction hash).
        """
        self.clear()
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        if not found_locations:
            logger.warning("LabeledDataCache: no locations to build from")
            return

        images, metadata_rows = self._extract_zarr_images(found_locations, label_df)
        self._write_cache(images, metadata_rows, DataSourceType.ZARR, cfg, label_df=label_df)
        logger.info("LabeledDataCache: built Zarr cache with {} images", len(images))

    def build_from_cutana(
        self,
        found_locations: dict[str, tuple[str, int]],
        label_df: pd.DataFrame,
        cfg: DotMap,
    ) -> None:
        """Extract raw labeled cutouts from Cutana catalogues and write to cache.

        Groups source IDs by catalogue file and streams cutouts via
        Cutana with normalisation *disabled* (``CONVERSION_ONLY`` + float32
        output) at :data:`LABELED_CACHE_RESOLUTION`.  That way the
        stored values survive any change to ``normalisation_method``,
        ``output_dtype`` or ``channel_combination`` — those are applied
        at read time by
        :func:`~anomaly_match.data_io.container_loaders.decode_cutana_raw_images`.

        Args:
            found_locations: Map of ``source_id -> (catalogue_path, row_index)``
                from validation.
            label_df: Label DataFrame with ``id``, ``label`` columns.
            cfg: Configuration.  Only raw-extraction fields
                (fits_extension, padding, flux conversion,
                interpolation_order) affect the cached pixels;
                normalisation / channel-combination fields are ignored
                at build time.
        """
        self.clear()
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        if not found_locations:
            logger.warning("LabeledDataCache: no locations to build from")
            return

        # Work from a cfg override that disables normalisation and pins
        # the resolution to the cache size — so the rest of the build
        # pipeline (orchestrator config, hash, write_cache) all see the
        # same raw-extraction settings even if a UI widget callback
        # mutates the shared cfg mid-build on another thread.
        extraction_cfg = _build_raw_extraction_cfg(cfg)

        images, metadata_rows = self._extract_cutana_images(
            found_locations, label_df, extraction_cfg
        )
        self._write_cache(
            images, metadata_rows, DataSourceType.CUTANA, extraction_cfg, label_df=label_df
        )
        logger.info(
            "LabeledDataCache: built Cutana cache with {} image(s) at {} px (raw float32)",
            len(images),
            extraction_cfg.normalisation.image_size[0],
        )

    def append_from_zarr(
        self,
        new_locations: dict[str, tuple[str, int]],
        new_label_df: pd.DataFrame,
        full_label_df: pd.DataFrame,
    ) -> None:
        """Append newly labeled Zarr images to an existing cache.

        Args:
            new_locations: New ``id -> (store_path, local_index)`` entries.
            new_label_df: Label DataFrame for the new entries.
            full_label_df: The complete label DataFrame (whole CSV) used to
                refresh ``label_csv_hash`` — see :meth:`_append_to_cache`.

        Raises:
            RuntimeError: If the cache has not been built yet.
        """
        if not self.is_populated:
            raise RuntimeError("Cannot append to an unpopulated cache")

        existing_meta = self.get_label_df()
        existing_names = set(existing_meta["id"])

        # Only extract truly new entries
        new_locs = {fn: loc for fn, loc in new_locations.items() if fn not in existing_names}
        if not new_locs:
            logger.debug("LabeledDataCache: no new entries to append")
            return

        images, metadata_rows = self._extract_zarr_images(new_locs, new_label_df)
        self._append_to_cache(images, metadata_rows, full_label_df)
        logger.info("LabeledDataCache: appended {} images", len(images))

    def append_from_cutana(
        self,
        new_locations: dict[str, tuple[str, int]],
        new_label_df: pd.DataFrame,
        cfg: DotMap,
        full_label_df: pd.DataFrame,
    ) -> None:
        """Append newly labeled Cutana cutouts to an existing cache.

        Args:
            new_locations: New ``source_id -> (catalogue_path, row_index)`` entries.
            new_label_df: Label DataFrame for the new entries.
            cfg: Configuration for Cutana orchestrator.
            full_label_df: The complete label DataFrame (whole CSV) used to
                refresh ``label_csv_hash`` — see :meth:`_append_to_cache`.

        Raises:
            RuntimeError: If the cache has not been built yet.
        """
        if not self.is_populated:
            raise RuntimeError("Cannot append to an unpopulated cache")

        existing_meta = self.get_label_df()
        existing_ids = set(existing_meta["id"])

        new_locs = {sid: loc for sid, loc in new_locations.items() if sid not in existing_ids}
        if not new_locs:
            logger.debug("LabeledDataCache: no new entries to append")
            return

        # Build the same raw-extraction cfg as ``build_from_cutana`` so
        # the appended cutouts match the on-disk shape/dtype.
        extraction_cfg = _build_raw_extraction_cfg(cfg)
        # Pin the cache resolution to whatever the existing images use
        # (stored shape may exceed the default because an earlier build
        # saw a higher user image_size).  The hash guard rebuilds when
        # other extraction params drift, so this is safe.
        stored_shape = self.get_cache_info().get("image_shape")
        if stored_shape:
            extraction_cfg.normalisation.image_size = list(stored_shape[:2])
            extraction_cfg.fitsbolt_cfg = None
            extraction_cfg = get_fitsbolt_config(extraction_cfg)
        images, metadata_rows = self._extract_cutana_images(new_locs, new_label_df, extraction_cfg)
        self._append_to_cache(images, metadata_rows, full_label_df)
        logger.info("LabeledDataCache: appended {} images", len(images))

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_raw_images(self) -> list[tuple[str, np.ndarray]]:
        """Read all cached raw images.

        Returns:
            List of ``(id, raw_image_array)`` tuples.
        """
        if not self.is_populated:
            return []

        zarr_path = self._cache_dir / self.IMAGES_ZARR
        root = zarr.open_group(str(zarr_path), mode="r")
        images_arr = root["images"]

        meta_df = self.get_label_df()
        ids = meta_df["id"].tolist()

        # One zarr chunk file per cutout over NFS, so this loop is the long
        # silent gap before training's "Decoding labeled cache" begins.  Route
        # a tqdm meter through ``tqdm_logging_file`` (same path as the decode
        # meter) so the training status label and progress bar advance during
        # the read instead of sitting on "Loading datasets...".
        results = []
        for i, item_id in enumerate(
            tqdm(
                ids,
                desc=_CACHE_READ_DESC,
                unit="img",
                file=tqdm_logging_file(),
                mininterval=1.0,
            )
        ):
            results.append((item_id, np.array(images_arr[i])))
        return results

    def get_raw_images_by_id(self, ids: list[str]) -> list[tuple[str, np.ndarray]]:
        """Read a specific subset of cached raw images by id.

        Callers are expected to have already checked that the cache is
        populated and that every requested id is in it — use
        :meth:`is_populated` and :meth:`get_label_df` for that.  The
        method raises on both invariants so a mis-wired call site fails
        loudly instead of returning a silently shorter list.

        Args:
            ids: Item ids to read.  Must be non-empty and every id must
                exist in the cache.  Order is preserved in the output.

        Returns:
            List of ``(id, raw_image_array)`` tuples, one per input id,
            in the input order.

        Raises:
            ValueError: If *ids* is empty.
            RuntimeError: If the cache has not been built yet.
            KeyError: If any requested id is not in the cache.
        """
        if not ids:
            raise ValueError("get_raw_images_by_id called with empty ids list")
        if not self.is_populated:
            raise RuntimeError("Cannot read from an unpopulated labeled cache")

        meta_df = self.get_label_df()
        index_by_id = {str(row_id): idx for idx, row_id in enumerate(meta_df["id"])}

        missing = [str(item_id) for item_id in ids if str(item_id) not in index_by_id]
        if missing:
            preview = missing[:5]
            suffix = " ..." if len(missing) > 5 else ""
            raise KeyError(f"{len(missing)} id(s) not in labeled cache: {preview}{suffix}")

        zarr_path = self._cache_dir / self.IMAGES_ZARR
        root = zarr.open_group(str(zarr_path), mode="r")
        images_arr = root["images"]

        # Each cutout is a separate zarr chunk file, so reading the preview's
        # few hundred ids over NFS is a long silent stretch behind the
        # "Loading preview..." spinner.  This runs in the UI kernel (not the
        # training subprocess), where tqdm-to-stderr wouldn't reach the
        # loguru-based progress tap — so emit throttled INFO lines the setup
        # spinner's cutana-hint matcher recognises instead.
        total = len(ids)
        log_every = _cache_read_log_stride(total)
        results: list[tuple[str, np.ndarray]] = []
        for done, item_id in enumerate(ids, start=1):
            results.append((str(item_id), np.array(images_arr[index_by_id[str(item_id)]])))
            if done == total or done % log_every == 0:
                logger.info("{}: {}/{} cutouts", _CACHE_READ_DESC, done, total)
        return results

    def get_label_df(self) -> pd.DataFrame:
        """Read the cached label metadata.

        Returns:
            DataFrame with ``id``, ``label`` columns (plus source-specific
            location columns like ``store_path`` or ``catalogue_path``).
        """
        meta_path = self._cache_dir / self.METADATA_PARQUET
        if not meta_path.exists():
            return pd.DataFrame(columns=["id", "label"])
        return pd.read_parquet(meta_path)

    def get_cache_info(self) -> dict:
        """Read cache metadata (source type, extraction hash, etc.).

        Returns:
            Dict with cache metadata, or empty dict if not populated.
        """
        info_path = self._cache_dir / self.CACHE_INFO_JSON
        if not info_path.exists():
            return {}
        with open(info_path) as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # Cache invalidation
    # ------------------------------------------------------------------

    def needs_rebuild(
        self,
        cfg: DotMap,
        label_df: pd.DataFrame | None = None,
        found_ids: set[str] | None = None,
    ) -> bool:
        """Check if extraction-affecting parameters changed since cache was built.

        For Zarr: never rebuilds on config changes (raw array data is
        fixed), but still rebuilds on label CSV changes when *label_df*
        is supplied.

        For Cutana: rebuild when any *raw*-extraction parameter changed
        (FITS extension list, padding, flux conversion, interpolation
        order — see :func:`_compute_extraction_hash`), when the requested
        ``image_size`` exceeds the stored resolution (resize down is fine;
        we can't invent detail that wasn't cached), **or** when the
        source's resolved band count no longer matches the cached channel
        count (a band was added to or removed from the catalogues'
        ``fits_file_paths``).  Normalisation method, output dtype, channel
        combination and ``n_output_channels`` are all applied at read time
        and so never force a rebuild.

        When *label_df* is supplied, also compare its hash against
        ``cache_info.json['label_csv_hash']`` and rebuild on mismatch —
        this lets a persistent cache (next to the label CSV) detect when
        the CSV has been edited between sessions.

        When *found_ids* is supplied, compare it against the cached id
        set and rebuild on mismatch.  This catches the case where the
        CSV is unchanged but the data source grew (e.g. the user dropped
        another Cutana catalogue into ``data_dir``) — previously the
        cache went stale silently until the CSV hash or extraction
        params happened to change.

        Every rebuild decision is logged with its reason so spurious
        rebuilds are diagnosable from the session log alone.

        Args:
            cfg: Current configuration.
            label_df: Label DataFrame with ``id``/``label`` columns.  If
                supplied, its hash is compared against the stored one to
                detect CSV edits.
            found_ids: IDs the current validation pass matched against
                the source.  When supplied, mismatch against the cached
                ids triggers a rebuild.

        Returns:
            True if the cache should be rebuilt.
        """
        if not self.is_populated:
            logger.info("Labeled cache rebuild: cache not populated")
            return True

        info = self.get_cache_info()

        # Upgrade caches written in an older on-disk format (e.g. the pre-v2
        # one-file-per-cutout layout, which has no version field) by rebuilding
        # once into the current sharded+compressed format.
        stored_version = info.get("cache_format_version", 1)
        if stored_version != _CACHE_FORMAT_VERSION:
            logger.info(
                "Labeled cache rebuild: format version changed (stored={} current={})",
                stored_version,
                _CACHE_FORMAT_VERSION,
            )
            return True

        source_type = info.get("source_type", "")
        stored_hash = info.get("extraction_hash", "")

        current_hash = _compute_extraction_hash(source_type, cfg)
        if stored_hash != current_hash:
            logger.info(
                "Labeled cache rebuild: extraction hash changed "
                "(stored={} current={}) — current params: {}",
                stored_hash,
                current_hash,
                _extraction_params_for_log(source_type, cfg),
            )
            return True

        if label_df is not None:
            stored_csv_hash = info.get("label_csv_hash", "")
            current_csv_hash = _compute_label_csv_hash(label_df)
            if stored_csv_hash != current_csv_hash:
                logger.info(
                    "Labeled cache rebuild: label CSV hash changed "
                    "(stored={} current={}) — labels were edited since build",
                    stored_csv_hash,
                    current_csv_hash,
                )
                return True

        if found_ids is not None:
            cached_ids = set(self.get_label_df()["id"].astype(str))
            if cached_ids != found_ids:
                added = len(found_ids - cached_ids)
                removed = len(cached_ids - found_ids)
                logger.info(
                    "Labeled cache rebuild: source-matched id set changed "
                    "(cached={}, matched now={}, +{} -{})",
                    len(cached_ids),
                    len(found_ids),
                    added,
                    removed,
                )
                return True

        if source_type == DataSourceType.CUTANA:
            stored_shape = info.get("image_shape")
            # image_shape is whatever the writer stored, so treat its length as
            # untrusted: a cache written without a channel axis (or by a foreign
            # writer) must degrade to "can't tell" rather than IndexError here.
            if stored_shape and len(stored_shape) >= 2:
                requested = list(cfg.normalisation.image_size)
                if stored_shape[0] < requested[0] or stored_shape[1] < requested[1]:
                    logger.info(
                        "Labeled cache rebuild: requested image_size {} larger "
                        "than cached resolution {} — can't upsample, rebuilding",
                        requested,
                        stored_shape,
                    )
                    return True

                # The cache stores one channel per resolved FITS band, and the
                # read path passes those bands straight through (channel
                # combination is applied on top, never used to invent a missing
                # band).  So if the source now resolves to a different band count
                # than the cache was built with — e.g. a NIR-J band was added to
                # the catalogues' ``fits_file_paths`` — the cached cutouts no
                # longer match the fresh unlabeled stream, and training would
                # stack 3-band labeled cutouts against a 4-band model.
                # ``fits_extension`` is already in the extraction hash, so this
                # only fires when the *catalogue's* per-source band list changed
                # while the config did not.
                #
                # Comparing a count resolved from the user *cfg* against one
                # stored from ``_build_raw_extraction_cfg(cfg)`` is only sound
                # because that helper rewrites normalisation_method /
                # output_dtype / image_size / fitsbolt_cfg but leaves
                # ``fits_extension`` alone, and band resolution keys only off
                # ``fits_extension``.  If that ever changes, the two counts
                # diverge permanently and every call rebuilds — pinned by
                # test_raw_extraction_cfg_preserves_fits_extension.
                #
                # Checked before resolving: a shape without a channel axis has
                # nothing to compare against, so skip the catalogue read too.
                if len(stored_shape) >= 3:
                    current_bands = _current_source_band_count(cfg)
                    if current_bands is not None and stored_shape[2] != current_bands:
                        logger.info(
                            "Labeled cache rebuild: source band count changed "
                            "(cached={} current={}) — the catalogues' fits_file_paths "
                            "now resolve to a different number of bands",
                            stored_shape[2],
                            current_bands,
                        )
                        return True
        return False

    def clear(self) -> None:
        """Delete all cache files."""
        for name in (self.IMAGES_ZARR, self.METADATA_PARQUET, self.CACHE_INFO_JSON):
            path = self._cache_dir / name
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()

    # ------------------------------------------------------------------
    # High-level orchestration
    # ------------------------------------------------------------------

    def build(
        self,
        found_locations: dict[str, tuple],
        label_df: pd.DataFrame,
        source_type: str,
        cfg: DotMap,
    ) -> None:
        """Build the cache from validated locations, dispatching by source type.

        Args:
            found_locations: Map from ID to source-specific location tuple.
            label_df: Label DataFrame with ``id``, ``label`` columns.
            source_type: A :class:`DataSourceType` value.
            cfg: Configuration for extraction.

        Raises:
            ValueError: If *source_type* is not a container type.
        """
        if source_type == DataSourceType.ZARR:
            self.build_from_zarr(found_locations, label_df, cfg)
        elif source_type == DataSourceType.CUTANA:
            self.build_from_cutana(found_locations, label_df, cfg)
        else:
            raise ValueError(f"Cannot build cache for source type: {source_type!r}")

    def update(
        self,
        new_labels: dict[str, str],
        cfg: DotMap,
    ) -> None:
        """Update the cache with newly labeled items (rebuild or append).

        Decides whether to rebuild from scratch (if extraction parameters
        changed) or incrementally append new entries.

        Args:
            new_labels: Dict mapping ``id -> csv_label_string``.
            cfg: Current configuration.

        Raises:
            RuntimeError: If the cache is not populated.
        """
        if not self.is_populated:
            raise RuntimeError("Cannot update an unpopulated cache")

        source_dir = cfg.data_dir
        source_type = _auto_detect_source_type(source_dir)
        if source_type not in (DataSourceType.ZARR, DataSourceType.CUTANA):
            return

        new_label_df = pd.DataFrame(
            [{"id": item_id, "label": label} for item_id, label in new_labels.items()]
        )
        all_labels = pd.read_csv(cfg.label_file)

        if self.needs_rebuild(cfg, label_df=all_labels):
            logger.info("Labeled cache needs rebuild (extraction params or CSV changed)")
            result = validate_labeled_files(source_dir, all_labels, source_type)
            self.clear()
            self.build(result.found_locations, all_labels, source_type, cfg)
        else:
            result = validate_labeled_files(source_dir, new_label_df, source_type)
            if result.found_locations:
                if source_type == DataSourceType.ZARR:
                    self.append_from_zarr(result.found_locations, new_label_df, all_labels)
                else:
                    self.append_from_cutana(result.found_locations, new_label_df, cfg, all_labels)

        logger.info("Labeled cache updated ({} images)", self.get_cache_info().get("num_images", 0))

    # ------------------------------------------------------------------
    # Internal: extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_zarr_images(
        found_locations: dict[str, tuple[str, int]],
        label_df: pd.DataFrame,
    ) -> tuple[list[np.ndarray], list[dict]]:
        """Read raw arrays from Zarr stores at known positions.

        Args:
            found_locations: Map of ``id -> (store_path, local_index)``.
            label_df: Label DataFrame with ``id`` and ``label`` columns.

        Returns:
            Tuple of (images_list, metadata_rows).
        """
        label_map = _build_label_map(label_df)

        # Group by store path for efficient sequential reads
        by_store: dict[str, list[tuple[str, int]]] = {}
        for item_id, (store_path, idx) in found_locations.items():
            by_store.setdefault(store_path, []).append((item_id, idx))

        images: list[np.ndarray] = []
        metadata_rows: list[dict] = []

        for store_path, items in by_store.items():
            root = zarr.open_group(store_path, mode="r")
            images_arr = root["images"]

            # Sort by index for sequential I/O
            items.sort(key=lambda x: x[1])

            for item_id, idx in items:
                raw_img = np.array(images_arr[idx])
                images.append(raw_img)
                metadata_rows.append(
                    {
                        "id": item_id,
                        # Direct index, not .get(): every id here came from
                        # validate_labeled_files(..., label_df, ...), so a miss
                        # is a broken invariant that must fail loudly rather
                        # than silently caching an empty label.
                        "label": str(label_map[item_id]),
                        "store_path": store_path,
                        "local_index": idx,
                    }
                )

        return images, metadata_rows

    @staticmethod
    def _extract_cutana_images(
        found_locations: dict[str, tuple[str, int]],
        label_df: pd.DataFrame,
        cfg: DotMap,
    ) -> tuple[list[np.ndarray], list[dict]]:
        """Stream cutouts from Cutana catalogues for labeled source IDs.

        Groups by catalogue file and uses the Cutana orchestrator to
        stream cutouts for the labeled subset only.  Emits an INFO-level
        progress line per catalogue so the setup-screen spinner mirrors
        real work — the full-catalogue parquet read that precedes each
        cutout stream can take many seconds, and without a log line the
        UI would sit on a static ``Loading preview...`` caption for the
        whole rebuild (see
        :class:`anomaly_match_ui.utils.progress_tap.LogProgressTap`).

        Args:
            found_locations: Map of ``source_id -> (catalogue_path, row_index)``.
            label_df: Label DataFrame with ``id`` and ``label`` columns.
            cfg: AnomalyMatch configuration for orchestrator setup.

        Returns:
            Tuple of (images_list, metadata_rows).
        """
        label_map = _build_label_map(label_df)

        # Group by catalogue file
        by_catalogue: dict[str, list[str]] = {}
        for sid, (cat_path, _row_idx) in found_locations.items():
            by_catalogue.setdefault(cat_path, []).append(sid)

        images: list[np.ndarray] = []
        metadata_rows: list[dict] = []
        n_cat = len(by_catalogue)
        total = len(found_locations)

        logger.info("Building labeled cache: extracting from {} catalogue(s)", n_cat)

        for i, (cat_path, source_ids) in enumerate(by_catalogue.items(), start=1):
            cat_name = Path(cat_path).name
            # Cumulative cutout count so the user sees overall progress, not
            # just the per-catalogue / per-FITS-set noise: each Cutana call
            # covers one catalogue and reports only its own handful of tiles.
            logger.info(
                "Extracting labeled data: {}/{} cutouts (catalogue {}/{}: {}, {} source(s))",
                len(images),
                total,
                i,
                n_cat,
                cat_name,
                len(source_ids),
            )
            cat_df = _load_catalogue_filtered(cat_path, source_ids)
            if cat_df.empty:
                continue

            cutout_results = _stream_cutana_cutouts(cat_df, cfg)

            for sid, raw_img in cutout_results:
                images.append(raw_img)
                metadata_rows.append(
                    {
                        "id": sid,
                        # Direct index, not .get(): every sid here came from
                        # validate_labeled_files(..., label_df, ...), so a miss
                        # is a broken invariant that must fail loudly rather
                        # than silently caching an empty label.
                        "label": str(label_map[sid]),
                        "catalogue_path": cat_path,
                    }
                )

        # Final tick: the per-catalogue lines log the count *before* each
        # catalogue's append (signal before the slow parquet read), so they
        # lag by one and never reach the total — this closes it out.
        logger.info(
            "Extracting labeled data: {}/{} cutouts — extraction complete", len(images), total
        )
        return images, metadata_rows

    # ------------------------------------------------------------------
    # Internal: cache I/O
    # ------------------------------------------------------------------

    def _write_cache(
        self,
        images: list[np.ndarray],
        metadata_rows: list[dict],
        source_type: DataSourceType,
        cfg: DotMap,
        label_df: pd.DataFrame,
    ) -> None:
        """Write images and metadata to cache files.

        Args:
            images: List of raw image arrays (all must share shape/dtype).
            metadata_rows: List of dicts with ``id``, ``label``, and
                source-specific location fields.
            source_type: Container source type (``ZARR`` or ``CUTANA``).
            cfg: Configuration (used to compute extraction hash).
            label_df: Full label DataFrame driving the build, used to
                compute ``label_csv_hash`` so needs_rebuild sees the full
                input — not the extraction-subset metadata, which would
                drift when Cutana silently drops cutouts and force a
                spurious rebuild on every session.
        """
        if not images:
            return

        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # Write images zarr -- all images must have the same shape/dtype.
        # Per-cutout chunks keep random-access reads cheap, but pack them into
        # shard files (and compress) so the labeled set is a few file opens
        # over NFS rather than one per cutout (see _CACHE_SHARD_SIZE).
        sample = images[0]
        zarr_path = self._cache_dir / self.IMAGES_ZARR
        root = zarr.open_group(str(zarr_path), mode="w")
        shard_count = min(_CACHE_SHARD_SIZE, len(images))
        arr = root.create_array(
            "images",
            shape=(len(images), *sample.shape),
            chunks=(1, *sample.shape),
            shards=(shard_count, *sample.shape),
            compressors=ZstdCodec(level=_CACHE_COMPRESSION_LEVEL),
            dtype=sample.dtype,
        )
        # Write a whole shard per assignment.  A per-cutout write into a
        # sharded+compressed array read-modify-writes (decompress, add one,
        # recompress) the entire shard each time — O(shard_size) per cutout,
        # ~180 s for the full labeled set; shard-batched writes do it in ~8 s.
        # Batch by shard (not all at once) to cap the transient memory at one
        # shard rather than a full second copy of the stack.
        for start in range(0, len(images), _CACHE_SHARD_SIZE):
            end = min(start + _CACHE_SHARD_SIZE, len(images))
            arr[start:end] = np.stack(images[start:end])

        # Write metadata parquet
        meta_df = pd.DataFrame(metadata_rows)
        meta_df.to_parquet(self._cache_dir / self.METADATA_PARQUET, index=False)

        # Write cache info — DataSourceType is ``(str, Enum)`` so json.dump
        # serialises it as its string value and we can rebuild it with
        # ``DataSourceType(...)`` on read.
        info = {
            "source_type": source_type.value,
            "cache_format_version": _CACHE_FORMAT_VERSION,
            "extraction_hash": _compute_extraction_hash(source_type, cfg),
            "label_csv_hash": _compute_label_csv_hash(label_df),
            "num_images": len(images),
            "image_shape": list(sample.shape),
            "image_dtype": str(sample.dtype),
        }
        with open(self._cache_dir / self.CACHE_INFO_JSON, "w") as f:
            json.dump(info, f, indent=2)

    def _append_to_cache(
        self,
        images: list[np.ndarray],
        metadata_rows: list[dict],
        full_label_df: pd.DataFrame,
    ) -> None:
        """Append new images and metadata to existing cache.

        Args:
            images: New raw image arrays to append.
            metadata_rows: Corresponding metadata dicts.
            full_label_df: The complete label DataFrame (the whole CSV, not
                just the appended subset) used to refresh ``label_csv_hash``.
                ``needs_rebuild`` recomputes the hash from the full CSV, so the
                stored value must be computed the same way — computing it from
                the extraction-subset metadata drifts whenever the CSV holds
                ids the source can't match or duplicate rows, forcing a
                spurious rebuild on the next session.
        """
        if not images:
            return

        zarr_path = self._cache_dir / self.IMAGES_ZARR
        root = zarr.open_group(str(zarr_path), mode="r+")
        arr = root["images"]

        old_len = arr.shape[0]
        new_len = old_len + len(images)
        arr.resize((new_len, *arr.shape[1:]))
        # One batched assignment, not per-cutout: writing single elements into
        # the sharded+compressed array rewrites the touched shard each time.
        arr[old_len:new_len] = np.stack(images)

        # Append metadata
        existing_meta = pd.read_parquet(self._cache_dir / self.METADATA_PARQUET)
        new_meta = pd.DataFrame(metadata_rows)
        combined = pd.concat([existing_meta, new_meta], ignore_index=True)
        combined.to_parquet(self._cache_dir / self.METADATA_PARQUET, index=False)

        # Update cache info — refresh num_images and the label-CSV hash so
        # a subsequent needs_rebuild() check against the CSV the gallery
        # just merged into sees a match, not a spurious mismatch.  The hash
        # must come from the full CSV (mirroring _write_cache), not the
        # extraction-subset metadata (see the docstring).
        info_path = self._cache_dir / self.CACHE_INFO_JSON
        with open(info_path) as f:
            info = json.load(f)
        info["num_images"] = new_len
        info["label_csv_hash"] = _compute_label_csv_hash(full_label_df)
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _build_label_map(label_df: pd.DataFrame) -> dict[str, str]:
    """Return a ``str(id) -> label`` map for cutout extraction.

    The extraction loops key ids as strings (``found_locations`` uses string
    ids), but ``pd.read_csv`` infers numeric ids as int64.  Keying the map by
    the DataFrame's native dtype would then miss every string lookup and
    silently store an empty label per cutout — so the ids are cast to str here,
    in one place, for both the Zarr and Cutana paths.

    Args:
        label_df: Label DataFrame with ``id`` and ``label`` columns.

    Returns:
        Map from stringified id to its label.
    """
    return {str(k): v for k, v in zip(label_df["id"], label_df["label"])}


def _compute_label_csv_hash(label_df: pd.DataFrame) -> str:
    """Return a stable 16-char hash of the ``(id, label)`` pairs.

    Used by :meth:`LabeledDataCache.needs_rebuild` to detect that the
    label CSV has been edited between sessions.  Sorting by ``id`` makes
    the hash invariant to row order, so resaving the same labels after a
    reshuffle does not trigger a rebuild.  Labels are cast to ``str`` so
    pandas column dtype (object vs category) does not perturb the hash.

    Args:
        label_df: DataFrame with ``id`` and ``label`` columns; must be
            non-empty (an empty label set is invalid upstream — see
            :func:`~anomaly_match.data_io.source_scanning.read_label_map`).

    Returns:
        16-char hex prefix of SHA256 over the sorted id/label pairs.

    Raises:
        ValueError: If *label_df* has no rows.
    """
    if label_df.empty:
        raise ValueError(
            "_compute_label_csv_hash called with an empty DataFrame — "
            "callers must guard against this; an empty label set should "
            "have been rejected earlier in the pipeline"
        )
    pairs = (
        label_df[["id", "label"]]
        .astype({"id": str, "label": str})
        .sort_values("id")
        .itertuples(index=False, name=None)
    )
    h = hashlib.sha256()
    for pair in pairs:
        h.update(f"{pair[0]}\x00{pair[1]}\n".encode("utf-8"))
    return h.hexdigest()[:16]


def _build_raw_extraction_cfg(cfg: DotMap) -> DotMap:
    """Return a deep-copied cfg with normalisation disabled for cache builds.

    The labeled cache stores raw float32 cutouts at
    :data:`LABELED_CACHE_RESOLUTION` (or larger if the user has ever
    requested a higher ``image_size``).  This helper builds the cfg
    override that drives both Cutana's orchestrator and the extraction
    hash toward that contract — normalisation method set to
    ``CONVERSION_ONLY``, output dtype set to float32, image size pinned
    to ``max(LABELED_CACHE_RESOLUTION, user_size)``, and the fitsbolt
    config regenerated from the overridden normalisation block.

    The original *cfg* is never mutated.

    Args:
        cfg: User-facing configuration.

    Returns:
        Deep-copied configuration with raw-extraction fields set.
    """
    extraction_cfg = copy.deepcopy(cfg)
    user_size = list(extraction_cfg.normalisation.image_size or [])
    requested = user_size[0] if user_size else LABELED_CACHE_RESOLUTION
    cache_side = max(LABELED_CACHE_RESOLUTION, requested)

    extraction_cfg.normalisation.image_size = [cache_side, cache_side]
    extraction_cfg.normalisation.normalisation_method = NormalisationMethod.CONVERSION_ONLY
    extraction_cfg.normalisation.output_dtype = np.float32
    # Force a rebuild of the fitsbolt config so Cutana's
    # external_fitsbolt_cfg sees the CONVERSION_ONLY / float32 pair
    # rather than the user's normalisation settings.
    extraction_cfg.fitsbolt_cfg = None
    return get_fitsbolt_config(extraction_cfg)


def _compute_extraction_hash(source_type: DataSourceType | str, cfg: DotMap) -> str:
    """Compute a hash of *raw*-extraction-affecting parameters.

    For Zarr: always returns a fixed hash (raw array data is independent
    of config — normalisation is applied at training time).

    For Cutana: the cache stores unnormalised float32 cutouts, so the
    hash only covers parameters that alter the raw pixel values Cutana
    produces — which FITS extensions are loaded, the padding/interpolation
    used during reprojection, and whether flux conversion is applied.
    Resolution is tracked separately via the stored ``image_shape``
    (see :meth:`LabeledDataCache.needs_rebuild`), so a bigger
    ``image_size`` forces a rebuild but a smaller one doesn't.
    Normalisation method, output dtype, channel combination and
    ``n_output_channels`` are applied at read time and deliberately
    excluded.

    Args:
        source_type: :class:`DataSourceType` value or its string form
            (as read from ``cache_info.json``).
        cfg: Configuration object.

    Returns:
        Hex digest string.
    """
    if source_type == DataSourceType.ZARR:
        return "zarr_raw_fixed"

    payload = json.dumps(_cutana_hash_params(cfg), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _current_source_band_count(cfg: DotMap) -> int | None:
    """Return the number of bands the Cutana source currently resolves to.

    Defers the count to
    :func:`~anomaly_match.datasets.cutana_source._resolve_extension_names` — the
    same resolver the cache build uses to decide how many channels to store — so
    the two can't drift (an explicit or integer-index ``fits_extension``
    selection is honoured identically).  It is pointed at a *validated* catalogue
    path so it never falls back to a bogus ``PRIMARY``/1-band count off a stray
    file.

    Best-effort: returns ``None`` when ``data_dir`` holds no readable Cutana
    catalogue (e.g. an image folder), so an unresolvable source degrades to
    "don't rebuild on this signal" rather than forcing a spurious rebuild.

    Args:
        cfg: Configuration whose ``fits_extension`` and ``data_dir`` locate the
            bands.

    Returns:
        The resolved band count, or ``None`` if it can't be determined.
    """
    catalogue_path = first_cutana_catalogue(cfg.data_dir)
    if catalogue_path is None:
        return None
    extension_names, _ = _resolve_extension_names(cfg, catalogue_path=catalogue_path)
    # ``or None`` maps 0 to "unknown".  _resolve_extension_names documents its
    # result as always non-empty, but an empty ``fits_extension`` list short-
    # circuits its resolution branch and returns []; without this, a 0 would
    # never equal the stored channel count and every call would rebuild.
    return len(extension_names) or None


def _cutana_hash_params(cfg: DotMap) -> dict:
    """Return the dict of raw-extraction params that go into the Cutana hash.

    Keyed by :data:`EXTRACTION_AFFECTING_NORM_FIELDS` — the setup-screen preview
    reads the same constant to decide whether a normalisation edit invalidates
    the raw cutouts it holds in memory, so the two must not drift apart.
    """
    norm = cfg.normalisation
    # ``sorted`` so ``_extraction_params_for_log`` reads the same order every
    # run: a frozenset iterates by string hash, which is randomised per process.
    params = {field: getattr(norm, field) for field in sorted(EXTRACTION_AFFECTING_NORM_FIELDS)}
    # The one field that is not already JSON-serialisable.
    params["fits_extension"] = _serialise_fits_extension(params["fits_extension"])
    return params


def _extraction_params_for_log(source_type: DataSourceType | str, cfg: DotMap) -> dict:
    """Return a loggable dict of the extraction params currently in use.

    Used by :meth:`LabeledDataCache.needs_rebuild` to explain *why* a
    rebuild was triggered when the hash mismatches.
    """
    if source_type == DataSourceType.ZARR:
        return {}
    return _cutana_hash_params(cfg)


def _serialise_fits_extension(fits_ext: object) -> object:
    """Convert fits_extension to a JSON-serialisable form.

    Args:
        fits_ext: FITS extension config value (may be None, int, str, list).

    Returns:
        JSON-safe representation.
    """
    if fits_ext is None:
        return None
    if isinstance(fits_ext, (str, int)):
        return fits_ext
    if isinstance(fits_ext, (list, tuple)):
        return [_serialise_fits_extension(e) for e in fits_ext]
    if isinstance(fits_ext, np.integer):
        return int(fits_ext)
    return str(fits_ext)


def _load_catalogue_filtered(cat_path: str, source_ids: list[str]) -> pd.DataFrame:
    """Load a Cutana catalogue and filter to specific source IDs.

    For parquet files, pushes the ``SourceID`` filter into pyarrow so
    entire row groups that don't contain any requested id are skipped at
    I/O time.  On billion-row Euclid catalogues this turns a multi-minute
    full-catalogue scan into a sub-second materialisation of just the
    matching rows.

    Args:
        cat_path: Path to CSV or parquet catalogue.
        source_ids: SourceIDs to keep.

    Returns:
        Filtered DataFrame with Cutana catalogue columns.
    """
    path = Path(cat_path)
    sid_set = {str(s) for s in source_ids}

    if path.suffix == ".parquet":
        df = _read_parquet_filtered(path, sid_set)
    else:
        df = pd.read_csv(path)

    return df[df["SourceID"].astype(str).isin(sid_set)]


def _pq_read(path: Path, **kwargs) -> pd.DataFrame:
    """Wrap :func:`pd.read_parquet` with the ``memory_map=False`` default.

    Centralises the NFS-pinning workaround — pyarrow's default memmap
    keeps the underlying file region live until every materialised
    DataFrame that references it is garbage-collected, and on NFS that
    occasionally hung a subsequent rebuild's reopen of the same file.
    The parquets involved are small enough that reading them fully into
    RAM is a better trade than the occasional deadlock.

    A future call site that imports :func:`_pq_read` instead of
    :func:`pd.read_parquet` directly inherits the flag automatically.

    Args:
        path: Parquet file to read.
        **kwargs: Forwarded to :func:`pd.read_parquet`.

    Returns:
        DataFrame loaded from *path* without memory-mapping the source file.
    """
    return pd.read_parquet(path, memory_map=False, **kwargs)


def _pq_read_schema(path: Path):
    """Wrap :func:`pyarrow.parquet.read_schema` with ``memory_map=False``.

    See :func:`_pq_read` for the NFS-pinning rationale.

    Args:
        path: Parquet file to inspect.

    Returns:
        ``pyarrow.Schema`` describing *path* (read without memory-mapping).
    """
    return pq.read_schema(path, memory_map=False)


def _read_parquet_filtered(path: Path, sid_set: set[str]) -> pd.DataFrame:
    """Read a parquet catalogue with a ``SourceID`` filter pushed to pyarrow.

    Uses :func:`_pq_read` / :func:`_pq_read_schema` throughout — see those
    helpers for the NFS-pinning rationale that motivates the
    ``memory_map=False`` flag they centralise.

    Falls back to a full-file read when the push-down fails (e.g. the
    parquet's SourceID column is stored as a non-string type that
    doesn't match the ``str`` values in *sid_set*, or the row-group
    statistics required for pruning aren't present).  The post-read
    ``isin`` filter in the caller handles whichever subset is returned.

    Args:
        path: Path to the parquet file.
        sid_set: Set of ``SourceID`` strings to match.

    Returns:
        DataFrame containing at least the matching rows (may also
        contain extras when the fallback path triggers).
    """
    try:
        schema = _pq_read_schema(path)
        sid_field = schema.field("SourceID")
    except Exception as exc:
        logger.debug("Parquet schema read failed on {}: {}", path, exc)
        return _pq_read(path)

    # Coerce filter values to the column's native type so pyarrow can
    # match against row-group statistics.  Callers always hand us the
    # ``str(source_id)`` form; integer-typed SourceID columns need the
    # numeric parse so the pushdown matches row-group min/max stats.
    if pat.is_integer(sid_field.type):
        try:
            values = [int(s) for s in sid_set]
        except (TypeError, ValueError):
            return _pq_read(path)
    else:
        values = list(sid_set)

    if not values:
        return schema.empty_table().to_pandas()

    try:
        return _pq_read(path, filters=[("SourceID", "in", values)])
    except Exception as exc:
        logger.debug("Parquet pushdown filter failed on {}: {}", path, exc)
        return _pq_read(path)


def _stream_cutana_cutouts(
    sub_catalogue_df: pd.DataFrame,
    cfg: DotMap,
) -> list[tuple[str, np.ndarray]]:
    """Stream cutouts for a filtered catalogue using Cutana's orchestrator.

    Uses the same orchestrator configuration as ``CutanaSource`` but operates
    on a pre-filtered sub-catalogue.

    Args:
        sub_catalogue_df: Filtered catalogue DataFrame.
        cfg: AnomalyMatch configuration.

    Returns:
        List of ``(source_id, raw_image_array)`` tuples.
    """
    # For sub-catalogues below the direct-cutout threshold we use
    # Cutana's in-process ``create_cutouts_direct`` path.  It skips the
    # subprocess spawn + IPC + job tracking of StreamingOrchestrator,
    # which turns 10-30 s label-cache builds into sub-second ones.
    if len(sub_catalogue_df) < DIRECT_CUTOUT_MAX_SOURCES:
        batches = extract_cutouts_direct(sub_catalogue_df, cfg)
        return _collect_cutouts(batches)

    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, "sub_catalogue.parquet")
    sub_catalogue_df.to_parquet(tmp_path, index=False)

    try:
        cutana_cfg = build_cutana_orchestrator_config(tmp_path, cfg)
        orchestrator = StreamingOrchestrator(cutana_cfg)
        orchestrator.init_streaming(
            batch_size=min(100, len(sub_catalogue_df)),
            write_to_disk=False,
            min_workers=cfg.cutana_min_workers,
            max_workers=cfg.cutana_max_workers,
        )

        results: list[tuple[str, np.ndarray]] = []
        n_batches = orchestrator.get_batch_count()
        for _ in range(n_batches):
            results.extend(_collect_cutouts([orchestrator.next_batch()]))

        return results
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _collect_cutouts(batches: list[dict]) -> list[tuple[str, np.ndarray]]:
    """Flatten Cutana batch dicts into ``(source_id, raw_img)`` tuples.

    Args:
        batches: One or more batch dicts with ``"cutouts"`` and
            ``"metadata"`` keys.

    Returns:
        List of ``(source_id, image_array)`` tuples.
    """
    return [item for batch in batches for item in _iter_batch_items(batch)]


def is_labeled_cache_populated(cache_path: str | Path) -> bool:
    """Check if a labeled-cache directory contains all required artefacts.

    Args:
        cache_path: Path to the labeled cache directory.

    Returns:
        ``True`` if ``images.zarr``, ``metadata.parquet`` and
        ``cache_info.json`` all exist inside *cache_path*.
    """
    p = Path(cache_path)
    return (
        (p / LabeledDataCache.IMAGES_ZARR).exists()
        and (p / LabeledDataCache.METADATA_PARQUET).exists()
        and (p / LabeledDataCache.CACHE_INFO_JSON).exists()
    )
