#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.

"""Main dataset class for anomaly detection with labeled and unlabeled data."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotmap import DotMap
from loguru import logger
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from anomaly_match.data_io.container_loaders import (
    decode_cutana_raw_images,
    decode_zarr_image,
)
from anomaly_match.data_io.labeled_data_cache import LabeledDataCache
from anomaly_match.data_io.metadata_handler import MetadataHandler
from anomaly_match.utils.tqdm_logging import tqdm_logging_file

from .Label import LABEL_ANOMALY, LABEL_NORMAL, LABEL_REMOVED, VALID_CSV_LABELS, Label
from .training_data_source import DataSourceType, ImageFolderSource, TrainingDataSource

# Cutouts are decoded a chunk at a time so the "Decoding labeled cache" progress
# meter keeps climbing instead of freezing for the whole (now parallel) decode.
# Each chunk is decoded across the pod's cores in one batched call, so the chunk
# only needs to be large enough to keep that thread pool full.
_DECODE_CHUNK_SIZE = 256


class AnomalyDetectionDataset:
    """Dataset for binary classification of normal vs anomaly images.

    Handles labeled/unlabeled data management, train/test splitting, and
    label updates. Image loading is delegated to a ``TrainingDataSource``.

    Args:
        cfg: Configuration object containing dataset parameters.
        data_source: Optional data source for loading images. If ``None``,
            an ``ImageFolderSource`` is created from ``cfg.data_dir``.
        labeled_cache_path: Optional path to a ``LabeledDataCache`` directory.
            When provided, labeled images are read from the cache and
            normalisation is applied at load time. The *data_source* is then
            used only for unlabeled sampling (should be created with
            ``lazy=True`` for container sources).
    """

    def __init__(
        self,
        cfg: DotMap,
        data_source: TrainingDataSource | None = None,
        labeled_cache_path: str | None = None,
    ) -> None:
        logger.debug(f"Loading AnomalyDetectionDataset from {cfg.data_dir}")

        self._labeled_cache_path = labeled_cache_path
        self.seed = cfg.seed
        self.size = cfg.normalisation.image_size
        self.root_dir = cfg.data_dir
        self.cfg = cfg
        self.test_ratio = cfg.test_ratio
        self.N_to_load = cfg.N_to_load
        self.label_file = (
            cfg.label_file if cfg.label_file else os.path.join(self.root_dir, "labeled_data.csv")
        )

        # Create data source (backward compatible default)
        if data_source is None:
            data_source = ImageFolderSource(cfg)
        self._data_source = data_source

        # For ImageFolderSource, expose discovered filenames; for other
        # sources the full filename list is built after loading.
        if isinstance(data_source, ImageFolderSource):
            self.all_filenames = data_source.all_filenames
        else:
            self.all_filenames = []

        # Auto-detect channel count from data source.  A channel_combination
        # matrix defines the output count by its rows (a 1x3 matrix turns RGB
        # into one channel), so raising n_output_channels to the source's band
        # count would contradict the images the matrix actually produces.
        detected_channels = data_source.detect_num_channels()
        if (
            detected_channels is not None
            and detected_channels > cfg.normalisation.n_output_channels
            and not isinstance(cfg.normalisation.channel_combination, (list, tuple, np.ndarray))
        ):
            logger.info(
                f"Detected {detected_channels} channels from images "
                f"(config had n_output_channels={cfg.normalisation.n_output_channels}), updating"
            )
            old_channels = cfg.normalisation.n_output_channels
            cfg.normalisation.n_output_channels = detected_channels
            cfg.num_channels = detected_channels
            for attr in ("norm_asinh_scale", "norm_asinh_clip"):
                val = cfg.normalisation[attr]
                if isinstance(val, list) and len(val) == old_channels:
                    cfg.normalisation[attr] = val + [val[-1]] * (detected_channels - old_channels)
        self.num_channels = cfg.normalisation.n_output_channels

        self.N_to_load = min(self.N_to_load, data_source.get_total_count())

        # Initialize metadata handler
        self.metadata_handler = MetadataHandler(cfg.metadata_file, self.all_filenames)

        logger.info(f"Found {data_source.get_total_count()} total images in {self.root_dir}")

        self.split_indices = None
        self.data_dict: dict[str, tuple[np.ndarray, Label]] = {}

        # Load data via the data source
        self._load_from_source()

        # Load the labels from the CSV file
        self._load_csv_and_apply_labels()

        # Split the data into training and testing sets
        self._split_data()

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_from_source(self) -> None:
        """Load labeled and unlabeled images via the data source.

        When a labeled cache path is set, labeled images are loaded from
        the cache (with normalisation applied) instead of from the data
        source. The data source is then used only for unlabeled sampling.
        """
        label_df = pd.read_csv(self.label_file)
        # Keys must be strings: CutanaSource returns source IDs as str
        # from orchestrator metadata, so dict lookups need str keys.
        labeled_dict = {str(k): v for k, v in label_df.set_index("id")["label"].to_dict().items()}

        if self._labeled_cache_path:
            self._load_labeled_from_cache(labeled_dict)
        else:
            # Load labeled images from data source
            for fn, img in self._data_source.get_labeled_images(label_df):
                if fn in labeled_dict:
                    label = Label.NORMAL if labeled_dict[fn] == LABEL_NORMAL else Label.ANOMALY
                    self.data_dict[fn] = (img, label)

        logger.info(f"Loaded {len(self.data_dict)} labeled images")

        # Load unlabeled batch — can take minutes for streaming sources (Cutana)
        logger.info(f"Loading up to {self.N_to_load} unlabeled images...")
        # Defense-in-depth: the sampler already excludes labeled ids, but if one
        # still resurfaces, never overwrite its labeled entry with the streamed
        # cutout — that array can have a different band count than the labeled
        # cache and would put a mismatched-channel image into the labeled set
        # (torch.stack then fails on mixed shapes).
        labeled_fns = set(self.data_dict.keys())
        n_skipped_labeled = 0
        n_skipped_duplicate = 0
        for fn, img in self._data_source.get_unlabeled_batch(self.N_to_load, labeled_fns):
            # Tested against the pre-loop snapshot, not data_dict: the dict grows
            # with each accepted cutout, so testing it alone would score a
            # sampler-emitted duplicate as a labeled leak.
            if fn in labeled_fns:
                n_skipped_labeled += 1
                continue
            if fn in self.data_dict:
                n_skipped_duplicate += 1
                continue
            self.data_dict[fn] = (img, Label.UNKNOWN)
        if n_skipped_labeled:
            logger.warning(
                "{} streamed cutout(s) skipped: id already labeled — the unlabeled "
                "sampler leaked a labeled source and the batch is short by that many",
                n_skipped_labeled,
            )
        if n_skipped_duplicate:
            # Not a leak: the sampler doesn't dedup SourceID across tiles, so a
            # source in an overlap region can be drawn from two of them.
            logger.info(
                "{} streamed cutout(s) skipped: id already sampled this batch — "
                "the batch is short by that many",
                n_skipped_duplicate,
            )

        n_unlabeled = sum(1 for _, lbl in self.data_dict.values() if lbl == Label.UNKNOWN)
        logger.info(f"Loaded {n_unlabeled} unlabeled images")

        # For non-folder sources, build all_filenames from what was loaded
        if not self.all_filenames:
            self.all_filenames = list(self.data_dict.keys())

    def _load_labeled_from_cache(self, labeled_dict: dict[str, str]) -> None:
        """Load labeled images from a LabeledDataCache, applying normalisation.

        Raw images from the cache are processed through the same normalisation
        pipeline used by the data sources.

        Args:
            labeled_dict: Map of ``id -> label_string`` for label assignment.

        Raises:
            RuntimeError: If the labeled cache is not populated.
        """
        cache = LabeledDataCache(Path(self._labeled_cache_path))
        if not cache.is_populated:
            raise RuntimeError(
                f"Labeled data cache at {self._labeled_cache_path} is not populated. "
                "Return to the Training Setup screen and re-validate the data source "
                "to rebuild the cache."
            )

        # Both cache flavours now store unnormalised data: Zarr holds the
        # store's raw arrays, and the Cutana cache (post-#392-refactor)
        # holds per-band float32 cutouts at the cache resolution.  The
        # read path applies the full fitsbolt pipeline
        # (normalise + resize + combine) so normalisation, output dtype,
        # channel combination and image size can all change without
        # forcing a cache rebuild.
        # Decoding applies the full fitsbolt pipeline (resize + normalise +
        # channel combination) per cached cutout.  The resize anti-aliasing
        # dominates and releases the GIL, so the Cutana path decodes a whole
        # chunk across the pod's cores in one batched call (issue #500) instead
        # of the previous single-threaded per-image loop that took minutes on a
        # thousand-plus labels.  The legacy Zarr cache has no batched decoder, so
        # it decodes per item.
        if cache.source_type == DataSourceType.CUTANA:
            batch_decode = decode_cutana_raw_images
        else:

            def batch_decode(raw_images, cfg):
                return [decode_zarr_image(raw, cfg) for raw in raw_images]

        labeled_items = [
            (item_id, raw) for item_id, raw in cache.get_raw_images() if item_id in labeled_dict
        ]
        # Feed the decoder in chunks so the meter advances steadily and the
        # ``tqdm_logging_file`` relay keeps the UI alive (see its module
        # docstring), rather than jumping 0 -> 100% after one big call.
        meter = tqdm(
            total=len(labeled_items),
            desc="Decoding labeled cache",
            unit="img",
            file=tqdm_logging_file(),
            mininterval=1.0,
        )
        try:
            for start in range(0, len(labeled_items), _DECODE_CHUNK_SIZE):
                chunk = labeled_items[start : start + _DECODE_CHUNK_SIZE]
                decoded = batch_decode([raw for _, raw in chunk], self.cfg)
                for (item_id, _), img in zip(chunk, decoded):
                    label = Label.NORMAL if labeled_dict[item_id] == LABEL_NORMAL else Label.ANOMALY
                    self.data_dict[item_id] = (img, label)
                meter.update(len(chunk))
        finally:
            meter.close()

        logger.debug(f"Loaded {len(self.data_dict)} labeled images from cache")

    # ------------------------------------------------------------------
    # CSV label handling
    # ------------------------------------------------------------------

    def _load_csv_and_apply_labels(self) -> None:
        """Load CSV label file and apply labels to the dataset."""
        logger.debug(f"Reloading CSV files from {self.label_file} and applying labels")
        assert os.path.exists(self.label_file), (
            f"No label file found at {self.label_file}. Please provide a csv file in the format: id,"
            + "label with labels 'normal' and 'anomaly'."
        )
        labeled_data = pd.read_csv(self.label_file)

        assert "label" in labeled_data.columns, "CSV file must contain column 'label'"
        assert "id" in labeled_data.columns, "CSV file must contain column 'id'"

        assert set(labeled_data["label"].unique()) <= VALID_CSV_LABELS, (
            "Labels should be either 'normal', 'anomaly' or 'removed' but found"
            + str(set(labeled_data["label"].unique()))
        )

        normal_count = labeled_data["label"].value_counts().get(LABEL_NORMAL, 0)
        anomaly_count = labeled_data["label"].value_counts().get(LABEL_ANOMALY, 0)
        removed_count = labeled_data["label"].value_counts().get(LABEL_REMOVED, 0)
        logger.debug(
            f"Label distribution in CSV file: Normal: {normal_count}, Anomaly: {anomaly_count}, "
            f"Removed: {removed_count}"
        )

        self.update_labels(labeled_data)

    # ------------------------------------------------------------------
    # Train/test splitting
    # ------------------------------------------------------------------

    def _split_data(self) -> None:
        """Split data into training and testing sets using only labeled images."""
        logger.debug(f"Splitting data with seed={self.seed}")
        assert self.split_indices is None, "Data was already split before"

        labeled_data = {
            filename: (img, label)
            for filename, (img, label) in self.data_dict.items()
            if label != Label.UNKNOWN
        }

        filenames = np.array(list(labeled_data.keys()))
        labels = np.array([label for _, label in labeled_data.values()])
        logger.trace(f"Labels to split: {labels}")

        stratify = labels if len(set(labels)) > 1 else None

        logger.trace(f"Number of labeled data points: {len(filenames)}")
        if self.test_ratio > 0:
            filenames_train, filenames_test = train_test_split(
                filenames,
                test_size=self.test_ratio,
                random_state=self.seed,
                stratify=stratify,
            )
        else:
            filenames_train, filenames_test = filenames, []

        self.split_indices = {
            "train": set(filenames_train),
            "test": set(filenames_test),
        }

    # ------------------------------------------------------------------
    # Data access properties
    # ------------------------------------------------------------------

    @property
    def unlabeled_filepaths(self) -> list[str]:
        """Return unlabeled filepaths."""
        return [
            os.path.join(self.root_dir, filename)
            for filename in self.all_filenames
            if (filename not in self.data_dict or self.data_dict[filename][1] == Label.UNKNOWN)
        ]

    @property
    def unlabeled_filenames(self) -> list[str]:
        """Return unlabeled filenames."""
        return [
            filename
            for filename in self.all_filenames
            if (filename not in self.data_dict or self.data_dict[filename][1] == Label.UNKNOWN)
        ]

    @property
    def unlabeled(self) -> list:
        """Return unlabeled data in format [[imgs],[filenames]]."""
        filenames = [
            filename for filename in self.data_dict if self.data_dict[filename][1] == Label.UNKNOWN
        ]
        return [
            [self.data_dict[filename][0] for filename in filenames],
            filenames,
        ]

    @property
    def train_data(self) -> list:
        """Return training data in format [[filenames],[imgs],[labels]]."""
        filenames = list(self.split_indices["train"])
        return [
            filenames,
            [self.data_dict[filename][0] for filename in filenames],
            [self.data_dict[filename][1] for filename in filenames],
        ]

    @property
    def test_data(self) -> list:
        """Return testing data in format [[filenames],[imgs],[labels]]."""
        filenames = list(self.split_indices["test"])
        return [
            filenames,
            [self.data_dict[filename][0] for filename in filenames],
            [self.data_dict[filename][1] for filename in filenames],
        ]

    # ------------------------------------------------------------------
    # Label updates (active learning)
    # ------------------------------------------------------------------

    def update_labels(self, new_labels_df: pd.DataFrame) -> None:
        """Update the dataset with new labels for previously unlabeled images.

        Args:
            new_labels_df: DataFrame with an ``id`` (or ``filename``) column and ``label``.

        Raises:
            ValueError: If an invalid label is encountered.
        """
        logger.debug(f"Updating labels for {new_labels_df.shape[0]} images")
        existing_values_changed = 0
        missing_filenames: list[str] = []

        for _, row in new_labels_df.iterrows():
            filename = os.path.basename(str(row["id"]))
            label = row["label"].lower()

            if label == LABEL_NORMAL:
                label_enum = Label.NORMAL
            elif label == LABEL_ANOMALY:
                label_enum = Label.ANOMALY
            elif label == LABEL_REMOVED:
                continue
            else:
                raise ValueError(f"Invalid label {label} for {filename}")

            if filename in self.data_dict:
                image, current_label = self.data_dict[filename]
                if current_label != label_enum and current_label != Label.UNKNOWN:
                    self.data_dict[filename] = (image, label_enum)
                    existing_values_changed += 1
                elif current_label == Label.UNKNOWN:
                    self.data_dict[filename] = (image, label_enum)
            else:
                # Accumulate instead of logging per row — Cutana catalogues
                # with thousands of labels that didn't match the source
                # previously produced one WARNING line per orphan and
                # flooded the training screen.  One summary line at the
                # end is enough to flag the condition without the spam.
                missing_filenames.append(filename)

        logger.debug(f"Updated {existing_values_changed} existing labels")
        if missing_filenames:
            preview = ", ".join(missing_filenames[:3])
            if len(missing_filenames) > 3:
                preview += f" (+{len(missing_filenames) - 3} more)"
            logger.warning(
                "{} labelled id(s) not present in the loaded dataset — these rows "
                "are ignored for training: {}",
                len(missing_filenames),
                preview,
            )

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def get_metadata_for_file(self, filename: str) -> dict | None:
        """Get metadata for a specific file.

        Args:
            filename: The filename to get metadata for.

        Returns:
            Metadata for the file, or None if not found.
        """
        return self.metadata_handler.get_metadata_for_file(filename)

    def get_all_metadata(self) -> pd.DataFrame | None:
        """Get all metadata.

        Returns:
            The full metadata DataFrame, or None if no metadata loaded.
        """
        return self.metadata_handler.get_all_metadata()
