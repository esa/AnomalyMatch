#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.

"""Abstract data source for training with multiple storage backends.

Provides a `TrainingDataSource` ABC that decouples image loading from the
dataset labeling/splitting logic in `AnomalyDetectionDataset`. Concrete
implementations handle image folders, Zarr arrays, and Cutana catalogues.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
from dotmap import DotMap
from loguru import logger

from anomaly_match.data_io.find_images_in_folder import get_image_names_from_folder
from anomaly_match.data_io.load_images import (
    detect_num_channels,
    load_and_process_wrapper,
)


class DataSourceType(str, Enum):
    """Storage backend types for training data."""

    IMAGE_FOLDER = "image_folder"
    ZARR = "zarr"
    CUTANA = "cutana"


class TrainingDataSource(ABC):
    """Abstract base class for loading training data from various storage formats.

    Each implementation handles a different storage backend (filesystem images,
    Zarr arrays, Cutana catalogues). The source delivers
    preprocessed numpy arrays that `AnomalyDetectionDataset` can label and split.

    Args:
        cfg: AnomalyMatch configuration with normalisation settings.
    """

    def __init__(self, cfg: DotMap) -> None:
        self._cfg = cfg
        # Optional hook fired right after an unlabeled batch's source sizes are
        # recorded — before the slow cutout streaming.  The training subprocess
        # sets this to emit the size histogram early, so the plot is visible
        # while cutouts stream instead of only after.  Only Cutana records
        # sizes, so it's the only source that fires this.
        self.unlabeled_sizes_callback: Callable[[], None] | None = None

    @abstractmethod
    def get_labeled_images(self, label_df: pd.DataFrame) -> list[tuple[str, np.ndarray]]:
        """Load images whose filenames appear in the label DataFrame.

        Args:
            label_df: DataFrame with at least ``filename`` and ``label`` columns.

        Returns:
            List of (filename, image_array) tuples. image_array is HWC numpy,
            already resized and normalised per cfg.
        """

    @abstractmethod
    def get_unlabeled_batch(self, n: int, exclude: set[str]) -> list[tuple[str, np.ndarray]]:
        """Load a batch of up to *n* unlabeled images.

        Args:
            n: Maximum number of images to return.
            exclude: Filenames to skip (already labeled).

        Returns:
            List of (filename, image_array) tuples.
        """

    @abstractmethod
    def get_total_count(self) -> int:
        """Return the total number of images available in the data source."""

    @abstractmethod
    def detect_num_channels(self) -> int | None:
        """Detect the number of image channels from a sample.

        Returns:
            Number of channels, or ``None`` when detection is not applicable
            (e.g. FITS files where channels depend on configuration).
        """

    def get_last_unlabeled_sizes(self) -> np.ndarray | None:
        """Source sizes (diameters) of the most recent unlabeled batch.

        Only Cutana catalogues carry a per-source size; other backends have no
        such column and return ``None``.  The training subprocess uses this to
        surface a size-distribution histogram of the sampled unlabeled pool.

        Returns:
            Finite source diameters from the last ``get_unlabeled_batch`` call,
            or ``None`` when the source exposes no size.
        """
        return None

    def get_last_unlabeled_size_unit(self) -> str | None:
        """Unit of :meth:`get_last_unlabeled_sizes` (``"pixel"`` / ``"arcsec"``).

        Returns:
            The size unit, or ``None`` when the source exposes no size.
        """
        return None

    def get_last_unlabeled_population_sizes(self) -> np.ndarray | None:
        """Candidate-population diameters of the last unlabeled batch's tiles.

        The population is every source in the tiles the sampler drew from — the
        baseline the training screen overlays the sampled pool against to show
        whether size stratification flattened the pool.  Only Cutana exposes it;
        other backends return ``None``.

        Returns:
            Finite population diameters, or ``None`` when the source is unsized.
        """
        return None


# ---------------------------------------------------------------------------
# Concrete implementations
# ---------------------------------------------------------------------------


class ImageFolderSource(TrainingDataSource):
    """Load training data from a directory of image files.

    Args:
        cfg: Configuration; ``cfg.data_dir`` must point to an image directory.
    """

    def __init__(self, cfg: DotMap) -> None:
        super().__init__(cfg)
        self._root_dir = cfg.data_dir
        self._all_filenames = get_image_names_from_folder(self._root_dir, recursive=False)

        rng = np.random.RandomState(cfg.seed)
        rng.shuffle(self._all_filenames)
        self._filename_set = set(self._all_filenames)

    @property
    def all_filenames(self) -> list[str]:
        """All discovered image filenames (shuffled)."""
        return self._all_filenames

    def get_labeled_images(self, label_df: pd.DataFrame) -> list[tuple[str, np.ndarray]]:
        """Load all images whose filenames appear in *label_df*.

        Returns:
            List of (filename, image_array) tuples for matched files.
        """
        labeled_files = [f for f in label_df["id"] if f in self._filename_set]
        if not labeled_files:
            return []

        filepaths = [os.path.join(self._root_dir, f) for f in labeled_files]
        results = load_and_process_wrapper(filepaths, self._cfg, desc="Loading labeled images")
        return [(os.path.basename(fp), img) for fp, img in results]

    def get_unlabeled_batch(self, n: int, exclude: set[str]) -> list[tuple[str, np.ndarray]]:
        """Load up to *n* images that are not in *exclude*.

        Returns:
            List of (filename, image_array) tuples.
        """
        candidates = [f for f in self._all_filenames if f not in exclude]
        batch_files = candidates[:n]
        if not batch_files:
            return []

        filepaths = [os.path.join(self._root_dir, f) for f in batch_files]
        results = load_and_process_wrapper(filepaths, self._cfg, desc="Loading unlabeled batch")
        return [(os.path.basename(fp), img) for fp, img in results]

    def get_total_count(self) -> int:
        """Return number of image files in the directory."""
        return len(self._all_filenames)

    def detect_num_channels(self) -> int | None:
        """Auto-detect channels from a sample image.

        Returns:
            Number of channels detected from the first image.
        """
        return detect_num_channels(self._root_dir, self._all_filenames)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _auto_detect_source_type(data_dir: str) -> DataSourceType:
    """Infer the training data source type from *data_dir*.

    Checks the path extension first, then scans directory contents to
    detect Zarr stores or Cutana catalogues.

    Args:
        data_dir: Path to the data directory.

    Returns:
        The detected ``DataSourceType``.
    """
    # Strip trailing separators so ".zarr/" still matches ".zarr"
    lower = data_dir.rstrip(os.sep + "/").lower()
    if lower.endswith(".zarr"):
        return DataSourceType.ZARR
    if lower.endswith((".csv", ".parquet")):
        return DataSourceType.CUTANA

    # Directory: check contents
    if os.path.isdir(data_dir):
        try:
            entries = os.listdir(data_dir)
        except OSError:
            return DataSourceType.IMAGE_FOLDER
        lower_entries = [e.lower() for e in entries]
        # Zarr store inside a directory
        if any(e.endswith(".zarr") for e in lower_entries):
            return DataSourceType.ZARR
        # Check if any CSV/parquet files are actual Cutana catalogues
        from anomaly_match.datasets.cutana_source import (  # noqa: PLC0415 — lazy: avoid circular import
            _is_cutana_catalogue,
        )

        for entry in entries:
            if entry.lower().endswith((".csv", ".parquet")):
                if _is_cutana_catalogue(Path(os.path.join(data_dir, entry))):
                    return DataSourceType.CUTANA

    return DataSourceType.IMAGE_FOLDER


def create_training_data_source(cfg: DotMap, *, lazy: bool = False) -> TrainingDataSource:
    """Create the appropriate data source from configuration.

    Uses ``cfg.training_data_source`` if set explicitly, otherwise auto-detects
    from ``cfg.data_dir``.

    Args:
        cfg: AnomalyMatch configuration.
        lazy: If True, skip O(N) indexing for container sources. Labeled
            images must come from a ``LabeledDataCache`` instead. Image
            folder sources ignore this flag.

    Returns:
        A concrete ``TrainingDataSource`` instance.

    Raises:
        ValueError: If the source type is unknown.
    """
    source_type = cfg.training_data_source
    if source_type is None:
        source_type = _auto_detect_source_type(cfg.data_dir)

    logger.info("Training data source: {} (data_dir={})", source_type, cfg.data_dir)

    if source_type == DataSourceType.IMAGE_FOLDER:
        return ImageFolderSource(cfg)
    elif source_type == DataSourceType.ZARR:
        from anomaly_match.datasets.zarr_source import (  # noqa: PLC0415
            ZarrSource,  # lazy: avoid importing heavy optional dependency
        )

        return ZarrSource(cfg, lazy=lazy)
    elif source_type == DataSourceType.CUTANA:
        from anomaly_match.datasets.cutana_source import (  # noqa: PLC0415
            CutanaSource,  # lazy: avoid importing heavy optional dependency
        )

        return CutanaSource(cfg, lazy=lazy)
    else:
        raise ValueError(f"Unknown training data source type: {source_type!r}")
