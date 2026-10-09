#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.

"""Zarr data source for training from pixel-array containers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from dotmap import DotMap
from loguru import logger

from anomaly_match.data_io.container_loaders import decode_zarr_image
from anomaly_match.datasets.training_data_source import TrainingDataSource


class ZarrSource(TrainingDataSource):
    """Load training data from one or more Zarr arrays with optional metadata.

    Supports two layouts:

    * **Single store** -- ``cfg.data_dir`` points directly to a ``.zarr``
      directory containing an ``images`` array.
    * **Multi-store directory** -- ``cfg.data_dir`` points to a parent
      directory containing multiple ``.zarr`` stores, each with an ``images``
      array.  All stores are indexed transparently.

    When ``lazy=True``, the expensive O(N) filename index and shuffled-index
    permutation are skipped. Unlabeled sampling uses random ``(store, index)``
    generation instead. Labeled images must come from a ``LabeledDataCache``,
    not from this source.

    Args:
        cfg: Configuration; ``cfg.data_dir`` must point to a ``.zarr``
            directory or a parent directory containing ``.zarr`` stores.
        lazy: If True, skip O(N) indexing for billion-scale sources.
    """

    def __init__(self, cfg: DotMap, *, lazy: bool = False) -> None:
        super().__init__(cfg)
        self._zarr_path = Path(cfg.data_dir)
        self._lazy = lazy

        # Discover stores: single .zarr or parent with multiple .zarr children
        self._stores: list[_ZarrStoreInfo] = self._discover_stores()
        if not self._stores:
            raise ValueError(f"No Zarr stores with 'images' array found in {self._zarr_path}")

        self._total = sum(s.count for s in self._stores)

        if lazy:
            # Lightweight: only store sizes, no filename or shuffle index
            self._name_to_loc = None
            self._filenames = None
            self._shuffled_indices = None
            store_desc = ", ".join(f"{s.path.name}({s.count})" for s in self._stores)
            logger.debug(f"ZarrSource(lazy): {self._total} images from [{store_desc}]")
        else:
            # Full mode: build O(N) filename index + shuffled indices
            self._name_to_loc: dict[str, tuple[int, int]] = {}
            self._filenames: list[str] = []
            for si, store in enumerate(self._stores):
                for li, fn in enumerate(store.filenames):
                    self._name_to_loc[fn] = (si, li)
                    self._filenames.append(fn)

            rng = np.random.RandomState(cfg.seed)
            self._shuffled_indices = rng.permutation(self._total)

            store_desc = ", ".join(f"{s.path.name}({s.count})" for s in self._stores)
            logger.debug(f"ZarrSource: indexed {self._total} images from [{store_desc}]")

    # ------------------------------------------------------------------
    # Store discovery
    # ------------------------------------------------------------------

    def _discover_stores(self) -> list[_ZarrStoreInfo]:
        """Find all Zarr stores under the configured path.

        Returns:
            List of store info objects, one per Zarr store.
        """
        zarr_path = self._zarr_path

        # Case 1: path is itself a .zarr directory
        if zarr_path.suffix == ".zarr" or str(zarr_path).rstrip("/\\").lower().endswith(".zarr"):
            return self._open_single_store(zarr_path)

        # Case 2: parent directory containing .zarr children
        if zarr_path.is_dir():
            children = sorted(zarr_path.glob("*.zarr"))
            if children:
                stores = []
                for child in children:
                    stores.extend(self._open_single_store(child))
                return stores

        return []

    def _open_single_store(self, path: Path) -> list[_ZarrStoreInfo]:
        """Open a single Zarr store and return its info.

        Args:
            path: Path to the ``.zarr`` directory.

        Returns:
            Single-element list with the store info, or empty if invalid.
        """
        root = zarr.open_group(str(path), mode="r")
        if "images" not in root:
            logger.warning(f"ZarrSource: no 'images' array in {path}, skipping")
            return []

        images = root["images"]
        count = images.shape[0]
        filenames = self._load_filenames(root, path, count)
        return [
            _ZarrStoreInfo(path=path, root=root, images=images, count=count, filenames=filenames)
        ]

    # ------------------------------------------------------------------
    # Filename loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_filenames(root: zarr.Group, zarr_path: Path, total: int) -> list[str]:
        """Discover filenames from parquet metadata or generate from indices.

        Args:
            root: Open Zarr group.
            zarr_path: Filesystem path to the Zarr store.
            total: Number of images in the store.

        Returns:
            List of filenames, one per image.
        """
        zarr_prefix = zarr_path.parent.name if zarr_path.name == "images.zarr" else zarr_path.stem

        metadata_file = None

        # Check zarr attributes
        if "metadata_file" in root.attrs:
            candidate = Path(root.attrs["metadata_file"])
            if not candidate.is_absolute():
                candidate = zarr_path.parent / candidate.name
            if candidate.exists():
                metadata_file = candidate

        # Fallback: <zarr_stem>_metadata.parquet next to zarr
        if metadata_file is None:
            candidate = zarr_path.parent / f"{zarr_path.stem}_metadata.parquet"
            if candidate.exists():
                metadata_file = candidate

        # Fallback for batch folders: images_metadata.parquet
        if metadata_file is None and zarr_path.name == "images.zarr":
            candidate = zarr_path.parent / "images_metadata.parquet"
            if candidate.exists():
                metadata_file = candidate

        if metadata_file is not None:
            logger.debug(f"ZarrSource: loading metadata from {metadata_file}")
            try:
                df = pd.read_parquet(metadata_file)
                for col in ("original_filename", "filename", "source_id"):
                    if col in df.columns:
                        filenames = df[col].tolist()
                        if len(filenames) == total:
                            return filenames
                        logger.warning(
                            f"Metadata count ({len(filenames)}) != image count ({total})"
                        )
                        break
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")

        # Generate filenames from indices
        return [f"{zarr_prefix}__image_{i:06d}" for i in range(total)]

    # ------------------------------------------------------------------
    # TrainingDataSource interface
    # ------------------------------------------------------------------

    def get_labeled_images(self, label_df: pd.DataFrame) -> list[tuple[str, np.ndarray]]:
        """Load images for filenames present in *label_df*.

        Not available in lazy mode -- use a ``LabeledDataCache`` instead.

        Returns:
            List of (filename, image_array) tuples.

        Raises:
            RuntimeError: If called in lazy mode.
        """
        if self._lazy:
            raise RuntimeError(
                "get_labeled_images() not available in lazy mode. Use a LabeledDataCache instead."
            )

        results: list[tuple[str, np.ndarray]] = []
        for fn in label_df["id"]:
            if fn in self._name_to_loc:
                si, li = self._name_to_loc[fn]
                img = decode_zarr_image(self._stores[si].images[li], self._cfg)
                results.append((fn, img))

        logger.debug(f"ZarrSource: loaded {len(results)} labeled images")
        return results

    def get_unlabeled_batch(
        self,
        n: int,
        exclude: set[str] | None = None,
        exclude_positions: set[tuple[int, int]] | None = None,
    ) -> list[tuple[str, np.ndarray]]:
        """Load up to *n* unlabeled images.

        In normal mode, skips filenames in *exclude*. In lazy mode, skips
        ``(store_idx, local_idx)`` pairs in *exclude_positions* using
        random sampling without a pre-computed permutation.

        Args:
            n: Maximum number of images to return.
            exclude: Filenames to skip (normal mode).
            exclude_positions: ``(store_idx, local_idx)`` pairs to skip
                (lazy mode).

        Returns:
            List of (filename, image_array) tuples.
        """
        if self._lazy:
            return self._get_unlabeled_batch_lazy(n, exclude_positions or set())

        results: list[tuple[str, np.ndarray]] = []
        exclude = exclude or set()
        for idx in self._shuffled_indices:
            if len(results) >= n:
                break
            fn = self._filenames[idx]
            if fn not in exclude:
                si, li = self._name_to_loc[fn]
                img = decode_zarr_image(self._stores[si].images[li], self._cfg)
                results.append((fn, img))

        logger.debug(f"ZarrSource: loaded {len(results)} unlabeled images")
        return results

    def _get_unlabeled_batch_lazy(
        self,
        n: int,
        exclude_positions: set[tuple[int, int]],
    ) -> list[tuple[str, np.ndarray]]:
        """Sample unlabeled images via random index generation (lazy mode).

        Generates random ``(store_idx, local_idx)`` pairs weighted by store
        size. Does not require a pre-built filename index or shuffled
        permutation, keeping memory constant regardless of dataset size.

        Returns:
            List of (filename, image_array) tuples.
        """
        rng = np.random.RandomState(self._cfg.seed)
        store_sizes = np.array([s.count for s in self._stores])
        store_weights = store_sizes / store_sizes.sum()

        seen: set[tuple[int, int]] = set()
        results: list[tuple[str, np.ndarray]] = []
        max_attempts = n * 10  # avoid infinite loop on tiny datasets

        for _ in range(max_attempts):
            if len(results) >= n:
                break
            si = int(rng.choice(len(self._stores), p=store_weights))
            li = int(rng.randint(0, self._stores[si].count))

            if (si, li) in exclude_positions or (si, li) in seen:
                continue
            seen.add((si, li))

            img = decode_zarr_image(self._stores[si].images[li], self._cfg)
            # Generate filename from store prefix + index
            store = self._stores[si]
            prefix = store.path.parent.name if store.path.name == "images.zarr" else store.path.stem
            fn = f"{prefix}__image_{li:06d}"
            results.append((fn, img))

        logger.debug(f"ZarrSource(lazy): sampled {len(results)} unlabeled images")
        return results

    def get_total_count(self) -> int:
        """Return total number of images across all Zarr stores.

        Returns:
            Image count.
        """
        return self._total

    def detect_num_channels(self) -> int | None:
        """Infer channel count from the first store's array shape.

        Returns:
            Number of channels, or ``None`` if the array is empty.
        """
        if self._total == 0:
            return None
        shape = self._stores[0].images.shape
        # (N, H, W, C) or (N, C, H, W)
        if len(shape) == 4:
            # Smaller of last vs second dim is likely channels
            return min(shape[1], shape[3])
        return None


class _ZarrStoreInfo:
    """Lightweight container for metadata about one opened Zarr store."""

    __slots__ = ("path", "root", "images", "count", "filenames")

    def __init__(
        self,
        path: Path,
        root: zarr.Group,
        images: zarr.Array,
        count: int,
        filenames: list[str],
    ) -> None:
        self.path = path
        self.root = root
        self.images = images
        self.count = count
        self.filenames = filenames
