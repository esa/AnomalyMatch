#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Least Recently Used (LRU) image cache for on-demand loading of prediction source images.

Images are loaded from disk using the same fitsbolt pipeline used
during training, ensuring visual consistency between training and
prediction views.
"""

from __future__ import annotations

import os
from collections import OrderedDict

import numpy as np
from dotmap import DotMap
from loguru import logger

from anomaly_match.image_processing.image_utils import ensure_uint8_hwc


class ImageCache:
    """LRU cache that loads source images on demand.

    Keeps up to *cache_size* images in memory.  When the cache is full
    the least-recently-used entry is evicted.  Images are loaded from
    *search_dir* using the fitsbolt pipeline configured in *cfg*.

    Args:
        cfg: Configuration object with ``normalisation`` settings and
            ``fitsbolt_cfg`` for the image processing pipeline.
        search_dir: Root directory to search for image files.
        cache_size: Maximum number of images to keep in memory.
    """

    def __init__(
        self,
        cfg: DotMap,
        search_dir: str,
        cache_size: int = 200,
    ) -> None:
        self._cfg = cfg
        self._search_dir = search_dir
        self._cache_size = cache_size
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_image(self, filename: str) -> np.ndarray | None:
        """Return the image for *filename*, loading from disk if needed.

        On cache hit the entry is promoted to most-recently-used.  On
        cache miss the image is loaded via fitsbolt's
        ``load_and_process_single_wrapper`` and inserted into the cache,
        evicting the LRU entry if the cache is full.

        Args:
            filename: Source filename (basename or relative path).

        Returns:
            HWC uint8 numpy array, or ``None`` if the file cannot be
            loaded.
        """
        # Cache hit — promote to most-recently-used
        if filename in self._cache:
            self._cache.move_to_end(filename)
            return self._cache[filename]

        # Cache miss — load from disk
        image = self._load_from_disk(filename)
        if image is None:
            return None

        # Ensure uint8 HWC
        image = _ensure_uint8_hwc(image)

        # Insert and evict if necessary
        self._cache[filename] = image
        if len(self._cache) > self._cache_size:
            evicted_key, _ = self._cache.popitem(last=False)
            logger.trace("ImageCache evicted {}", evicted_key)

        return image

    def prefetch(self, filenames: list[str]) -> None:
        """Pre-load a batch of images into the cache.

        Already-cached filenames are skipped.

        Args:
            filenames: List of filenames to pre-load.
        """
        for fn in filenames:
            if fn not in self._cache:
                self.get_image(fn)

    def put(self, filename: str, image: np.ndarray) -> None:
        """Manually insert an image into the cache.

        Useful for caching images loaded via alternative paths (e.g.
        Cutana cutouts) that bypass ``get_image``.

        Args:
            filename: Cache key.
            image: HWC uint8 numpy array.
        """
        image = _ensure_uint8_hwc(image)
        self._cache[filename] = image
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    def clear(self) -> None:
        """Drop all cached images."""
        self._cache.clear()

    @property
    def size(self) -> int:
        """Return number of currently cached images.

        Returns:
            Current cache occupancy.
        """
        return len(self._cache)

    @property
    def max_size(self) -> int:
        """Return maximum cache capacity.

        Returns:
            Maximum number of images the cache can hold.
        """
        return self._cache_size

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_from_disk(self, filename: str) -> np.ndarray | None:
        """Attempt to load *filename* from disk.

        Tries the full path first, then ``search_dir / filename``, and
        finally ``search_dir / basename(filename)``.

        Args:
            filename: Source filename to load.

        Returns:
            Raw image array, or ``None`` on failure.
        """
        from anomaly_match.data_io.load_images import (  # noqa: PLC0415
            load_and_process_single_wrapper,  # lazy: avoid circular import
        )

        # Resolve filename: may be absolute, relative to search_dir,
        # or a nested path where only the basename exists in search_dir
        candidates = [
            filename,
            os.path.join(self._search_dir, filename),
            os.path.join(self._search_dir, os.path.basename(filename)),
        ]

        for path in candidates:
            if os.path.isfile(path):
                try:
                    return load_and_process_single_wrapper(
                        path,
                        self._cfg,
                        desc="image_cache",
                        show_progress=False,
                    )
                except Exception as exc:
                    logger.warning("ImageCache failed to load {}: {}", path, exc)
                    return None

        logger.trace("ImageCache: file not found for {}", filename)
        return None


def _ensure_uint8_hwc(image: np.ndarray) -> np.ndarray:
    """Convert *image* to uint8 HWC format if necessary.

    Delegates to the shared :func:`ensure_uint8_hwc` utility.

    Returns:
        uint8 HWC numpy array.
    """
    return ensure_uint8_hwc(image)
