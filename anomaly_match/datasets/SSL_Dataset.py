#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.

"""Semi-supervised dataset splitting for labeled and unlabeled data."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from loguru import logger

from anomaly_match.image_processing.transforms import (
    get_prediction_transforms,
    get_weak_transforms,
)

from .AnomalyDetectionDataset import AnomalyDetectionDataset
from .BasicDataset import BasicDataset
from .training_data_source import TrainingDataSource

if TYPE_CHECKING:
    from collections.abc import Callable

    from dotmap import DotMap


class SSL_Dataset:
    """SSL_Dataset class separates labeled and unlabeled data.

    Returns BasicDataset: torch.utils.data.Dataset (see datasets.dataset.py).

    Args:
        cfg: Configuration object containing dataset parameters.
        train: True means the dataset is training dataset (default=True).
        data_source: Optional data source for loading images. If ``None``,
            an ``ImageFolderSource`` is created from ``cfg.data_dir``.
        labeled_cache_path: Optional path to a ``LabeledDataCache`` directory.
            Passed through to ``AnomalyDetectionDataset``.
    """

    def __init__(
        self,
        cfg: DotMap,
        train: bool = True,
        data_source: TrainingDataSource | None = None,
        labeled_cache_path: str | None = None,
    ) -> None:
        self.seed = cfg.seed
        self.test_ratio = cfg.test_ratio
        self.N_to_load = cfg.N_to_load
        self.train = train
        self.num_classes = 2
        self.size = cfg.normalisation.image_size
        self.data_dir = cfg.data_dir
        self.label_file = cfg.label_file
        self.dset = None
        self.cfg = cfg
        self._data_source = data_source
        self._labeled_cache_path = labeled_cache_path

    def get_data(self) -> tuple[list, list, list | None, list[str], list[str] | None]:
        """Return data (images) and targets (labels)."""
        if self.dset is None:
            self.dset = AnomalyDetectionDataset(
                cfg=self.cfg,
                data_source=self._data_source,
                labeled_cache_path=self._labeled_cache_path,
            )
        else:
            logger.debug("Dataset already loaded.")

        self.num_channels = self.dset.num_channels

        if self.train:
            filenames, imgs, targets = self.dset.train_data
            unlabeled, unlabeled_filenames = self.dset.unlabeled
            self.transform = get_weak_transforms(num_channels=self.num_channels)
        else:
            filenames, imgs, targets = self.dset.test_data
            unlabeled, unlabeled_filenames = None, None  # no unlabeled data in test
            self.transform = get_prediction_transforms(num_channels=self.num_channels)

        return imgs, targets, unlabeled, filenames, unlabeled_filenames

    def get_dset(
        self, use_strong_transform: bool = False, strong_transform: Callable | None = None
    ) -> BasicDataset:
        """Return a BasicDataset containing the returns of get_data.

        Args:
            use_strong_transform: If True, returned dataset generates a pair of weak and
                strong augmented images.
            strong_transform: list of strong_transform (augmentation) if use_strong_transform is True

        Returns:
            BasicDataset configured for evaluation.
        """
        assert not self.train, "get_dset is only for evaluation dataset"

        data, targets, _, filenames, _ = self.get_data()

        logger.debug("Loading evaluation dataset")
        logger.debug(f"Label distribution (0: {targets.count(0)}, 1: {targets.count(1)})")

        return BasicDataset(
            data,
            filenames,
            targets,
            self.num_classes,
            self.transform,
            use_strong_transform,
            strong_transform,
            num_channels=self.num_channels,
        )

    def get_ssl_dset(
        self,
        use_strong_transform: bool = True,
        strong_transform: Callable | None = None,
    ) -> tuple[BasicDataset, BasicDataset]:
        """Split training samples into labeled and unlabeled samples.

        The labeled data is balanced samples over classes.

        Args:
            use_strong_transform: If True, unlabeled dataset returns weak & strong augmented image pair.
                                  If False, unlabeled datasets returns only weak augmented image.
            strong_transform: list of strong transform (RandAugment in FixMatch)

        Returns:
            BasicDataset (for labeled data), BasicDataset (for unlabeled data)

        Raises:
            ValueError: If labeled data contains only one class (all normal
                or all anomaly). FixMatch requires both classes.
        """
        assert self.train, "get_ssl_dset is only for training dataset"

        data, targets, unlabeled, labeled_filenames, unlabeled_filenames = self.get_data()

        logger.debug("Number of labeled training samples: {}".format(len(labeled_filenames)))
        logger.debug("Number of unlabeled samples: {}".format(len(unlabeled_filenames)))
        logger.debug(f"Label distribution (0: {targets.count(0)}, 1: {targets.count(1)})")

        # Add assertion to ensure unlabeled data exists
        assert len(unlabeled_filenames) > 0, (
            "No unlabeled data were provided. Semi-supervised learning requires unlabeled data, "
            + " i.e. images that are not classified in labeled_data.csv."
        )

        # FixMatch requires both classes — abort early with a clear message
        if targets.count(0) == 0 or targets.count(1) == 0:
            raise ValueError(
                f"Labeled data contain only one class "
                f"(normal: {targets.count(0)}, anomaly: {targets.count(1)}). "
                f"Semi-supervised training requires at least one label of each class. "
                f"Please label some images as "
                f"{'anomalies' if targets.count(1) == 0 else 'normal'} before training."
            )

        lb_dset = BasicDataset(
            data,
            labeled_filenames,
            targets,
            self.num_classes,
            self.transform,
            use_strong_transform=False,
            strong_transform=None,
            num_channels=self.num_channels,
        )

        ulb_dset = BasicDataset(
            unlabeled,
            unlabeled_filenames,
            torch.zeros(len(unlabeled)) - 1,  # dummy label
            self.num_classes,
            self.transform,
            use_strong_transform,
            strong_transform,
            num_channels=self.num_channels,
        )

        return lb_dset, ulb_dset
