#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file ‘LICENCE.txt’.
import torch
from loguru import logger

from .BasicDataset import BasicDataset
from .AnomalyDetectionDataset import AnomalyDetectionDataset

from torchvision import transforms


def get_transform(train=True):
    """Get weak augmentation transforms.

    Args:
        train (bool, optional): Whether training, in test only normalization is applied.

    Returns:
        torchvision.transforms.Compose: transforms.
    """

    if train:
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(0, translate=(0, 0.125)),
            ]
        )
    else:
        return transforms.Compose([transforms.ToTensor()])


class SSL_Dataset:
    """
    SSL_Dataset class separates labeled and unlabeled data,
    and return BasicDataset: torch.utils.data.Dataset (see datasets.dataset.py)
    """

    def __init__(
        self,
        test_ratio,
        N_to_load,
        train=True,
        data_dir="./data",
        seed=42,
        size=[300, 300],
        label_file=None,
    ):
        """
        Args
            mean: mean of the dataset
            test_ratio: ratio of test data
            N_to_load: number of samples to load
            train: True means the dataset is training dataset (default=True)
            data_dir: path of directory, where data is downloaed or stored.
            seed: seed to use for the train / test split. Not available for cifar which is presplit
            size: size of the image to resize to
            label_file: path to the CSV file with labels. If None, 'labeled_data.csv' in data_dir is used
        """

        self.seed = seed
        self.test_ratio = test_ratio
        self.N_to_load = N_to_load
        self.train = train
        self.num_classes = 2
        self.size = size
        self.data_dir = data_dir
        self.label_file = label_file
        self.dset = None

    def get_data(self):
        """
        get_data returns data (images) and targets (labels)
        """
        if self.dset is None:
            self.dset = AnomalyDetectionDataset(
                test_ratio=self.test_ratio,
                root_dir=self.data_dir,
                seed=self.seed,
                size=self.size,
                N_to_load=self.N_to_load,
                label_file=self.label_file,
            )
        else:
            logger.debug("Dataset already loaded.")

        self.num_channels = self.dset.num_channels

        if self.train:
            filenames, imgs, targets = self.dset.train_data
            unlabeled, unlabeled_filenames = self.dset.unlabeled
        else:
            filenames, imgs, targets = self.dset.test_data
            unlabeled, unlabeled_filenames = None, None  # no unlabeled data in test
        self.transform = get_transform(self.train)
        return imgs, targets, unlabeled, filenames, unlabeled_filenames

    def get_dset(self, use_strong_transform=False, strong_transform=None):
        """
        get_dset returns class BasicDataset, containing the returns of get_data.

        Args
            use_strong_tranform: If True, returned dataset generates a pair of weak and strong augmented images.
            strong_transform: list of strong_transform (augmentation) if use_strong_transform is True
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
        )

    def get_ssl_dset(
        self,
        use_strong_transform=True,
        strong_transform=None,
    ):
        """
        get_ssl_dset split training samples into labeled and unlabeled samples.
        The labeled data is balanced samples over classes.

        Args:
            use_strong_transform: If True, unlabeld dataset returns weak & strong augmented image pair.
                                  If False, unlabeled datasets returns only weak augmented image.
            strong_transform: list of strong transform (RandAugment in FixMatch)

        Returns:
            BasicDataset (for labeled data), BasicDataset (for unlabeld data)
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

        # If labeled data only contain nominal / anomaly data, print a warning
        if targets.count(0) == 0 or targets.count(1) == 0:
            logger.warning(
                f"Labeled data contain only one class (0: {targets.count(0)}, 1: {targets.count(1)})"
            )
            logger.warning("Consider adding more labels for the missing class")

        lb_dset = BasicDataset(
            data,
            labeled_filenames,
            targets,
            self.num_classes,
            self.transform,
            use_strong_transform=False,
            strong_transform=None,
        )

        ulb_dset = BasicDataset(
            unlabeled,
            unlabeled_filenames,
            torch.zeros(len(unlabeled)) - 1,  # dummy label
            self.num_classes,
            self.transform,
            use_strong_transform,
            strong_transform,
        )

        return lb_dset, ulb_dset

    def update_dsets(
        self,
        label_update=None,
        N_to_load=None,
        use_strong_transform=True,
        strong_transform=None,
    ):
        """
        Update the labels of the labeled dataset with the labels from the label_update dataframe
        and load the next batch of data.

        Args:
            label_update: DataFrame containing the new labels.
            N_to_load: number of samples to load
            use_strong_transform: If True, unlabeld dataset returns weak & strong augmented image pair.
                                  If False, unlabeled datasets returns only weak augmented image.
            strong_transform: list of strong transform (RandAugment in FixMatch)

        Returns:
            BasicDataset (for labeled data), BasicDataset (for unlabeld data)
        """
        assert self.train, "update_dsets is only for training dataset"
        assert self.dset is not None, "Dataset is not loaded yet"

        if label_update is not None:
            logger.info("Updating dataset with new labels")
            self.dset.update_labels(label_update)
        self.dset.load_next_batch(N_to_load)

        filenames, imgs, targets = self.dset.train_data
        unlabeled, unlabeled_filenames = self.dset.unlabeled

        # Add assertion to ensure unlabeled data exists after batch update
        assert (
            "No unlabeled data were provided. Semi-supervised learning requires unlabeled data, "
            + " i.e. images that are not classified in labeled_data.csv."
        )

        if label_update is not None:
            logger.debug("Number of labeled training samples: {}".format(len(filenames)))
            logger.info(f"Label distribution (0: {targets.count(0)}, 1: {targets.count(1)})")
        logger.debug("Number of unlabeled samples: {}".format(len(unlabeled_filenames)))

        lb_dset = BasicDataset(
            imgs,
            filenames,
            targets,
            self.num_classes,
            self.transform,
            use_strong_transform=False,
            strong_transform=None,
        )

        ulb_dset = BasicDataset(
            unlabeled,
            unlabeled_filenames,
            torch.zeros(len(unlabeled)) - 1,  # dummy label
            self.num_classes,
            self.transform,
            use_strong_transform,
            strong_transform,
        )

        return lb_dset, ulb_dset
