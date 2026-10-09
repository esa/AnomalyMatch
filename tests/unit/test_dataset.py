#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
import os
import tempfile

import numpy as np
import pandas as pd
import pytest
import torch
from loguru import logger
from PIL import Image

import anomaly_match as am
from anomaly_match.datasets.AnomalyDetectionDataset import AnomalyDetectionDataset
from anomaly_match.datasets.BasicDataset import BasicDataset
from anomaly_match.datasets.Label import Label
from anomaly_match.datasets.SSL_Dataset import SSL_Dataset
from anomaly_match.image_processing.transforms import (
    get_prediction_transforms,
    get_weak_transforms,
)


@pytest.fixture(scope="module")
def base_config():
    """Fixture providing base configuration for tests."""
    cfg = am.get_default_cfg()
    cfg.data_dir = "tests/test_data/grayscale/"
    cfg.normalisation.image_size = [64, 64]
    cfg.normalisation.n_output_channels = 3
    cfg.num_train_iter = 2
    cfg.test_ratio = 0.5
    cfg.N_to_load = 10
    cfg.normalisation.fits_extension = None
    cfg.label_file = None
    return cfg


@pytest.fixture(scope="function")
def test_config(base_config):
    """Provides a fresh copy of the base configuration for each test, which need to modify it."""
    import copy

    return copy.deepcopy(base_config)


@pytest.fixture(scope="module")
def sample_data():
    """Fixture providing sample data for BasicDataset tests."""
    # Create sample data as numpy array
    imgs = np.random.randint(0, 255, (10, 64, 64, 3), dtype=np.uint8)
    filenames = [f"img_{i}.jpg" for i in range(10)]
    targets = [0, 1] * 5  # Alternating normal/anomaly labels
    return imgs, filenames, targets


@pytest.fixture(scope="function")
def multi_extension_dataset():
    """Create a temporary directory with images of different extensions for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test images with different extensions - only supported formats
        extensions = [".jpg", ".jpeg", ".png", ".tiff"]
        test_images = []

        # Create a simple test image
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        img[20:40, 20:40, 0] = 255  # Red square

        # Save image in different formats
        for i, ext in enumerate(extensions):
            filename = f"test_image_{i}{ext}"
            filepath = os.path.join(temp_dir, filename)
            Image.fromarray(img).save(filepath)
            test_images.append(filename)

        # Create a more comprehensive CSV file with multiple labels of each class
        # to allow for stratified train/test split
        csv_path = os.path.join(temp_dir, "labeled_data.csv")

        # Label at least 2 images as normal and 2 as anomaly
        labels = ["normal", "normal", "anomaly", "anomaly"]
        files_to_label = test_images[: len(labels)]

        df = pd.DataFrame({"id": files_to_label, "label": labels})
        df.to_csv(csv_path, index=False)

        yield temp_dir, test_images, extensions


def test_anomaly_detection_dataset_initialization(base_config):
    """Test AnomalyDetectionDataset initialization and basic properties."""
    dataset = AnomalyDetectionDataset(base_config)

    assert dataset is not None
    assert dataset.size == base_config.normalisation.image_size
    assert dataset.test_ratio == base_config.test_ratio
    assert dataset.num_channels == 3
    assert hasattr(dataset, "data_dict")


def test_multiple_file_extensions_support(multi_extension_dataset, test_config):
    """Test support for multiple file extensions."""
    temp_dir, test_images, extensions = multi_extension_dataset
    test_config.data_dir = str(temp_dir)  # Ensure path is string
    test_config.test_ratio = 0.5  # to handle small datasets

    # Create dataset with default extensions
    dataset = AnomalyDetectionDataset(test_config)

    # Check if all images were found
    assert len(dataset.all_filenames) == len(extensions), (
        "Not all images with different extensions were found"
    )

    # Verify that all expected files are included
    for filename in test_images:
        assert filename in dataset.all_filenames, f"Image {filename} was not found in dataset"


def test_anomaly_detection_dataset_splits(base_config):
    """Test dataset splitting functionality."""
    dataset = AnomalyDetectionDataset(base_config)

    train_data = dataset.train_data
    test_data = dataset.test_data

    assert len(train_data) == 3  # filenames, images, labels
    assert len(test_data) == 3

    # Verify no overlap between train and test sets
    train_files = set(train_data[0])
    test_files = set(test_data[0])
    assert len(train_files.intersection(test_files)) == 0


def test_basic_dataset(sample_data):
    """Test BasicDataset functionality."""
    imgs, filenames, targets = sample_data

    # Create transforms
    transform = get_prediction_transforms()  # empty transform for testing

    dataset = BasicDataset(imgs, filenames, targets, num_classes=2, transform=transform)

    assert len(dataset) == len(imgs)

    # Test __getitem__
    img, target, filename = dataset[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, 64, 64)  # CHW format after transform
    assert target == targets[0]
    assert filename == filenames[0]


def test_ssl_dataset(base_config):
    """Test SSL_Dataset initialization and splitting."""
    ssl_dataset = SSL_Dataset(
        cfg=base_config,
        train=True,
    )

    # Test getting SSL datasets
    labeled_dataset, unlabeled_dataset = ssl_dataset.get_ssl_dset()

    assert labeled_dataset is not None
    assert unlabeled_dataset is not None
    assert hasattr(ssl_dataset, "num_classes")
    assert hasattr(ssl_dataset, "num_channels")


def test_dataset_label_updates(base_config):
    """Test label updating functionality."""
    dataset = AnomalyDetectionDataset(base_config)

    # Create new labels
    new_labels = pd.DataFrame({"id": [dataset.all_filenames[0]], "label": ["normal"]})

    # Update labels
    dataset.update_labels(new_labels)

    # Verify label was updated
    filename = dataset.all_filenames[0]
    if filename in dataset.data_dict:
        _, label = dataset.data_dict[filename]
        assert label == Label.NORMAL


def test_basic_dataset_augmentation(sample_data):
    """Test dataset augmentation functionality."""
    imgs, filenames, targets = sample_data

    # Create weak and strong transforms
    weak_transform = get_weak_transforms()
    # Test with strong augmentation
    dataset = BasicDataset(
        imgs,
        filenames,
        targets,
        num_classes=2,
        transform=weak_transform,
        use_strong_transform=True,
    )

    # When using strong augmentation, should return weak and strong augmented images
    weak_img, strong_img, target = dataset[0]
    assert isinstance(weak_img, torch.Tensor)
    assert isinstance(strong_img, torch.Tensor)
    assert weak_img.shape == (3, 64, 64)
    assert strong_img.shape == (3, 64, 64)
    assert target == targets[0]


def test_ssl_dataset_consistency(base_config):
    """Test consistency of SSL dataset splits."""
    ssl_dataset = SSL_Dataset(
        cfg=base_config,
        train=True,
    )

    # Get datasets twice and verify they're the same
    labeled_1, unlabeled_1 = ssl_dataset.get_ssl_dset()
    labeled_2, unlabeled_2 = ssl_dataset.get_ssl_dset()

    assert len(labeled_1) == len(labeled_2)
    assert len(unlabeled_1) == len(unlabeled_2)


def test_anomaly_detection_dataset_data_loading(base_config):
    """Test data loading functionality."""
    dataset = AnomalyDetectionDataset(base_config)

    # Test getting unlabeled data
    unlabeled = dataset.unlabeled
    assert len(unlabeled) == 2  # [images, filenames]

    # Test getting train data
    labeled = dataset.train_data
    assert len(labeled) == 3  # [filenames, images, labels]

    # Test getting test data
    test = dataset.test_data
    assert len(test) == 3  # [filenames, images, labels]


def test_anomaly_detection_dataset_properties(base_config):
    """Test data access properties."""
    dataset = AnomalyDetectionDataset(base_config)

    # Test getting unlabeled data
    unlabeled = dataset.unlabeled
    assert len(unlabeled) == 2  # [images, filenames]

    # Test properties
    assert isinstance(dataset.unlabeled_filenames, list)
    assert isinstance(dataset.unlabeled_filepaths, list)


# ── Label CSV id column tests ──────────────────────────────────


class TestLabelCSVIdColumn:
    """Verify label CSV with id column loads correctly."""

    def test_image_folder_with_id_column(self, test_config):
        """Image folder CSV with id column loads all labeled images."""
        test_config.label_file = "tests/test_data/grayscale/labeled_data.csv"
        test_config.N_to_load = 20
        test_config.num_workers = 0
        dset = AnomalyDetectionDataset(test_config)

        labeled = {k for k, v in dset.data_dict.items() if v[1] != Label.UNKNOWN}
        assert len(labeled) == 10

    def test_data_dict_keyed_by_id(self, test_config):
        """data_dict keys are the id values from the CSV."""
        test_config.label_file = "tests/test_data/grayscale/labeled_data.csv"
        test_config.N_to_load = 20
        test_config.num_workers = 0
        dset = AnomalyDetectionDataset(test_config)

        assert "Abell2390_VIS_2.jpeg" in dset.data_dict

    def test_update_labels_works_with_id_column(self, test_config):
        """update_labels() finds images by id."""
        test_config.label_file = "tests/test_data/grayscale/labeled_data.csv"
        test_config.N_to_load = 20
        test_config.num_workers = 0
        dset = AnomalyDetectionDataset(test_config)

        new_labels = pd.DataFrame({"id": ["Abell2390_VIS_2.jpeg"], "label": ["normal"]})
        dset.update_labels(new_labels)

        _, label = dset.data_dict["Abell2390_VIS_2.jpeg"]
        assert label == Label.NORMAL


def test_anomaly_detection_dataset_data_access(base_config):
    """Test data access methods."""
    dataset = AnomalyDetectionDataset(base_config)

    # Test accessing labeled and unlabeled data
    unlabeled = dataset.unlabeled
    assert len(unlabeled) == 2  # [images, filenames]

    train_data = dataset.train_data
    assert len(train_data) == 3  # [filenames, images, labels]


def test_anomaly_detection_dataset_file_operations(base_config):
    """Test file operations."""
    dataset = AnomalyDetectionDataset(base_config)

    # Test file access methods
    assert isinstance(dataset.unlabeled_filenames, list)
    assert isinstance(dataset.unlabeled_filepaths, list)


def test_ssl_dataset_single_class_raises(test_config):
    """SSL_Dataset.get_ssl_dset() raises when labels contain only one class."""
    with tempfile.TemporaryDirectory() as tmp:
        # Create images
        for i in range(6):
            img = np.zeros((64, 64, 3), dtype=np.uint8)
            Image.fromarray(img).save(os.path.join(tmp, f"img_{i}.png"))

        # Label only as normal — no anomalies
        csv_path = os.path.join(tmp, "labeled_data.csv")
        pd.DataFrame({"id": ["img_0.png", "img_1.png"], "label": ["normal", "normal"]}).to_csv(
            csv_path, index=False
        )

        test_config.data_dir = tmp
        test_config.label_file = csv_path
        test_config.N_to_load = 4
        test_config.test_ratio = 0.0

        ssl = SSL_Dataset(cfg=test_config, train=True)
        with pytest.raises(ValueError, match="only one class"):
            ssl.get_ssl_dset()


class _FakeBandMismatchSource:
    """Data source whose unlabeled stream resurfaces a labeled id at a different band count.

    A leaky sampler that re-emits a labeled id — here at 4 bands while the
    labeled cutout is 3 bands — so the guard that skips it can be exercised.
    """

    def __init__(self):
        self._labeled = [("A", np.zeros((8, 8, 3), dtype=np.uint8))]
        self._unlabeled = [
            ("A", np.ones((8, 8, 4), dtype=np.uint8)),  # labeled id resurfacing at 4 bands
            ("B", np.ones((8, 8, 4), dtype=np.uint8)),
        ]

    def detect_num_channels(self):
        return 4

    def get_total_count(self):
        return 3

    def get_labeled_images(self, label_df):
        return self._labeled

    def get_unlabeled_batch(self, n, exclude=None):
        return self._unlabeled


def test_unlabeled_stream_does_not_clobber_labeled(tmp_path, test_config):
    """A resurfaced labeled id keeps its labeled cutout, not the streamed copy.

    Regression: overwriting the labeled entry with the streamed cutout planted a
    4-band array in the 3-band labeled set (torch.stack crash). The label is
    repaired by ``update_labels`` one step later, so the *shape* is the real
    oracle — the label assertion alone passes even without the fix.
    """
    test_config.normalisation.n_output_channels = 4
    test_config.metadata_file = None
    test_config.N_to_load = 10
    test_config.test_ratio = 0.0
    label_csv = tmp_path / "labels.csv"
    pd.DataFrame({"id": ["A"], "label": ["normal"]}).to_csv(label_csv, index=False)
    test_config.label_file = str(label_csv)

    dset = AnomalyDetectionDataset(cfg=test_config, data_source=_FakeBandMismatchSource())

    img_a, _ = dset.data_dict["A"]
    assert img_a.shape[-1] == 3, (
        "labeled 'A' must keep its 3-band cutout, not the 4-band stream copy"
    )
    # The genuinely-unlabeled id is still added; the leaked labeled id is not.
    assert dset.data_dict["B"][1] == Label.UNKNOWN
    assert "A" not in {fn for fn in dset.data_dict if dset.data_dict[fn][1] == Label.UNKNOWN}
    assert dset.data_dict["B"][0].shape[-1] == 4


class _FakeDuplicateUnlabeledSource:
    """Data source whose unlabeled stream emits the same *unlabeled* id twice.

    The sampler doesn't dedup ``SourceID`` across tiles, so a source in an
    overlap region can be drawn from two of them.
    """

    def __init__(self):
        self._labeled = [("A", np.zeros((8, 8, 3), dtype=np.uint8))]
        self._unlabeled = [
            ("B", np.ones((8, 8, 3), dtype=np.uint8)),
            ("B", np.ones((8, 8, 3), dtype=np.uint8)),  # same unlabeled id again
        ]

    def detect_num_channels(self):
        return 3

    def get_total_count(self):
        return 3

    def get_labeled_images(self, label_df):
        return self._labeled

    def get_unlabeled_batch(self, n, exclude=None):
        return self._unlabeled


def test_duplicate_unlabeled_id_not_reported_as_labeled_leak(tmp_path, test_config):
    """A repeated *unlabeled* id must not be blamed on the labeled sampler.

    ``data_dict`` grows as cutouts are accepted, so testing membership against
    it alone counts a sampler duplicate as a resurfaced labeled id and emits a
    "sampler leaked a labeled source" warning that is simply untrue.
    """
    test_config.normalisation.n_output_channels = 3
    test_config.metadata_file = None
    test_config.N_to_load = 10
    test_config.test_ratio = 0.0
    label_csv = tmp_path / "labels.csv"
    pd.DataFrame({"id": ["A"], "label": ["normal"]}).to_csv(label_csv, index=False)
    test_config.label_file = str(label_csv)

    messages: list[str] = []
    sink_id = logger.add(messages.append, level="INFO", format="{message}")
    try:
        dset = AnomalyDetectionDataset(cfg=test_config, data_source=_FakeDuplicateUnlabeledSource())
    finally:
        logger.remove(sink_id)

    log = "\n".join(messages)
    assert "leaked a labeled source" not in log, (
        "a duplicate unlabeled id was misreported as a labeled leak"
    )
    assert "already sampled this batch" in log
    # The duplicate is skipped, not added twice under a mangled key.
    assert dset.data_dict["B"][1] == Label.UNKNOWN
    assert [fn for fn, (_, lbl) in dset.data_dict.items() if lbl == Label.UNKNOWN] == ["B"]
