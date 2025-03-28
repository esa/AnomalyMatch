#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file ‘LICENCE.txt’.
import pytest
import numpy as np
import pandas as pd
import torch
import anomaly_match as am
from anomaly_match.datasets.Label import Label
from anomaly_match.datasets.AnomalyDetectionDataset import AnomalyDetectionDataset
from anomaly_match.datasets.BasicDataset import BasicDataset
from anomaly_match.datasets.SSL_Dataset import SSL_Dataset
from torchvision import transforms
import os
import tempfile
from PIL import Image


@pytest.fixture(scope="module")
def base_config():
    """Fixture providing base configuration for tests."""
    cfg = am.get_default_cfg()
    cfg.data_dir = "tests/test_data/"
    cfg.size = [64, 64]
    cfg.num_train_iter = 2
    cfg.test_ratio = 0.5
    return cfg


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
        # Create test images with different extensions
        extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff"]
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

        df = pd.DataFrame({"filename": files_to_label, "label": labels})
        df.to_csv(csv_path, index=False)

        yield temp_dir, test_images, extensions


def test_anomaly_detection_dataset_initialization(base_config):
    """Test AnomalyDetectionDataset initialization and basic properties."""
    dataset = AnomalyDetectionDataset(
        test_ratio=base_config.test_ratio,
        root_dir=base_config.data_dir,
        size=base_config.size,
        N_to_load=10,
    )

    assert dataset is not None
    assert dataset.size == base_config.size
    assert dataset.test_ratio == base_config.test_ratio
    assert dataset.num_channels == 3
    assert hasattr(dataset, "data_dict")
    assert hasattr(dataset, "mean")
    assert hasattr(dataset, "std")


def test_multiple_file_extensions_support(multi_extension_dataset):
    """Test support for multiple file extensions."""
    temp_dir, test_images, extensions = multi_extension_dataset

    # Create dataset with default extensions
    dataset = AnomalyDetectionDataset(
        test_ratio=0.5,  # Use 50/50 split for better handling of small datasets
        root_dir=temp_dir,
        size=[64, 64],
        N_to_load=10,
    )

    # Check if all images were found
    assert len(dataset.all_filenames) == len(
        extensions
    ), "Not all images with different extensions were found"

    # Verify that all expected files are included
    for filename in test_images:
        assert filename in dataset.all_filenames, f"Image {filename} was not found in dataset"


def test_file_extension_filtering(multi_extension_dataset):
    """Test filtering by specific file extensions."""
    temp_dir, test_images, extensions = multi_extension_dataset

    # Test with limited extensions
    limited_extensions = [".jpg", ".png"]
    dataset = AnomalyDetectionDataset(
        test_ratio=0.0,  # No test split to avoid sklearn errors with small datasets
        root_dir=temp_dir,
        size=[64, 64],
        file_extensions=limited_extensions,
        N_to_load=10,
    )

    # Check if only images with specified extensions were found
    expected_count = sum(
        1 for img in test_images if any(img.lower().endswith(ext) for ext in limited_extensions)
    )
    assert (
        len(dataset.all_filenames) == expected_count
    ), f"Expected {expected_count} images but found {len(dataset.all_filenames)}"

    # Verify correct images are included
    for filename in test_images:
        if any(filename.lower().endswith(ext) for ext in limited_extensions):
            assert filename in dataset.all_filenames, f"Image {filename} should be in dataset"
        else:
            assert (
                filename not in dataset.all_filenames
            ), f"Image {filename} should not be in dataset"


def test_read_and_resize_different_formats(multi_extension_dataset):
    """Test reading and resizing images of different formats."""
    temp_dir, test_images, _ = multi_extension_dataset

    dataset = AnomalyDetectionDataset(
        test_ratio=0.0,  # No test split to avoid sklearn errors with small datasets
        root_dir=temp_dir,
        size=[64, 64],
        N_to_load=10,
    )

    # Test that all images can be loaded and are properly resized
    for filename in test_images:
        filepath = os.path.join(temp_dir, filename)
        image = dataset._read_and_resize_image(filepath)

        # Check image dimensions and type
        assert image.shape == (64, 64, 3), f"Image {filename} was not resized correctly"
        assert image.dtype == np.uint8, f"Image {filename} has incorrect data type"


def test_anomaly_detection_dataset_splits(base_config):
    """Test dataset splitting functionality."""
    dataset = AnomalyDetectionDataset(
        test_ratio=0.5, root_dir=base_config.data_dir, size=base_config.size, N_to_load=10
    )

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
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

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
        test_ratio=base_config.test_ratio,
        N_to_load=10,
        train=True,
        data_dir=base_config.data_dir,
        seed=42,
        size=base_config.size,
    )

    # Test getting SSL datasets
    labeled_dataset, unlabeled_dataset = ssl_dataset.get_ssl_dset()

    assert labeled_dataset is not None
    assert unlabeled_dataset is not None
    assert hasattr(ssl_dataset, "num_classes")
    assert hasattr(ssl_dataset, "num_channels")


def test_dataset_label_updates(base_config):
    """Test label updating functionality."""
    dataset = AnomalyDetectionDataset(
        test_ratio=base_config.test_ratio,
        root_dir=base_config.data_dir,
        size=base_config.size,
        N_to_load=10,
    )

    # Create new labels
    new_labels = pd.DataFrame({"filename": [dataset.all_filenames[0]], "label": ["normal"]})

    # Update labels
    dataset.update_labels(new_labels)

    # Verify label was updated
    filename = dataset.all_filenames[0]
    if filename in dataset.data_dict:
        _, label = dataset.data_dict[filename]
        assert label == Label.NORMAL


def test_dataset_batch_loading(base_config):
    """Test batch loading functionality."""
    dataset = AnomalyDetectionDataset(
        test_ratio=base_config.test_ratio,
        root_dir=base_config.data_dir,
        size=base_config.size,
        N_to_load=5,  # Small batch size for testing
    )

    initial_unlabeled = len(dataset.unlabeled[1])
    dataset.load_next_batch()

    # After loading next batch, should have different or same number of unlabeled images
    # depending on total dataset size
    current_unlabeled = len(dataset.unlabeled[1])
    assert current_unlabeled <= initial_unlabeled


def test_basic_dataset_augmentation(sample_data):
    """Test dataset augmentation functionality."""
    imgs, filenames, targets = sample_data

    # Create weak and strong transforms
    weak_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
        ]
    )

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
        test_ratio=base_config.test_ratio,
        N_to_load=10,
        train=True,
        data_dir=base_config.data_dir,
        seed=42,
        size=base_config.size,
    )

    # Get datasets twice and verify they're the same
    labeled_1, unlabeled_1 = ssl_dataset.get_ssl_dset()
    labeled_2, unlabeled_2 = ssl_dataset.get_ssl_dset()

    assert len(labeled_1) == len(labeled_2)
    assert len(unlabeled_1) == len(unlabeled_2)


def test_anomaly_detection_dataset_data_loading(base_config):
    """Test data loading functionality."""
    dataset = AnomalyDetectionDataset(
        test_ratio=base_config.test_ratio,
        root_dir=base_config.data_dir,
        size=base_config.size,
        N_to_load=10,
    )

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
    dataset = AnomalyDetectionDataset(
        test_ratio=base_config.test_ratio,
        root_dir=base_config.data_dir,
        size=base_config.size,
        N_to_load=10,
    )

    # Test getting unlabeled data
    unlabeled = dataset.unlabeled
    assert len(unlabeled) == 2  # [images, filenames]

    # Test properties
    assert isinstance(dataset.unlabeled_filenames, list)
    assert isinstance(dataset.unlabeled_filepaths, list)


def test_anomaly_detection_dataset_hdf5(base_config, tmp_path):
    """Test HDF5 save/load functionality."""
    dataset = AnomalyDetectionDataset(
        test_ratio=base_config.test_ratio,
        root_dir=base_config.data_dir,
        size=base_config.size,
        N_to_load=10,
    )

    # Test save/load functionality
    save_path = tmp_path / "test_save.hdf5"
    dataset.save_as_hdf5(str(save_path))
    assert save_path.exists()

    # Test loading
    new_dataset = AnomalyDetectionDataset(
        test_ratio=base_config.test_ratio,
        root_dir=base_config.data_dir,
        size=base_config.size,
        N_to_load=10,
    )
    new_dataset.load_from_hdf5(str(save_path))
    assert len(new_dataset.data_dict) > 0


def test_anomaly_detection_dataset_data_access(base_config):
    """Test data access methods."""
    dataset = AnomalyDetectionDataset(
        test_ratio=base_config.test_ratio,
        root_dir=base_config.data_dir,
        size=base_config.size,
        N_to_load=10,
    )

    # Test accessing labeled and unlabeled data
    unlabeled = dataset.unlabeled
    assert len(unlabeled) == 2  # [images, filenames]

    train_data = dataset.train_data
    assert len(train_data) == 3  # [filenames, images, labels]


def test_anomaly_detection_dataset_file_operations(base_config):
    """Test file operations."""
    dataset = AnomalyDetectionDataset(
        test_ratio=base_config.test_ratio,
        root_dir=base_config.data_dir,
        size=base_config.size,
        N_to_load=10,
    )

    # Test file access methods
    assert isinstance(dataset.unlabeled_filenames, list)
    assert isinstance(dataset.unlabeled_filepaths, list)


def test_anomaly_detection_dataset_hdf5_operations(base_config, tmp_path):
    """Test HDF5 operations."""
    dataset = AnomalyDetectionDataset(
        test_ratio=base_config.test_ratio,
        root_dir=base_config.data_dir,
        size=base_config.size,
        N_to_load=10,
    )

    # Test HDF5 save/load
    save_path = tmp_path / "test_save.hdf5"
    dataset.save_as_hdf5(str(save_path))
    assert save_path.exists()
