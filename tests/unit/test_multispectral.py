#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Unit tests for multispectral (N-channel) image support."""

import numpy as np
import torch

from anomaly_match.datasets.AnomalyDetectionDataset import AnomalyDetectionDataset
from anomaly_match.datasets.BasicDataset import BasicDataset
from anomaly_match.datasets.SSL_Dataset import SSL_Dataset
from anomaly_match.image_processing.transforms import (
    get_prediction_transforms,
    get_strong_transforms,
    get_weak_transforms,
)


class TestMultispectralDataset:
    """Tests for multispectral dataset loading and handling."""

    def test_dataset_loads_4_channel_images(self, multispectral_config):
        """Test that AnomalyDetectionDataset correctly loads 4-channel images."""
        dataset = AnomalyDetectionDataset(multispectral_config)

        assert dataset is not None
        assert dataset.num_channels == 4, f"Expected 4 channels, got {dataset.num_channels}"
        assert dataset.size == multispectral_config.normalisation.image_size

        # Check that images in data_dict have correct shape
        for filename, (image, label) in dataset.data_dict.items():
            assert image.shape[-1] == 4, f"Image {filename} has wrong channels: {image.shape}"
            assert image.shape[:2] == tuple(dataset.size), (
                f"Image {filename} has wrong size: {image.shape}"
            )

    def test_channel_auto_detection(self, multispectral_config):
        """Test that n_output_channels is auto-detected from 4-channel images."""
        # Config starts at default (3), should be updated after dataset load
        assert multispectral_config.normalisation.n_output_channels == 3

        dataset = AnomalyDetectionDataset(multispectral_config)

        # After loading, config should be updated to 4
        assert multispectral_config.normalisation.n_output_channels == 4
        assert dataset.num_channels == 4

    def test_dataset_mean_std_4_channels(self, multispectral_config):
        """Test that mean and std are computed correctly for 4 channels."""
        dataset = AnomalyDetectionDataset(multispectral_config)

        assert len(dataset.mean) == 4, f"Mean should have 4 values, got {len(dataset.mean)}"
        assert len(dataset.std) == 4, f"Std should have 4 values, got {len(dataset.std)}"

        # Check that values are reasonable (between 0 and 1 for normalized)
        assert all(0 <= m <= 1 for m in dataset.mean), f"Mean values out of range: {dataset.mean}"
        assert all(0 <= s <= 1 for s in dataset.std), f"Std values out of range: {dataset.std}"

    def test_ssl_dataset_4_channels(self, multispectral_config):
        """Test SSL_Dataset initialization with 4-channel data."""
        ssl_dataset = SSL_Dataset(cfg=multispectral_config, train=True)
        assert ssl_dataset.num_classes == 2

        # Auto-detection happens when dataset is loaded (lazy)
        labeled_dataset, unlabeled_dataset = ssl_dataset.get_ssl_dset()
        assert labeled_dataset is not None
        assert unlabeled_dataset is not None
        assert multispectral_config.normalisation.n_output_channels == 4


class TestMultispectralAugmentation:
    """Tests for multispectral augmentation compatibility."""

    def test_basic_dataset_4_channel_no_transform(self, multispectral_config):
        """Test BasicDataset works with 4-channel data without transforms."""
        # Create sample 4-channel data
        imgs = np.random.randint(0, 255, (5, 64, 64, 4), dtype=np.uint8)
        filenames = [f"img_{i}.npy" for i in range(5)]
        targets = [0, 1, 0, 1, 0]

        transform = get_prediction_transforms(num_channels=4)
        dataset = BasicDataset(
            imgs, filenames, targets, num_classes=2, transform=transform, num_channels=4
        )

        img, target, filename = dataset[0]

        assert isinstance(img, torch.Tensor)
        assert img.shape[0] == 4, f"Expected 4 channels, got shape {img.shape}"
        assert img.shape[1:] == (64, 64), f"Expected (64, 64) spatial, got {img.shape}"

    def test_basic_dataset_4_channel_weak_transform(self, multispectral_config):
        """Test BasicDataset works with 4-channel data and weak transforms."""
        imgs = np.random.randint(0, 255, (5, 64, 64, 4), dtype=np.uint8)
        filenames = [f"img_{i}.npy" for i in range(5)]
        targets = [0, 1, 0, 1, 0]

        transform = get_weak_transforms(num_channels=4)
        dataset = BasicDataset(
            imgs, filenames, targets, num_classes=2, transform=transform, num_channels=4
        )

        img, target, filename = dataset[0]

        assert isinstance(img, torch.Tensor)
        assert img.shape[0] == 4, f"Expected 4 channels after weak transform, got {img.shape}"

    def test_basic_dataset_4_channel_strong_transform(self, multispectral_config):
        """Test BasicDataset works with 4-channel data and strong transforms."""
        imgs = np.random.randint(0, 255, (5, 64, 64, 4), dtype=np.uint8)
        filenames = [f"img_{i}.npy" for i in range(5)]
        targets = [0, 1, 0, 1, 0]

        weak_transform = get_weak_transforms(num_channels=4)
        strong_transform = get_strong_transforms(num_channels=4)

        dataset = BasicDataset(
            imgs,
            filenames,
            targets,
            num_classes=2,
            transform=weak_transform,
            use_strong_transform=True,
            strong_transform=strong_transform,
            num_channels=4,
        )

        weak_img, strong_img, target = dataset[0]

        assert isinstance(weak_img, torch.Tensor)
        assert isinstance(strong_img, torch.Tensor)
        assert weak_img.shape[0] == 4, f"Weak augmented should have 4 channels: {weak_img.shape}"
        assert strong_img.shape[0] == 4, (
            f"Strong augmented should have 4 channels: {strong_img.shape}"
        )


class TestMultispectralModel:
    """Tests for multispectral model architecture."""

    def test_network_builder_4_channels(self):
        """Test that network builder creates model accepting 4 channels."""
        from anomaly_match.utils.get_net_builder import get_net_builder

        net_builder = get_net_builder("efficientnet-lite0", pretrained=True, in_channels=4)
        model = net_builder(num_classes=2, in_channels=4)

        # Set to eval mode to avoid batch norm issues with batch size 1
        model.eval()

        # Create dummy input with 4 channels
        dummy_input = torch.randn(1, 4, 64, 64)
        with torch.no_grad():
            output = model(dummy_input)

        assert output.shape == (1, 2), f"Expected output shape (1, 2), got {output.shape}"

    def test_network_builder_pretrained_weight_transfer(self):
        """Test that pretrained weights are adapted for 4-channel input by timm."""
        from anomaly_match.utils.get_net_builder import get_net_builder

        # Get 4-channel pretrained model
        net_builder_4ch = get_net_builder("efficientnet-lite0", pretrained=True, in_channels=4)
        model_4ch = net_builder_4ch(num_classes=2, in_channels=4)

        # Verify conv_stem was adapted to 4 input channels
        assert model_4ch.conv_stem.weight.data.shape[1] == 4, (
            "conv_stem should have 4 input channels"
        )

        # Verify weights are not all zeros (pretrained weights were adapted, not just zero-padded)
        assert model_4ch.conv_stem.weight.data.abs().sum() > 0, "Adapted weights should be non-zero"


class TestMultispectralPrediction:
    """Tests for multispectral prediction pipeline."""

    def test_model_prediction_4_channels(self, multispectral_config):
        """Test model prediction with 4-channel images."""
        from anomaly_match.models.FixMatch import FixMatch
        from anomaly_match.utils.get_net_builder import get_net_builder

        # Build and initialize model
        net_builder = get_net_builder(
            multispectral_config.net,
            pretrained=multispectral_config.pretrained,
            in_channels=4,
        )
        model = FixMatch(
            net_builder=net_builder,
            num_classes=2,
            in_channels=4,
            ema_m=multispectral_config.ema_m,
            T=multispectral_config.temperature,
            p_cutoff=multispectral_config.p_cutoff,
            lambda_u=multispectral_config.ulb_loss_ratio,
        )

        # Create test batch
        test_batch = torch.randn(4, 4, 64, 64)  # (batch, channels, H, W)

        # Get predictions
        model.eval_model.eval()
        with torch.no_grad():
            predictions = model.eval_model(test_batch)

        assert predictions.shape == (4, 2), f"Expected (4, 2), got {predictions.shape}"
        # Check predictions are valid probabilities after softmax
        probs = torch.softmax(predictions, dim=1)
        assert torch.allclose(probs.sum(dim=1), torch.ones(4)), "Probabilities should sum to 1"


class TestMultispectralDisplay:
    """Tests for multispectral display functionality."""

    def test_prepare_4_channel_for_display(self):
        """Test converting 4-channel image to RGB for display."""
        from anomaly_match_ui.utils.display_transforms import prepare_for_display

        # Create 4-channel image
        img_4ch = np.random.randint(0, 255, (64, 64, 4), dtype=np.uint8)

        # Convert to displayable RGB
        rgb_img = prepare_for_display(img_4ch)

        assert rgb_img.shape == (64, 64, 3), f"Expected RGB output, got {rgb_img.shape}"
        assert rgb_img.dtype == np.uint8

    def test_prepare_4_channel_with_rgb_mapping(self):
        """Test converting 4-channel image with custom RGB channel mapping."""
        from anomaly_match_ui.utils.display_transforms import prepare_for_display

        # Create 4-channel image
        img_4ch = np.random.randint(0, 255, (64, 64, 4), dtype=np.uint8)

        # Use channels 0, 2, 3 as RGB
        rgb_mapping = [0, 2, 3]
        rgb_img = prepare_for_display(img_4ch, rgb_mapping=rgb_mapping)

        assert rgb_img.shape == (64, 64, 3)
        # Verify mapping is correct
        assert np.array_equal(rgb_img[:, :, 0], img_4ch[:, :, 0])
        assert np.array_equal(rgb_img[:, :, 1], img_4ch[:, :, 2])
        assert np.array_equal(rgb_img[:, :, 2], img_4ch[:, :, 3])


class TestMultispectralHDF5:
    """Tests for HDF5 save/load with multispectral data."""

    def test_hdf5_save_load_4_channels(self, multispectral_config, tmp_path):
        """Test saving and loading 4-channel dataset to/from HDF5."""
        dataset = AnomalyDetectionDataset(multispectral_config)

        # Save to HDF5
        hdf5_path = tmp_path / "test_4ch.hdf5"
        dataset.save_as_hdf5(str(hdf5_path))

        assert hdf5_path.exists()

        # Load back
        new_dataset = AnomalyDetectionDataset(multispectral_config)
        new_dataset.load_from_hdf5(str(hdf5_path))

        # Verify data integrity
        assert len(new_dataset.data_dict) == len(dataset.data_dict)
        for filename in dataset.data_dict:
            if filename in new_dataset.data_dict:
                orig_img, _ = dataset.data_dict[filename]
                loaded_img, _ = new_dataset.data_dict[filename]
                assert orig_img.shape == loaded_img.shape
                assert np.array_equal(orig_img, loaded_img)
