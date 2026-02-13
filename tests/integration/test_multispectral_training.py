#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Integration tests for multispectral (N-channel) model training."""

import pytest

from anomaly_match.datasets.SSL_Dataset import SSL_Dataset


@pytest.mark.slow
class TestMultispectralTraining:
    """Tests for multispectral training pipeline."""

    def test_model_training_4_channels(self, multispectral_config):
        """Test that FixMatch can be set up with 4-channel data."""
        from anomaly_match.models.FixMatch import FixMatch
        from anomaly_match.utils.get_net_builder import get_net_builder

        # Load datasets
        ssl_dataset = SSL_Dataset(cfg=multispectral_config, train=True)
        labeled_dset, unlabeled_dset = ssl_dataset.get_ssl_dset()

        # Verify datasets have correct channel count
        assert labeled_dset.num_channels == 4
        assert unlabeled_dset.num_channels == 4

        # Build network
        net_builder = get_net_builder(
            multispectral_config.net,
            pretrained=multispectral_config.pretrained,
            in_channels=4,
        )

        # Create model
        model = FixMatch(
            net_builder=net_builder,
            num_classes=2,
            in_channels=4,
            ema_m=multispectral_config.ema_m,
            T=multispectral_config.temperature,
            p_cutoff=multispectral_config.p_cutoff,
            lambda_u=multispectral_config.ulb_loss_ratio,
        )

        # Set up data loaders
        model.set_data_loader(
            cfg=multispectral_config,
            lb_dset=labeled_dset,
            ulb_dset=unlabeled_dset,
            eval_dset=None,
        )

        # Verify model is set up correctly for 4-channel input
        assert model.train_model is not None
        assert model.eval_model is not None
        # The first conv layer should accept 4 channels
        # TestCNN uses _conv_stem, timm models use conv_stem
        train_model = model.train_model
        first_conv = getattr(train_model, "conv_stem", getattr(train_model, "_conv_stem", None))
        assert first_conv is not None
        assert first_conv.in_channels == 4
