#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
from pathlib import Path

import timm
import torch.nn as nn
from loguru import logger

# Local cache directory for pretrained weights, avoids flock issues on NFS filesystems.
_PACKAGE_DIR = Path(__file__).resolve().parent.parent
_PRETRAINED_CACHE_DIR = _PACKAGE_DIR / "pretrained_cache"

# Mapping from AnomalyMatch net names to timm model identifiers.
# efficientnet-lite variants use the tf_ prefix for TF-style same-padding,
# which is backward compatible with the previous efficientnet_lite_pytorch package.
_TIMM_MODEL_MAP = {
    "efficientnet-lite0": "tf_efficientnet_lite0",
    "efficientnet-lite1": "tf_efficientnet_lite1",
    "efficientnet-lite2": "tf_efficientnet_lite2",
    "efficientnet-lite3": "tf_efficientnet_lite3",
    "efficientnet-lite4": "tf_efficientnet_lite4",
    "efficientnet-b0": "efficientnet_b0",
    "efficientnet-b1": "efficientnet_b1",
    "efficientnet-b2": "efficientnet_b2",
    "efficientnet-b3": "efficientnet_b3",
    "efficientnet-b4": "efficientnet_b4",
    "efficientnet-b5": "efficientnet_b5",
    "efficientnet-b6": "efficientnet_b6",
    "efficientnet-b7": "efficientnet_b7",
}

# Pretrained tag for timm (appended to model name for pretrained loading)
_TIMM_PRETRAINED_TAG = {
    "tf_efficientnet_lite0": "tf_efficientnet_lite0.in1k",
    "efficientnet_b0": "efficientnet_b0.ra_in1k",
    "efficientnet_b1": "efficientnet_b1.ft_in1k",
    "efficientnet_b2": "efficientnet_b2.ra_in1k",
    "efficientnet_b3": "efficientnet_b3.ra2_in1k",
    "efficientnet_b4": "efficientnet_b4.ra2_in1k",
}


class TestCNN(nn.Module):
    """Minimal CNN for fast testing. Not for production use."""

    def __init__(self, num_classes=2, in_channels=3):
        super().__init__()
        self._conv_stem = nn.Conv2d(in_channels, 8, 3, stride=2, padding=1)
        self.features = nn.Sequential(
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self._fc = nn.Linear(8, num_classes)

    def forward(self, x):
        x = self._conv_stem(x)
        x = self.features(x)
        x = x.flatten(1)
        return self._fc(x)


def _resolve_timm_name(net_name, pretrained):
    """Resolve AnomalyMatch net name to timm model identifier.

    Args:
        net_name: AnomalyMatch-style net name (e.g. "efficientnet-lite0")
        pretrained: Whether pretrained weights are requested

    Returns:
        str: timm model name (with pretrained tag if applicable)

    Raises:
        ValueError: If net_name is not supported
    """
    timm_base = _TIMM_MODEL_MAP.get(net_name)
    if timm_base is None:
        supported = list(_TIMM_MODEL_MAP.keys())
        raise ValueError(
            f"Unsupported network architecture: {net_name}. Supported architectures: {supported}"
        )

    if pretrained:
        timm_name = _TIMM_PRETRAINED_TAG.get(timm_base)
        if timm_name is None:
            logger.warning(
                f"No pretrained weights available for {net_name}. Using random initialization."
            )
            return timm_base, False
        return timm_name, True

    return timm_base, False


def get_net_builder(net_name, pretrained=False, in_channels=3):
    """Create a neural network builder function for the specified architecture.

    This function returns a builder function that creates a neural network with the
    specified architecture when called with num_classes and in_channels parameters.
    Uses timm (pytorch-image-models) as the backend for all EfficientNet variants.

    Args:
        net_name (str): Name of the network architecture, supported values:
            - efficientnet-lite0 through efficientnet-lite4
            - efficientnet-b0 through efficientnet-b7
            - test-cnn (for testing only)
        pretrained (bool, optional): If True, loads pretrained weights. Default is False.
        in_channels (int, optional): Number of input channels. Default is 3.

    Returns:
        callable: A function that builds the network when called with (num_classes, in_channels)

    Raises:
        ValueError: If an unsupported network architecture is specified
    """
    if net_name == "test-cnn":
        logger.debug("Using test-cnn model (for testing only)")

        def build_test_cnn(num_classes, in_channels, pretrained=None):
            return TestCNN(num_classes=num_classes, in_channels=in_channels)

        return build_test_cnn

    timm_name, use_pretrained = _resolve_timm_name(net_name, pretrained)
    logger.debug(
        f"Using {'pretrained' if use_pretrained else 'non-pretrained'} {net_name} "
        f"(timm: {timm_name})"
    )

    def build_model(
        num_classes, in_channels, _timm_name=timm_name, _pretrained=use_pretrained, pretrained=None
    ):
        effective_pretrained = pretrained if pretrained is not None else _pretrained
        if effective_pretrained:
            try:
                return timm.create_model(
                    _timm_name,
                    pretrained=True,
                    num_classes=num_classes,
                    in_chans=in_channels,
                    cache_dir=str(_PRETRAINED_CACHE_DIR),
                )
            except Exception:
                logger.warning(
                    f"Bundled pretrained weights not available (clone with git-lfs to avoid "
                    f"re-downloading). Downloading {_timm_name} from HuggingFace."
                )
                return timm.create_model(
                    _timm_name,
                    pretrained=True,
                    num_classes=num_classes,
                    in_chans=in_channels,
                )
        return timm.create_model(
            _timm_name,
            pretrained=False,
            num_classes=num_classes,
            in_chans=in_channels,
        )

    return build_model
