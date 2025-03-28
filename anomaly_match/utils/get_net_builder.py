#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file ‘LICENCE.txt’.
from efficientnet_pytorch import EfficientNet
import efficientnet_lite_pytorch
from efficientnet_lite0_pytorch_model import EfficientnetLite0ModelFile

from loguru import logger


def get_net_builder(net_name, pretrained=False, in_channels=3):
    """Create a neural network builder function for the specified architecture.

    This function returns a builder function that creates a neural network with the
    specified architecture when called with num_classes and in_channels parameters.
    Currently supports various EfficientNet variants.

    Args:
        net_name (str): Name of the network architecture, supported values:
            - efficientnet-lite0, efficientnet-lite1, etc.
            - efficientnet-b0, efficientnet-b1, etc.
        pretrained (bool, optional): If True, loads pretrained weights. Default is False.
        in_channels (int, optional): Number of input channels. Default is 3.

    Returns:
        callable: A function that builds the network when called with (num_classes, in_channels)

    Raises:
        ValueError: If an unsupported network architecture is specified
    """
    if "efficientnet-lite" in net_name:
        if pretrained:
            if net_name == "efficientnet-lite0":
                logger.debug(f"Using pretrained {net_name} model")
                weights_path = EfficientnetLite0ModelFile.get_model_file_path()

                return lambda num_classes, in_channels: efficientnet_lite_pytorch.EfficientNet.from_pretrained(
                    "efficientnet-lite0",
                    weights_path=weights_path,
                    num_classes=num_classes,
                    in_channels=in_channels,
                )
            else:
                logger.warning(
                    f"Only efficientnet-lite0 pretrained is supported. Using non-pretrained {net_name} instead."
                )
                return lambda num_classes, in_channels: efficientnet_lite_pytorch.EfficientNet.from_name(
                    net_name, num_classes=num_classes, in_channels=in_channels
                )
        else:
            logger.debug(f"Using non-pretrained {net_name} model")
            return (
                lambda num_classes, in_channels: efficientnet_lite_pytorch.EfficientNet.from_name(
                    net_name, num_classes=num_classes, in_channels=in_channels
                )
            )

    elif "efficientnet" in net_name:
        if pretrained:
            logger.debug(f"Using pretrained {net_name} model")
            return lambda num_classes, in_channels: EfficientNet.from_pretrained(
                net_name, num_classes=num_classes, in_channels=in_channels
            )

        else:
            logger.debug(f"Using non-pretrained {net_name} model")
            return lambda num_classes, in_channels: EfficientNet.from_name(
                net_name, num_classes=num_classes, in_channels=in_channels
            )
    else:
        error_msg = (
            f"Unsupported network architecture: {net_name}. "
            f"Supported architectures: efficientnet-b[0-7] and efficientnet-lite[0-4]"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
