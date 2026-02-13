#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.

import pytest

from anomaly_match import get_default_cfg
from prediction_utils import MEMORY_COEFFICIENTS, estimate_batch_size

FAKE_VRAM_BYTES = 16 * 1024**3  # 16GB


@pytest.fixture
def test_config():
    cfg = get_default_cfg()
    cfg.normalisation.image_size = [100, 100]
    cfg.net = "efficientnet-lite0"
    cfg.num_channels = 3

    return cfg


@pytest.fixture
def no_cuda(monkeypatch):
    """Cuda GPU mock showing no GPU available"""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: None)


@pytest.fixture
def fake_cuda(monkeypatch):
    """Cuda GPU mock with fixed 16 GB of vram"""
    import torch

    class Props:
        total_memory = FAKE_VRAM_BYTES

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: Props())


def verify_batch_size_estimation(
    test_config, estimated_batch_size, available_vram, safety_margin, net_name=None
):
    """Helper function to verify batch size estimation"""
    S2 = test_config.normalisation.image_size[0] * test_config.normalisation.image_size[1]
    coeffs = MEMORY_COEFFICIENTS[net_name or test_config.net]

    # Calculate memory occupied according to model with the estimated batch size
    memory_occupied = (
        coeffs["a"] * estimated_batch_size * S2 * test_config.num_channels
        + coeffs["b"] * estimated_batch_size
        + coeffs["c"]
    )

    # How much memory actually is avaiable according to the specification
    usable_vram = available_vram * (1 - safety_margin)

    # What the batch size should be if everything works properly
    true_batch_size = max(
        1,
        int(
            (usable_vram - coeffs["c"])
            / (coeffs["a"] * S2 * test_config.num_channels + coeffs["b"])
        ),
    )

    assert true_batch_size == estimated_batch_size
    assert memory_occupied <= usable_vram


def test_estimate_batch_size_manual_vram(test_config):
    """Test batch size estimation with manual vram specification"""

    # Setup
    available_vram = 1024  # MB
    safety_margin = 0.2
    estimated_batch_size = estimate_batch_size(
        test_config, available_vram=available_vram, safety_margin=safety_margin
    )

    verify_batch_size_estimation(
        test_config=test_config,
        estimated_batch_size=estimated_batch_size,
        available_vram=available_vram,
        safety_margin=safety_margin,
    )


def test_estimate_batch_size_unknown_model(test_config):
    """Test batch size estimation with unknown model"""

    # Setup
    available_vram = 1024  # MB
    test_config.net = "UnknownNet"
    safety_margin = 0.2
    estimated_batch_size = estimate_batch_size(
        test_config, available_vram=available_vram, safety_margin=safety_margin
    )

    verify_batch_size_estimation(
        test_config=test_config,
        estimated_batch_size=estimated_batch_size,
        available_vram=available_vram,
        safety_margin=safety_margin,
        net_name="efficientnet-lite0",
    )


def test_estimate_batch_size_cuda_16gb(test_config, fake_cuda):
    """Test batch size estimation with efficientnet-lite0 and 16GB vram cuda GPU"""

    # Setup
    safety_margin = 0.2
    estimated_batch_size = estimate_batch_size(test_config, safety_margin=safety_margin)

    verify_batch_size_estimation(
        test_config=test_config,
        estimated_batch_size=estimated_batch_size,
        safety_margin=safety_margin,
        available_vram=FAKE_VRAM_BYTES / 1024**2,
    )


def test_estimate_batch_size_no_cuda(test_config, no_cuda):
    """Test batch size estimation with eddicientnet-lite0 and no cuda GPU available"""

    # Setup
    safety_margin = 0.2
    estimated_batch_size = estimate_batch_size(test_config, safety_margin=safety_margin)

    verify_batch_size_estimation(
        test_config=test_config,
        estimated_batch_size=estimated_batch_size,
        safety_margin=safety_margin,
        available_vram=4096,
    )


def test_estimate_batch_size_invalid_memory(test_config):
    """Test safeguard with invalid memory available"""

    batch_size = estimate_batch_size(test_config, available_vram=0)

    assert batch_size == 1


def test_estimate_batch_size_invalid_model_coefficients(test_config, monkeypatch):
    import prediction_utils

    # Inject invalid coefficients
    monkeypatch.setattr(
        prediction_utils,
        "MEMORY_COEFFICIENTS",
        {"efficientnet-lite0": {"a": -0.1, "b": -0.1, "c": -0.1}},
    )

    batch_size = estimate_batch_size(test_config, available_vram=4096)

    assert batch_size == 1
