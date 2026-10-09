#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Regression tests for the batched Cutana channel-combination helper.

The batched ``apply_channel_combination_to_cutana_batch`` replaced a per-image
Python loop on the prediction/training hot path (issue #458).  Its output must
stay bit-identical to stacking the single-image variant across the three
shapes that occur in practice: a real combination matrix, the single-band →
multi-channel broadcast, and the single-band passthrough.
"""

import numpy as np
import pytest
import torch

import anomaly_match as am
from anomaly_match.data_io.container_loaders import (
    apply_channel_combination_to_cutana_batch,
    apply_channel_combination_to_cutana_image,
)
from prediction_utils import cutana_batch_to_model_tensor_gpu


def _cfg(n_output, channel_combination):
    cfg = am.get_default_cfg()
    cfg.normalisation.n_output_channels = n_output
    cfg.normalisation.channel_combination = channel_combination
    return cfg


@pytest.mark.parametrize(
    "n_in, n_output, channel_combination",
    [
        # VIS+NIR: 4 bands blended into 3 output channels.
        (4, 3, np.array([[1, 0, 0, 0], [0, 1, 0.5, 0], [0, 0, 0.5, 1]], dtype=np.float32)),
        # Single band broadcast to 3 channels (no matrix configured).
        (1, 3, None),
        # Single band passthrough.
        (1, 1, None),
    ],
)
def test_batch_matches_stacked_per_image(n_in, n_output, channel_combination):
    cfg = _cfg(n_output, channel_combination)
    rng = np.random.RandomState(0)
    batch = rng.randint(0, 256, (5, 8, 8, n_in), dtype=np.uint8)

    batched = apply_channel_combination_to_cutana_batch(batch, cfg)
    per_image = np.stack([apply_channel_combination_to_cutana_image(img, cfg) for img in batch])

    assert batched.shape == (5, 8, 8, n_output)
    assert batched.dtype == np.uint8
    assert np.array_equal(batched, per_image)


def test_single_band_broadcast_replicates_channels():
    cfg = _cfg(3, None)
    rng = np.random.RandomState(1)
    batch = rng.randint(0, 256, (2, 8, 8, 1), dtype=np.uint8)

    out = apply_channel_combination_to_cutana_batch(batch, cfg)

    assert np.array_equal(out[..., 0], out[..., 1])
    assert np.array_equal(out[..., 0], out[..., 2])
    assert np.array_equal(out[..., 0], batch[..., 0])


def test_non_hwc_image_passes_through():
    """The single-image wrapper leaves a channel-less array untouched."""
    cfg = _cfg(3, np.array([[0], [0], [1]], dtype=np.float32))
    img = np.arange(64, dtype=np.uint8).reshape(8, 8)  # 2-D, no channel axis

    out = apply_channel_combination_to_cutana_image(img, cfg)

    assert np.array_equal(out, img)


_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize(
    "n_in, n_output, channel_combination",
    [
        (4, 3, np.array([[1, 0, 0, 0], [0, 1, 0.5, 0], [0, 0, 0.5, 1]], dtype=np.float32)),
        (4, 3, np.array([[1, 0, 0, 0], [0, 0.5, 0.5, 0], [0, 0, 0.5, 0.5]], dtype=np.float32)),
        (1, 3, None),  # broadcast
        (1, 1, None),  # passthrough
    ],
)
def test_cutana_model_tensor_matches_cpu_path(device, n_in, n_output, channel_combination):
    """The on-device combine+scale reproduces the CPU/fitsbolt model tensor.

    Runs on CPU (the function falls back when CUDA is unavailable) and, when a
    GPU is present, also on CUDA — so CI without a GPU still exercises it.
    """
    cfg = _cfg(n_output, channel_combination)
    rng = np.random.RandomState(2)
    batch = rng.randint(0, 256, (5, 8, 8, n_in), dtype=np.uint8)

    got = cutana_batch_to_model_tensor_gpu(batch, cfg, device=device).cpu()

    combined = apply_channel_combination_to_cutana_batch(batch, cfg)  # (N,H,W,Cout) uint8
    ref = torch.from_numpy(combined.transpose(0, 3, 1, 2).astype(np.float32) / 255.0)

    assert got.shape == ref.shape == (5, n_output, 8, 8)
    assert torch.allclose(got, ref, atol=1e-5)


def test_cutana_model_tensor_uint8_reduction_value():
    """4-band → 1-channel average lands at the analytically expected [0,1] value."""
    cfg = _cfg(1, np.array([[0.25, 0.25, 0.25, 0.25]], dtype=np.float32))
    rng = np.random.RandomState(3)
    batch = rng.randint(0, 256, (2, 8, 8, 4), dtype=np.uint8)

    out = cutana_batch_to_model_tensor_gpu(batch, cfg, device="cpu")

    # 0.25-each blend = per-pixel mean of the 4 bands; the uint8 path truncates
    # the combined value (fitsbolt quantisation) before the /255 scaling.
    blended = (batch.astype(np.float32) * 0.25).sum(axis=3)
    expected = torch.from_numpy(np.trunc(blended) / 255.0)[:, None, :, :]

    assert out.shape == (2, 1, 8, 8)
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=0)


def test_cutana_model_tensor_float_input_not_rescaled():
    """Float input already in [0,1] must not be divided by 255 again."""
    cfg = _cfg(3, np.eye(3, dtype=np.float32))
    batch = np.random.random((2, 8, 8, 3)).astype(np.float32)

    out = cutana_batch_to_model_tensor_gpu(batch, cfg, device="cpu")

    assert out.dtype == torch.float32
    assert 0.0 <= out.min().item() <= out.max().item() <= 1.0
    assert out.max().item() > 0.1, "float input was crushed — likely divided by 255"


def test_band_mismatch_without_matrix_raises_clearly():
    """A model whose checkpoint dropped its channel_combination (older format)
    would feed the conv stem the wrong channel count — fail loudly with an
    actionable message instead of crashing deep in the model."""
    # 4 bands in, model expects 3 channels, but no channel_combination is set.
    cfg = _cfg(3, None)
    batch = np.random.randint(0, 256, (2, 8, 8, 4), dtype=np.uint8)

    with pytest.raises(ValueError, match="channel_combination"):
        cutana_batch_to_model_tensor_gpu(batch, cfg, device="cpu")


def test_wrong_shaped_matrix_raises():
    """A channel_combination whose row count != n_output_channels must be
    rejected before it feeds the model the wrong channel count (the 3x4 -> 2x4
    truncation from stale UI state)."""
    # Model wants 3 channels, 4 input bands, but the matrix has only 2 rows.
    cfg = _cfg(3, np.array([[0, 0.66, 0.33, 0], [1, 0, 0, 0]], dtype=np.float32))
    batch = np.random.randint(0, 256, (2, 8, 8, 4), dtype=np.uint8)

    with pytest.raises(ValueError, match=r"n_output_channels=3.*n_bands=4|shape"):
        cutana_batch_to_model_tensor_gpu(batch, cfg, device="cpu")
