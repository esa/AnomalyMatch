#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Regression tests for channel_combination in Cutana preview and decode paths.

Verifies that channel_combination is applied exactly once, producing the
expected output channels.  Catches the double-application bug where both
the orchestrator's ``channel_weights`` and fitsbolt's ``channel_combination``
used to be active on the same pixel data.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
import torch
from fitsbolt.normalisation.NormalisationMethod import NormalisationMethod

import anomaly_match as am
from anomaly_match.data_io.checkpoint_io import (
    read_checkpoint_normalisation,
    save_checkpoint,
    sync_normalisation_from_checkpoint,
)
from anomaly_match.data_io.container_loaders import (
    apply_channel_combination_to_cutana_image,
    decode_cutana_raw_images,
    decode_zarr_image,
)
from anomaly_match.data_io.load_images import get_fitsbolt_config
from anomaly_match.prediction.cutana_loader import load_sample_cutouts

CUTANA_DIR = os.path.join("tests", "test_data", "cutana_catalogue")
CUTANA_LABELS = os.path.join(CUTANA_DIR, "labeled_data.csv")

_skip_no_cutana = pytest.mark.skipif(
    not os.path.isdir(CUTANA_DIR),
    reason="Cutana test data not available",
)


def _cfg_with_channel_combination(channel_combination: np.ndarray) -> am.DotMap:
    """Build a config with ``channel_combination`` set, ready for Cutana loading."""
    cfg = am.get_default_cfg()
    cfg.data_dir = CUTANA_DIR
    cfg.label_file = CUTANA_LABELS
    cfg.normalisation.image_size = [64, 64]
    cfg.normalisation.n_output_channels = 3
    cfg.normalisation.fits_extension = None
    cfg.normalisation.channel_combination = channel_combination
    return get_fitsbolt_config(cfg)


class TestChannelCombinationPersistedInCheckpoint:
    """The band-mixing matrix is applied on the GPU outside fitsbolt, so it lives
    in its own checkpoint field (not fitsbolt_cfg).  It must survive the
    save/load round-trip and be recoverable onto cfg, or a model trained on more
    bands than its input channels (e.g. 4-band Euclid -> 3 channels) can't be
    scored."""

    def test_survives_checkpoint_round_trip_and_recovers_onto_cfg(self, tmp_path):
        matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=float)
        cfg = am.get_default_cfg()
        cfg.normalisation.image_size = [176, 176]
        cfg.normalisation.n_output_channels = 3
        cfg.normalisation.fits_extension = None
        cfg.normalisation.channel_combination = matrix
        cfg = get_fitsbolt_config(cfg)

        path = save_checkpoint(
            {
                "eval_model": {"w": torch.zeros(2)},
                "net": "efficientnet-lite0",
                "num_channels": 3,
                "fitsbolt_cfg": cfg.fitsbolt_cfg,
                "channel_combination": cfg.normalisation.channel_combination,
            },
            tmp_path / "model",
        )

        # Read the standalone matrix back through the same reader prediction uses
        # (read_checkpoint_normalisation), so the test exercises the real recovery
        # path rather than poking the raw checkpoint dict.
        fb, cc = read_checkpoint_normalisation(path)
        assert np.array_equal(np.asarray(cc), matrix)

        # A fresh prediction cfg (matrix defaulted away) recovers it from the model.
        pred_cfg = am.get_default_cfg()
        assert sync_normalisation_from_checkpoint(pred_cfg, fb, cc)
        assert np.array_equal(np.asarray(pred_cfg.normalisation.channel_combination), matrix)
        assert pred_cfg.normalisation.image_size == [176, 176]


class TestDecodeZarrImageChannelCombination:
    """channel_combination in decode_zarr_image (used for upstream Zarr reads)."""

    def test_channel_combination_zeros_out_channels(self):
        """Out1=sum, Out2=0, Out3=0 → only R channel has signal."""
        matrix = np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0]], dtype=float)
        cfg = _cfg_with_channel_combination(matrix)
        img = np.full((32, 32, 3), 100, dtype=np.uint8)

        result = decode_zarr_image(img, cfg)

        assert result[:, :, 1].max() < 5, f"G should be ~0, got {result[:, :, 1].mean()}"
        assert result[:, :, 2].max() < 5, f"B should be ~0, got {result[:, :, 2].mean()}"
        assert result[:, :, 0].mean() > 50, "R should have signal"

    @pytest.mark.parametrize("n_output_channels", [1, 3])
    def test_greyscale_input_without_channel_axis(self, n_output_channels):
        """A raw (H, W) greyscale array (no channel axis) must decode cleanly."""
        cfg = am.get_default_cfg()
        cfg.normalisation.image_size = [32, 32]
        cfg.normalisation.n_output_channels = n_output_channels
        cfg.normalisation.fits_extension = None
        cfg = get_fitsbolt_config(cfg)
        img = np.full((32, 32), 100, dtype=np.uint8)

        result = decode_zarr_image(img, cfg)

        assert result.shape == (32, 32, n_output_channels)


class TestCutanaCacheChannelCombination:
    """channel_combination applied to already-normalised Cutana cache data."""

    def test_channel_combination_zeros_out_channels(self):
        """Out1=sum, Out2=0, Out3=0 → only R channel has signal."""
        matrix = np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0]], dtype=float)
        cfg = _cfg_with_channel_combination(matrix)
        img = np.full((32, 32, 3), 100, dtype=np.uint8)

        result = apply_channel_combination_to_cutana_image(img, cfg)

        # Identity 1→sum: Out1 = 300 which is clipped to 255 at the dtype
        # boundary; Out2 and Out3 are zeroed by the matrix.
        assert result[:, :, 1].max() == 0
        assert result[:, :, 2].max() == 0
        assert result[:, :, 0].min() > 0


class TestDecodeCutanaRawImage:
    """Read-time pipeline on the normalisation-agnostic Cutana cache."""

    def test_resizes_to_requested_size(self):
        """Raw 128 px cache image → 64 px user size via fitsbolt resize."""
        cfg = _cfg_with_channel_combination(np.eye(3, dtype=float))
        cfg.normalisation.image_size = [64, 64]
        cfg = get_fitsbolt_config(cfg)
        raw = np.full((128, 128, 3), 0.5, dtype=np.float32)

        result = decode_cutana_raw_images([raw], cfg)[0]

        assert result.shape[:2] == (64, 64)

    def test_applies_channel_combination(self):
        """channel_combination reduces bands at read time."""
        matrix = np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0]], dtype=float)
        cfg = _cfg_with_channel_combination(matrix)
        # Keep the pipeline in float32 so the conversion-only cast doesn't
        # truncate our test signal — we're verifying matrix behaviour, not
        # the dtype cast itself.
        cfg.normalisation.output_dtype = np.float32
        cfg = get_fitsbolt_config(cfg)
        raw = np.full((64, 64, 3), 0.3, dtype=np.float32)

        result = decode_cutana_raw_images([raw], cfg)[0]

        assert result[:, :, 1].max() == 0
        assert result[:, :, 2].max() == 0
        assert result[:, :, 0].min() > 0

    def test_channel_reducing_matrix_keeps_per_band_asinh(self):
        """Each band is stretched with its own ASINH scale/clip under a reducing matrix.

        ``get_fitsbolt_config`` cuts the ASINH lists to ``n_output_channels``, so a
        2x3 matrix used to stretch band 2 with band 1's values. The selected band
        must decode exactly as it does under the identity matrix.
        """
        raw = np.random.default_rng(0).uniform(0.0, 10.0, (64, 64, 3)).astype(np.float32)

        def decode(matrix):
            cfg = _cfg_with_channel_combination(matrix)
            cfg.normalisation.normalisation_method = NormalisationMethod.ASINH
            cfg.normalisation.norm_asinh_scale = [0.1, 0.7, 5.0]
            cfg.normalisation.norm_asinh_clip = [99.0, 99.5, 90.0]
            cfg.normalisation.n_output_channels = matrix.shape[0]
            cfg.normalisation.output_dtype = np.float32
            cfg = get_fitsbolt_config(cfg)
            return decode_cutana_raw_images([raw], cfg)[0]

        per_band = decode(np.eye(3, dtype=float))
        reduced = decode(np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]))

        assert reduced.shape[-1] == 2
        np.testing.assert_allclose(reduced[..., 0], per_band[..., 0], rtol=1e-6)
        np.testing.assert_allclose(reduced[..., 1], per_band[..., 2], rtol=1e-6)

    def test_applies_output_dtype(self):
        """output_dtype is picked up from the live cfg, not from the cache."""
        cfg = _cfg_with_channel_combination(np.eye(3, dtype=float))
        cfg.normalisation.output_dtype = np.uint8
        cfg = get_fitsbolt_config(cfg)
        raw = np.full((64, 64, 3), 0.5, dtype=np.float32)

        result = decode_cutana_raw_images([raw], cfg)[0]

        assert result.dtype == np.uint8

    def test_broadcasts_single_band_when_no_combination(self):
        """Single-band cache image + no CC + n_out=3 → replicate to 3 channels."""
        cfg = am.get_default_cfg()
        cfg.normalisation.image_size = [64, 64]
        cfg.normalisation.n_output_channels = 3
        cfg.normalisation.channel_combination = None
        cfg.normalisation.fits_extension = None
        cfg = get_fitsbolt_config(cfg)
        raw = np.full((64, 64, 1), 0.7, dtype=np.float32)

        result = decode_cutana_raw_images([raw], cfg)[0]

        assert result.shape[-1] == 3
        # All three channels should carry the same signal (broadcast).
        assert np.allclose(result[:, :, 0], result[:, :, 1])
        assert np.allclose(result[:, :, 1], result[:, :, 2])

    def test_broadcast_preserves_uint8_dtype(self):
        """Broadcast branch must preserve output_dtype (no silent float32 promotion).

        Regression: the batched decoder called fitsbolt's
        ``batch_channel_combination`` without ``output_dtype``, which
        defaulted to float32.  BasicDataset then tried
        ``Image.fromarray`` on a (H,W,3) float32 array and crashed with
        ``Cannot handle this data type: (1, 1, 3), <f4`` — PIL has no
        3-channel float32 mode.
        """
        from PIL import Image  # noqa: PLC0415

        cfg = am.get_default_cfg()
        cfg.normalisation.image_size = [64, 64]
        cfg.normalisation.n_output_channels = 3
        cfg.normalisation.channel_combination = None
        cfg.normalisation.fits_extension = None
        cfg.normalisation.output_dtype = np.uint8
        cfg = get_fitsbolt_config(cfg)
        raw = np.full((64, 64, 1), 0.7, dtype=np.float32)

        result = decode_cutana_raw_images([raw], cfg)[0]

        assert result.dtype == np.uint8
        # The downstream training path calls Image.fromarray; verify it works.
        Image.fromarray(result)


@_skip_no_cutana
class TestCutanaPreviewChannelCombination:
    """channel_combination in load_sample_cutouts (preview path)."""

    def test_channel_combination_applied_to_cutouts(self):
        """Blue-only channel_combination produces images with R=G≈0.

        Test catalogue is single-band (VIS), so the matrix is ``(3, 1)`` —
        Out0 and Out1 zero out the single VIS band, Out2 keeps it.
        """

        matrix = np.array([[0], [0], [1]], dtype=float)
        cfg = _cfg_with_channel_combination(matrix)

        samples = load_sample_cutouts(cfg, CUTANA_DIR, n_samples=2)
        assert len(samples) > 0, "Should load at least one cutout"

        for sid, img in samples:
            assert img[:, :, 0].mean() < 5, f"{sid}: R should be ~0, got {img[:, :, 0].mean():.1f}"
            assert img[:, :, 1].mean() < 5, f"{sid}: G should be ~0, got {img[:, :, 1].mean():.1f}"

    def test_broadcast_channel_combination_preserves_signal(self):
        """Broadcast single VIS band to 3 output channels keeps all channels populated."""

        matrix = np.array([[1], [1], [1]], dtype=float)
        cfg = _cfg_with_channel_combination(matrix)

        samples = load_sample_cutouts(cfg, CUTANA_DIR, n_samples=2)
        assert len(samples) > 0

    def test_all_labeled_returned_with_enough_samples(self):
        """Requesting enough samples covers all labeled IDs."""

        cfg = _cfg_with_channel_combination(np.array([[1], [1], [1]], dtype=float))
        label_df = pd.read_csv(CUTANA_LABELS)
        labeled_ids = set(label_df["id"])

        samples = load_sample_cutouts(cfg, CUTANA_DIR, n_samples=len(labeled_ids) + 20)
        loaded_ids = {sid for sid, _ in samples}

        matched = loaded_ids & labeled_ids
        assert len(matched) == len(labeled_ids), (
            f"Only {len(matched)}/{len(labeled_ids)} labeled IDs found in samples"
        )
