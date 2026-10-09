#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.

"""Unit tests for checkpoint_io: safetensors-based model checkpoint serialization."""

import numpy as np
import pytest
import torch
from dotmap import DotMap
from fitsbolt.normalisation.NormalisationMethod import NormalisationMethod

import anomaly_match as am
from anomaly_match.data_io.checkpoint_io import (
    load_checkpoint,
    read_checkpoint_normalisation,
    read_model_normalisation,
    save_checkpoint,
    sync_normalisation_from_checkpoint,
)
from anomaly_match.data_io.load_images import get_fitsbolt_config


def _make_state_dict(seed=0):
    """Create a small deterministic state_dict for testing."""
    torch.manual_seed(seed)
    return {
        "layer.weight": torch.randn(4, 3),
        "layer.bias": torch.randn(4),
        "bn.running_mean": torch.zeros(4),
        "bn.running_var": torch.ones(4),
        "bn.num_batches_tracked": torch.tensor(0, dtype=torch.long),
    }


def _make_full_checkpoint(**overrides):
    """Create a complete checkpoint dict with sensible defaults."""
    checkpoint = {
        "train_model": _make_state_dict(seed=0),
        "eval_model": _make_state_dict(seed=1),
        "optimizer": None,
        "scheduler": None,
        "it": 42,
        "total_it": 100,
        "best_eval_acc": 0.95,
        "best_it": 80,
        "num_channels": 3,
        "net": "efficientnet-lite0",
        "normalisation_method": NormalisationMethod.CONVERSION_ONLY,
        "last_normalisation_method": NormalisationMethod.LOG,
        "fitsbolt_cfg": None,
    }
    checkpoint.update(overrides)
    return checkpoint


class TestSaveLoadRoundTrip:
    """Test that save_checkpoint -> load_checkpoint round-trips all data correctly."""

    def test_model_weights_roundtrip(self, tmp_path):
        """Verify train_model and eval_model state_dicts survive round-trip."""
        original = _make_full_checkpoint()
        path = save_checkpoint(original, tmp_path / "model")

        loaded = load_checkpoint(path)

        for key in ("train_model", "eval_model"):
            for param_name in original[key]:
                assert torch.equal(original[key][param_name], loaded[key][param_name]), (
                    f"{key}.{param_name} mismatch after round-trip"
                )

    def test_scalar_metadata_roundtrip(self, tmp_path):
        """Verify scalar metadata (it, total_it, etc.) survives round-trip."""
        original = _make_full_checkpoint()
        path = save_checkpoint(original, tmp_path / "model")
        loaded = load_checkpoint(path)

        assert loaded["it"] == 42
        assert loaded["total_it"] == 100
        assert loaded["best_eval_acc"] == 0.95
        assert loaded["num_channels"] == 3
        assert loaded["net"] == "efficientnet-lite0"

    def test_normalisation_enum_roundtrip(self, tmp_path):
        """Verify NormalisationMethod enum values survive round-trip."""
        original = _make_full_checkpoint()
        path = save_checkpoint(original, tmp_path / "model")
        loaded = load_checkpoint(path)

        assert loaded["normalisation_method"] == NormalisationMethod.CONVERSION_ONLY
        assert loaded["last_normalisation_method"] == NormalisationMethod.LOG
        assert isinstance(loaded["normalisation_method"], NormalisationMethod)

    def test_optimizer_state_roundtrip(self, tmp_path):
        """Verify optimizer state (including momentum tensors) survives round-trip."""
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        loss = model(torch.randn(1, 3)).sum()
        loss.backward()
        optimizer.step()

        opt_state = optimizer.state_dict()
        original = _make_full_checkpoint(optimizer=opt_state)
        path = save_checkpoint(original, tmp_path / "model")
        loaded = load_checkpoint(path)

        assert loaded["optimizer"]["param_groups"][0]["lr"] == 0.01
        for param_idx in opt_state["state"]:
            for key in opt_state["state"][param_idx]:
                orig_val = opt_state["state"][param_idx][key]
                loaded_val = loaded["optimizer"]["state"][param_idx][key]
                if isinstance(orig_val, torch.Tensor):
                    assert torch.equal(orig_val, loaded_val)

    def test_fitsbolt_cfg_roundtrip(self, tmp_path):
        """Verify fitsbolt DotMap config survives round-trip."""
        fb_cfg = DotMap(
            {
                "output_dtype": np.uint8,
                "size": [64, 64],
                "normalisation_method": NormalisationMethod.CONVERSION_ONLY,
                "n_output_channels": 3,
                "channel_combination": np.array([[1, 0], [0, 1], [0.5, 0.5]]),
            }
        )
        original = _make_full_checkpoint(fitsbolt_cfg=fb_cfg)
        path = save_checkpoint(original, tmp_path / "model")
        loaded = load_checkpoint(path)

        loaded_fb = loaded["fitsbolt_cfg"]
        assert isinstance(loaded_fb, DotMap)
        assert loaded_fb.normalisation_method == NormalisationMethod.CONVERSION_ONLY
        assert loaded_fb.output_dtype == np.uint8
        assert np.array_equal(loaded_fb.channel_combination, fb_cfg.channel_combination)

    def test_none_values_roundtrip(self, tmp_path):
        """Verify None values survive round-trip correctly."""
        original = _make_full_checkpoint(
            optimizer=None,
            scheduler=None,
            fitsbolt_cfg=None,
            best_eval_acc=None,
            normalisation_method=None,
        )
        path = save_checkpoint(original, tmp_path / "model")
        loaded = load_checkpoint(path)

        assert loaded["optimizer"] is None
        assert loaded["scheduler"] is None
        assert loaded["fitsbolt_cfg"] is None
        assert loaded["best_eval_acc"] is None
        assert loaded["normalisation_method"] is None


class TestFileFormat:
    """Test file format details."""

    def test_extension_forced_to_safetensors(self, tmp_path):
        """save_checkpoint forces .safetensors extension."""
        path = save_checkpoint(_make_full_checkpoint(), tmp_path / "model.pth")
        assert path.suffix == ".safetensors"
        assert path.exists()

    def test_load_nonexistent_raises(self, tmp_path):
        """Loading a nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_checkpoint(tmp_path / "nonexistent.safetensors")

    def test_shared_memory_tensors(self, tmp_path):
        """Tensors that share memory (e.g. EMA copy) are saved without error."""
        shared = _make_state_dict(seed=0)
        original = _make_full_checkpoint(
            train_model=shared,
            eval_model=shared,  # same object, shares memory
        )
        path = save_checkpoint(original, tmp_path / "model")
        loaded = load_checkpoint(path)
        assert "train_model" in loaded
        assert "eval_model" in loaded


class TestReadModelNormalisation:
    """The lightweight, tensor-free normalisation-metadata reader."""

    def _save_with_fitsbolt(self, tmp_path, method, image_size, n_channels):
        cfg = am.get_default_cfg()
        cfg.normalisation.normalisation_method = method
        cfg.normalisation.image_size = image_size
        cfg.normalisation.n_output_channels = n_channels
        cfg = get_fitsbolt_config(cfg)
        checkpoint = _make_full_checkpoint(
            num_channels=n_channels,
            normalisation_method=method,
            fitsbolt_cfg=cfg.fitsbolt_cfg,
        )
        return save_checkpoint(checkpoint, tmp_path / "model")

    def test_reads_embedded_normalisation(self, tmp_path):
        path = self._save_with_fitsbolt(tmp_path, NormalisationMethod.ASINH, [96, 96], n_channels=2)
        summary = read_model_normalisation(path)
        assert summary["image_size"] == [96, 96]
        assert summary["normalisation_method"] == NormalisationMethod.ASINH
        assert summary["n_output_channels"] == 2
        assert summary["num_channels"] == 2
        assert summary["net"] == "efficientnet-lite0"

    def test_missing_metadata_returns_none_fields(self, tmp_path):
        """A checkpoint without fitsbolt_cfg yields None normalisation fields."""
        path = save_checkpoint(_make_full_checkpoint(fitsbolt_cfg=None), tmp_path / "old")
        summary = read_model_normalisation(path)
        assert summary["image_size"] is None
        assert summary["normalisation_method"] is None
        assert summary["n_output_channels"] is None

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_model_normalisation(tmp_path / "nope.safetensors")


class TestSyncNormalisationFromCheckpoint:
    """The checkpoint is the single source of truth for prediction normalisation."""

    def _model_fitsbolt_cfg(self, method, image_size, n_channels):
        cfg = am.get_default_cfg()
        cfg.normalisation.normalisation_method = method
        cfg.normalisation.image_size = image_size
        cfg.normalisation.n_output_channels = n_channels
        return get_fitsbolt_config(cfg).fitsbolt_cfg

    def test_overwrites_divergent_cfg(self):
        """A stale cfg (wrong resolution/method/channels) is fully overwritten."""
        fb = self._model_fitsbolt_cfg(NormalisationMethod.ASINH, [150, 150], 1)

        cfg = am.get_default_cfg()
        cfg.normalisation.image_size = [999, 999]  # stale UI default
        cfg.normalisation.normalisation_method = NormalisationMethod.ZSCALE
        cfg.normalisation.n_output_channels = 3

        assert sync_normalisation_from_checkpoint(cfg, fb, None) is True
        assert cfg.normalisation.image_size == [150, 150]
        assert cfg.normalisation.normalisation_method == NormalisationMethod.ASINH
        assert cfg.normalisation.n_output_channels == 1
        assert cfg.fitsbolt_cfg is fb

    def test_channel_combination_synced_and_guarded(self):
        """The band-mixing matrix (a separate checkpoint field, applied on the
        GPU outside fitsbolt) is carried onto cfg; junk becomes None so the
        combine sites never receive an empty DotMap."""
        fb = self._model_fitsbolt_cfg(NormalisationMethod.CONVERSION_ONLY, [64, 64], 2)
        matrix = np.array([[1.0, 0.0, 0.0], [0.5, 0.5, 0.0]])

        cfg = am.get_default_cfg()
        assert sync_normalisation_from_checkpoint(cfg, fb, matrix)
        assert np.array_equal(np.asarray(cfg.normalisation.channel_combination), matrix)

        # A None matrix clears any stale cfg value.
        cfg2 = am.get_default_cfg()
        cfg2.normalisation.channel_combination = [[1.0]]
        assert sync_normalisation_from_checkpoint(cfg2, fb, None)
        assert cfg2.normalisation.channel_combination is None

    def test_channel_combination_row_count_must_match_channels(self):
        """A matrix whose row count disagrees with n_output_channels is rejected.

        Guards the 3x4 -> 2x4 truncation seen from stale UI state: a 2-row matrix
        on a 3-channel model would feed the model the wrong channel count.
        """
        fb = self._model_fitsbolt_cfg(NormalisationMethod.CONVERSION_ONLY, [64, 64], 3)
        bad = np.array([[0.0, 0.66, 0.33, 0.0], [1.0, 0.0, 0.0, 0.0]])  # 2 rows, model wants 3

        cfg = am.get_default_cfg()
        with pytest.raises(ValueError, match="2 rows but the model has 3 output channels"):
            sync_normalisation_from_checkpoint(cfg, fb, bad)

        # The matching 3-row matrix is accepted.
        good = np.array([[0.0, 0.66, 0.33, 0.0], [0.0, 0.0, 0.33, 0.66], [1.0, 0.0, 0.0, 0.0]])
        cfg_ok = am.get_default_cfg()
        assert sync_normalisation_from_checkpoint(cfg_ok, fb, good)
        assert np.array_equal(np.asarray(cfg_ok.normalisation.channel_combination), good)

    def test_non_array_channel_combination_guarded_to_none(self):
        """A non-array (e.g. the DotMap-None pitfall) is guarded to None."""
        fb = self._model_fitsbolt_cfg(NormalisationMethod.CONVERSION_ONLY, [64, 64], 2)
        cfg3 = am.get_default_cfg()
        assert sync_normalisation_from_checkpoint(cfg3, fb, {})
        assert cfg3.normalisation.channel_combination is None

    def test_embedded_matrix_is_not_used_without_standalone_field(self):
        """No fallback: a matrix embedded in fitsbolt_cfg (old FITS checkpoints)
        is ignored — only the standalone field is read, so an unsupported old
        checkpoint surfaces as None and fails hard downstream."""
        fb = DotMap(
            {
                "size": [176, 176],
                "normalisation_method": int(NormalisationMethod.ASINH),
                "n_output_channels": 3,
                "channel_combination": np.array(
                    [[1.0, 0, 0, 0], [0, 1.0, 0.5, 0], [0, 0, 0.5, 1.0]]
                ),
            },
            _dynamic=False,
        )
        cfg = am.get_default_cfg()
        assert sync_normalisation_from_checkpoint(cfg, fb, None)
        assert cfg.normalisation.channel_combination is None

    def test_returns_false_for_legacy(self):
        cfg = am.get_default_cfg()
        assert sync_normalisation_from_checkpoint(cfg, None, None) is False

    def test_read_checkpoint_normalisation_roundtrip(self, tmp_path):
        fb = self._model_fitsbolt_cfg(NormalisationMethod.LOG, [128, 128], 2)
        matrix = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
        checkpoint = _make_full_checkpoint(fitsbolt_cfg=fb, channel_combination=matrix)
        path = save_checkpoint(checkpoint, tmp_path / "model")

        loaded_fb, loaded_cc = read_checkpoint_normalisation(path)
        assert loaded_fb is not None
        assert list(loaded_fb["size"]) == [128, 128]
        assert loaded_fb["n_output_channels"] == 2
        assert np.array_equal(np.asarray(loaded_cc), matrix)

        legacy = save_checkpoint(_make_full_checkpoint(fitsbolt_cfg=None), tmp_path / "legacy")
        legacy_fb, legacy_cc = read_checkpoint_normalisation(legacy)
        assert legacy_fb is None and legacy_cc is None


class TestSecurity:
    """Verify the format is safe against code execution attacks."""

    def test_no_pickle_in_file(self, tmp_path):
        """The saved file must not contain pickle opcodes."""
        path = save_checkpoint(_make_full_checkpoint(), tmp_path / "model")
        data = path.read_bytes()
        # Pickle protocol markers (0x80 = protocol 2+, 'cos\n' = protocol 0)
        assert not data[8:].startswith(b"\x80\x02")  # not pickle protocol 2
        assert not data[8:].startswith(b"cos\n")  # not pickle protocol 0
