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

from anomaly_match.data_io.checkpoint_io import load_checkpoint, save_checkpoint


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
    """Test that save_checkpoint → load_checkpoint round-trips all data correctly."""

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
        assert loaded["best_it"] == 80
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
        # Build a real optimizer state
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        # Step once to create momentum buffers
        loss = model(torch.randn(1, 3)).sum()
        loss.backward()
        optimizer.step()

        opt_state = optimizer.state_dict()
        original = _make_full_checkpoint(optimizer=opt_state)
        path = save_checkpoint(original, tmp_path / "model")
        loaded = load_checkpoint(path)

        # Check param_groups
        assert loaded["optimizer"]["param_groups"][0]["lr"] == 0.01
        assert loaded["optimizer"]["param_groups"][0]["momentum"] == 0.9

        # Check state tensors
        for param_idx in opt_state["state"]:
            for key in opt_state["state"][param_idx]:
                orig_val = opt_state["state"][param_idx][key]
                loaded_val = loaded["optimizer"]["state"][param_idx][key]
                if isinstance(orig_val, torch.Tensor):
                    assert torch.equal(orig_val, loaded_val)

    def test_scheduler_state_roundtrip(self, tmp_path):
        """Verify scheduler state survives round-trip."""
        sched_state = {
            "T_max": 200,
            "eta_min": 0,
            "last_epoch": 50,
            "_step_count": 51,
            "base_lrs": [0.01],
            "_last_lr": [0.005],
        }
        original = _make_full_checkpoint(scheduler=sched_state)
        path = save_checkpoint(original, tmp_path / "model")
        loaded = load_checkpoint(path)

        assert loaded["scheduler"]["T_max"] == 200
        assert loaded["scheduler"]["last_epoch"] == 50

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

    def test_labeled_data_csv_roundtrip(self, tmp_path):
        """Verify labeled_data_csv string survives round-trip."""
        csv = "filename,label\nimg1.jpg,anomaly\nimg2.jpg,normal\n"
        original = _make_full_checkpoint(labeled_data_csv=csv)
        path = save_checkpoint(original, tmp_path / "model")
        loaded = load_checkpoint(path)

        assert loaded["labeled_data_csv"] == csv

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

    def test_safetensors_extension_preserved(self, tmp_path):
        """If .safetensors extension is already correct, it's preserved."""
        path = save_checkpoint(_make_full_checkpoint(), tmp_path / "model.safetensors")
        assert path.suffix == ".safetensors"

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
        # Should not raise RuntimeError about shared tensors
        path = save_checkpoint(original, tmp_path / "model")
        loaded = load_checkpoint(path)
        assert "train_model" in loaded
        assert "eval_model" in loaded


class TestSecurity:
    """Verify the format is safe against code execution attacks."""

    def test_no_pickle_in_file(self, tmp_path):
        """The saved file must not contain pickle opcodes."""
        path = save_checkpoint(_make_full_checkpoint(), tmp_path / "model")
        data = path.read_bytes()
        # Pickle protocol markers (0x80 = protocol 2+, 'cos\n' = protocol 0)
        # safetensors files start with a little-endian u64 header size
        assert not data[8:].startswith(b"\x80\x02")  # not pickle protocol 2
        assert not data[8:].startswith(b"cos\n")  # not pickle protocol 0

    def test_metadata_is_plain_json(self, tmp_path):
        """All metadata in the safetensors header is valid JSON strings."""
        import json

        from safetensors import safe_open

        path = save_checkpoint(_make_full_checkpoint(), tmp_path / "model")
        with safe_open(str(path), framework="pt") as f:
            metadata = f.metadata()

        for key, value in metadata.items():
            # Every metadata value must be a valid JSON string
            parsed = json.loads(value)
            assert parsed is not None or value == "null"
