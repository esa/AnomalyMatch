#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.

import json
import shutil
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from dotmap import DotMap
from fitsbolt.normalisation.NormalisationMethod import NormalisationMethod

from anomaly_match.data_io.checkpoint_io import load_checkpoint
from anomaly_match.data_io.SessionIOHandler import SessionIOHandler
from anomaly_match.pipeline.SessionTracker import SessionTracker
from anomaly_match.utils.get_net_builder import get_net_builder
from tests.test_data.generate_test_model import STATE_DICT_MANIFEST_PATH, state_dict_manifest


class MockModel(nn.Module):
    """Simple mock model for testing."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)

    def forward(self, x):
        return self.linear(x)


class MockFixMatch:
    """Mock FixMatch model for testing."""

    def __init__(self, it_value=100):
        self.train_model = MockModel()
        self.eval_model = MockModel()
        self.optimizer = torch.optim.Adam(self.train_model.parameters())
        self.scheduler = None
        self.it = it_value
        self.total_it = 200
        self.best_eval_acc = 0.85
        self.best_it = 150
        self.last_normalisation_method = NormalisationMethod.CONVERSION_ONLY


class TestModelIOIntegration:
    """Test model saving and loading through SessionIOHandler."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.session_io = SessionIOHandler(str(self.temp_dir))
        self.session_tracker = SessionTracker("test_session")
        self.mock_model = MockFixMatch()
        # Create test config from default
        from anomaly_match.utils.get_default_cfg import get_default_cfg

        self.cfg = get_default_cfg()
        self.cfg.model_path = str(self.temp_dir / "test_model.safetensors")

    def teardown_method(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_save_model_with_session_tracker(self):
        """Test saving model with session tracker."""
        # Start a session iteration to ensure session_iterations is not empty
        self.session_tracker.start_new_session_iteration()
        self.session_tracker.add_labeled_sample("test.jpg", "anomaly")

        # Save model
        model_path = self.session_io.save_model(self.mock_model, self.cfg, self.session_tracker)

        # Verify file was created
        assert Path(model_path).exists()
        assert "test_session" in model_path  # Should be in session directory

        # Verify model path was updated in session tracker
        assert self.session_tracker.session_iterations
        assert self.session_tracker.session_iterations[-1].model_state_path == model_path

    def test_save_model_with_config_path(self):
        """Test saving model using config path."""
        # Save model without session tracker
        saved_path = self.session_io.save_model(self.mock_model, self.cfg, session_tracker=None)

        # Verify file was created at config path
        assert Path(saved_path).exists()
        assert saved_path == self.cfg.model_path

    def test_load_model_success(self):
        """Test successful model loading."""
        # First save a model
        self.session_io.save_model(self.mock_model, self.cfg, session_tracker=None)

        # Create new model instance with different initial values
        new_model = MockFixMatch(it_value=50)  # Different it value
        original_it = new_model.it

        # Load model
        success = self.session_io.load_model(new_model, self.cfg)

        # Verify loading was successful
        assert success
        assert new_model.it == self.mock_model.it  # Should be loaded value
        assert new_model.it != original_it  # Should be different from original

    def test_load_model_with_normalisation_update(self):
        """Test model loading with normalisation method update."""
        # Set different normalisation in model vs config
        self.mock_model.last_normalisation_method = NormalisationMethod.LOG

        # Update config to match model for saving
        save_cfg = DotMap(self.cfg)
        save_cfg.normalisation.normalisation_method = NormalisationMethod.LOG

        # Save model
        self.session_io.save_model(self.mock_model, save_cfg, session_tracker=None)

        # Create config with different normalisation for loading
        test_cfg = DotMap(self.cfg)
        test_cfg.normalisation.normalisation_method = NormalisationMethod.CONVERSION_ONLY

        # Load model
        new_model = MockFixMatch()
        success = self.session_io.load_model(new_model, test_cfg)

        # Verify normalisation was updated from model
        assert success
        assert test_cfg.normalisation.normalisation_method == NormalisationMethod.LOG
        assert new_model.last_normalisation_method == NormalisationMethod.LOG

    def test_load_model_nonexistent_file(self):
        """Test loading from nonexistent file."""
        self.cfg.model_path = str(self.temp_dir / "nonexistent.pth")

        success = self.session_io.load_model(self.mock_model, self.cfg)

        assert not success


class TestLabeledDataInCheckpoint:
    """Test saving and loading labeled data embedded in model checkpoints."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.session_io = SessionIOHandler(str(self.temp_dir))
        self.session_tracker = SessionTracker("test_session")
        self.mock_model = MockFixMatch()
        from anomaly_match.utils.get_default_cfg import get_default_cfg

        self.cfg = get_default_cfg()

    def teardown_method(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_save_and_load_labeled_data_roundtrip(self):
        """Save model with labels, load checkpoint, verify labels match."""
        self.session_tracker.start_new_session_iteration()
        self.session_tracker.add_labeled_sample("img_001.fits", "anomaly")
        self.session_tracker.add_labeled_sample("img_002.fits", "normal")
        self.session_tracker.add_labeled_sample("img_003.fits", "anomaly")

        model_path = self.session_io.save_model(self.mock_model, self.cfg, self.session_tracker)

        checkpoint = load_checkpoint(model_path)
        recovered = SessionIOHandler.get_labeled_data_from_checkpoint(checkpoint)

        assert recovered is not None
        assert len(recovered) == 3
        assert list(recovered.columns) == ["id", "label", "iteration"]
        assert set(recovered["id"]) == {"img_001.fits", "img_002.fits", "img_003.fits"}
        anomalies = recovered[recovered["label"] == "anomaly"]
        assert len(anomalies) == 2

    def test_checkpoint_without_labels_raises(self):
        """Extracting labels from checkpoint saved without session_tracker raises."""
        self.cfg.model_path = str(self.temp_dir / "model_no_labels.safetensors")
        model_path = self.session_io.save_model(self.mock_model, self.cfg, session_tracker=None)

        checkpoint = load_checkpoint(model_path)
        with pytest.raises(ValueError, match="does not contain labeled data"):
            SessionIOHandler.get_labeled_data_from_checkpoint(checkpoint)

    def test_save_with_empty_labels_warns(self):
        """Saving model with session_tracker but no labeled data warns."""
        self.session_tracker.start_new_session_iteration()

        model_path = self.session_io.save_model(self.mock_model, self.cfg, self.session_tracker)

        checkpoint = load_checkpoint(model_path)
        assert "labeled_data_csv" not in checkpoint


class TestStoredModelLoading:
    """Regression tests pinning the shared checkpoint to a known architecture.

    The checkpoint is now written by the same ``get_net_builder`` these tests
    read it back with, so a round trip on its own can no longer fail. The fixed
    point is ``tests/test_data/test_model_state_dict.json``, which records every
    ``state_dict`` key and shape from the architecture this suite was written
    against — a backbone swap (the timm migration being the real example) has to
    fail here rather than pass silently. Refresh it deliberately with
    ``python tests/test_data/generate_test_model.py --update-manifest``.
    """

    def test_stored_model_matches_pinned_state_dict(self, test_model_path):
        """The generated checkpoint still has the pinned keys and tensor shapes."""
        pinned = json.loads(STATE_DICT_MANIFEST_PATH.read_text(encoding="utf-8"))
        actual = state_dict_manifest(test_model_path)

        for part in ("train_model", "eval_model"):
            assert actual[part].keys() == pinned[part].keys(), (
                f"{part} state_dict keys changed — architecture differs from the pinned "
                f"manifest. Added: {sorted(actual[part].keys() - pinned[part].keys())}; "
                f"removed: {sorted(pinned[part].keys() - actual[part].keys())}"
            )
            mismatched = {
                key: (pinned[part][key], shape)
                for key, shape in actual[part].items()
                if shape != pinned[part][key]
            }
            assert not mismatched, f"{part} tensor shapes changed (pinned, actual): {mismatched}"

    def test_stored_model_has_expected_keys(self, test_model_path):
        """Verify the stored checkpoint contains expected top-level keys."""
        checkpoint = load_checkpoint(test_model_path)

        assert "eval_model" in checkpoint, (
            f"Checkpoint missing 'eval_model' key. Found: {list(checkpoint.keys())}"
        )
        assert "train_model" in checkpoint

    def test_stored_model_loads_into_efficientnet_lite0(self, test_model_path):
        """Verify stored model state_dict is compatible with the current architecture."""
        checkpoint = load_checkpoint(test_model_path)

        net_builder = get_net_builder("efficientnet-lite0", pretrained=False, in_channels=3)
        model = net_builder(num_classes=2, in_channels=3)

        # This will raise RuntimeError if keys don't match (the exact regression
        # that would occur if the model was saved with a different architecture)
        model.load_state_dict(checkpoint["eval_model"])
        model.load_state_dict(checkpoint["train_model"])
