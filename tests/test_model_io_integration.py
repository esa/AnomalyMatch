#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.

import tempfile
import shutil
from pathlib import Path
import torch
import torch.nn as nn
from dotmap import DotMap

from anomaly_match.data_io.SessionIOHandler import SessionIOHandler
from anomaly_match.pipeline.SessionTracker import SessionTracker
from anomaly_match.image_processing.NormalisationMethod import NormalisationMethod


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
        self.cfg.model_path = str(self.temp_dir / "test_model.pth")

    def teardown_method(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_save_model_with_session_tracker(self):
        """Test saving model with session tracker."""
        # Start a session iteration to ensure session_iterations is not empty
        self.session_tracker.start_new_session_iteration()

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
        save_cfg.normalisation_method = NormalisationMethod.LOG

        # Save model
        self.session_io.save_model(self.mock_model, save_cfg, session_tracker=None)

        # Create config with different normalisation for loading
        test_cfg = DotMap(self.cfg)
        test_cfg.normalisation_method = NormalisationMethod.CONVERSION_ONLY

        # Load model
        new_model = MockFixMatch()
        success = self.session_io.load_model(new_model, test_cfg)

        # Verify normalisation was updated from model
        assert success
        assert test_cfg.normalisation_method == NormalisationMethod.LOG
        assert new_model.last_normalisation_method == NormalisationMethod.LOG

    def test_load_model_nonexistent_file(self):
        """Test loading from nonexistent file."""
        self.cfg.model_path = str(self.temp_dir / "nonexistent.pth")

        success = self.session_io.load_model(self.mock_model, self.cfg)

        assert not success

    def test_load_model_checkpoint(self):
        """Test loading model checkpoint."""
        # Create and save a checkpoint
        model_state = {
            "train_model_state_dict": self.mock_model.train_model.state_dict(),
            "eval_model_state_dict": self.mock_model.eval_model.state_dict(),
            "total_it": self.mock_model.total_it,
        }

        checkpoint_path = self.session_io.save_model_checkpoint(
            model_state, self.session_tracker, "test_checkpoint.pkl"
        )

        # Load checkpoint
        loaded_checkpoint = self.session_io.load_model_checkpoint(checkpoint_path)

        # Verify checkpoint was loaded
        assert loaded_checkpoint is not None
        assert "train_model_state_dict" in loaded_checkpoint
        assert "total_it" in loaded_checkpoint
        assert loaded_checkpoint["total_it"] == self.mock_model.total_it

    def test_load_model_checkpoint_nonexistent(self):
        """Test loading nonexistent checkpoint."""
        checkpoint = self.session_io.load_model_checkpoint(str(self.temp_dir / "nonexistent.pkl"))

        assert checkpoint is None
