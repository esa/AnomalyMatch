#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.

import os
import tempfile
from unittest.mock import Mock

import pandas as pd
import pytest
import torch

from anomaly_match.data_io.checkpoint_io import load_checkpoint
from anomaly_match.data_io.SessionIOHandler import SessionIOHandler
from anomaly_match.pipeline.SessionTracker import SessionTracker


class TestRunAndLabelSavingMigration:
    """Test the migration of save_run and save_labels functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def session_io(self, temp_dir):
        """Create a SessionIOHandler instance."""
        return SessionIOHandler(base_save_path=temp_dir)

    @pytest.fixture
    def session_tracker(self):
        """Create a SessionTracker instance."""
        return SessionTracker("test_session")

    @pytest.fixture
    def mock_model(self):
        """Create a mock FixMatch model."""
        model = Mock()
        model.train_model = Mock()
        model.eval_model = Mock()
        model.optimizer = Mock()
        model.scheduler = Mock()
        model.it = 100
        model.total_it = 100
        model.best_eval_acc = 0.95
        model.best_it = 90
        model.last_normalisation_method = "min_max"

        # Create realistic state_dict returns using actual tensors
        model.train_model.state_dict.return_value = {"layer1.weight": torch.randn(10, 5)}
        model.eval_model.state_dict.return_value = {"layer1.weight": torch.randn(10, 5)}
        model.optimizer.state_dict.return_value = {"state": {}, "param_groups": [{"lr": 0.001}]}
        model.scheduler.state_dict.return_value = {"last_epoch": 100}

        # Handle distributed training case by not having .module attribute
        model.train_model.module = None
        model.eval_model.module = None

        return model

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration."""
        config = Mock()
        config.normalisation_method = "min_max"
        config.model_path = "test_model.safetensors"
        # Explicitly set fitsbolt_cfg to None to avoid pickling issues with Mock
        config.fitsbolt_cfg = None
        return config

    def test_save_run_basic(self, session_io, mock_model, temp_dir):
        """Test basic save_run functionality."""
        save_name = "test_model.safetensors"
        save_path = temp_dir

        result_path = session_io.save_run(mock_model, save_name, save_path)

        # Check that the model was saved (save_checkpoint forces .safetensors extension)
        expected_path = os.path.join(save_path, save_name)
        assert result_path == expected_path
        assert os.path.exists(expected_path)

        # Verify the saved model can be loaded
        checkpoint = load_checkpoint(expected_path)
        assert "train_model" in checkpoint
        assert "eval_model" in checkpoint
        assert "optimizer" in checkpoint
        assert "scheduler" in checkpoint
        assert checkpoint["it"] == 100
        assert checkpoint["total_it"] == 100

    def test_save_run_with_session_tracker(self, session_io, mock_model, session_tracker, temp_dir):
        """Test save_run with session tracker integration."""
        save_name = "test_model.safetensors"
        save_path = temp_dir

        # Start a session iteration
        session_tracker.start_new_session_iteration()

        result_path = session_io.save_run(
            mock_model, save_name, save_path, session_tracker=session_tracker
        )

        # Check that session tracker was updated
        assert len(session_tracker.session_iterations) == 1
        assert session_tracker.session_iterations[0].model_state_path == result_path

    def test_save_run_with_config(self, session_io, mock_model, mock_config, temp_dir):
        """Test save_run with configuration saving."""
        save_name = "test_model.safetensors"
        save_path = temp_dir

        # Mock the config saving function
        with pytest.MonkeyPatch().context() as m:
            mock_save_config = Mock()
            m.setattr("anomaly_match.data_io.save_config.save_config_toml", mock_save_config)

            session_io.save_run(mock_model, save_name, save_path, cfg=mock_config)

            # Verify config saving was attempted
            mock_save_config.assert_called_once()

    def test_save_labels_to_output_dir(self, session_io, temp_dir):
        """Test save_labels_to_output_dir functionality."""
        # Create test labeled data
        labeled_data = pd.DataFrame(
            {
                "filename": ["img1.jpg", "img2.jpg", "img3.jpg"],
                "label": ["normal", "anomaly", "normal"],
            }
        )

        output_dir = os.path.join(temp_dir, "output")

        result_path = session_io.save_labels_to_output_dir(labeled_data, output_dir)

        # Check that the labels were saved
        expected_path = os.path.join(output_dir, "labeled_data.csv")
        assert result_path == expected_path
        assert os.path.exists(expected_path)

        # Verify the saved data
        loaded_data = pd.read_csv(expected_path)
        pd.testing.assert_frame_equal(loaded_data, labeled_data)

    def test_save_labels_with_session_tracker(self, session_io, session_tracker, temp_dir):
        """Test save_labels_to_output_dir with session tracker integration."""
        # Create test labeled data
        labeled_data = pd.DataFrame(
            {"filename": ["img1.jpg", "img2.jpg"], "label": ["normal", "anomaly"]}
        )

        output_dir = os.path.join(temp_dir, "output")

        session_io.save_labels_to_output_dir(
            labeled_data, output_dir, session_tracker=session_tracker
        )

        # Check that session tracker was updated (with iteration column added)
        expected_df = labeled_data.copy()
        expected_df["iteration"] = -1  # Default iteration for initial data
        pd.testing.assert_frame_equal(session_tracker.labeled_data_df, expected_df)

    def test_session_tracker_update_labeled_data(self, session_tracker):
        """Test SessionTracker.update_labeled_data method."""
        labeled_data = pd.DataFrame(
            {"filename": ["img1.jpg", "img2.jpg"], "label": ["normal", "anomaly"]}
        )

        session_tracker.update_labeled_data(labeled_data)

        # Check that labeled data was updated (with iteration column added)
        expected_df = labeled_data.copy()
        expected_df["iteration"] = -1  # Default iteration for initial data
        pd.testing.assert_frame_equal(session_tracker.labeled_data_df, expected_df)

    def test_integration_training_run_flow(
        self, session_io, session_tracker, mock_model, mock_config, temp_dir
    ):
        """Test the complete integration flow for training run saving."""
        save_name = "final_model.safetensors"
        save_path = temp_dir

        # Simulate a training session
        session_tracker.start_new_session_iteration()
        session_tracker.update_model_iteration(loss=0.1)

        # Save the training run
        model_path = session_io.save_run(
            mock_model, save_name, save_path, cfg=mock_config, session_tracker=session_tracker
        )

        # Verify everything was saved correctly
        assert os.path.exists(model_path)
        assert session_tracker.session_iterations[0].model_state_path == model_path

        # Verify model checkpoint structure
        checkpoint = load_checkpoint(model_path)
        assert all(key in checkpoint for key in ["train_model", "eval_model", "optimizer", "it"])

    def test_integration_label_saving_flow(self, session_io, session_tracker, temp_dir):
        """Test the complete integration flow for label saving."""
        # Create labeled data
        labeled_data = pd.DataFrame(
            {
                "filename": ["sample1.jpg", "sample2.jpg", "sample3.jpg"],
                "label": ["normal", "anomaly", "normal"],
            }
        )

        output_dir = os.path.join(temp_dir, "labels_output")

        # Save labels through SessionIOHandler
        csv_path = session_io.save_labels_to_output_dir(
            labeled_data, output_dir, session_tracker=session_tracker
        )

        # Verify everything was saved correctly (with iteration column added)
        expected_df = labeled_data.copy()
        expected_df["iteration"] = -1  # Default iteration for initial data
        pd.testing.assert_frame_equal(session_tracker.labeled_data_df, expected_df)

        # Verify CSV file content
        loaded_data = pd.read_csv(csv_path)
        pd.testing.assert_frame_equal(loaded_data, labeled_data)
