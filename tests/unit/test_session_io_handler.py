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
from unittest.mock import patch

import pandas as pd
import pytest

from anomaly_match.data_io.checkpoint_io import load_checkpoint
from anomaly_match.data_io.SessionIOHandler import SessionIOHandler, print_session
from anomaly_match.pipeline.SessionTracker import SessionTracker


class TestSessionIOHandler:
    """Test cases for SessionIOHandler class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.base_save_path = str(Path(self.temp_dir) / "test_sessions")

        # Create SessionIOHandler with custom base path
        self.io_handler = SessionIOHandler(base_save_path=self.base_save_path)

        # Create a real SessionTracker for testing
        self.session_tracker = SessionTracker("test_session")
        self.session_tracker.start_new_session_iteration()
        self.session_tracker.update_model_iteration(0.5)
        self.session_tracker.add_labeled_sample("test1.jpg", "anomaly")
        self.session_tracker.add_labeled_sample("test2.jpg", "normal")
        self.session_tracker.update_test_performance({"AUROC": 0.85, "AUPRC": 0.78})

    def teardown_method(self):
        """Clean up after each test method."""
        shutil.rmtree(self.temp_dir)

    def test_init_default_path(self):
        """Test SessionIOHandler initialization with default path."""
        handler = SessionIOHandler()
        expected_path = Path("anomaly_match_results/sessions")
        assert handler.base_save_path == expected_path
        # Clean up the created directory
        shutil.rmtree("anomaly_match_results", ignore_errors=True)

    def test_init_custom_path(self):
        """Test SessionIOHandler initialization with custom path."""
        custom_path = str(Path(self.temp_dir) / "custom_sessions")
        handler = SessionIOHandler(base_save_path=custom_path)
        assert handler.base_save_path == Path(custom_path)
        assert handler.base_save_path.exists()

    def test_get_session_save_path(self):
        """Test getting session save path."""
        save_path = self.io_handler.get_session_save_path(self.session_tracker)

        # Should be base_path / session_name_timestamp
        expected_name_pattern = f"{self.session_tracker.session_name}_"
        assert save_path.name.startswith(expected_name_pattern)
        assert save_path.parent == self.io_handler.base_save_path

    def test_save_session_complete(self):
        """Test saving a complete session."""
        save_path = self.io_handler.save_session(self.session_tracker)

        # Check that session directory was created
        assert save_path.exists()
        assert save_path.is_dir()

        # Check that all expected files were created
        assert (save_path / "session_metadata.json").exists()
        assert (save_path / "labeled_data.csv").exists()

        # Verify session metadata content
        with open(save_path / "session_metadata.json", "r") as f:
            metadata = json.load(f)

        assert "session_info" in metadata
        assert "all_iterations" in metadata
        assert metadata["session_info"]["session_name"] == "test_session"

        # Verify labeled data CSV
        df = pd.read_csv(save_path / "labeled_data.csv")
        assert len(df) == 2
        assert "test1.jpg" in df["filename"].values
        assert "test2.jpg" in df["filename"].values

    def test_save_session_custom_path(self):
        """Test saving session to custom path."""
        custom_path = Path(self.temp_dir) / "custom_session"
        save_path = self.io_handler.save_session(self.session_tracker, save_path=custom_path)

        assert save_path == custom_path
        assert save_path.exists()
        assert (save_path / "session_metadata.json").exists()

    def test_save_model_checkpoint(self):
        """Test saving model checkpoint."""
        import torch

        model_state = {
            "train_model": {"layer.weight": torch.randn(2, 2)},
            "eval_model": {"layer.weight": torch.randn(2, 2)},
            "optimizer": None,
            "scheduler": None,
            "it": 10,
            "total_it": 10,
            "best_eval_acc": None,
            "best_it": None,
            "num_channels": 3,
            "net": "efficientnet-lite0",
            "normalisation_method": None,
            "last_normalisation_method": None,
            "fitsbolt_cfg": None,
        }

        checkpoint_path = self.io_handler.save_model_checkpoint(model_state, self.session_tracker)

        # Check checkpoint was saved
        assert Path(checkpoint_path).exists()
        assert "checkpoints" in checkpoint_path
        assert checkpoint_path.endswith(".safetensors")

        # Verify checkpoint content
        loaded_state = load_checkpoint(checkpoint_path)
        assert loaded_state["it"] == 10
        assert "train_model" in loaded_state

        # Verify that session tracker was updated - check the last iteration
        assert len(self.session_tracker.session_iterations) > 0
        last_iter = self.session_tracker.session_iterations[-1]
        assert last_iter.model_state_path == checkpoint_path

    def test_save_model_checkpoint_custom_name(self):
        """Test saving model checkpoint with custom name."""
        import torch

        model_state = {
            "train_model": {"layer.weight": torch.randn(2, 2)},
            "eval_model": {"layer.weight": torch.randn(2, 2)},
            "optimizer": None,
            "scheduler": None,
            "it": 0,
            "total_it": 0,
            "best_eval_acc": None,
            "best_it": None,
            "num_channels": 3,
            "net": "efficientnet-lite0",
            "normalisation_method": None,
            "last_normalisation_method": None,
            "fitsbolt_cfg": None,
        }
        custom_name = "custom_checkpoint.safetensors"

        checkpoint_path = self.io_handler.save_model_checkpoint(
            model_state, self.session_tracker, checkpoint_name=custom_name
        )

        assert checkpoint_path.endswith(custom_name)
        assert Path(checkpoint_path).exists()

    def test_load_session_complete_cycle(self):
        """Test complete save/load cycle."""
        # First save a session
        original_save_path = self.io_handler.save_session(self.session_tracker)

        # Then load it back
        loaded_tracker = self.io_handler.load_session(original_save_path)

        # Verify loaded session matches original
        assert loaded_tracker.session_name == self.session_tracker.session_name
        assert loaded_tracker.total_model_iterations == self.session_tracker.total_model_iterations

        # Check labeled data was preserved
        original_df = self.session_tracker.get_labeled_data_df()
        loaded_df = loaded_tracker.get_labeled_data_df()
        assert len(loaded_df) == len(original_df)
        assert loaded_df["filename"].tolist() == original_df["filename"].tolist()

    def test_load_session_nonexistent_path(self):
        """Test loading session from nonexistent path."""
        nonexistent_path = Path(self.temp_dir) / "nonexistent"

        with pytest.raises(FileNotFoundError):
            self.io_handler.load_session(nonexistent_path)

    def test_load_session_missing_metadata(self):
        """Test loading session with missing metadata file."""
        # Create directory without metadata
        session_dir = Path(self.temp_dir) / "invalid_session"
        session_dir.mkdir()

        with pytest.raises(FileNotFoundError):
            self.io_handler.load_session(session_dir)

    def test_list_sessions_empty(self):
        """Test listing sessions when none exist."""
        sessions = self.io_handler.list_sessions()
        assert sessions == []

    def test_list_sessions_with_data(self):
        """Test listing sessions with existing data."""
        # Save multiple sessions
        tracker1 = SessionTracker("session1")
        tracker2 = SessionTracker("session2")

        path1 = self.io_handler.save_session(tracker1)
        path2 = self.io_handler.save_session(tracker2)

        sessions = self.io_handler.list_sessions()
        assert len(sessions) == 2
        assert path1 in sessions
        assert path2 in sessions

    def test_get_session_summary(self):
        """Test getting session summary."""
        save_path = self.io_handler.save_session(self.session_tracker)
        summary = self.io_handler.get_session_summary(save_path)

        assert "session_name" in summary
        assert summary["session_name"] == "test_session"
        assert "total_model_iterations" in summary

    def test_get_session_summary_invalid_path(self):
        """Test getting session summary for invalid path."""
        invalid_path = Path(self.temp_dir) / "invalid"
        summary = self.io_handler.get_session_summary(invalid_path)

        assert "error" in summary
        assert "Session metadata not found" in summary["error"]


class TestPrintSession:
    """Test cases for print_session function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.base_save_path = str(Path(self.temp_dir) / "test_sessions")

        # Create a test session
        session_tracker = SessionTracker("test_session")
        session_tracker.start_new_session_iteration()
        session_tracker.update_model_iteration(0.8)
        session_tracker.update_model_iteration(0.6)
        session_tracker.add_labeled_sample("img1.jpg", "anomaly")
        session_tracker.add_labeled_sample("img2.jpg", "normal")
        session_tracker.update_test_performance({"AUROC": 0.92, "AUPRC": 0.88})
        session_tracker.update_model_state_path("models/final_model.safetensors")

        # Start second iteration
        session_tracker.start_new_session_iteration()
        session_tracker.update_model_iteration(0.4)
        session_tracker.add_labeled_sample("img3.jpg", "anomaly")
        session_tracker.update_test_performance({"AUROC": 0.95, "AUPRC": 0.91})

        # Save session
        io_handler = SessionIOHandler(self.base_save_path)
        self.session_path = io_handler.save_session(session_tracker)

    def teardown_method(self):
        """Clean up after tests."""
        shutil.rmtree(self.temp_dir)

    @patch("builtins.print")
    def test_print_session_valid_path(self, mock_print):
        """Test print_session with valid session path."""
        print_session(str(self.session_path))

        # Check that print was called
        assert mock_print.called

        # Get all print calls - some calls might be print() with no args
        print_calls = []
        for call in mock_print.call_args_list:
            if call[0]:  # If there are positional arguments
                print_calls.append(str(call[0][0]))
            else:  # Empty print() call
                print_calls.append("")

        output = "\n".join(print_calls)

        # Check key information is present
        assert "test_session" in output
        assert "ANOMALY MATCH SESSION REPORT" in output
        assert "TRAINING SUMMARY" in output
        assert "LABELING SUMMARY" in output

    @patch("builtins.print")
    def test_print_session_nonexistent_path(self, mock_print):
        """Test print_session with non-existent path."""
        print_session("/nonexistent/path")

        # Should print error message
        assert mock_print.called
        all_calls = [call[0][0] for call in mock_print.call_args_list]
        output = "\n".join(all_calls)
        assert "Error: Session path does not exist" in output

    @patch("builtins.print")
    def test_print_session_path_object(self, mock_print):
        """Test print_session with Path object."""
        print_session(self.session_path)

        # Should work with Path objects
        assert mock_print.called
        print_calls = []
        for call in mock_print.call_args_list:
            if call[0]:  # If there are positional arguments
                print_calls.append(str(call[0][0]))
            else:  # Empty print() call
                print_calls.append("")

        output = "\n".join(print_calls)
        assert "test_session" in output

    @patch("builtins.print")
    def test_print_session_invalid_metadata(self, mock_print):
        """Test print_session with corrupted metadata."""
        # Create directory with invalid metadata
        invalid_dir = Path(self.temp_dir) / "invalid_session"
        invalid_dir.mkdir()

        # Create invalid JSON file
        metadata_file = invalid_dir / "session_metadata.json"
        metadata_file.write_text("invalid json content")

        print_session(str(invalid_dir))

        # Should handle error gracefully
        assert mock_print.called
        print_calls = []
        for call in mock_print.call_args_list:
            if call[0]:  # If there are positional arguments
                print_calls.append(str(call[0][0]))
            else:  # Empty print() call
                print_calls.append("")

        output = "\n".join(print_calls)
        assert "Error loading session" in output


class TestSessionIOHandlerIntegration:
    """Integration tests for SessionIOHandler."""

    def setup_method(self):
        """Set up integration test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.base_save_path = str(Path(self.temp_dir) / "integration_sessions")
        self.io_handler = SessionIOHandler(self.base_save_path)

    def teardown_method(self):
        """Clean up integration test environment."""
        shutil.rmtree(self.temp_dir)

    def test_full_workflow_integration(self):
        """Test complete workflow: create session, save, load, verify."""
        # Create a comprehensive session
        tracker = SessionTracker("integration_test")

        # First iteration
        tracker.start_new_session_iteration()
        tracker.update_model_iteration(0.9)
        tracker.update_model_iteration(0.7)
        tracker.add_labeled_sample("img1.jpg", "anomaly")
        tracker.add_labeled_sample("img2.jpg", "normal")
        tracker.add_labeled_sample("img3.jpg", "normal")
        tracker.update_test_performance({"AUROC": 0.88, "AUPRC": 0.82})

        # Second iteration
        tracker.start_new_session_iteration()
        tracker.update_model_iteration(0.5)
        tracker.add_labeled_sample("img4.jpg", "anomaly")
        tracker.update_test_performance({"AUROC": 0.93, "AUPRC": 0.89})
        tracker.update_model_state_path("models/best_model.safetensors")

        # Save session
        saved_path = self.io_handler.save_session(tracker)

        # Save model checkpoint
        import torch

        model_state = {
            "train_model": {"layer.weight": torch.randn(2, 2)},
            "eval_model": {"layer.weight": torch.randn(2, 2)},
            "optimizer": None,
            "scheduler": None,
            "it": 50,
            "total_it": 50,
            "best_eval_acc": None,
            "best_it": None,
            "num_channels": 3,
            "net": "efficientnet-lite0",
            "normalisation_method": None,
            "last_normalisation_method": None,
            "fitsbolt_cfg": None,
        }
        checkpoint_path = self.io_handler.save_model_checkpoint(model_state, tracker)

        # Load session back
        loaded_tracker = self.io_handler.load_session(saved_path)

        # Comprehensive verification
        assert loaded_tracker.session_name == "integration_test"
        assert loaded_tracker.total_model_iterations == tracker.total_model_iterations
        assert len(loaded_tracker.get_labeled_data_df()) == 4
        assert len(loaded_tracker.session_iterations) == 2

        # Check model checkpoint exists and can be loaded
        assert Path(checkpoint_path).exists()
        loaded_model = load_checkpoint(checkpoint_path)
        assert loaded_model["it"] == 50

    def test_multiple_sessions_management(self):
        """Test managing multiple sessions."""
        # Create multiple sessions
        sessions = []
        for i in range(3):
            tracker = SessionTracker(f"session_{i}")
            tracker.start_new_session_iteration()
            tracker.update_model_iteration(0.5 + i * 0.1)
            tracker.add_labeled_sample(f"img_{i}.jpg", "anomaly" if i % 2 == 0 else "normal")

            saved_path = self.io_handler.save_session(tracker)
            sessions.append(saved_path)

        # List all sessions
        all_sessions = self.io_handler.list_sessions()
        assert len(all_sessions) == 3

        # Verify each session can be loaded
        for session_path in all_sessions:
            summary = self.io_handler.get_session_summary(session_path)
            assert "session_name" in summary
            assert summary["session_name"].startswith("session_")

            # Load full session
            loaded_tracker = self.io_handler.load_session(session_path)
            assert loaded_tracker.session_name.startswith("session_")
