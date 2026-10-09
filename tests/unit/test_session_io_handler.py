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
        assert "test1.jpg" in df["id"].values
        assert "test2.jpg" in df["id"].values

    def test_save_session_custom_path(self):
        """Test saving session to custom path."""
        custom_path = Path(self.temp_dir) / "custom_session"
        save_path = self.io_handler.save_session(self.session_tracker, save_path=custom_path)

        assert save_path == custom_path
        assert save_path.exists()
        assert (save_path / "session_metadata.json").exists()

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
        assert loaded_df["id"].tolist() == original_df["id"].tolist()

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

    def test_filter_labels_csv_to_ids_drops_unknown_ids(self):
        """Filter keeps only rows whose id is in the valid set."""
        csv_path = Path(self.temp_dir) / "labels.csv"
        pd.DataFrame(
            {"id": ["a", "b", "c", "d"], "label": ["anomaly", "normal", "anomaly", "normal"]}
        ).to_csv(csv_path, index=False)

        kept, dropped = SessionIOHandler.filter_labels_csv_to_ids(str(csv_path), {"a", "c"})

        assert (kept, dropped) == (2, 2)
        df = pd.read_csv(csv_path)
        assert set(df["id"]) == {"a", "c"}

    def test_filter_labels_csv_to_ids_keeps_all_when_all_known(self):
        """No rows dropped when every id is valid."""
        csv_path = Path(self.temp_dir) / "labels.csv"
        pd.DataFrame({"id": ["a", "b"], "label": ["anomaly", "normal"]}).to_csv(
            csv_path, index=False
        )

        kept, dropped = SessionIOHandler.filter_labels_csv_to_ids(str(csv_path), {"a", "b"})

        assert (kept, dropped) == (2, 0)

    def test_filter_labels_csv_to_ids_handles_integer_ids(self):
        """IDs coerced to str so int-valued Cutana SourceIDs match string cache ids."""
        csv_path = Path(self.temp_dir) / "labels.csv"
        pd.DataFrame({"id": [123, 456, 789], "label": ["anomaly", "normal", "anomaly"]}).to_csv(
            csv_path, index=False
        )

        kept, dropped = SessionIOHandler.filter_labels_csv_to_ids(str(csv_path), {"123", "789"})

        assert (kept, dropped) == (2, 1)
        df = pd.read_csv(csv_path)
        assert set(df["id"].astype(str)) == {"123", "789"}

    def test_filter_labels_csv_to_ids_survives_blank_id_row(self):
        """One blank id must not float64-ify the column and drop every label.

        Regression for the second half of #556: this runs in place right
        before training, so a post-hoc ``astype(str)`` turning 123 into
        "123.0" silently wiped the whole label set and trained on nothing.
        """
        # Written literally: a DataFrame with a None id is already float64, so
        # it would serialise "123.0" and test the wrong thing.  This is what
        # merge_gallery_labels actually puts on disk.
        csv_path = Path(self.temp_dir) / "labels.csv"
        csv_path.write_text("id,label\n123,anomaly\n456,normal\n,normal\n789,anomaly\n")

        kept, dropped = SessionIOHandler.filter_labels_csv_to_ids(str(csv_path), {"123", "789"})

        # The blank-id row is unmatchable and drops out with 456.
        assert (kept, dropped) == (2, 2)
        df = pd.read_csv(csv_path, dtype={"id": str})
        assert set(df["id"]) == {"123", "789"}

    def test_filter_labels_csv_to_ids_preserves_zero_padded_ids(self):
        """Zero-padded ids must match and survive the in-place rewrite as-is."""
        csv_path = Path(self.temp_dir) / "labels.csv"
        pd.DataFrame({"id": ["007", "042"], "label": ["anomaly", "normal"]}).to_csv(
            csv_path, index=False
        )

        kept, dropped = SessionIOHandler.filter_labels_csv_to_ids(str(csv_path), {"007"})

        assert (kept, dropped) == (1, 1)
        df = pd.read_csv(csv_path, dtype={"id": str})
        assert list(df["id"]) == ["007"]

    @pytest.mark.parametrize("blank_id_row", [False, True], ids=["int_ids", "float_ids"])
    def test_merge_gallery_labels_overwrites_numeric_ids(self, blank_id_row):
        """Relabelling a numeric-id source overwrites its row instead of adding one.

        Regression for #556: ``read_csv`` infers int64 for numeric catalogue
        ids while the gallery hands back str, so the merge saw 51 and "51" as
        different keys and grew the CSV by one row per relabelled source.

        The ``float_ids`` case is the same bug reached by a different
        inference: one blank id anywhere in the file makes pandas type the
        whole column ``float64``, where a post-hoc ``astype(str)`` would
        produce ``"51.0"`` and match nothing.

        Args:
            blank_id_row: Append a row with an empty id, so pandas infers
                ``float64`` for the id column instead of ``int64``.
        """
        src = Path(self.temp_dir) / "labelled_data.csv"
        frame = pd.DataFrame(
            {"id": list(range(1, 501)), "label": ["anomaly"] * 50 + ["normal"] * 450}
        )
        if blank_id_row:
            frame = pd.concat([frame, pd.DataFrame([{"id": "", "label": "normal"}])])
        frame.to_csv(src, index=False)
        original = pd.read_csv(src)
        expected_rows = 501 if blank_id_row else 500

        out = Path(self.temp_dir) / "merged.csv"
        self.io_handler.merge_gallery_labels(
            str(src), {str(i): "anomaly" for i in range(51, 56)}, str(out)
        )

        merged = pd.read_csv(out, dtype={"id": str})
        assert len(merged) == expected_rows, "relabelling must not grow the CSV"
        assert not merged.loc[merged["id"].notna(), "id"].duplicated().any()
        # The 5 sources moved out of 'normal' rather than being added twice.
        expected_counts = {"normal": 446 if blank_id_row else 445, "anomaly": 55}
        assert merged["label"].value_counts().to_dict() == expected_counts
        relabelled = merged["id"].isin([str(i) for i in range(51, 56)])
        assert relabelled.sum() == 5, "ids must round-trip as written, not as floats"
        assert set(merged.loc[relabelled, "label"]) == {"anomaly"}
        # The user's input CSV is read-only — only the session copy is written.
        pd.testing.assert_frame_equal(original, pd.read_csv(src))

    def test_merge_gallery_labels_rejects_csv_without_id_column(self):
        """A labels CSV missing the merge key is a hard error, not a silent replace."""
        src = Path(self.temp_dir) / "labelled_data.csv"
        pd.DataFrame({"filename": ["a.jpg"], "label": ["normal"]}).to_csv(src, index=False)

        out = Path(self.temp_dir) / "merged.csv"
        with pytest.raises(ValueError, match="no 'id' column"):
            self.io_handler.merge_gallery_labels(str(src), {"b.jpg": "anomaly"}, str(out))

    def test_merge_gallery_labels_overwrites_string_ids(self):
        """The image-folder path (filename ids) must keep overwriting correctly."""
        src = Path(self.temp_dir) / "labelled_data.csv"
        pd.DataFrame(
            {"id": ["a.jpg", "b.jpg", "c.jpg"], "label": ["normal", "normal", "anomaly"]}
        ).to_csv(src, index=False)

        out = Path(self.temp_dir) / "merged.csv"
        self.io_handler.merge_gallery_labels(str(src), {"b.jpg": "anomaly"}, str(out))

        merged = pd.read_csv(out)
        assert len(merged) == 3
        assert dict(zip(merged["id"], merged["label"])) == {
            "a.jpg": "normal",
            "b.jpg": "anomaly",
            "c.jpg": "anomaly",
        }

    def test_merge_gallery_labels_removed_drops_numeric_id(self):
        """A 'removed' override must also match an int-typed existing id."""
        src = Path(self.temp_dir) / "labelled_data.csv"
        pd.DataFrame({"id": [10, 20], "label": ["anomaly", "normal"]}).to_csv(src, index=False)

        out = Path(self.temp_dir) / "merged.csv"
        self.io_handler.merge_gallery_labels(str(src), {"10": "removed"}, str(out))

        merged = pd.read_csv(out)
        assert len(merged) == 2
        assert dict(zip(merged["id"].astype(str), merged["label"])) == {
            "10": "removed",
            "20": "normal",
        }


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

        # Load session back
        loaded_tracker = self.io_handler.load_session(saved_path)

        # Comprehensive verification
        assert loaded_tracker.session_name == "integration_test"
        assert loaded_tracker.total_model_iterations == tracker.total_model_iterations
        assert len(loaded_tracker.get_labeled_data_df()) == 4
        assert len(loaded_tracker.session_iterations) == 2

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
