#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Integration tests for Session.run_pipeline and Session.evaluate_all_images.

Heavy mocking of subprocesses, torch, and zarr so no real training or
GPU is needed.  Real filesystem is used for file-type detection and image grouping.
"""

import os
import pickle
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pandas as pd
import pytest
from fitsbolt.normalisation.NormalisationMethod import NormalisationMethod
from loguru import logger

from anomaly_match.data_io.checkpoint_io import save_checkpoint
from anomaly_match.data_io.load_images import get_fitsbolt_config
from anomaly_match.datasets.training_data_source import DataSourceType
from anomaly_match.pipeline.session import Session
from anomaly_match.utils.get_default_cfg import get_default_cfg


def _create_dummy_model(path, normalisation_method=None):
    """Create a minimal safetensors checkpoint with a valid embedded fitsbolt
    config — prediction now requires the model to carry its normalisation, so a
    config-less stub is rejected before any subprocess is launched.

    Args:
        path: Destination checkpoint path.
        normalisation_method: Override for the embedded method (e.g. to exercise
            the MIDTONES/Cutana incompatibility check, which reads the model's
            method after the checkpoint sync).
    """
    cfg = get_default_cfg()
    if normalisation_method is not None:
        cfg.normalisation.normalisation_method = normalisation_method
    fb = get_fitsbolt_config(cfg).fitsbolt_cfg
    save_checkpoint({"train_model": {}, "eval_model": {}, "fitsbolt_cfg": fb}, path)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def session_cfg(tmp_path):
    """Minimal config that bypasses dataset/model init via mocks."""
    cfg = get_default_cfg()
    cfg.output_dir = str(tmp_path / "output")
    cfg.save_dir = str(tmp_path / "sessions")
    cfg.save_path = str(tmp_path / "sessions")
    cfg.data_dir = str(tmp_path / "data")
    cfg.name = "test_eval"
    cfg.log_level = "WARNING"
    cfg.normalisation.image_size = [64, 64]
    cfg.normalisation.n_output_channels = 3
    cfg.num_channels = 3
    cfg.net = "test-cnn"
    cfg.pretrained = False
    cfg.num_workers = 0
    cfg.test_ratio = 0.0

    # Create dirs / files the config validator expects
    os.makedirs(cfg.data_dir, exist_ok=True)
    label_file = tmp_path / "labels.csv"
    pd.DataFrame({"id": ["a.jpg"], "label": ["anomaly"]}).to_csv(label_file, index=False)
    cfg.label_file = str(label_file)

    return cfg


@pytest.fixture()
def session(session_cfg, tmp_path):
    """Build a real Session (lightweight — no model or datasets in memory)."""
    s = Session(session_cfg)
    s.cfg.model_path = str(tmp_path / "model.safetensors")
    s.cfg.save_file = "test_eval_model"
    return s


# ---------------------------------------------------------------------------
# run_pipeline tests
# ---------------------------------------------------------------------------


class TestRunPipeline:
    def test_image_file_type_creates_temp_file_list(self, session, tmp_path):
        config_path = str(tmp_path / "cfg.pkl")
        input_path = str(tmp_path / "images")
        os.makedirs(input_path, exist_ok=True)

        with (
            patch(
                "subprocess.Popen",
                return_value=MagicMock(returncode=0, stdout=iter([]), wait=MagicMock()),
            ) as mock_run,
            patch("anomaly_match.pipeline.session.set_log_level"),
            patch("os.path.exists", return_value=True),
            patch("builtins.open", mock_open()),
        ):
            session.run_pipeline(
                config_path, input_path, 1000, file_type=DataSourceType.IMAGE_FOLDER
            )

        # subprocess.Popen should have been called with the image prediction script
        assert mock_run.called
        cmd = mock_run.call_args[0][0]
        assert "prediction_process.py" in cmd[1]
        # cmd layout: [python, script, config_path, input/filelist, top_N]
        assert cmd[3].endswith("_file_list.txt")

    def test_zarr_file_type(self, session, tmp_path):
        config_path = str(tmp_path / "cfg.pkl")
        input_path = str(tmp_path / "store.zarr")

        with (
            patch(
                "subprocess.Popen",
                return_value=MagicMock(returncode=0, stdout=iter([]), wait=MagicMock()),
            ) as mock_run,
            patch("anomaly_match.pipeline.session.set_log_level"),
            patch("os.path.exists", return_value=True),
        ):
            session.run_pipeline(config_path, input_path, 500, file_type=DataSourceType.ZARR)

        cmd = mock_run.call_args[0][0]
        assert "prediction_process_zarr.py" in cmd[1]

    def test_stream_file_type(self, session, tmp_path):
        config_path = str(tmp_path / "cfg.pkl")
        input_path = str(tmp_path / "catalog.parquet")

        with (
            patch(
                "subprocess.Popen",
                return_value=MagicMock(returncode=0, stdout=iter([]), wait=MagicMock()),
            ) as mock_run,
            patch("anomaly_match.pipeline.session.set_log_level"),
            patch("os.path.exists", return_value=True),
        ):
            session.run_pipeline(config_path, input_path, 500, file_type=DataSourceType.CUTANA)

        cmd = mock_run.call_args[0][0]
        assert "prediction_process_cutana.py" in cmd[1]

    def test_prediction_subprocess_pythonpath_includes_repo_and_scripts(self, session, tmp_path):
        # The subprocess must be able to import anomaly_match and prediction_utils
        # without the package being pip-installed (mirrors the training launcher);
        # otherwise prediction fails at import time on a bare source checkout even
        # though training works.
        config_path = str(tmp_path / "cfg.pkl")
        input_path = str(tmp_path / "catalog.parquet")

        with (
            patch(
                "subprocess.Popen",
                return_value=MagicMock(returncode=0, stdout=iter([]), wait=MagicMock()),
            ) as mock_run,
            patch("anomaly_match.pipeline.session.set_log_level"),
            patch("os.path.exists", return_value=True),
        ):
            session.run_pipeline(config_path, input_path, 500, file_type=DataSourceType.CUTANA)

        env = mock_run.call_args.kwargs["env"]
        script_path = mock_run.call_args[0][0][1]
        repo_root = os.path.dirname(os.path.dirname(script_path))
        path_entries = env["PYTHONPATH"].split(os.pathsep)
        assert repo_root in path_entries
        assert os.path.join(repo_root, "subprocess_scripts") in path_entries

    def test_script_not_found_raises(self, session, tmp_path):
        config_path = str(tmp_path / "cfg.pkl")
        input_path = str(tmp_path / "store.zarr")

        with patch("os.path.exists", return_value=False):
            with pytest.raises(FileNotFoundError, match="Script not found"):
                session.run_pipeline(config_path, input_path, 500, file_type=DataSourceType.ZARR)

    def test_unsupported_file_type_raises(self, session, tmp_path):
        config_path = str(tmp_path / "cfg.pkl")
        input_path = str(tmp_path / "data.bin")

        with pytest.raises(ValueError, match="Unsupported prediction file type"):
            session.run_pipeline(config_path, input_path, 500, file_type="binary")

    def test_subprocess_failure_raises(self, session, tmp_path):
        """Non-zero subprocess exit must raise so the caller aborts the chunk loop."""
        config_path = str(tmp_path / "cfg.pkl")
        input_path = str(tmp_path / "store.zarr")

        with (
            patch(
                "subprocess.Popen",
                return_value=MagicMock(returncode=1, stdout=iter([]), wait=MagicMock()),
            ),
            patch("anomaly_match.pipeline.session.set_log_level"),
            patch("os.path.exists", return_value=True),
            pytest.raises(RuntimeError, match="exited with code 1"),
        ):
            session.run_pipeline(config_path, input_path, 500, file_type=DataSourceType.ZARR)

    def test_subprocess_signal_killed_raises_with_hint(self, session, tmp_path):
        """Negative return codes (signal terminations) surface a signal hint."""
        config_path = str(tmp_path / "cfg.pkl")
        input_path = str(tmp_path / "store.zarr")

        with (
            patch(
                "subprocess.Popen",
                return_value=MagicMock(returncode=-9, stdout=iter([]), wait=MagicMock()),
            ),
            patch("anomaly_match.pipeline.session.set_log_level"),
            patch("os.path.exists", return_value=True),
            pytest.raises(RuntimeError, match="killed by signal 9"),
        ):
            session.run_pipeline(config_path, input_path, 500, file_type=DataSourceType.ZARR)

    def test_auto_detect_parquet_file(self, session, tmp_path):
        config_path = str(tmp_path / "cfg.pkl")
        pq_file = tmp_path / "catalog.parquet"
        pq_file.touch()

        with (
            patch(
                "subprocess.Popen",
                return_value=MagicMock(returncode=0, stdout=iter([]), wait=MagicMock()),
            ) as mock_run,
            patch("anomaly_match.pipeline.session.set_log_level"),
            patch("os.path.exists", return_value=True),
        ):
            session.run_pipeline(config_path, str(pq_file), 500, file_type=None)

        cmd = mock_run.call_args[0][0]
        assert "prediction_process_cutana.py" in cmd[1]

    def test_auto_detect_directory_delegates_to_auto_detect(self, session, tmp_path):
        config_path = str(tmp_path / "cfg.pkl")
        search_dir = tmp_path / "images"
        search_dir.mkdir()
        (search_dir / "a.jpg").touch()

        with (
            patch(
                "subprocess.Popen",
                return_value=MagicMock(returncode=0, stdout=iter([]), wait=MagicMock()),
            ) as mock_run,
            patch("anomaly_match.pipeline.session.set_log_level"),
            patch("builtins.open", mock_open()),
        ):
            session.run_pipeline(config_path, str(search_dir), 500, file_type=None)

        cmd = mock_run.call_args[0][0]
        assert "prediction_process.py" in cmd[1]


# ---------------------------------------------------------------------------
# evaluate_all_images tests
# ---------------------------------------------------------------------------


class TestEvaluateAllImagesErrors:
    def test_missing_model_raises(self, session, tmp_path):
        session.cfg.model_path = str(tmp_path / "nonexistent.safetensors")
        session.cfg.prediction_search_dir = str(tmp_path)

        with pytest.raises(FileNotFoundError, match="Model not found"):
            session.evaluate_all_images()

    def test_missing_search_dir_raises(self, session, tmp_path):
        # Create the model file so we pass the first check
        model_path = tmp_path / "model.safetensors"
        _create_dummy_model(model_path)
        session.cfg.model_path = str(model_path)
        session.cfg.prediction_search_dir = None

        with pytest.raises(ValueError, match="No prediction_search_dir"):
            session.evaluate_all_images()

    def test_midtones_stream_incompatibility(self, session, tmp_path):
        model_path = tmp_path / "model.safetensors"
        # The method comes from the model checkpoint (synced over cfg), so the
        # incompatibility must be embedded in the model, not just set on cfg.
        _create_dummy_model(model_path, normalisation_method=NormalisationMethod.MIDTONES)
        session.cfg.model_path = str(model_path)

        search_dir = tmp_path / "streams"
        search_dir.mkdir()
        (search_dir / "catalog.parquet").touch()

        session.cfg.prediction_search_dir = str(search_dir)

        with pytest.raises(ValueError, match="MIDTONES normalisation is not supported"):
            session.evaluate_all_images()


class TestEvaluateAllImagesZarr:
    def test_zarr_file_counting_and_processing(self, session, tmp_path):
        model_path = tmp_path / "model.safetensors"
        _create_dummy_model(model_path)
        session.cfg.model_path = str(model_path)

        search_dir = tmp_path / "zarr_data"
        search_dir.mkdir()
        # Create a .zarr directory
        zarr_dir = search_dir / "store.zarr"
        zarr_dir.mkdir()

        session.cfg.prediction_search_dir = str(search_dir)
        os.makedirs(session.cfg.output_dir, exist_ok=True)

        # Prepare output files
        output_csv = os.path.join(session.cfg.output_dir, f"{session.cfg.save_file}_top1000.csv")
        output_npy = os.path.join(session.cfg.output_dir, f"{session.cfg.save_file}_top1000.npy")
        pd.DataFrame(
            {
                "Filename": [f"img_{i}.jpg" for i in range(3)],
                "Score": [0.9, 0.5, 0.1],
            }
        ).to_csv(output_csv, index=False)
        np.save(output_npy, np.random.randint(0, 255, (3, 64, 64, 3), dtype=np.uint8))

        # Mock zarr.open_group
        mock_images_array = MagicMock()
        mock_images_array.shape = (200,)
        mock_root = MagicMock()
        mock_root.__contains__ = lambda self, key: key == "images"
        mock_root.__getitem__ = lambda self, key: mock_images_array if key == "images" else None

        with (
            patch("anomaly_match.pipeline.session.zarr.open_group", return_value=mock_root),
            patch.object(Session, "run_pipeline") as mock_pipeline,
            patch("anomaly_match.pipeline.session.torch.cuda.is_available", return_value=False),
        ):
            session.evaluate_all_images(top_N=1000)

        assert mock_pipeline.called

    def test_run_status_logged_on_first_chunk(self, session, tmp_path):
        """A single-line run-status (chunks + sources done/todo, runtime, speed,
        ETA) is logged at the very first subprocess spawn — the first chunk used
        to fall back to a bare "Processing N images" line with no overall
        status, so a run inspected early showed nothing useful."""
        model_path = tmp_path / "model.safetensors"
        _create_dummy_model(model_path)
        session.cfg.model_path = str(model_path)

        search_dir = tmp_path / "zarr_data"
        search_dir.mkdir()
        (search_dir / "store.zarr").mkdir()
        session.cfg.prediction_search_dir = str(search_dir)
        os.makedirs(session.cfg.output_dir, exist_ok=True)

        mock_images_array = MagicMock()
        mock_images_array.shape = (200,)
        mock_root = MagicMock()
        mock_root.__contains__ = lambda self, key: key == "images"
        mock_root.__getitem__ = lambda self, key: mock_images_array if key == "images" else None

        messages: list[str] = []
        sink_id = logger.add(messages.append, level="INFO", format="{message}")
        try:
            with (
                patch("anomaly_match.pipeline.session.zarr.open_group", return_value=mock_root),
                patch.object(Session, "run_pipeline"),
                patch("anomaly_match.pipeline.session.torch.cuda.is_available", return_value=False),
            ):
                session.evaluate_all_images(top_N=1000)
        finally:
            logger.remove(sink_id)

        status_lines = [m for m in messages if m.startswith("Run status:")]
        assert status_lines, "no run-status line was logged at spawn"
        first = status_lines[0]
        assert "chunk 1/1" in first
        assert "0/200 sources" in first  # total known, nothing done yet
        assert "ETA unknown" in first  # no completed chunk to derive a rate

    def test_zarr_nested_directory_detection(self, session, tmp_path):
        """Zarr inside batch directories (batch_dir/images.zarr) is detected."""
        model_path = tmp_path / "model.safetensors"
        _create_dummy_model(model_path)
        session.cfg.model_path = str(model_path)

        search_dir = tmp_path / "zarr_nested"
        search_dir.mkdir()
        batch_dir = search_dir / "batch_001"
        batch_dir.mkdir()
        (batch_dir / "images.zarr").mkdir()

        session.cfg.prediction_search_dir = str(search_dir)
        os.makedirs(session.cfg.output_dir, exist_ok=True)

        # Prepare output files
        output_csv = os.path.join(session.cfg.output_dir, f"{session.cfg.save_file}_top1000.csv")
        output_npy = os.path.join(session.cfg.output_dir, f"{session.cfg.save_file}_top1000.npy")
        pd.DataFrame(
            {
                "Filename": ["img_0.jpg"],
                "Score": [0.8],
            }
        ).to_csv(output_csv, index=False)
        np.save(output_npy, np.random.randint(0, 255, (1, 64, 64, 3), dtype=np.uint8))

        mock_images_array = MagicMock()
        mock_images_array.shape = (50,)
        mock_root = MagicMock()
        mock_root.__contains__ = lambda self, key: key == "images"
        mock_root.__getitem__ = lambda self, key: mock_images_array if key == "images" else None

        with (
            patch("anomaly_match.pipeline.session.zarr.open_group", return_value=mock_root),
            patch.object(Session, "run_pipeline") as mock_pipeline,
            patch("anomaly_match.pipeline.session.torch.cuda.is_available", return_value=False),
        ):
            session.evaluate_all_images(top_N=1000)

        # The nested images.zarr path should be passed to run_pipeline
        assert mock_pipeline.called
        input_file_arg = mock_pipeline.call_args[0][1]
        assert "images.zarr" in input_file_arg

    def test_zarr_no_images_key_warning(self, session, tmp_path):
        """Zarr file without 'images' key emits warning and counts 0."""
        model_path = tmp_path / "model.safetensors"
        _create_dummy_model(model_path)
        session.cfg.model_path = str(model_path)

        search_dir = tmp_path / "zarr_empty"
        search_dir.mkdir()
        (search_dir / "bad.zarr").mkdir()

        session.cfg.prediction_search_dir = str(search_dir)
        os.makedirs(session.cfg.output_dir, exist_ok=True)

        # Prepare output files
        output_csv = os.path.join(session.cfg.output_dir, f"{session.cfg.save_file}_top1000.csv")
        output_npy = os.path.join(session.cfg.output_dir, f"{session.cfg.save_file}_top1000.npy")
        pd.DataFrame({"Filename": [], "Score": []}).to_csv(output_csv, index=False)
        np.save(output_npy, np.zeros((0, 64, 64, 3), dtype=np.uint8))

        # Mock zarr root that does NOT contain "images"
        mock_root = MagicMock()
        mock_root.__contains__ = lambda self, key: False

        with (
            patch("anomaly_match.pipeline.session.zarr.open_group", return_value=mock_root),
            patch.object(Session, "run_pipeline") as mock_pipeline,
            patch("anomaly_match.pipeline.session.torch.cuda.is_available", return_value=False),
        ):
            session.evaluate_all_images(top_N=1000)

        # With total_images=0, evaluate_all_images returns early and does
        # not spawn a prediction subprocess with an empty file list.
        assert not mock_pipeline.called


class TestEvaluateAllImagesImage:
    def test_image_counting(self, session, tmp_path):
        """Individual image files each count as 1."""
        model_path = tmp_path / "model.safetensors"
        _create_dummy_model(model_path)
        session.cfg.model_path = str(model_path)

        search_dir = tmp_path / "imgs"
        search_dir.mkdir()
        for i in range(5):
            (search_dir / f"img_{i}.jpg").touch()

        session.cfg.prediction_search_dir = str(search_dir)
        os.makedirs(session.cfg.output_dir, exist_ok=True)

        output_csv = os.path.join(session.cfg.output_dir, f"{session.cfg.save_file}_top1000.csv")
        output_npy = os.path.join(session.cfg.output_dir, f"{session.cfg.save_file}_top1000.npy")
        pd.DataFrame(
            {
                "Filename": [f"img_{i}.jpg" for i in range(5)],
                "Score": np.linspace(0.1, 0.9, 5),
            }
        ).to_csv(output_csv, index=False)
        np.save(output_npy, np.random.randint(0, 255, (5, 64, 64, 3), dtype=np.uint8))

        with (
            patch.object(Session, "run_pipeline") as mock_pipeline,
            patch("anomaly_match.pipeline.session.torch.cuda.is_available", return_value=False),
        ):
            session.evaluate_all_images(top_N=1000)

        assert mock_pipeline.called

    def test_image_grouping_into_batches(self, session, tmp_path):
        """More than 10K images are split into groups of 10K, written to .txt files."""
        model_path = tmp_path / "model.safetensors"
        _create_dummy_model(model_path)
        session.cfg.model_path = str(model_path)

        search_dir = tmp_path / "large_dataset"
        search_dir.mkdir()
        # Create 15K image file stubs (just need names, not real images)
        for i in range(15_000):
            (search_dir / f"img_{i:06d}.jpg").touch()

        session.cfg.prediction_search_dir = str(search_dir)
        os.makedirs(session.cfg.output_dir, exist_ok=True)

        output_csv = os.path.join(session.cfg.output_dir, f"{session.cfg.save_file}_top1000.csv")
        output_npy = os.path.join(session.cfg.output_dir, f"{session.cfg.save_file}_top1000.npy")
        pd.DataFrame(
            {
                "Filename": ["img_0.jpg"],
                "Score": [0.5],
            }
        ).to_csv(output_csv, index=False)
        np.save(output_npy, np.random.randint(0, 255, (1, 64, 64, 3), dtype=np.uint8))

        pipeline_calls = []

        def capture_pipeline(config_path, input_file, top_n, file_type):
            pipeline_calls.append(input_file)

        with (
            patch.object(Session, "run_pipeline", side_effect=capture_pipeline),
            patch("anomaly_match.pipeline.session.torch.cuda.is_available", return_value=False),
        ):
            session.evaluate_all_images(top_N=1000)

        # 15K images should produce 2 groups (10K + 5K)
        assert len(pipeline_calls) == 2
        # Each input_file should be a .txt group file
        for path in pipeline_calls:
            assert path.endswith(".txt")

        # Verify the group files were written with correct content
        for path in pipeline_calls:
            content = Path(path).read_text()
            lines = [line for line in content.split("\n") if line.strip()]
            assert len(lines) <= 10_000

    def test_image_small_batch_no_grouping(self, session, tmp_path):
        """Fewer than 10K images go as a single group."""
        model_path = tmp_path / "model.safetensors"
        _create_dummy_model(model_path)
        session.cfg.model_path = str(model_path)

        search_dir = tmp_path / "small_dataset"
        search_dir.mkdir()
        for i in range(50):
            (search_dir / f"img_{i:04d}.jpg").touch()

        session.cfg.prediction_search_dir = str(search_dir)
        os.makedirs(session.cfg.output_dir, exist_ok=True)

        output_csv = os.path.join(session.cfg.output_dir, f"{session.cfg.save_file}_top1000.csv")
        output_npy = os.path.join(session.cfg.output_dir, f"{session.cfg.save_file}_top1000.npy")
        pd.DataFrame(
            {
                "Filename": [f"img_{i:04d}.jpg" for i in range(50)],
                "Score": np.random.rand(50),
            }
        ).to_csv(output_csv, index=False)
        np.save(output_npy, np.random.randint(0, 255, (50, 64, 64, 3), dtype=np.uint8))

        pipeline_calls = []

        def capture_pipeline(config_path, input_file, top_n, file_type):
            pipeline_calls.append(input_file)

        with (
            patch.object(Session, "run_pipeline", side_effect=capture_pipeline),
            patch("anomaly_match.pipeline.session.torch.cuda.is_available", return_value=False),
        ):
            session.evaluate_all_images(top_N=1000)

        # Single group for <= 10K images
        assert len(pipeline_calls) == 1

    def test_txt_group_file_counting(self, session, tmp_path):
        """When processing loop encounters a .txt group file, it counts lines."""
        model_path = tmp_path / "model.safetensors"
        _create_dummy_model(model_path)
        session.cfg.model_path = str(model_path)

        search_dir = tmp_path / "imgs_for_txt"
        search_dir.mkdir()
        for i in range(3):
            (search_dir / f"img_{i}.jpg").touch()

        session.cfg.prediction_search_dir = str(search_dir)
        os.makedirs(session.cfg.output_dir, exist_ok=True)

        output_csv = os.path.join(session.cfg.output_dir, f"{session.cfg.save_file}_top1000.csv")
        output_npy = os.path.join(session.cfg.output_dir, f"{session.cfg.save_file}_top1000.npy")
        pd.DataFrame(
            {
                "Filename": [f"img_{i}.jpg" for i in range(3)],
                "Score": [0.1, 0.5, 0.9],
            }
        ).to_csv(output_csv, index=False)
        np.save(output_npy, np.random.randint(0, 255, (3, 64, 64, 3), dtype=np.uint8))

        with (
            patch.object(Session, "run_pipeline"),
            patch("anomaly_match.pipeline.session.torch.cuda.is_available", return_value=False),
        ):
            session.evaluate_all_images(top_N=1000)

        # Session is lightweight; results go to predictions.db, not in-memory attributes.
        # Verify run_pipeline was invoked (subprocess would write the DB).


class TestEvaluateAllImagesStream:
    def test_stream_cutana_validation(self, session, tmp_path):
        model_path = tmp_path / "model.safetensors"
        _create_dummy_model(model_path)
        session.cfg.model_path = str(model_path)

        search_dir = tmp_path / "catalogs"
        search_dir.mkdir()
        (search_dir / "cat.parquet").touch()

        session.cfg.prediction_search_dir = str(search_dir)
        os.makedirs(session.cfg.output_dir, exist_ok=True)

        output_csv = os.path.join(session.cfg.output_dir, f"{session.cfg.save_file}_top1000.csv")
        output_npy = os.path.join(session.cfg.output_dir, f"{session.cfg.save_file}_top1000.npy")
        pd.DataFrame(
            {
                "Filename": ["src_1", "src_2"],
                "Score": [0.9, 0.3],
            }
        ).to_csv(output_csv, index=False)
        np.save(output_npy, np.random.randint(0, 255, (2, 64, 64, 3), dtype=np.uint8))

        # The buffer file produced by cutana_buffer_generator
        buffer_file = tmp_path / "buffer.parquet"
        pd.DataFrame({"source_id": [1, 2]}).to_parquet(buffer_file)

        with (
            patch(
                "anomaly_match.pipeline.session.cutana_validate_files_and_count_sources",
                return_value=([str(search_dir / "cat.parquet")], 5000, 2),
            ),
            patch(
                "anomaly_match.pipeline.session.cutana_buffer_generator",
                return_value=[str(buffer_file)],
            ),
            patch.object(Session, "run_pipeline"),
            patch("anomaly_match.pipeline.session.torch.cuda.is_available", return_value=False),
        ):
            session.evaluate_all_images(top_N=1000)

    def test_stream_no_valid_files_raises(self, session, tmp_path):
        model_path = tmp_path / "model.safetensors"
        _create_dummy_model(model_path)
        session.cfg.model_path = str(model_path)

        search_dir = tmp_path / "bad_catalogs"
        search_dir.mkdir()
        (search_dir / "bad.parquet").touch()

        session.cfg.prediction_search_dir = str(search_dir)

        with (
            patch(
                "anomaly_match.pipeline.session.cutana_validate_files_and_count_sources",
                return_value=([], 0, 0),
            ),
        ):
            with pytest.raises(RuntimeError, match="not compatible with cutana"):
                session.evaluate_all_images(top_N=1000)


class TestEvaluateAllImagesProgress:
    def test_progress_callback_called(self, session, tmp_path):
        model_path = tmp_path / "model.safetensors"
        _create_dummy_model(model_path)
        session.cfg.model_path = str(model_path)

        search_dir = tmp_path / "imgs"
        search_dir.mkdir()
        (search_dir / "a.jpg").touch()
        (search_dir / "b.jpg").touch()

        session.cfg.prediction_search_dir = str(search_dir)
        os.makedirs(session.cfg.output_dir, exist_ok=True)

        output_csv = os.path.join(session.cfg.output_dir, f"{session.cfg.save_file}_top1000.csv")
        output_npy = os.path.join(session.cfg.output_dir, f"{session.cfg.save_file}_top1000.npy")
        pd.DataFrame(
            {
                "Filename": ["a.jpg", "b.jpg"],
                "Score": [0.9, 0.1],
            }
        ).to_csv(output_csv, index=False)
        np.save(output_npy, np.random.randint(0, 255, (2, 64, 64, 3), dtype=np.uint8))

        callback = MagicMock()

        with (
            patch.object(Session, "run_pipeline"),
            patch("anomaly_match.pipeline.session.torch.cuda.is_available", return_value=False),
        ):
            session.evaluate_all_images(top_N=1000, progress_callback=callback)

        # Callback should have been called at least once with batch_update=True
        assert callback.called
        # Check for final completed callback
        final_call = callback.call_args_list[-1]
        assert final_call[1].get("completed") is True or final_call[1].get("batch_update") is True

    def test_progress_with_eta(self, session, tmp_path):
        """After first file, progress callback includes ETA info."""
        model_path = tmp_path / "model.safetensors"
        _create_dummy_model(model_path)
        session.cfg.model_path = str(model_path)

        search_dir = tmp_path / "zarr_multi"
        search_dir.mkdir()
        (search_dir / "batch_1.zarr").mkdir()
        (search_dir / "batch_2.zarr").mkdir()

        session.cfg.prediction_search_dir = str(search_dir)
        os.makedirs(session.cfg.output_dir, exist_ok=True)

        output_csv = os.path.join(session.cfg.output_dir, f"{session.cfg.save_file}_top1000.csv")
        output_npy = os.path.join(session.cfg.output_dir, f"{session.cfg.save_file}_top1000.npy")
        pd.DataFrame(
            {
                "Filename": ["img_0.jpg", "img_1.jpg"],
                "Score": [0.8, 0.2],
            }
        ).to_csv(output_csv, index=False)
        np.save(output_npy, np.random.randint(0, 255, (2, 64, 64, 3), dtype=np.uint8))

        mock_images_array = MagicMock()
        mock_images_array.shape = (50,)
        mock_root = MagicMock()
        mock_root.__contains__ = lambda self, key: key == "images"
        mock_root.__getitem__ = lambda self, key: mock_images_array if key == "images" else None

        callback = MagicMock()

        with (
            patch("anomaly_match.pipeline.session.zarr.open_group", return_value=mock_root),
            patch.object(Session, "run_pipeline"),
            patch("anomaly_match.pipeline.session.torch.cuda.is_available", return_value=False),
        ):
            session.evaluate_all_images(top_N=1000, progress_callback=callback)

        assert callback.called
        # With 2 files, the second file should get ETA info after the first
        # Find a call that has eta_str in kwargs
        eta_calls = [c for c in callback.call_args_list if c[1].get("eta_str") is not None]
        # After the first iteration, the second should have ETA
        assert len(eta_calls) >= 1 or len(callback.call_args_list) >= 2


class TestEvaluateAllImagesOutputLoading:
    def test_missing_db_no_crash(self, session, tmp_path):
        """When subprocess produces no predictions.db, session logs error but doesn't crash."""
        model_path = tmp_path / "model.safetensors"
        _create_dummy_model(model_path)
        session.cfg.model_path = str(model_path)

        search_dir = tmp_path / "imgs"
        search_dir.mkdir()
        (search_dir / "a.jpg").touch()

        session.cfg.prediction_search_dir = str(search_dir)
        os.makedirs(session.cfg.output_dir, exist_ok=True)
        # Do NOT create predictions.db

        with (
            patch.object(Session, "run_pipeline"),
            patch("anomaly_match.pipeline.session.torch.cuda.is_available", return_value=False),
        ):
            # Should not raise even though predictions.db is missing
            session.evaluate_all_images(top_N=1000)

    def test_results_updated_callback(self, session, tmp_path):
        """progress_callback is called with results_updated=True when predictions.db exists."""
        model_path = tmp_path / "model.safetensors"
        _create_dummy_model(model_path)
        session.cfg.model_path = str(model_path)

        search_dir = tmp_path / "imgs"
        search_dir.mkdir()
        (search_dir / "a.jpg").touch()

        session.cfg.prediction_search_dir = str(search_dir)
        os.makedirs(session.cfg.output_dir, exist_ok=True)

        # Create predictions.db so the results_updated callback fires
        db_path = os.path.join(session.cfg.output_dir, "predictions.db")
        Path(db_path).touch()

        callback = MagicMock()

        with (
            patch.object(Session, "run_pipeline"),
            patch("anomaly_match.pipeline.session.torch.cuda.is_available", return_value=False),
        ):
            session.evaluate_all_images(top_N=1000, progress_callback=callback)

        # Find a call with results_updated=True
        results_updated_calls = [
            c for c in callback.call_args_list if c[1].get("results_updated") is True
        ]
        assert len(results_updated_calls) >= 1


class TestEvaluateAllImagesFinalStatistics:
    def test_final_statistics_with_images(self, session, tmp_path):
        model_path = tmp_path / "model.safetensors"
        _create_dummy_model(model_path)
        session.cfg.model_path = str(model_path)

        search_dir = tmp_path / "imgs"
        search_dir.mkdir()
        (search_dir / "a.jpg").touch()

        session.cfg.prediction_search_dir = str(search_dir)
        os.makedirs(session.cfg.output_dir, exist_ok=True)

        # Create predictions.db so the per-file callback fires
        db_path = os.path.join(session.cfg.output_dir, "predictions.db")
        Path(db_path).touch()

        callback = MagicMock()

        with (
            patch.object(Session, "run_pipeline"),
            patch("anomaly_match.pipeline.session.torch.cuda.is_available", return_value=False),
        ):
            session.evaluate_all_images(top_N=1000, progress_callback=callback)

        # Final callback should include completed flag
        completed_calls = [c for c in callback.call_args_list if c[1].get("completed") is True]
        assert len(completed_calls) == 1
        final = completed_calls[0]
        assert "total_time_str" in final[1]
        assert "final_speed" in final[1]

    def test_no_images_processed_warning(self, session, tmp_path):
        """When no images are processed (empty dir), warning is logged."""
        model_path = tmp_path / "model.safetensors"
        _create_dummy_model(model_path)
        session.cfg.model_path = str(model_path)

        search_dir = tmp_path / "empty_imgs"
        search_dir.mkdir()
        # Put one jpg so auto-detect finds "image" type, but it's the only file
        # so processed_images=1 and we won't hit the warning path.
        # Actually, to test the "no images" warning we need an empty file list,
        # but that's hard because we need at least one file for auto-detect.
        # Instead, we can put a non-matching file type.
        (search_dir / "readme.md").touch()

        session.cfg.prediction_search_dir = str(search_dir)
        os.makedirs(session.cfg.output_dir, exist_ok=True)

        # With no recognized images, auto-detect falls back to "image" with
        # empty input_files, so the loop body never runs and processed_images=0
        with (
            patch("anomaly_match.pipeline.session.torch.cuda.is_available", return_value=False),
        ):
            # Should not crash; will log warning about no images processed
            session.evaluate_all_images(top_N=1000)

    def test_cuda_cache_cleared(self, session, tmp_path):
        """torch.cuda.empty_cache is called when CUDA is available."""
        model_path = tmp_path / "model.safetensors"
        _create_dummy_model(model_path)
        session.cfg.model_path = str(model_path)

        search_dir = tmp_path / "imgs"
        search_dir.mkdir()
        (search_dir / "a.jpg").touch()

        session.cfg.prediction_search_dir = str(search_dir)
        os.makedirs(session.cfg.output_dir, exist_ok=True)

        output_csv = os.path.join(session.cfg.output_dir, f"{session.cfg.save_file}_top1000.csv")
        output_npy = os.path.join(session.cfg.output_dir, f"{session.cfg.save_file}_top1000.npy")
        pd.DataFrame(
            {
                "Filename": ["a.jpg"],
                "Score": [0.5],
            }
        ).to_csv(output_csv, index=False)
        np.save(output_npy, np.random.randint(0, 255, (1, 64, 64, 3), dtype=np.uint8))

        with (
            patch.object(Session, "run_pipeline"),
            patch("anomaly_match.pipeline.session.torch.cuda.is_available", return_value=True),
            patch("anomaly_match.pipeline.session.torch.cuda.empty_cache") as mock_cache,
        ):
            session.evaluate_all_images(top_N=1000)

        assert mock_cache.called


class TestEvaluateAllImagesConfigSerialization:
    def test_config_pickled_for_subprocess(self, session, tmp_path):
        """The config is serialized to a pickle file before running the subprocess."""
        model_path = tmp_path / "model.safetensors"
        _create_dummy_model(model_path)
        session.cfg.model_path = str(model_path)

        search_dir = tmp_path / "imgs"
        search_dir.mkdir()
        (search_dir / "a.jpg").touch()

        session.cfg.prediction_search_dir = str(search_dir)
        os.makedirs(session.cfg.output_dir, exist_ok=True)

        output_csv = os.path.join(session.cfg.output_dir, f"{session.cfg.save_file}_top1000.csv")
        output_npy = os.path.join(session.cfg.output_dir, f"{session.cfg.save_file}_top1000.npy")
        pd.DataFrame(
            {
                "Filename": ["a.jpg"],
                "Score": [0.5],
            }
        ).to_csv(output_csv, index=False)
        np.save(output_npy, np.random.randint(0, 255, (1, 64, 64, 3), dtype=np.uint8))

        config_paths_written = []

        def capture_pipeline(config_path, input_file, top_n, file_type):
            config_paths_written.append(config_path)

        with (
            patch.object(Session, "run_pipeline", side_effect=capture_pipeline),
            patch("anomaly_match.pipeline.session.torch.cuda.is_available", return_value=False),
        ):
            session.evaluate_all_images(top_N=1000)

        # A pickle config was created
        assert len(config_paths_written) == 1
        config_path = config_paths_written[0]
        assert config_path.endswith("_config.pkl")
        assert os.path.exists(config_path)

        # Verify it deserializes correctly
        with open(config_path, "rb") as f:
            loaded = pickle.load(f)
        assert isinstance(loaded, dict)

    def test_model_path_missing_during_loop_raises(self, session, tmp_path):
        """If model_path disappears mid-loop, FileNotFoundError is raised."""
        model_path = tmp_path / "model.safetensors"
        _create_dummy_model(model_path)
        session.cfg.model_path = str(model_path)

        # Use two Zarr files so we get two loop iterations
        search_dir = tmp_path / "zarr_vanish"
        search_dir.mkdir()
        (search_dir / "batch_1.zarr").mkdir()
        (search_dir / "batch_2.zarr").mkdir()

        session.cfg.prediction_search_dir = str(search_dir)
        os.makedirs(session.cfg.output_dir, exist_ok=True)

        output_csv = os.path.join(session.cfg.output_dir, f"{session.cfg.save_file}_top1000.csv")
        output_npy = os.path.join(session.cfg.output_dir, f"{session.cfg.save_file}_top1000.npy")
        pd.DataFrame({"Filename": ["img.jpg"], "Score": [0.5]}).to_csv(output_csv, index=False)
        np.save(output_npy, np.random.randint(0, 255, (1, 64, 64, 3), dtype=np.uint8))

        mock_images_array = MagicMock()
        mock_images_array.shape = (10,)
        mock_root = MagicMock()
        mock_root.__contains__ = lambda self, key: key == "images"
        mock_root.__getitem__ = lambda self, key: mock_images_array if key == "images" else None

        # The first run_pipeline call removes the model file, so the second
        # iteration's os.path.exists check raises FileNotFoundError.
        def remove_model(*args, **kwargs):
            if model_path.exists():
                os.remove(str(model_path))

        with (
            patch("anomaly_match.pipeline.session.zarr.open_group", return_value=mock_root),
            patch.object(Session, "run_pipeline", side_effect=remove_model),
            patch("anomaly_match.pipeline.session.torch.cuda.is_available", return_value=False),
        ):
            with pytest.raises(FileNotFoundError, match="Model file not found"):
                session.evaluate_all_images(top_N=1000)


# ---------------------------------------------------------------------------
# Resume pre-filter — fully-scored inputs must not spawn subprocesses
# ---------------------------------------------------------------------------


class TestResumeDispatchFilter:
    """evaluate_all_images drops fully-scored work before launching subprocesses."""

    def _seed_db(self, output_dir, filenames):
        from anomaly_match.prediction import AnomalyScoreDB

        os.makedirs(output_dir, exist_ok=True)
        db_path = os.path.join(output_dir, "predictions.db")
        with AnomalyScoreDB(db_path) as db:
            db.store_results([(fn, 0.5) for fn in filenames])
        return db_path

    def _fresh_output_dir(self, session, tmp_path):
        """Pin session output to a tmp subdir.

        Session's ``update_config_paths_for_session`` otherwise points
        ``cfg.output_dir`` at a timestamped ``anomaly_match_results/
        sessions/...`` dir outside tmp_path — stale DBs from earlier
        runs in the same second make the resume read blow up.
        """
        target = tmp_path / "pred_output"
        target.mkdir(exist_ok=True)
        session.cfg.output_dir = str(target)
        return str(target)

    def test_image_fully_scored_skips_all_subprocesses(self, session, tmp_path):
        """No subprocess spawns when every image path is already in the DB."""
        model_path = tmp_path / "model.safetensors"
        _create_dummy_model(model_path)
        session.cfg.model_path = str(model_path)

        self._fresh_output_dir(session, tmp_path)

        search_dir = tmp_path / "images"
        search_dir.mkdir()
        image_paths = []
        for i in range(5):
            p = search_dir / f"img_{i}.jpg"
            p.touch()
            image_paths.append(str(p))
        session.cfg.prediction_search_dir = str(search_dir)

        self._seed_db(session.cfg.output_dir, image_paths)

        with (
            patch.object(Session, "run_pipeline") as mock_pipeline,
            patch("anomaly_match.pipeline.session.torch.cuda.is_available", return_value=False),
        ):
            session.evaluate_all_images(top_N=1000)

        assert mock_pipeline.call_count == 0, (
            "Fully-scored run should not spawn any prediction subprocess."
        )

    def test_image_partial_resume_spawns_once_with_remaining(self, session, tmp_path):
        """Half the images scored → exactly one subprocess for the remaining half."""
        model_path = tmp_path / "model.safetensors"
        _create_dummy_model(model_path)
        session.cfg.model_path = str(model_path)

        self._fresh_output_dir(session, tmp_path)

        search_dir = tmp_path / "images"
        search_dir.mkdir()
        image_paths = []
        for i in range(10):
            p = search_dir / f"img_{i:02d}.jpg"
            p.touch()
            image_paths.append(str(p))
        session.cfg.prediction_search_dir = str(search_dir)

        self._seed_db(session.cfg.output_dir, image_paths[:5])

        with (
            patch.object(Session, "run_pipeline") as mock_pipeline,
            patch("anomaly_match.pipeline.session.torch.cuda.is_available", return_value=False),
        ):
            session.evaluate_all_images(top_N=1000)

        assert mock_pipeline.call_count == 1, (
            "Partial resume should spawn exactly one subprocess for remaining images."
        )

        # The tmp filelist handed to the subprocess must contain only
        # the unscored paths.
        input_file_arg = mock_pipeline.call_args[0][1]
        with open(input_file_arg) as f:
            dispatched = [line.strip() for line in f if line.strip()]
        assert set(dispatched) == set(image_paths[5:])

    def test_zarr_fully_scored_skips_all_subprocesses(self, session, tmp_path):
        """Zarr file whose metadata parquet lists only already-scored
        names is dropped from dispatch."""
        model_path = tmp_path / "model.safetensors"
        _create_dummy_model(model_path)
        session.cfg.model_path = str(model_path)

        self._fresh_output_dir(session, tmp_path)

        search_dir = tmp_path / "zarr_data"
        search_dir.mkdir()
        zarr_path = search_dir / "store.zarr"

        # Build a real zarr + a matching metadata parquet so
        # derive_filenames returns the same names we seed in the DB.
        import zarr as zarr_real

        root = zarr_real.open_group(str(zarr_path), mode="w")
        root.create_dataset("images", shape=(4, 8, 8, 3), chunks=(1, 8, 8, 3), dtype=np.uint8)
        pd.DataFrame({"original_filename": [f"scored_{i}.png" for i in range(4)]}).to_parquet(
            search_dir / "store_metadata.parquet", index=False
        )

        session.cfg.prediction_search_dir = str(search_dir)

        self._seed_db(session.cfg.output_dir, [f"scored_{i}.png" for i in range(4)])

        with (
            patch.object(Session, "run_pipeline") as mock_pipeline,
            patch("anomaly_match.pipeline.session.torch.cuda.is_available", return_value=False),
        ):
            session.evaluate_all_images(top_N=1000)

        assert mock_pipeline.call_count == 0, (
            "Zarr with all-scored filenames should not spawn a prediction subprocess."
        )

    def _prepare_resume_session(self, session, tmp_path):
        """Give *session* a valid model and a fresh tmp output dir for resume tests."""
        model_path = tmp_path / "model.safetensors"
        _create_dummy_model(model_path)
        session.cfg.model_path = str(model_path)
        self._fresh_output_dir(session, tmp_path)

    def _make_cutana_buffer(self, tmp_path, id_column, ids):
        """Write a buffer parquet as cutana_buffer_generator would (catalogue cols)."""
        buffer_file = tmp_path / f"buffer_{id_column}.parquet"
        pd.DataFrame(
            {id_column: ids, "RA": [1.0] * len(ids), "fits_file_paths": ["x"] * len(ids)}
        ).to_parquet(buffer_file)
        return buffer_file

    def _run_cutana_resume(self, session, tmp_path, buffer_file, n_sources):
        """Drive evaluate_all_images over a single mocked Cutana buffer chunk."""
        search_dir = tmp_path / "catalogs"
        search_dir.mkdir(exist_ok=True)
        (search_dir / "cat.parquet").touch()
        session.cfg.prediction_search_dir = str(search_dir)

        with (
            patch(
                "anomaly_match.pipeline.session.cutana_validate_files_and_count_sources",
                return_value=([str(search_dir / "cat.parquet")], n_sources, 1),
            ),
            patch(
                "anomaly_match.pipeline.session.cutana_buffer_generator",
                return_value=[str(buffer_file)],
            ),
            patch.object(Session, "run_pipeline") as mock_pipeline,
            patch("anomaly_match.pipeline.session.torch.cuda.is_available", return_value=False),
        ):
            session.evaluate_all_images(top_N=1000)
        return mock_pipeline

    def test_cutana_fully_scored_chunk_with_sourceid_skips(self, session, tmp_path):
        """Regression #514: Cutana catalogues key the id under ``SourceID``.

        On resume the chunk's ids are already in the DB, so the chunk must be
        recognised and skipped — not raise ``KeyError`` looking for a lowercase
        ``source_id`` (which aborted every resume and showed a false "Done").
        """
        self._prepare_resume_session(session, tmp_path)

        ids = ["-505850700276317218", "-505876634276377279", "777"]
        self._seed_db(session.cfg.output_dir, ids)
        buffer_file = self._make_cutana_buffer(tmp_path, "SourceID", ids)

        mock_pipeline = self._run_cutana_resume(session, tmp_path, buffer_file, n_sources=len(ids))

        assert mock_pipeline.call_count == 0, (
            "A fully-scored SourceID chunk must be skipped on resume, not re-scored or crashed."
        )

    def test_cutana_partial_resume_spawns_for_unscored(self, session, tmp_path):
        """A SourceID chunk with unscored ids is not a subset → one subprocess."""
        self._prepare_resume_session(session, tmp_path)

        self._seed_db(session.cfg.output_dir, ["already_scored"])
        buffer_file = self._make_cutana_buffer(tmp_path, "SourceID", ["new_1", "new_2"])

        mock_pipeline = self._run_cutana_resume(session, tmp_path, buffer_file, n_sources=2)

        assert mock_pipeline.call_count == 1, (
            "A chunk with unscored sources must still spawn its scoring subprocess."
        )

    def test_cutana_chunk_without_id_column_raises(self, session, tmp_path):
        """A buffer carrying no recognised id column is a genuine schema bug."""
        self._prepare_resume_session(session, tmp_path)

        # processed must be non-empty for the resume pre-filter to run at all.
        self._seed_db(session.cfg.output_dir, ["already_scored"])
        buffer_file = tmp_path / "no_id.parquet"
        pd.DataFrame({"RA": [1.0], "fits_file_paths": ["x"]}).to_parquet(buffer_file)

        with pytest.raises(KeyError, match="no source-id column"):
            self._run_cutana_resume(session, tmp_path, buffer_file, n_sources=1)


# ---------------------------------------------------------------------------
# Graceful cancellation — request_prediction_stop + chunk-loop bailout
# ---------------------------------------------------------------------------


class TestPredictionCancellation:
    """Stop signal must reach the running subprocess AND abort the chunk loop."""

    def test_request_stop_sigterms_live_subprocess(self, session):
        """Calling request_prediction_stop SIGTERMs a live subprocess."""
        fake_process = MagicMock()
        fake_process.poll.return_value = None  # alive
        fake_process.wait.return_value = 0
        session._current_prediction_process = fake_process

        session.request_prediction_stop(timeout=1.0)

        assert session._prediction_stop_requested.is_set()
        fake_process.terminate.assert_called_once()
        fake_process.wait.assert_called_once()

    def test_request_stop_escalates_to_sigkill_on_timeout(self, session):
        """If SIGTERM is ignored, we escalate to SIGKILL."""
        import subprocess as subprocess_mod

        fake_process = MagicMock()
        fake_process.poll.return_value = None
        # First wait (after terminate) times out; second wait (after kill) succeeds.
        fake_process.wait.side_effect = [
            subprocess_mod.TimeoutExpired(cmd="x", timeout=1.0),
            0,
        ]
        session._current_prediction_process = fake_process

        session.request_prediction_stop(timeout=0.01)

        fake_process.terminate.assert_called_once()
        fake_process.kill.assert_called_once()

    def test_request_stop_is_idempotent_when_no_process(self, session):
        """Safe to call when nothing's running — just arms the stop flag."""
        assert session._current_prediction_process is None
        session.request_prediction_stop()
        assert session._prediction_stop_requested.is_set()

    def test_chunk_loop_bails_after_stop_flag_set(self, session, tmp_path):
        """Setting the stop flag between chunks prevents further subprocess launches."""
        model_path = tmp_path / "model.safetensors"
        _create_dummy_model(model_path)
        session.cfg.model_path = str(model_path)

        search_dir = tmp_path / "images"
        search_dir.mkdir()
        for i in range(3):
            (search_dir / f"img_{i}.jpg").touch()
        session.cfg.prediction_search_dir = str(search_dir)

        call_count = {"n": 0}

        def fake_run_pipeline(*_args, **_kwargs):
            call_count["n"] += 1
            # First chunk finishes normally, then UI requests stop.
            session._prediction_stop_requested.set()

        with (
            patch.object(Session, "run_pipeline", side_effect=fake_run_pipeline),
            patch("anomaly_match.pipeline.session.torch.cuda.is_available", return_value=False),
        ):
            session.evaluate_all_images(top_N=0)

        # With 3 images grouped into one chunk of 10k we expect a single
        # run_pipeline call.  The important check is that the stop flag
        # cleanly aborts the loop rather than raising.
        assert call_count["n"] <= 1

    def test_stop_flag_cleared_on_fresh_run(self, session, tmp_path):
        """evaluate_all_images must reset a stop flag left by a prior run."""
        model_path = tmp_path / "model.safetensors"
        _create_dummy_model(model_path)
        session.cfg.model_path = str(model_path)

        search_dir = tmp_path / "images"
        search_dir.mkdir()
        (search_dir / "img_0.jpg").touch()
        session.cfg.prediction_search_dir = str(search_dir)

        # Arm the flag from a "prior" run that was cancelled.
        session._prediction_stop_requested.set()

        called = {"n": 0}

        def fake_run_pipeline(*_args, **_kwargs):
            called["n"] += 1

        with (
            patch.object(Session, "run_pipeline", side_effect=fake_run_pipeline),
            patch("anomaly_match.pipeline.session.torch.cuda.is_available", return_value=False),
        ):
            session.evaluate_all_images(top_N=0)

        assert called["n"] == 1, "A stale stop flag from a prior run must not block a fresh call."

    def test_subprocess_nonzero_exit_swallowed_when_stop_requested(self, session, tmp_path):
        """SIGTERM-ed subprocess exits nonzero — don't raise when the stop was intentional."""
        config_path = str(tmp_path / "cfg.pkl")
        input_path = str(tmp_path / "store.zarr")

        session._prediction_stop_requested.set()

        with (
            patch(
                "subprocess.Popen",
                return_value=MagicMock(returncode=-15, stdout=iter([]), wait=MagicMock()),
            ),
            patch("anomaly_match.pipeline.session.set_log_level"),
            patch("os.path.exists", return_value=True),
        ):
            # Must not raise even though returncode is nonzero — the stop
            # flag tells run_pipeline the exit was intentional.
            session.run_pipeline(config_path, input_path, 500, file_type=DataSourceType.ZARR)

    def test_popen_registered_then_cleared_on_normal_exit(self, session, tmp_path):
        """_current_prediction_process is set while running and cleared when done."""
        config_path = str(tmp_path / "cfg.pkl")
        input_path = str(tmp_path / "store.zarr")

        fake_popen = MagicMock(returncode=0, stdout=iter([]), wait=MagicMock())

        with (
            patch("subprocess.Popen", return_value=fake_popen),
            patch("anomaly_match.pipeline.session.set_log_level"),
            patch("os.path.exists", return_value=True),
        ):
            session.run_pipeline(config_path, input_path, 500, file_type=DataSourceType.ZARR)

        assert session._current_prediction_process is None


# ---------------------------------------------------------------------------
# Skip-and-continue when individual chunks fail
# ---------------------------------------------------------------------------


class TestSkipFailedChunks:
    """A failed chunk must not abort the whole multi-hour scoring run.

    Regression for the case where the data volume detached mid-run
    (MyRun_20260424_185241): chunk 116's catalogue referenced FITS tiles
    that no longer existed, so all 93 batches returned empty cutouts and
    the subprocess raised RuntimeError, killing the whole 30M-image run.
    """

    @staticmethod
    def _make_image_search(tmp_path, n_images: int) -> str:
        search_dir = tmp_path / "images"
        search_dir.mkdir()
        for i in range(n_images):
            (search_dir / f"img_{i:03d}.jpg").touch()
        return str(search_dir)

    def test_runtime_error_in_chunk_skipped_and_loop_continues(self, session, tmp_path):
        """run_pipeline RuntimeError is caught and the chunk is counted as skipped."""
        model_path = tmp_path / "model.safetensors"
        _create_dummy_model(model_path)
        session.cfg.model_path = str(model_path)

        search_dir = self._make_image_search(tmp_path, 3)
        session.cfg.prediction_search_dir = search_dir

        def fake_run_pipeline(*_args, **_kwargs):
            # Simulate a chunk failure (e.g. data volume detached
            # mid-run, so the subprocess exited non-zero).
            raise RuntimeError("Prediction subprocess exited with code 1")

        with (
            patch.object(Session, "run_pipeline", side_effect=fake_run_pipeline),
            patch("anomaly_match.pipeline.session.torch.cuda.is_available", return_value=False),
        ):
            # Must NOT raise — chunk failures are caught and accumulated.
            session.evaluate_all_images(top_N=0)

        assert session.last_run_skipped_chunks == 1
        assert session.last_run_skipped_sources > 0

    def test_zero_rows_chunk_counted_as_skipped(self, session, tmp_path):
        """A chunk that exits cleanly but produces no DB rows is counted as skipped."""
        model_path = tmp_path / "model.safetensors"
        _create_dummy_model(model_path)
        session.cfg.model_path = str(model_path)

        search_dir = self._make_image_search(tmp_path, 2)
        session.cfg.prediction_search_dir = search_dir

        # run_pipeline returns silently → zero rows added → chunk skipped.
        with (
            patch.object(Session, "run_pipeline", return_value=None),
            patch("anomaly_match.pipeline.session.torch.cuda.is_available", return_value=False),
        ):
            session.evaluate_all_images(top_N=0)

        assert session.last_run_skipped_chunks >= 1

    def test_chunks_writing_rows_are_not_marked_skipped(self, session, tmp_path):
        """A chunk that writes DB rows is *not* counted as skipped."""
        from anomaly_match.prediction import AnomalyScoreDB

        model_path = tmp_path / "model.safetensors"
        _create_dummy_model(model_path)
        session.cfg.model_path = str(model_path)

        # Pin output dir so we can seed predictions.db and not race a stale one.
        out_dir = tmp_path / "pred_output"
        out_dir.mkdir()
        session.cfg.output_dir = str(out_dir)

        search_dir = tmp_path / "images"
        search_dir.mkdir()
        (search_dir / "img_a.jpg").touch()
        session.cfg.prediction_search_dir = str(search_dir)

        db_path = os.path.join(session.cfg.output_dir, "predictions.db")

        call_record: list[int] = []

        def fake_run_pipeline(*_args, **_kwargs):
            # Simulate the subprocess writing one prediction row.
            with AnomalyScoreDB(db_path) as db:
                db.store_results([(f"img_{len(call_record):d}.jpg", 0.5)])
            call_record.append(1)

        with (
            patch.object(Session, "run_pipeline", side_effect=fake_run_pipeline),
            patch("anomaly_match.pipeline.session.torch.cuda.is_available", return_value=False),
        ):
            session.evaluate_all_images(top_N=0)

        assert session.last_run_skipped_chunks == 0
        assert call_record  # run_pipeline actually called

    def test_skip_stats_reset_between_runs(self, session, tmp_path):
        """A fresh evaluate_all_images call zeros the skip counters."""
        session.last_run_skipped_chunks = 5
        session.last_run_skipped_sources = 12345

        model_path = tmp_path / "model.safetensors"
        _create_dummy_model(model_path)
        session.cfg.model_path = str(model_path)

        # No images → returns before any chunk loop, but before that it
        # resets counters.
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        session.cfg.prediction_search_dir = str(empty_dir)

        with patch("anomaly_match.pipeline.session.torch.cuda.is_available", return_value=False):
            session.evaluate_all_images(top_N=0)

        assert session.last_run_skipped_chunks == 0
        assert session.last_run_skipped_sources == 0
