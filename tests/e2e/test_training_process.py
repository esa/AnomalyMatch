#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""End-to-end tests for the training subprocess (scripts/training_process.py)."""

import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import threading

import numpy as np
import pytest
from fitsbolt.cfg.create_config import create_config as fb_create_cfg
from fitsbolt.normalisation.NormalisationMethod import NormalisationMethod

from anomaly_match.utils.get_default_cfg import get_default_cfg

pytestmark = pytest.mark.slow

# Subprocess needs repo root (for anomaly_match) and subprocess_scripts/
# (for prediction_utils) on PYTHONPATH — CI doesn't install the package.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SUBPROCESS_ENV = {
    **os.environ,
    "PYTHONPATH": os.pathsep.join([_REPO_ROOT, os.path.join(_REPO_ROOT, "subprocess_scripts")]),
}


@pytest.fixture
def training_config():
    """Create a minimal training config using test-cnn for speed."""
    cfg = get_default_cfg()
    cfg.normalisation.image_size = [64, 64]
    cfg.normalisation.n_output_channels = 3
    cfg.net = "test-cnn"
    cfg.pretrained = False
    cfg.num_channels = 3
    cfg.model_path = None
    cfg.gpu = 0
    cfg.output_dir = tempfile.mkdtemp()
    cfg.save_dir = tempfile.mkdtemp()
    cfg.normalisation.normalisation_method = NormalisationMethod.CONVERSION_ONLY
    cfg.log_level = "INFO"
    cfg.name = "test_training"
    cfg.seed = 42
    cfg.test_ratio = 0.0
    cfg.data_dir = "tests/test_data/grayscale/"
    cfg.label_file = "tests/test_data/grayscale/labeled_data.csv"
    cfg.num_workers = 0
    cfg.num_train_iter = 3  # Very few iterations for fast testing
    cfg.batch_size = 4
    cfg.uratio = 2

    cfg.fitsbolt_cfg = fb_create_cfg(
        output_dtype=np.uint8,
        size=cfg.normalisation.image_size,
        fits_extension=cfg.normalisation.fits_extension,
        interpolation_order=cfg.normalisation.interpolation_order,
        normalisation_method=cfg.normalisation.normalisation_method,
        channel_combination=cfg.normalisation.channel_combination,
        num_workers=max(cfg.num_workers, 1),
        norm_maximum_value=cfg.normalisation.norm_maximum_value,
        norm_minimum_value=cfg.normalisation.norm_minimum_value,
        norm_log_calculate_minimum_value=cfg.normalisation.norm_log_calculate_minimum_value,
        norm_crop_for_maximum_value=cfg.normalisation.norm_crop_for_maximum_value,
        norm_asinh_scale=cfg.normalisation.norm_asinh_scale,
        norm_asinh_clip=cfg.normalisation.norm_asinh_clip,
    )

    return cfg


def test_training_process_runs_to_completion(training_config, tmp_path):
    """Full subprocess run: config pickle → train → save → progress.jsonl."""
    cfg = training_config

    # Serialize config
    config_path = str(tmp_path / "config.pkl")
    with open(config_path, "wb") as f:
        pickle.dump(cfg.toDict(), f)

    # Labels CSV
    labels_csv = str(tmp_path / "labels.csv")
    shutil.copy("tests/test_data/grayscale/labeled_data.csv", labels_csv)

    # Progress file
    progress_file = str(tmp_path / "progress.jsonl")

    # Run the subprocess
    script = os.path.join("subprocess_scripts", "training_process.py")
    result = subprocess.run(
        [sys.executable, script, config_path, labels_csv, progress_file],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=_REPO_ROOT,
        env=_SUBPROCESS_ENV,
    )

    assert result.returncode == 0, (
        f"Training failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    )

    # Verify progress file has entries
    assert os.path.isfile(progress_file)
    with open(progress_file) as f:
        lines = [json.loads(line) for line in f if line.strip()]

    # Should have loading, training iterations, saving, and done entries
    statuses = [line["status"] for line in lines]
    assert "loading" in statuses
    assert "training" in statuses
    assert "done" in statuses

    # Final entry should have model_path
    done_entry = [entry for entry in lines if entry["status"] == "done"][-1]
    assert "model_path" in done_entry
    assert os.path.isfile(done_entry["model_path"])
    assert done_entry["elapsed"] > 0


def test_training_subprocess_with_ui_pipe_setup(tmp_path):
    """Replicate exact UI launch: stdout=DEVNULL, stderr=PIPE, efficientnet-lite0.

    This catches pipe-buffer hangs and thread-contention issues that only
    manifest when the subprocess is launched the way BackendInterface does.
    Verifies all expected outputs: progress file, model checkpoint, labels
    CSV, and training log in the session directory.
    """
    cfg = get_default_cfg()
    cfg.net = "efficientnet-lite0"
    cfg.pretrained = True
    cfg.num_train_iter = 3
    cfg.batch_size = 4
    cfg.uratio = 2
    cfg.num_workers = 0
    cfg.log_level = "INFO"

    # Set up session-like output directory with iteration subfolder,
    # mirroring what BackendInterface.launch_training_subprocess creates.
    session_dir = str(tmp_path / "session")
    iter_dir = os.path.join(session_dir, "iteration_0")
    os.makedirs(iter_dir)
    cfg.output_dir = session_dir
    cfg.save_dir = session_dir
    cfg.model_path = os.path.join(iter_dir, "model.safetensors")

    config_path = str(tmp_path / "config.pkl")
    with open(config_path, "wb") as f:
        pickle.dump(cfg.toDict(), f)

    labels_csv = str(tmp_path / "labels.csv")
    shutil.copy("tests/test_data/grayscale/labeled_data.csv", labels_csv)

    progress_file = str(tmp_path / "progress.jsonl")
    script = os.path.join("subprocess_scripts", "training_process.py")

    # Launch exactly like BackendInterface: stdout=DEVNULL, stderr=PIPE
    proc = subprocess.Popen(
        [sys.executable, script, config_path, labels_csv, progress_file],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        cwd=_REPO_ROOT,
        env=_SUBPROCESS_ENV,
    )

    # Drain stderr like the UI's _stderr_reader_loop does
    stderr_lines = []

    def drain_stderr():
        for raw_line in iter(proc.stderr.readline, b""):
            stderr_lines.append(raw_line.decode(errors="replace").rstrip())

    reader = threading.Thread(target=drain_stderr, daemon=True)
    reader.start()

    try:
        proc.wait(timeout=90)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        pytest.fail(
            "Training subprocess timed out (90s). Last stderr:\n" + "\n".join(stderr_lines[-10:])
        )

    assert proc.returncode == 0, f"Training failed (exit {proc.returncode}):\n" + "\n".join(
        stderr_lines[-20:]
    )

    # ── Verify progress file ─────────────────────────────────────────
    assert os.path.isfile(progress_file)
    with open(progress_file) as f:
        lines = [json.loads(line) for line in f if line.strip()]
    statuses = [entry["status"] for entry in lines]
    assert "loading" in statuses
    assert "training" in statuses
    assert "saving" in statuses
    assert "done" in statuses

    done_entry = [e for e in lines if e["status"] == "done"][-1]
    assert done_entry["elapsed"] > 0

    # ── Verify model checkpoint in iteration subdirectory ──────────
    model_path = done_entry["model_path"]
    assert os.path.isfile(model_path), f"Model checkpoint missing: {model_path}"
    assert iter_dir in model_path, (
        f"Model saved outside iteration dir: {model_path} not in {iter_dir}"
    )

    # ── Verify labels CSV in iteration subdirectory ──────────────────
    labels_out = os.path.join(iter_dir, "labelled_data.csv")
    assert os.path.isfile(labels_out), f"Labels CSV missing: {labels_out}"

    # ── Verify training log in iteration subdirectory ────────────────
    training_log = os.path.join(iter_dir, "training.log")
    assert os.path.isfile(training_log), f"Training log missing: {training_log}"
    with open(training_log) as f:
        log_content = f.read()
    assert "Training complete" in log_content, (
        f"Training log missing completion marker. Content:\n{log_content[-500:]}"
    )

    # Iteration dir should contain exactly these files (no stale prediction.log)
    iter_files = set(os.listdir(iter_dir))
    assert "prediction.log" not in iter_files, (
        "Training subprocess should not create prediction.log"
    )
    expected = {"model.safetensors", "labelled_data.csv", "training.log"}
    assert expected.issubset(iter_files), (
        f"Missing files in iteration dir. Expected {expected}, got {iter_files}"
    )


def test_training_process_creates_logs(training_config, tmp_path):
    """Verify the training subprocess creates log files."""
    cfg = training_config

    config_path = str(tmp_path / "config.pkl")
    with open(config_path, "wb") as f:
        pickle.dump(cfg.toDict(), f)

    labels_csv = str(tmp_path / "labels.csv")
    shutil.copy("tests/test_data/grayscale/labeled_data.csv", labels_csv)

    progress_file = str(tmp_path / "progress.jsonl")
    script = os.path.join("subprocess_scripts", "training_process.py")

    subprocess.run(
        [sys.executable, script, config_path, labels_csv, progress_file],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=_REPO_ROOT,
        env=_SUBPROCESS_ENV,
    )

    # Check log files were created in scripts/logs/
    logs_dir = os.path.join("scripts", "logs")
    if os.path.isdir(logs_dir):
        log_files = [f for f in os.listdir(logs_dir) if f.startswith("training_")]
        assert len(log_files) > 0, "Expected training log files in scripts/logs/"
