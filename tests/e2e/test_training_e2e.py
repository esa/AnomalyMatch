#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""End-to-end test for the full active-learning cycle: train → score → retrain.

Exercises the real BackendInterface + Session to:
1. Initialize a session
2. Launch training subprocess → wait for model checkpoint
3. Verify iteration output (model, labels, log)
4. Retrain (second iteration) → verify new iteration directory

Marked @slow because it trains a real model (test-cnn, 3 iterations each).
"""

import json
import os
import pickle
import shutil
import subprocess
import sys
import threading

import numpy as np
import pandas as pd
import pytest
from fitsbolt.cfg.create_config import create_config as fb_create_cfg
from fitsbolt.normalisation.NormalisationMethod import NormalisationMethod

from anomaly_match.datasets.Label import LABEL_ANOMALY, LABEL_NORMAL
from anomaly_match.pipeline.session import Session
from anomaly_match.utils.get_default_cfg import get_default_cfg
from anomaly_match_ui.utils.backend_interface import BackendInterface

pytestmark = pytest.mark.slow

# Subprocess needs repo root + subprocess_scripts on PYTHONPATH
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SUBPROCESS_ENV = {
    **os.environ,
    "PYTHONPATH": os.pathsep.join([_REPO_ROOT, os.path.join(_REPO_ROOT, "subprocess_scripts")]),
}


@pytest.fixture()
def session_with_config(tmp_path):
    """Create a real Session with test-cnn config for e2e testing."""
    cfg = get_default_cfg()
    cfg.normalisation.image_size = [64, 64]
    cfg.normalisation.n_output_channels = 3
    cfg.net = "test-cnn"
    cfg.pretrained = False
    cfg.num_channels = 3
    cfg.gpu = 0
    cfg.output_dir = str(tmp_path / "output")
    cfg.save_dir = str(tmp_path / "sessions")
    cfg.normalisation.normalisation_method = NormalisationMethod.CONVERSION_ONLY
    cfg.log_level = "WARNING"
    cfg.name = "e2e_train"
    cfg.seed = 42
    cfg.test_ratio = 0.0
    cfg.data_dir = "tests/test_data/grayscale/"
    cfg.label_file = "tests/test_data/grayscale/labeled_data.csv"
    cfg.num_workers = 0
    cfg.num_train_iter = 3  # Very few for speed
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

    session = Session(cfg)
    BackendInterface.set_session(session)
    yield session
    BackendInterface.set_session(None)


def _run_training_subprocess(cfg, labels_csv, tmp_path, timeout=120):
    """Launch training subprocess and wait for it to complete.

    Returns the (progress_lines, iter_dir) tuple.
    """
    os.makedirs(tmp_path, exist_ok=True)
    config_path = str(tmp_path / "config.pkl")
    with open(config_path, "wb") as f:
        pickle.dump(cfg.toDict(), f)

    progress_file = str(tmp_path / "progress.jsonl")
    script = os.path.join("subprocess_scripts", "training_process.py")

    proc = subprocess.Popen(
        [sys.executable, script, config_path, labels_csv, progress_file],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        cwd=_REPO_ROOT,
        env=_SUBPROCESS_ENV,
    )

    stderr_lines = []

    def drain_stderr():
        for raw_line in iter(proc.stderr.readline, b""):
            stderr_lines.append(raw_line.decode(errors="replace").rstrip())

    reader = threading.Thread(target=drain_stderr, daemon=True)
    reader.start()

    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        pytest.fail(
            f"Training subprocess timed out ({timeout}s). "
            f"Last stderr:\n" + "\n".join(stderr_lines[-10:])
        )

    assert proc.returncode == 0, f"Training failed (exit {proc.returncode}):\n" + "\n".join(
        stderr_lines[-20:]
    )

    # Parse progress
    assert os.path.isfile(progress_file)
    with open(progress_file) as f:
        lines = [json.loads(line) for line in f if line.strip()]

    iter_dir = os.path.dirname(cfg.model_path)
    return lines, iter_dir


def test_full_training_cycle(session_with_config, tmp_path):
    """Complete train → verify outputs → retrain cycle."""
    session = session_with_config
    cfg = session.cfg

    # ── Iteration 0: First training ────────────────────────────
    iteration = session.session_tracker.start_new_session_iteration()
    assert iteration == 0

    iter_dir_0 = os.path.join(cfg.output_dir, f"iteration_{iteration}")
    os.makedirs(iter_dir_0, exist_ok=True)
    cfg.model_path = os.path.join(iter_dir_0, "model.safetensors")

    labels_csv_0 = str(tmp_path / "labels_0.csv")
    shutil.copy("tests/test_data/grayscale/labeled_data.csv", labels_csv_0)

    lines_0, _ = _run_training_subprocess(cfg, labels_csv_0, tmp_path / "run_0")

    # Verify progress statuses
    statuses = [line["status"] for line in lines_0]
    assert "loading" in statuses
    assert "training" in statuses
    assert "done" in statuses

    # Verify done entry
    done_entry = [e for e in lines_0 if e["status"] == "done"][-1]
    assert done_entry["elapsed"] > 0
    assert "model_path" in done_entry

    # Verify iteration_0 outputs
    model_path = done_entry["model_path"]
    assert os.path.isfile(model_path), f"Model missing: {model_path}"

    labels_out = os.path.join(iter_dir_0, "labelled_data.csv")
    assert os.path.isfile(labels_out), f"Labels missing: {labels_out}"

    training_log = os.path.join(iter_dir_0, "training.log")
    assert os.path.isfile(training_log), f"Training log missing: {training_log}"
    with open(training_log) as f:
        log_content = f.read()
    assert "Training complete" in log_content

    # ── Simulate labeling (add new labels) ─────────────────────
    new_labels = {"Abell2390_VIS_7523.jpeg": LABEL_ANOMALY}
    merged_path = session.session_io.merge_gallery_labels(
        existing_label_file=labels_csv_0,
        new_labels=new_labels,
        output_path=str(tmp_path / "merged_labels.csv"),
    )

    # Verify merge added the new label
    merged_df = pd.read_csv(merged_path)
    assert "Abell2390_VIS_7523.jpeg" in merged_df["id"].values

    # ── Iteration 1: Retrain with updated labels ───────────────
    iteration = session.session_tracker.start_new_session_iteration()
    assert iteration == 1

    iter_dir_1 = os.path.join(cfg.output_dir, f"iteration_{iteration}")
    os.makedirs(iter_dir_1, exist_ok=True)
    cfg.model_path = os.path.join(iter_dir_1, "model.safetensors")
    cfg.label_file = merged_path

    lines_1, _ = _run_training_subprocess(cfg, merged_path, tmp_path / "run_1")

    # Verify iteration_1 outputs
    statuses_1 = [line["status"] for line in lines_1]
    assert "done" in statuses_1

    done_entry_1 = [e for e in lines_1 if e["status"] == "done"][-1]
    assert os.path.isfile(done_entry_1["model_path"])
    assert os.path.isfile(os.path.join(iter_dir_1, "labelled_data.csv"))
    assert os.path.isfile(os.path.join(iter_dir_1, "training.log"))

    # ── Verify both iterations exist independently ─────────────
    assert os.path.isdir(iter_dir_0)
    assert os.path.isdir(iter_dir_1)
    assert iter_dir_0 != iter_dir_1

    # Both should have model checkpoints
    assert os.path.isfile(os.path.join(iter_dir_0, "model.safetensors"))
    assert os.path.isfile(os.path.join(iter_dir_1, "model.safetensors"))

    # Session tracker should reflect both iterations
    info = session.get_session_info()
    assert info["total_session_iterations"] == 2


def test_backend_launch_training(session_with_config, tmp_path):
    """Verify BackendInterface.launch_training_subprocess creates correct structure."""
    session = session_with_config

    proc, temp_dir, progress_file = BackendInterface.launch_training_subprocess(
        num_train_iter=3,
    )

    assert proc is not None
    assert os.path.isdir(temp_dir)
    assert os.path.isfile(os.path.join(temp_dir, "config.pkl"))
    assert os.path.isfile(os.path.join(temp_dir, "labelled_data.csv"))
    assert progress_file.endswith("progress.jsonl")

    # Verify iteration directory was created
    iter_dir = os.path.join(session.cfg.output_dir, "iteration_0")
    assert os.path.isdir(iter_dir)

    # Wait for subprocess to complete
    stderr_lines = []

    def drain():
        for raw in iter(proc.stderr.readline, b""):
            stderr_lines.append(raw.decode(errors="replace").rstrip())

    reader = threading.Thread(target=drain, daemon=True)
    reader.start()

    try:
        proc.wait(timeout=120)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        pytest.fail("Timed out. Stderr:\n" + "\n".join(stderr_lines[-10:]))

    assert proc.returncode == 0, f"Training failed (exit {proc.returncode}):\n" + "\n".join(
        stderr_lines[-20:]
    )

    # Verify outputs
    assert os.path.isfile(os.path.join(iter_dir, "model.safetensors"))
    assert os.path.isfile(os.path.join(iter_dir, "labelled_data.csv"))
    assert os.path.isfile(os.path.join(iter_dir, "training.log"))


def test_label_merge_across_iterations(session_with_config, tmp_path):
    """Verify label merging works correctly across retrain cycles."""
    session = session_with_config

    # Start with initial labels
    initial_csv = str(tmp_path / "initial.csv")
    shutil.copy("tests/test_data/grayscale/labeled_data.csv", initial_csv)

    initial_df = pd.read_csv(initial_csv)
    initial_count = len(initial_df)

    # Simulate iteration 0: add new labels
    new_labels_0 = {
        "new_anomaly.png": LABEL_ANOMALY,
        "new_nominal.png": LABEL_NORMAL,
    }
    merged_0 = session.session_io.merge_gallery_labels(
        existing_label_file=initial_csv,
        new_labels=new_labels_0,
        output_path=str(tmp_path / "merged_0.csv"),
    )

    df_0 = pd.read_csv(merged_0)
    assert len(df_0) == initial_count + 2

    # Simulate iteration 1: override one label, add another
    new_labels_1 = {
        "new_anomaly.png": LABEL_NORMAL,  # Changed from anomaly → normal
        "another.png": LABEL_ANOMALY,
    }
    merged_1 = session.session_io.merge_gallery_labels(
        existing_label_file=merged_0,
        new_labels=new_labels_1,
        output_path=str(tmp_path / "merged_1.csv"),
    )

    df_1 = pd.read_csv(merged_1)
    assert len(df_1) == initial_count + 3  # +2 from iter 0, +1 new from iter 1

    # Verify the override
    overridden = df_1[df_1["id"] == "new_anomaly.png"]
    assert len(overridden) == 1
    assert overridden.iloc[0]["label"] == LABEL_NORMAL
