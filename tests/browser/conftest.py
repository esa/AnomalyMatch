#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Voila + Playwright fixtures for browser-based E2E testing.

Architecture:
    - A session-scoped Voila server serves a generated notebook that creates
      the AnomalyMatch UI with test-cnn configuration.
    - Each test gets a fresh Playwright page. Navigating to the Voila URL
      spawns a new Jupyter kernel, giving clean test isolation.
"""

import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright

from anomaly_match_ui.utils import ui_state
from anomaly_match_ui.utils.backend_interface import BackendInterface

VIDEO_DIR = Path(__file__).parent.parent.parent / "test-results"

TEST_DATA_DIR = Path(__file__).parent.parent / "test_data" / "grayscale"
_AM_THREAD_PREFIX = "am-"


# ========== Helpers ==========


def _find_free_port():
    """Find a free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _create_notebook(path, cells):
    """Write a minimal .ipynb with the given code cells.

    Args:
        path: Output notebook path.
        cells: List of code strings, one per cell.
    """
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": code.splitlines(keepends=True),
            }
            for code in cells
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    with open(path, "w") as f:
        json.dump(notebook, f)


def _wait_for_server(url, timeout=30):
    """Poll until the HTTP server at *url* responds."""
    import urllib.request

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status < 400:
                    return
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"Server at {url} did not start within {timeout}s")


def _join_am_threads(timeout: float) -> None:
    """Join all daemon threads spawned by AnomalyMatch screens."""
    deadline = time.monotonic() + timeout
    main = threading.main_thread()
    for t in threading.enumerate():
        if t is main or not t.is_alive():
            continue
        if not (t.name or "").startswith(_AM_THREAD_PREFIX):
            continue
        remaining = max(0.1, deadline - time.monotonic())
        t.join(timeout=remaining)


# ========== Notebook generation ==========


def _make_app_notebook_code(
    data_dir, save_dir, label_file="", model_path="", prediction_search_dir=""
):
    """Return Python code for the test notebook.

    The notebook creates a Session with test-cnn config and starts the UI.
    All paths are baked into the notebook so both training and prediction
    setup screens auto-validate without FileChooser interaction.
    """
    # Use forward slashes for cross-platform path compatibility in the notebook
    data_dir_s = str(data_dir).replace("\\", "/")
    save_dir_s = str(save_dir).replace("\\", "/")
    label_file_s = str(label_file).replace("\\", "/") if label_file else ""
    model_path_s = str(model_path).replace("\\", "/") if model_path else ""
    prediction_search_dir_s = (
        str(prediction_search_dir).replace("\\", "/") if prediction_search_dir else ""
    )

    return f"""\
import sys, os
# Ensure the repo root is on sys.path for editable installs / subprocess_scripts
sys.path.insert(0, {str(Path.cwd()).replace(chr(92), "/")!r})
os.chdir({str(Path.cwd()).replace(chr(92), "/")!r})

import anomaly_match as am
from anomaly_match_ui.app import start_ui

cfg = am.get_default_cfg()
cfg.data_dir = {data_dir_s!r}
cfg.label_file = {label_file_s!r}
cfg.save_dir = {save_dir_s!r}
cfg.normalisation.image_size = [64, 64]
cfg.normalisation.n_output_channels = 3
cfg.net = "test-cnn"
cfg.pretrained = False
cfg.num_train_iter = 2
cfg.num_workers = 0
cfg.test_ratio = 0.5
cfg.model_path = {model_path_s!r}
cfg.prediction_search_dir = {prediction_search_dir_s!r}

session = am.Session(cfg)
app = start_ui(session)
"""


# ========== Voila server ==========


@pytest.fixture(scope="session")
def ui_state_dir(tmp_path_factory):
    """Throwaway ``ANOMALYMATCH_CONFIG_DIR`` the served app persists UI state into.

    Session-scoped because the Voila server reads the env var once, at launch;
    :func:`_reset_persisted_normalisation` keeps tests from inheriting each
    other's writes.

    Isolation is what makes the suite runnable on a developer machine at all:
    ``TrainingSetupScreen._initial_dir`` lets a persisted last-browsed directory
    win over ``cfg.data_dir`` (#429), so the real ``~/.config/anomalymatch``
    would open the chooser somewhere other than ``TEST_DATA_DIR`` and the
    "Found N images" assertions would time out.
    """
    return tmp_path_factory.mktemp("am_ui_state")


@pytest.fixture(autouse=True)
def _reset_persisted_normalisation(ui_state_dir):
    """Drop remembered normalisation settings before each test.

    The setup screen restores them on build, so a test that starts a run would
    otherwise change the starting state of every test after it and make the
    suite order-dependent.  Persistence still survives page loads *within* a
    test, which is what the cross-kernel test asserts on.
    """
    state_path = ui_state_dir / "ui_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state.pop("normalisation_settings", None)
        state_path.write_text(json.dumps(state), encoding="utf-8")


@pytest.fixture(scope="session")
def voila_server(tmp_path_factory, test_model_path, ui_state_dir):
    """Start a Voila server serving the AnomalyMatch UI notebook.

    The server is started once per test session and shared across all browser
    tests. Each page load in Playwright spawns a fresh Jupyter kernel in
    Voila, so tests get clean isolation without server restart overhead.
    """
    notebook_dir = tmp_path_factory.mktemp("voila_notebooks")
    notebook_path = notebook_dir / "test_app.ipynb"
    save_dir = tmp_path_factory.mktemp("am_save")

    code = _make_app_notebook_code(
        data_dir=TEST_DATA_DIR,
        save_dir=save_dir,
        label_file=TEST_DATA_DIR / "labeled_data.csv",
        model_path=test_model_path,
        prediction_search_dir=TEST_DATA_DIR,
    )
    _create_notebook(notebook_path, [code])

    port = _find_free_port()
    # The UI persists chooser dirs and normalisation settings on user actions;
    # point that at a tmp dir so the suite neither reads nor writes the real
    # ``~/.config/anomalymatch``.  It outlives individual page loads, which is
    # what lets a test assert state carries across a fresh kernel.
    server_env = {**os.environ, ui_state.CONFIG_DIR_ENV_VAR: str(ui_state_dir)}
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "voila",
            str(notebook_path),
            f"--Voila.port={port}",
            "--no-browser",
            "--VoilaExecutor.timeout=120",
            "--show_tracebacks=True",
        ],
        # DEVNULL prevents pipe-buffer deadlock when kernels produce
        # lots of log output (training/scoring subprocesses).
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        # Kernels inherit this, so the UI persists chooser state into the
        # throwaway dir instead of the developer's real config.
        env=server_env,
    )

    base_url = f"http://localhost:{port}"
    try:
        _wait_for_server(base_url, timeout=30)
    except RuntimeError:
        proc.kill()
        raise RuntimeError(f"Voila server failed to start on port {port}.")

    yield base_url

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


# ========== Playwright ==========


@pytest.fixture(scope="session")
def _playwright_instance():
    """Manage the Playwright lifecycle (session-scoped)."""
    with sync_playwright() as p:
        yield p


@pytest.fixture(scope="session")
def browser(_playwright_instance):
    """Launch a headless Chromium browser (session-scoped for speed)."""
    b = _playwright_instance.chromium.launch()
    yield b
    b.close()


@pytest.fixture()
def page(request, browser, voila_server):
    """Create a fresh page and navigate to the Voila-served app.

    Each page load triggers a fresh Jupyter kernel in Voila, so each test
    starts with a clean AnomalyMatch session at the main menu.
    Video is recorded and retained only on test failure.
    """
    video_dir = VIDEO_DIR / request.node.name
    video_dir.mkdir(parents=True, exist_ok=True)
    context = browser.new_context(record_video_dir=str(video_dir))
    pg = context.new_page()
    pg.goto(voila_server, wait_until="domcontentloaded", timeout=60_000)
    # Wait for ipywidgets to render — look for the main VBox container
    pg.wait_for_selector(".widget-vbox", timeout=60_000)
    yield pg
    pg.close()
    context.close()
    # Keep video only on failure
    if not hasattr(request.node, "rep_call") or not request.node.rep_call.failed:
        shutil.rmtree(video_dir, ignore_errors=True)


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Attach test outcome to the request node for video retention."""
    outcome = yield
    rep = outcome.get_result()
    setattr(item, f"rep_{rep.when}", rep)


# ========== Cleanup ==========


@pytest.fixture(autouse=True)
def _cleanup_prediction_monitor(ui_state_dir):
    """Ensure prediction DB, cache, threads and chooser state are torn down.

    The config dir is session-scoped, so a test that clicks Select in a chooser
    would otherwise persist that directory for every later test's fresh kernel —
    the same override this isolation exists to prevent, just intra-session.
    """
    yield
    BackendInterface.close_prediction_monitor()
    _join_am_threads(timeout=2.0)
    (ui_state_dir / "ui_state.json").unlink(missing_ok=True)
