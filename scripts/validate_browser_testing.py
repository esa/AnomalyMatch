#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Launch the AnomalyMatch UI in Voila for interactive testing.

Starts a Voila server with test-cnn configuration and test data, then
prints the URL. Use with Playwright MCP or a browser to interact.

The server writes its URL to ``tmp/voila_url.txt`` so Playwright MCP
(or scripts) can discover the port automatically.

Usage::

    python scripts/validate_browser_testing.py          # prints URL, waits
    python scripts/validate_browser_testing.py --open   # also opens browser
    python scripts/validate_browser_testing.py --port 8899  # fixed port
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
TEST_DATA_DIR = REPO_ROOT / "tests" / "test_data" / "grayscale"
TEST_MODEL_PATH = REPO_ROOT / "tests" / "test_data" / "test_model.safetensors"
URL_FILE = REPO_ROOT / "tmp" / "voila_url.txt"
# Chooser state this session persists. Kept out of the real
# ~/.config/anomalymatch so driving the UI here cannot leave a last-browsed
# directory behind that later overrides cfg.data_dir (#429) — in the app, and
# in the browser tests that share the same state file.
# Wiped on every launch (see main): merely redirecting the path would move #429
# into tmp/ rather than remove it, since yesterday's chooser directory would
# still win over the cfg.data_dir the launcher notebook sets.
UI_STATE_DIR = REPO_ROOT / "tmp" / "voila_session_ui_state"

# Run from anywhere: this script is a developer entry point, not part of a
# package, so the repo root is not necessarily importable yet.
sys.path.insert(0, str(REPO_ROOT))

from anomaly_match_ui.utils.ui_state import CONFIG_DIR_ENV_VAR
from tests.test_data.generate_test_model import ensure_test_model


def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _create_notebook(path, code):
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": code.splitlines(keepends=True),
            }
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    with open(path, "w") as f:
        json.dump(notebook, f)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--open", action="store_true", help="Open the URL in a browser")
    parser.add_argument("--port", type=int, default=0, help="Fixed port (0 = auto)")
    args = parser.parse_args()

    ensure_test_model(TEST_MODEL_PATH)

    save_dir = REPO_ROOT / "tmp" / "voila_session"
    save_dir.mkdir(parents=True, exist_ok=True)

    _s = lambda p: str(p).replace("\\", "/")  # noqa: E731

    code = f"""\
import sys, os
sys.path.insert(0, {_s(REPO_ROOT)!r})
os.chdir({_s(REPO_ROOT)!r})

import anomaly_match as am
from anomaly_match_ui.app import start_ui

cfg = am.get_default_cfg()
cfg.data_dir = {_s(TEST_DATA_DIR)!r}
cfg.label_file = {_s(TEST_DATA_DIR / "labeled_data.csv")!r}
cfg.save_dir = {_s(save_dir)!r}
cfg.normalisation.image_size = [64, 64]
cfg.normalisation.n_output_channels = 3
cfg.net = "test-cnn"
cfg.pretrained = False
cfg.num_train_iter = 2
cfg.num_workers = 0
cfg.test_ratio = 0.5
cfg.model_path = {_s(TEST_MODEL_PATH)!r}
cfg.prediction_search_dir = {_s(TEST_DATA_DIR)!r}

session = am.Session(cfg)
app = start_ui(session)
"""
    nb_dir = REPO_ROOT / "tmp"
    nb_dir.mkdir(parents=True, exist_ok=True)
    nb_path = nb_dir / "voila_session.ipynb"
    _create_notebook(nb_path, code)

    # Start from empty, not merely from "exists": this session must open on
    # cfg.data_dir, never on wherever the previous session last browsed.
    shutil.rmtree(UI_STATE_DIR, ignore_errors=True)
    UI_STATE_DIR.mkdir(parents=True, exist_ok=True)

    port = args.port or _find_free_port()
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "voila",
            str(nb_path),
            f"--Voila.port={port}",
            "--no-browser",
            "--VoilaExecutor.timeout=120",
            "--show_tracebacks=True",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        # Kernels inherit this, so chooser state lands in tmp/ rather than the
        # developer's real config.
        env={**os.environ, CONFIG_DIR_ENV_VAR: str(UI_STATE_DIR)},
    )

    url = f"http://localhost:{port}"
    print(f"Starting Voila on port {port}...")

    import urllib.request

    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status < 400:
                    break
        except Exception:
            pass
        time.sleep(0.5)
    else:
        proc.kill()
        raise RuntimeError("Voila server failed to start")

    # Write URL to a well-known file so Playwright MCP can discover it
    URL_FILE.parent.mkdir(parents=True, exist_ok=True)
    URL_FILE.write_text(url)

    print(f"\nVoila ready: {url}")
    print(f"URL written to: {URL_FILE}")
    print("Press Ctrl+C to stop.\n")

    if args.open:
        webbrowser.open(url)

    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        proc.wait(timeout=5)
    finally:
        URL_FILE.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
