#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Shared fixtures for UI tests."""

import threading
import time

import pytest

from anomaly_match_ui.utils import ui_state
from anomaly_match_ui.utils.backend_interface import BackendInterface

_AM_THREAD_PREFIX = "am-"


@pytest.fixture(autouse=True)
def _isolated_ui_state(monkeypatch, tmp_path):
    """Point ui_state at tmp_path so tests never read or write the real ~/.config.

    Screens consult persisted state while building (last-browsed dirs,
    remembered normalisation), so without this a developer's own state file
    would change what the tests observe.
    """
    monkeypatch.setenv(ui_state.CONFIG_DIR_ENV_VAR, str(tmp_path))


@pytest.fixture(autouse=True)
def _cleanup_prediction_monitor():
    """Ensure prediction DB, cache, and threads are torn down after every UI test.

    Without this, a test failure can leave daemon threads polling a
    SQLite connection that gets garbage-collected during interpreter
    shutdown, causing a segfault (exit code 139) on Linux CI.
    """
    yield
    BackendInterface.close_prediction_monitor()
    _join_am_threads(timeout=2.0)


def _join_am_threads(timeout: float) -> None:
    """Join long-lived AnomalyMatch threads spawned by screens.

    Targets threads named with the ``am-`` (dash) prefix — currently
    just ``am-poll``.  ``ThreadPoolExecutor`` workers spawned by the
    gallery screens use an underscore prefix (``am`` / ``am_prefetch``)
    on purpose: those workers idle in ``queue.get()`` between
    submissions and would eat the full join timeout per test if matched
    here.  The poll thread is the only one that holds a SQLite
    connection across iterations and so the only one that risked the
    interpreter-shutdown segfault this fixture was added to prevent.
    """
    deadline = time.monotonic() + timeout
    main = threading.main_thread()
    for t in threading.enumerate():
        if t is main or not t.is_alive():
            continue
        if not (t.name or "").startswith(_AM_THREAD_PREFIX):
            continue
        remaining = max(0.1, deadline - time.monotonic())
        t.join(timeout=remaining)
