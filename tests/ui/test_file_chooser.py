#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Tests for AMFileChooser, including last-browsed directory persistence (#429)."""

import pytest

from anomaly_match_ui.utils import ui_state
from anomaly_match_ui.widgets.file_chooser import AMFileChooser

pytestmark = pytest.mark.ui


@pytest.fixture(autouse=True)
def isolated_state(monkeypatch, tmp_path):
    monkeypatch.setenv(ui_state.CONFIG_DIR_ENV_VAR, str(tmp_path))
    yield tmp_path / "ui_state.json"


def test_no_purpose_does_not_persist_anything(isolated_state, tmp_path):
    """When no ``purpose`` is supplied the chooser does not own a persistence
    callback and therefore cannot accidentally write to ui_state."""
    chooser = AMFileChooser(path=str(tmp_path))

    # Sanity check: without a purpose we record nothing on the wrapper.
    assert chooser._purpose == ""
    # And the wrapper does not expose any persistence helper attached to
    # this instance — only the ones with a purpose write through.
    assert not hasattr(chooser, "_purpose_callback_active") or not chooser._purpose
    assert not isolated_state.exists()


def test_purpose_persists_parent_dir_on_selection(isolated_state, tmp_path):
    """A chooser with ``purpose`` writes the parent dir of every confirmed selection."""
    folder = tmp_path / "browse"
    folder.mkdir()
    csv_file = folder / "labels.csv"
    csv_file.write_text("filename,label\n", encoding="utf-8")

    chooser = AMFileChooser(path=str(tmp_path), purpose="label_chooser")

    # Drive the persistence callback the same way the FileChooser does.
    chooser._file_chooser._selected_path = str(folder)
    chooser._file_chooser._selected_filename = "labels.csv"
    chooser._persist_browsed_dir(chooser._file_chooser)

    assert ui_state.get_last_browsed_dir("label_chooser") == str(folder)


def test_purpose_no_op_when_nothing_selected(isolated_state, tmp_path):
    """Persistence is skipped when ``selected`` is None — no crash, no write."""
    chooser = AMFileChooser(path=str(tmp_path), purpose="model_chooser")

    chooser._file_chooser._selected_path = None
    chooser._file_chooser._selected_filename = None
    chooser._persist_browsed_dir(chooser._file_chooser)

    assert not isolated_state.exists()


def test_register_callback_does_not_overwrite_persistence(isolated_state, tmp_path):
    """Regression test for #429.

    ``ipyfilechooser.FileChooser.register_callback`` *replaces* its
    callback rather than appending — so before this fix every screen
    that called ``register_callback`` clobbered the persistence hook
    and the last-browsed dir was never recorded.  The wrapper now
    fans out to a list of callbacks, so both fire.
    """
    folder = tmp_path / "browse"
    folder.mkdir()

    chooser = AMFileChooser(path=str(tmp_path), purpose="source_chooser")

    screen_calls: list[object] = []
    chooser.register_callback(lambda fc: screen_calls.append(fc))

    chooser._file_chooser._selected_path = str(folder)
    chooser._file_chooser._selected_filename = ""
    chooser._dispatch_callbacks(chooser._file_chooser)

    # Persistence happened (the persistence hook is in the callback list).
    assert ui_state.get_last_browsed_dir("source_chooser") == str(folder)
    # Screen-side callback also fired.
    assert len(screen_calls) == 1


def test_callback_failure_does_not_block_other_callbacks(isolated_state, tmp_path):
    """One callback raising must not stop the rest from running."""
    chooser = AMFileChooser(path=str(tmp_path), purpose="source_chooser")

    def _raises(_fc: object) -> None:
        raise RuntimeError("boom")

    chooser.register_callback(_raises)
    later_called: list[object] = []
    chooser.register_callback(lambda fc: later_called.append(fc))

    folder = tmp_path / "later"
    folder.mkdir()
    chooser._file_chooser._selected_path = str(folder)
    chooser._file_chooser._selected_filename = ""
    chooser._dispatch_callbacks(chooser._file_chooser)

    assert len(later_called) == 1
    # Persistence still ran (it's first in the dispatch order).
    assert ui_state.get_last_browsed_dir("source_chooser") == str(folder)
