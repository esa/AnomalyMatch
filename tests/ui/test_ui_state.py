#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Tests for ``anomaly_match_ui.utils.ui_state`` last-browsed dir persistence."""

import json

import numpy as np
import pytest

from anomaly_match.utils.validate_config import serialisable_config
from anomaly_match_ui.utils import ui_state

pytestmark = pytest.mark.ui


@pytest.fixture()
def isolated_state(tmp_path):
    """Path of the state file the autouse conftest fixture redirects ui_state to."""
    return tmp_path / "ui_state.json"


def test_get_returns_none_when_no_state_file(isolated_state):
    assert ui_state.get_last_browsed_dir("model_chooser") is None


def test_set_then_get_round_trip(isolated_state, tmp_path):
    target = tmp_path / "models"
    target.mkdir()

    ui_state.set_last_browsed_dir("model_chooser", str(target))

    assert ui_state.get_last_browsed_dir("model_chooser") == str(target)


def test_set_with_file_path_stores_parent_dir(isolated_state, tmp_path):
    """Selecting a file persists its parent directory, not the file path."""
    folder = tmp_path / "labels"
    folder.mkdir()
    csv_file = folder / "run42.csv"
    csv_file.write_text("filename,label\n", encoding="utf-8")

    ui_state.set_last_browsed_dir("label_chooser", str(csv_file))

    assert ui_state.get_last_browsed_dir("label_chooser") == str(folder)


def test_get_returns_none_for_stale_path(isolated_state, tmp_path):
    """A persisted directory that no longer exists is treated as absent."""
    target = tmp_path / "since_deleted"
    target.mkdir()
    ui_state.set_last_browsed_dir("source_chooser", str(target))
    target.rmdir()

    assert ui_state.get_last_browsed_dir("source_chooser") is None


def test_different_purposes_are_independent(isolated_state, tmp_path):
    a = tmp_path / "a"
    a.mkdir()
    b = tmp_path / "b"
    b.mkdir()

    ui_state.set_last_browsed_dir("model_chooser", str(a))
    ui_state.set_last_browsed_dir("source_chooser", str(b))

    assert ui_state.get_last_browsed_dir("model_chooser") == str(a)
    assert ui_state.get_last_browsed_dir("source_chooser") == str(b)


def test_corrupt_state_file_is_ignored(isolated_state):
    """Garbage in the JSON file must not crash the read path."""
    isolated_state.parent.mkdir(parents=True, exist_ok=True)
    isolated_state.write_text("{not valid json", encoding="utf-8")

    assert ui_state.get_last_browsed_dir("model_chooser") is None


def test_state_file_records_schema_version(isolated_state, tmp_path):
    target = tmp_path / "x"
    target.mkdir()
    ui_state.set_last_browsed_dir("model_chooser", str(target))

    data = json.loads(isolated_state.read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
    assert data["last_browsed_dirs"]["model_chooser"] == str(target)


# ── Normalisation settings ───────────────────────────────────────


def test_normalisation_returns_none_when_nothing_recorded(isolated_state):
    assert ui_state.get_normalisation_settings("training_setup") is None


def test_normalisation_round_trip(isolated_state):
    settings = {"normalisation_method": 3, "image_size": [224, 224], "channel_combination": None}
    ui_state.set_normalisation_settings("training_setup", settings, ["VIS", "NIR-H"])

    record = ui_state.get_normalisation_settings("training_setup")
    assert record["settings"] == settings
    assert record["extensions"] == ["VIS", "NIR-H"]


def test_normalisation_matrix_round_trips_through_json(isolated_state):
    """A serialised matrix reaches disk as nested lists and comes back intact."""
    matrix = serialisable_config({"channel_combination": np.eye(2)})["channel_combination"]
    ui_state.set_normalisation_settings(
        "training_setup", {"channel_combination": matrix}, ["VIS", "NIR-H"]
    )

    on_disk = json.loads(isolated_state.read_text(encoding="utf-8"))
    stored = on_disk["normalisation_settings"]["training_setup"]["settings"]
    assert stored["channel_combination"] == [[1.0, 0.0], [0.0, 1.0]]
    assert (
        ui_state.get_normalisation_settings("training_setup")["settings"]["channel_combination"]
        == stored["channel_combination"]
    )


def test_normalisation_purposes_are_independent(isolated_state):
    ui_state.set_normalisation_settings("training_setup", {"image_size": [64, 64]}, ["R"])
    ui_state.set_normalisation_settings("prediction_setup", {"image_size": [128, 128]}, ["G"])

    assert ui_state.get_normalisation_settings("training_setup")["settings"]["image_size"] == [
        64,
        64,
    ]
    assert ui_state.get_normalisation_settings("prediction_setup")["extensions"] == ["G"]


@pytest.mark.parametrize(
    "record",
    ["not-a-dict", {"settings": "nope", "extensions": []}, {"settings": {}}],
)
def test_malformed_normalisation_record_is_ignored(isolated_state, record):
    isolated_state.parent.mkdir(parents=True, exist_ok=True)
    isolated_state.write_text(
        json.dumps({"schema_version": 1, "normalisation_settings": {"training_setup": record}}),
        encoding="utf-8",
    )

    assert ui_state.get_normalisation_settings("training_setup") is None


def test_unserialisable_value_leaves_the_existing_state_intact(isolated_state, tmp_path):
    """A bad write must not cost the user the settings they already had.

    ``set_normalisation_settings`` is a plain JSON writer, so a caller that
    forgets ``serialisable_config`` hands it a value no encoder handles. Writing
    into the real file would truncate it at the first unencodable value and take
    the previously remembered settings — and the chooser directories sharing the
    file — down with it.
    """
    target = tmp_path / "models"
    target.mkdir()
    ui_state.set_last_browsed_dir("model_chooser", str(target))
    ui_state.set_normalisation_settings("training_setup", {"image_size": [64, 64]}, ["R"])

    ui_state.set_normalisation_settings("training_setup", {"matrix": object()}, ["R"])

    assert ui_state.get_last_browsed_dir("model_chooser") == str(target)
    assert ui_state.get_normalisation_settings("training_setup")["settings"] == {
        "image_size": [64, 64]
    }
    assert json.loads(isolated_state.read_text(encoding="utf-8"))["schema_version"] == 1
    assert not list(isolated_state.parent.glob("*.tmp"))
