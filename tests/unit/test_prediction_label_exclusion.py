#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Tests for excluding already-labelled training data from prediction scoring.

Labelled sources are filtered out of the unlabeled training pool but were still
re-scored at prediction time, so they reappeared at the top of the score gallery
once the model trained on them.  These tests cover the shared filter primitives
(``filter_unprocessed_batch`` exclusion + ``load_excluded_label_ids``) that the
three prediction subprocesses use to skip them.
"""

import pandas as pd
import pytest

import anomaly_match as am
from prediction_utils import (
    basename_exclusions,
    filter_unprocessed_batch,
    load_excluded_label_ids,
)


class _FakeDB:
    """Minimal stand-in exposing only the resume query the filter needs."""

    def __init__(self, already_scored):
        self._scored = set(already_scored)

    def get_unprocessed(self, keys):
        return [k for k in keys if k not in self._scored]


def test_filter_without_exclude_is_pure_resume():
    db = _FakeDB(already_scored={"b"})
    keep, n_skipped, n_excluded = filter_unprocessed_batch(["a", "b", "c"], db)
    assert keep == [True, False, True]
    assert n_skipped == 1
    assert n_excluded == 0


def test_filter_drops_excluded_keys():
    db = _FakeDB(already_scored={"b"})
    # "a" is labelled (excluded), "b" is already scored, "c" survives.
    keep, n_skipped, n_excluded = filter_unprocessed_batch(["a", "b", "c"], db, exclude={"a"})
    assert keep == [False, False, True]
    assert n_skipped == 1
    assert n_excluded == 1


def test_filter_separates_resume_and_label_counts():
    """A fresh run that only excludes labels must not report them as resume skips."""
    db = _FakeDB(already_scored=set())
    keep, n_skipped, n_excluded = filter_unprocessed_batch(["a", "b", "c"], db, exclude={"a", "b"})
    assert keep == [False, False, True]
    assert n_skipped == 0
    assert n_excluded == 2


def test_filter_resume_takes_precedence_over_exclude():
    db = _FakeDB(already_scored={"a"})
    # "a" is both already-scored and labelled — counted as a resume skip only.
    keep, n_skipped, n_excluded = filter_unprocessed_batch(["a", "c"], db, exclude={"a"})
    assert keep == [False, True]
    assert n_skipped == 1
    assert n_excluded == 0


def test_empty_exclude_set_keeps_resume_only():
    db = _FakeDB(already_scored=set())
    keep, n_skipped, n_excluded = filter_unprocessed_batch(["a", "b"], db, exclude=set())
    assert keep == [True, True]
    assert n_skipped == 0
    assert n_excluded == 0


def test_basename_exclusions_maps_labels_into_path_space():
    # Folder DB / resume keys are full paths; labels are bare basenames.
    keys = ["/data/search/img_1.png", "/data/search/img_2.png", "/data/search/img_3.png"]
    exclude = basename_exclusions(keys, {"img_2.png"})
    assert exclude == {"/data/search/img_2.png"}


def test_basename_exclusions_empty_when_nothing_labelled():
    keys = ["/data/search/img_1.png"]
    assert basename_exclusions(keys, set()) == set()


def test_folder_excluded_basename_drops_full_path_key():
    """End-to-end folder path: a labelled basename removes its full-path key
    from the scored set via basename_exclusions + filter_unprocessed_batch."""
    keys = ["/data/search/a.png", "/data/search/b.png", "/data/search/c.png"]
    db = _FakeDB(already_scored=set())
    exclude = basename_exclusions(keys, {"b.png"})

    keep, n_skipped, n_excluded = filter_unprocessed_batch(keys, db, exclude=exclude)

    kept_paths = [k for k, keepit in zip(keys, keep) if keepit]
    assert kept_paths == ["/data/search/a.png", "/data/search/c.png"]
    assert n_skipped == 0
    assert n_excluded == 1


def test_load_excluded_label_ids_reads_csv_ids(tmp_path):
    label_path = tmp_path / "labels.csv"
    pd.DataFrame({"id": ["123", "456"], "label": ["anomaly", "normal"]}).to_csv(
        label_path, index=False
    )
    cfg = am.get_default_cfg()
    cfg.label_file = str(label_path)

    excluded = load_excluded_label_ids(cfg)

    assert excluded == {"123", "456"}


def test_load_excluded_label_ids_stringifies_numeric_ids(tmp_path):
    """Cutana SourceIDs are integers in the CSV but compared as strings."""
    label_path = tmp_path / "labels.csv"
    pd.DataFrame({"id": [123, 456], "label": ["anomaly", "normal"]}).to_csv(label_path, index=False)
    cfg = am.get_default_cfg()
    cfg.label_file = str(label_path)

    excluded = load_excluded_label_ids(cfg)

    assert excluded == {"123", "456"}


def test_load_excluded_label_ids_empty_when_no_label_file():
    cfg = am.get_default_cfg()
    cfg.label_file = ""
    assert load_excluded_label_ids(cfg) == set()


def test_load_excluded_label_ids_raises_when_configured_file_missing(tmp_path):
    """A configured-but-missing label file must fail hard, not silently return
    an empty set — a stale/typo'd path would otherwise let labelled sources
    score again with no error, reintroducing the bug this guards against."""
    cfg = am.get_default_cfg()
    cfg.label_file = str(tmp_path / "does_not_exist.csv")
    with pytest.raises(FileNotFoundError):
        load_excluded_label_ids(cfg)
