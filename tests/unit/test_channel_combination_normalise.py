#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Tests for channel-combination weight normalisation (no silent clipping)."""

import numpy as np
import pytest
from loguru import logger

import anomaly_match as am
from anomaly_match.data_io import load_images
from anomaly_match.data_io.load_images import (
    _get_channel_combination_array,
    _warn_channel_combination_once,
    normalise_channel_combination,
)


def _advisories(channel_combination):
    """Capture the advisories the decode path logs for a matrix.

    The UI advisory was removed (a log warning suffices); the prediction path
    logs the same messages via :func:`_warn_channel_combination_once` from the
    :func:`normalise_channel_combination` results.  This drives that path and
    returns the logged advisory text, with the dedup guard cleared first so each
    call observes its own warnings.
    """
    if channel_combination is None:
        return []
    load_images._warned_channel_combinations.clear()
    captured: list[str] = []
    sink_id = logger.add(lambda m: captured.append(m.record["message"]), level="WARNING")
    try:
        _, rescaled_rows, has_negative = normalise_channel_combination(channel_combination)
        _warn_channel_combination_once(channel_combination, rescaled_rows, has_negative)
    finally:
        logger.remove(sink_id)
    prefix = "channel_combination: "
    return [message[len(prefix) :] for message in captured if message.startswith(prefix)]


def test_none_passes_through():
    matrix, rescaled, has_negative = normalise_channel_combination(None)
    assert matrix is None and rescaled == [] and has_negative is False


@pytest.mark.parametrize(
    "cc",
    [
        np.eye(3, dtype=np.float32),  # identity
        np.ones((3, 1), dtype=np.float32),  # single-band broadcast
        np.array([[0.5, 0.5], [0.25, 0.75]], dtype=np.float32),  # convex rows
    ],
)
def test_convex_and_identity_unchanged(cc):
    matrix, rescaled, has_negative = normalise_channel_combination(cc)
    assert rescaled == []
    assert has_negative is False
    np.testing.assert_array_equal(matrix, cc)


def test_overflowing_row_rescaled_to_sum_one():
    cc = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.5, 0.0], [0.0, 0.0, 0.5, 1.0]], np.float32)
    matrix, rescaled, has_negative = normalise_channel_combination(cc)

    assert rescaled == [1, 2]  # rows summing to 1.5
    assert has_negative is False
    np.testing.assert_allclose(matrix.sum(axis=1), [1.0, 1.0, 1.0], atol=1e-6)
    # relative weighting within each rescaled row is preserved
    np.testing.assert_allclose(matrix[1], [0.0, 1 / 1.5, 0.5 / 1.5, 0.0], atol=1e-6)
    np.testing.assert_allclose(matrix[2], [0.0, 0.0, 0.5 / 1.5, 1 / 1.5], atol=1e-6)
    # untouched row is bit-identical
    np.testing.assert_array_equal(matrix[0], cc[0])


def test_negative_weights_flagged_not_rescaled():
    cc = np.array([[1.5, -0.5]], dtype=np.float32)  # sum 1.0 but has a negative
    matrix, rescaled, has_negative = normalise_channel_combination(cc)
    assert rescaled == []
    assert has_negative is True
    np.testing.assert_array_equal(matrix, cc)


def test_get_channel_combination_array_returns_normalised():
    cfg = am.get_default_cfg()
    cfg.normalisation.channel_combination = [[2.0, 0.0], [0.0, 2.0]]  # rows sum to 2
    out = _get_channel_combination_array(cfg)
    np.testing.assert_allclose(out.sum(axis=1), [1.0, 1.0], atol=1e-6)


def test_advisories_silent_for_clip_safe():
    for cc in (None, np.eye(3, dtype=np.float32), np.array([[0.5, 0.5]], dtype=np.float32)):
        assert _advisories(cc) == []


def test_advisories_report_rescaled_rows_one_based():
    cc = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.5, 0.0], [0.0, 0.0, 0.5, 1.0]], np.float32)
    messages = _advisories(cc)
    assert len(messages) == 1
    # Rows are reported 1-based to match the widget's "Out N" labels.
    assert "2, 3" in messages[0]
    assert "rescaled" in messages[0].lower()


def test_advisories_flag_negative_weights():
    messages = _advisories(np.array([[1.5, -0.5]], dtype=np.float32))
    assert len(messages) == 1
    assert "negative" in messages[0].lower()


def test_advisories_flag_blank_row():
    # Row 1 has all-zero weights -> that output channel would be entirely blank.
    cc = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    messages = _advisories(cc)
    assert any("blank" in m.lower() for m in messages)
    assert any("1" in m for m in messages if "blank" in m.lower())
